use bytemuck::Zeroable;
use cgmath::{
    BaseNum, ElementWise, EuclideanSpace, MetricSpace, Point3, Vector2, Vector3, Vector4,
};
use half::f16;
use num_traits::Float;
use std::fmt::Debug;
use std::sync::Arc;
use egui::mutex::Mutex;

use std::{mem, thread};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GaussianCompressed {
    pub xyz: Point3<f32>,
    pub opacity: i8,
    pub scale_factor: i8,
    pub geometry_idx: u32,
    pub sh_idx: u32,
}
unsafe impl bytemuck::Zeroable for GaussianCompressed {}
unsafe impl bytemuck::Pod for GaussianCompressed {}

impl Default for GaussianCompressed {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    pub xyz: Point3<f32>,
    pub opacity: f16,
    _pad: f16,
    pub cov: [f16; 6],
}

unsafe impl bytemuck::Zeroable for Gaussian {}
unsafe impl bytemuck::Pod for Gaussian {}

impl Gaussian {
    pub fn new(xyz: Point3<f32>, opacity: f16, cov: [f16; 6]) -> Self {
        Self {
            xyz: xyz,
            opacity: opacity,
            cov: cov,
            _pad: f16::ZERO,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Covariance3D(pub [f16; 6]);

impl Default for Covariance3D {
    fn default() -> Self {
        Covariance3D::zeroed()
    }
}

pub struct PointCloudMetadata {
    pub num_points: usize,
    pub sh_deg: u32,
    pub center: Point3<f32>,
    pub up: Option<Vector3<f32>>,
    pub mip_splatting: Option<bool>,
    pub kernel_size: Option<f32>,
    pub background_color: Option<[f32; 3]>,
    pub quantization: Option<GaussianQuantization>,
}

struct DynamicPointCloudAttributes {
    valid_points: u32,
    bbox: Aabb<f32>,
    center: Point3<f32>,
    up: Option<Vector3<f32>>,
}

#[allow(dead_code)]
pub struct PointCloud {
    splat_2d_buffer: wgpu::Buffer,
    gaussian_buffer: wgpu::Buffer,
    sh_coef_buffer: wgpu::Buffer,

    bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    total_num_points: u32,
    sh_deg: u32,
    compressed: bool,

    dynamic_attributes: Arc<Mutex<DynamicPointCloudAttributes>>,
    mip_splatting: Option<bool>,
    kernel_size: Option<f32>,
    background_color: Option<wgpu::Color>,
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.total_num_points)
            .finish()
    }
}

impl PointCloud {
    pub fn new(device: &wgpu::Device, metadata: PointCloudMetadata) -> Result<Self, anyhow::Error> {
        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d gaussians buffer"),
            size: (metadata.num_points * mem::size_of::<Splat>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud rendering bind group"),
            layout: &Self::bind_group_layout_render(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 2,
                resource: splat_2d_buffer.as_entire_binding(),
            }],
        });

        let gaussian_buffer_size = (metadata.num_points * mem::size_of::<Gaussian>()) as u64;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3d gaussians buffer"),
            size: gaussian_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sh_coef_buffer_size = (metadata.num_points * mem::size_of::<[[f16; 3]; 16]>()) as u64;
        let sh_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sh coefs buffer"),
            size: sh_coef_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sh_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: splat_2d_buffer.as_entire_binding(),
            },
        ];

        // let bind_group = if pc.compressed() {
        //     let covars_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //         label: Some("Covariances buffer"),
        //         contents: bytemuck::cast_slice(pc.covars.as_ref().unwrap().as_slice()),
        //         usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //     });
        //     let quantization_uniform = UniformBuffer::new(
        //         device,
        //         pc.quantization.unwrap(),
        //         Some("quantization uniform buffer"),
        //     );
        //     bind_group_entries.push(wgpu::BindGroupEntry {
        //         binding: 3,
        //         resource: covars_buffer.as_entire_binding(),
        //     });
        //     bind_group_entries.push(wgpu::BindGroupEntry {
        //         binding: 4,
        //         resource: quantization_uniform.buffer().as_entire_binding(),
        //     });

        //     device.create_bind_group(&wgpu::BindGroupDescriptor {
        //         label: Some("point cloud bind group (compressed)"),
        //         layout: &Self::bind_group_layout_compressed(device),
        //         entries: &bind_group_entries,
        //     })
        // } else {
        //     device.create_bind_group(&wgpu::BindGroupDescriptor {
        //         label: Some("point cloud bind group"),
        //         layout: &Self::bind_group_layout(device),
        //         entries: &bind_group_entries,
        //     })
        // };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud bind group"),
            layout: &Self::bind_group_layout(device),
            entries: &bind_group_entries,
        });

        Ok(Self {
            splat_2d_buffer,
            gaussian_buffer: vertex_buffer,
            sh_coef_buffer: sh_buffer,
            bind_group,
            render_bind_group,
            total_num_points: metadata.num_points as u32,
            sh_deg: metadata.sh_deg,
            compressed: metadata.quantization.is_some(),
            dynamic_attributes: Arc::new(Mutex::new(DynamicPointCloudAttributes {
                valid_points: 0,
                bbox: Aabb::unit(),
                center: metadata.center,
                up: metadata.up,
            })),
            mip_splatting: metadata.mip_splatting,
            kernel_size: metadata.kernel_size,
            background_color: metadata.background_color.map(|c| wgpu::Color {
                r: c[0] as f64,
                g: c[1] as f64,
                b: c[2] as f64,
                a: 1.,
            }),
        })
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }

    pub fn total_num_points(&self) -> u32 {
        self.total_num_points
    }

    pub fn valid_num_points(&self) -> u32 {
        self.dynamic_attributes.lock().valid_points
    }

    pub fn sh_deg(&self) -> u32 {
        self.sh_deg
    }

    pub fn bbox(&self) -> Aabb<f32> {
        self.dynamic_attributes.lock().bbox
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
    pub(crate) fn render_bind_group(&self) -> &wgpu::BindGroup {
        &self.render_bind_group
    }

    pub fn bind_group_layout_compressed(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud bind group layout (compressed)"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud float bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn bind_group_layout_render(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud rendering bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub fn mip_splatting(&self) -> Option<bool> {
        self.mip_splatting
    }

    pub fn dilation_kernel_size(&self) -> Option<f32> {
        self.kernel_size
    }

    pub fn center(&self) -> Point3<f32> {
        self.dynamic_attributes.lock().center
    }

    pub fn up(&self) -> Option<Vector3<f32>> {
        self.dynamic_attributes.lock().up
    }

    pub fn upload_chunk(
        &self,
        gaussians: &[Gaussian],
        sh_coefs: &[[[f16; 3]; 16]],
        queue: &wgpu::Queue,
        bbox: &Aabb<f32>,
    ) {
        assert_eq!(
            gaussians.len(),
            sh_coefs.len(),
            "Number of gaussians and sh coefficients must match"
        );
        let new_points = gaussians.len() as u32;
        let offset = self.valid_num_points() as u64;
        queue.write_buffer(
            &self.gaussian_buffer,
            offset * std::mem::size_of::<Gaussian>() as u64,
            bytemuck::cast_slice(gaussians),
        );
        queue.write_buffer(
            &self.sh_coef_buffer,
            offset * std::mem::size_of::<[[f16; 3]; 16]>() as u64,
            bytemuck::cast_slice(sh_coefs),
        );
        {
            let mut attributes = self.dynamic_attributes.lock();
            attributes.bbox.grow_union(bbox);
            attributes.valid_points += new_points;
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat {
    pub v: Vector4<f16>,
    pub pos: Vector2<f16>,
    pub color: Vector4<f16>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Quantization {
    pub zero_point: i32,
    pub scale: f32,
    _pad: [u32; 2],
}

impl Quantization {
    pub fn new(zero_point: i32, scale: f32) -> Self {
        Quantization {
            zero_point,
            scale,
            ..Default::default()
        }
    }
}

impl Default for Quantization {
    fn default() -> Self {
        Self {
            zero_point: 0,
            scale: 1.,
            _pad: [0, 0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GaussianQuantization {
    pub color_dc: Quantization,
    pub color_rest: Quantization,
    pub opacity: Quantization,
    pub scaling_factor: Quantization,
    pub scaling: Quantization,
    pub rotation: Quantization,
}

#[repr(C)]
#[derive(Zeroable, Clone, Copy, Debug, PartialEq)]
pub struct Aabb<F: Float + BaseNum> {
    pub min: Point3<F>,
    pub max: Point3<F>,
}

impl<F: Float + BaseNum> Aabb<F> {
    pub fn new(min: Point3<F>, max: Point3<F>) -> Self {
        Self { min, max }
    }

    pub fn grow(&mut self, pos: &Point3<F>) {
        self.min.x = self.min.x.min(pos.x);
        self.min.y = self.min.y.min(pos.y);
        self.min.z = self.min.z.min(pos.z);

        self.max.x = self.max.x.max(pos.x);
        self.max.y = self.max.y.max(pos.y);
        self.max.z = self.max.z.max(pos.z);
    }

    pub fn corners(&self) -> [Point3<F>; 8] {
        [
            Vector3::new(F::zero(), F::zero(), F::zero()),
            Vector3::new(F::one(), F::zero(), F::zero()),
            Vector3::new(F::zero(), F::one(), F::zero()),
            Vector3::new(F::one(), F::one(), F::zero()),
            Vector3::new(F::zero(), F::zero(), F::one()),
            Vector3::new(F::one(), F::zero(), F::one()),
            Vector3::new(F::zero(), F::one(), F::one()),
            Vector3::new(F::one(), F::one(), F::one()),
        ]
        .map(|d| self.min + self.max.to_vec().mul_element_wise(d))
    }

    pub fn unit() -> Self {
        Self {
            min: Point3::new(-F::one(), -F::one(), -F::one()),
            max: Point3::new(F::one(), F::one(), F::one()),
        }
    }

    pub fn center(&self) -> Point3<F> {
        self.min.midpoint(self.max)
    }

    /// radius of a sphere that contains the aabb
    pub fn radius(&self) -> F {
        self.min.distance(self.max) / (F::one() + F::one())
    }

    pub fn size(&self) -> Vector3<F> {
        self.max - self.min
    }

    pub fn grow_union(&mut self, other: &Aabb<F>) {
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);

        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
    }
}

impl Into<Aabb<f32>> for Aabb<f16> {
    fn into(self) -> Aabb<f32> {
        Aabb {
            min: self.min.map(|v| v.into()),
            max: self.max.map(|v| v.into()),
        }
    }
}
