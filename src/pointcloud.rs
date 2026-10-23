use bytemuck::Zeroable;
use cgmath::{
    Array, BaseNum, ElementWise, EuclideanSpace, InnerSpace, MetricSpace, Point3, Vector2, Vector3,
    Vector4,
};
use half::f16;
use num_traits::Float;
use std::fmt::Debug;
use std::mem;
use wgpu::util::DeviceExt;

use crate::cpu_sort::CpuSorter;
use crate::io::{GaussianChunk, GenericGaussianPointCloud, PlyInfo};
use crate::renderer::supports_compute;
use crate::uniform::UniformBuffer;

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

#[allow(dead_code)]
pub struct PointCloud {
    /// projected gaussians, only needed by the compute based renderer
    splat_2d_buffer: Option<wgpu::Buffer>,
    /// 3d gaussians (and their sh coefficients), kept to append streamed chunks
    /// (only with compute shaders, without them the gaussians are in `cpu_sorted`)
    vertex_buffer: Option<wgpu::Buffer>,
    sh_buffer: Option<wgpu::Buffer>,
    /// the gaussians and their draw order, if the device has no compute shaders
    cpu_sorted: Option<CpuSortedGaussians>,

    /// bind groups with storage buffers, only available if the device has compute shaders
    bind_group: Option<wgpu::BindGroup>,
    render_bind_group: Option<wgpu::BindGroup>,
    /// total number of points of the point cloud (size of the buffers)
    num_points: u32,
    /// number of points that were uploaded so far, equal to `num_points` when fully loaded
    loaded_points: u32,
    plane_fit: PlaneFit,
    sh_deg: u32,
    bbox: Aabb<f32>,
    compressed: bool,

    center: Point3<f32>,
    up: Option<Vector3<f32>>,

    mip_splatting: Option<bool>,
    kernel_size: Option<f32>,
    background_color: Option<wgpu::Color>,
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.num_points)
            .finish()
    }
}

impl PointCloud {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pc: GenericGaussianPointCloud,
    ) -> Result<Self, anyhow::Error> {
        let compute = supports_compute(device);
        if !compute {
            let info = PlyInfo {
                num_points: pc.num_points,
                sh_deg: pc.sh_deg,
                mip_splatting: pc.mip_splatting,
                kernel_size: pc.kernel_size,
                background_color: pc.background_color,
            };
            let mut cloud = Self::new_streaming(device, &info)?;
            // (compressed point clouds are decoded, there are no compute shaders to do that)
            // In pieces: all points at once would need several times the memory of the point
            // cloud, which is too much for phones.
            const PIECE: usize = 1 << 16;
            for start in (0..pc.num_points).step_by(PIECE) {
                let end = (start + PIECE).min(pc.num_points);
                cloud.append(queue, &pc.to_chunk(start..end));
            }
            // the same box, center and up as with compute shaders
            cloud.bbox = pc.aabb.into();
            cloud.center = pc.center;
            cloud.up = pc.up;
            return Ok(cloud);
        }
        let (splat_2d_buffer, render_bind_group) =
            Self::create_splat_buffer(device, pc.num_points, compute);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3d gaussians buffer"),
            contents: pc.gaussian_buffer(),
            usage: Self::gaussian_usage(),
        });

        let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh coefs buffer"),
            contents: pc.sh_coefs_buffer(),
            usage: Self::gaussian_usage(),
        });

        let bind_group = splat_2d_buffer.as_ref().map(|splat_2d_buffer| {
            let mut bind_group_entries = vec![
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

            if pc.compressed() {
                let covars_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Covariances buffer"),
                    contents: bytemuck::cast_slice(pc.covars.as_ref().unwrap().as_slice()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                let quantization_uniform = UniformBuffer::new(
                    device,
                    pc.quantization.unwrap(),
                    Some("quantization uniform buffer"),
                );
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: covars_buffer.as_entire_binding(),
                });
                bind_group_entries.push(wgpu::BindGroupEntry {
                    binding: 4,
                    resource: quantization_uniform.buffer().as_entire_binding(),
                });

                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("point cloud bind group (compressed)"),
                    layout: &Self::bind_group_layout_compressed(device),
                    entries: &bind_group_entries,
                })
            } else {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("point cloud bind group"),
                    layout: &Self::bind_group_layout(device),
                    entries: &bind_group_entries,
                })
            }
        });

        Ok(Self {
            splat_2d_buffer,
            vertex_buffer: Some(vertex_buffer),
            sh_buffer: Some(sh_buffer),
            cpu_sorted: None,

            bind_group,
            render_bind_group,
            num_points: pc.num_points as u32,
            loaded_points: pc.num_points as u32,
            plane_fit: PlaneFit::default(),
            sh_deg: pc.sh_deg,
            compressed: pc.compressed(),
            bbox: pc.aabb.into(),
            center: pc.center,
            up: pc.up,
            mip_splatting: pc.mip_splatting,
            kernel_size: pc.kernel_size,
            background_color: pc.background_color.map(|c| wgpu::Color {
                r: c[0] as f64,
                g: c[1] as f64,
                b: c[2] as f64,
                a: 1.,
            }),
        })
    }

    /// The buffer for the projected gaussians and its bind group, only for the compute based
    /// renderer.
    fn create_splat_buffer(
        device: &wgpu::Device,
        num_points: usize,
        compute: bool,
    ) -> (Option<wgpu::Buffer>, Option<wgpu::BindGroup>) {
        if !compute {
            return (None, None);
        }
        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d gaussians buffer"),
            size: (num_points * mem::size_of::<Splat>()) as u64,
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
        (Some(splat_2d_buffer), Some(render_bind_group))
    }

    fn gaussian_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE
    }

    /// Creates an empty point cloud with buffers for `info.num_points` points, to be filled
    /// with [`Self::append`]. Only supports uncompressed point clouds.
    pub fn new_streaming(device: &wgpu::Device, info: &PlyInfo) -> Result<Self, anyhow::Error> {
        let compute = supports_compute(device);
        let (splat_2d_buffer, render_bind_group) =
            Self::create_splat_buffer(device, info.num_points, compute);
        let (vertex_buffer, sh_buffer, bind_group, cpu_sorted) = if compute {
            let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("3d gaussians buffer"),
                size: (info.num_points * mem::size_of::<Gaussian>()) as u64,
                usage: Self::gaussian_usage(),
                mapped_at_creation: false,
            });
            let sh_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sh coefs buffer"),
                size: (info.num_points * mem::size_of::<[[f16; 3]; 16]>()) as u64,
                usage: Self::gaussian_usage(),
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("point cloud bind group"),
                layout: &Self::bind_group_layout(device),
                entries: &[
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
                        resource: splat_2d_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            });
            (Some(vertex_buffer), Some(sh_buffer), Some(bind_group), None)
        } else {
            (
                None,
                None,
                None,
                Some(CpuSortedGaussians::new(device, info.num_points)?),
            )
        };
        Ok(Self {
            splat_2d_buffer,
            vertex_buffer,
            sh_buffer,
            cpu_sorted,
            bind_group,
            render_bind_group,
            num_points: info.num_points as u32,
            loaded_points: 0,
            plane_fit: PlaneFit::default(),
            sh_deg: info.sh_deg,
            // same start as GenericGaussianPointCloud, so both end up with the same box
            bbox: Aabb::zeroed(),
            compressed: false,
            center: Point3::new(0., 0., 0.),
            up: None,
            mip_splatting: info.mip_splatting,
            kernel_size: info.kernel_size,
            background_color: info.background_color.map(|c| wgpu::Color {
                r: c[0] as f64,
                g: c[1] as f64,
                b: c[2] as f64,
                a: 1.,
            }),
        })
    }

    /// Uploads the next chunk of points. Returns the number of points that were added.
    /// Points beyond the declared `num_points` are dropped.
    pub fn append(&mut self, queue: &wgpu::Queue, chunk: &GaussianChunk) -> u32 {
        let remaining = (self.num_points - self.loaded_points) as usize;
        let n = chunk.gaussians.len().min(remaining);
        if n == 0 {
            return 0;
        }
        let gaussians = &chunk.gaussians[..n];
        if let (Some(vertex_buffer), Some(sh_buffer)) = (&self.vertex_buffer, &self.sh_buffer) {
            queue.write_buffer(
                vertex_buffer,
                (self.loaded_points as usize * mem::size_of::<Gaussian>()) as u64,
                bytemuck::cast_slice(gaussians),
            );
            queue.write_buffer(
                sh_buffer,
                (self.loaded_points as usize * mem::size_of::<[[f16; 3]; 16]>()) as u64,
                bytemuck::cast_slice(&chunk.sh_coefs[..n]),
            );
        }
        if let Some(cpu_sorted) = &mut self.cpu_sorted {
            cpu_sorted.append(queue, self.loaded_points as usize, chunk, n);
        }
        for g in gaussians {
            self.bbox.grow(&g.xyz);
            self.plane_fit.add(g.xyz);
        }
        self.loaded_points += n as u32;
        self.center = self.plane_fit.centroid();
        if self.is_complete() {
            self.up = self
                .plane_fit
                .finish()
                .1
                .filter(|_| self.bbox.radius() >= 10.);
        }
        n as u32
    }

    /// whether all points are loaded
    pub fn is_complete(&self) -> bool {
        self.loaded_points == self.num_points
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }

    /// total number of points, including the ones that are not loaded yet
    pub fn num_points(&self) -> u32 {
        self.num_points
    }

    /// number of points that are available for rendering
    pub fn loaded_points(&self) -> u32 {
        self.loaded_points
    }

    pub fn sh_deg(&self) -> u32 {
        self.sh_deg
    }

    pub fn bbox(&self) -> &Aabb<f32> {
        &self.bbox
    }

    /// only available with compute shaders
    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        self.bind_group
            .as_ref()
            .expect("the point cloud bind group needs compute shaders")
    }
    /// only available with compute shaders
    pub(crate) fn render_bind_group(&self) -> &wgpu::BindGroup {
        self.render_bind_group
            .as_ref()
            .expect("the point cloud bind group needs compute shaders")
    }

    /// Updates the draw order of the gaussians (if they are sorted on the CPU, without compute
    /// shaders): takes the result of the last sort and starts a new one if the camera or the
    /// points changed. `view` is the third row of the view matrix, see
    /// [`crate::sort::DepthSorter::sort`]. Returns whether the draw order changed.
    pub fn update_sort(&mut self, queue: &wgpu::Queue, view: [f32; 4]) -> bool {
        let loaded_points = self.loaded_points;
        let complete = self.is_complete();
        match &mut self.cpu_sorted {
            Some(cpu_sorted) => cpu_sorted.update(queue, view, loaded_points, complete),
            None => false,
        }
    }

    /// the resources to draw the gaussians if they are sorted on the CPU
    pub(crate) fn cpu_sorted(&self) -> Option<&CpuSortedGaussians> {
        self.cpu_sorted.as_ref()
    }

    pub fn bind_group_layout_data(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gaussian data texture bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        })
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
        self.center
    }

    pub fn up(&self) -> Option<Vector3<f32>> {
        self.up
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
    #[cfg(feature = "npz")]
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

/// Fits a plane to a stream of points, see [`Self::finish`].
/// Accumulates in f64 relative to the first point, so points can be added in chunks.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlaneFit {
    n: u64,
    origin: [f64; 3],
    sum: [f64; 3],
    /// sums of xx, xy, xz, yy, yz, zz
    sum_prod: [f64; 6],
}

impl PlaneFit {
    pub fn add(&mut self, p: Point3<f32>) {
        let p = [p.x as f64, p.y as f64, p.z as f64];
        if self.n == 0 {
            self.origin = p;
        }
        let r = [
            p[0] - self.origin[0],
            p[1] - self.origin[1],
            p[2] - self.origin[2],
        ];
        self.n += 1;
        for i in 0..3 {
            self.sum[i] += r[i];
        }
        self.sum_prod[0] += r[0] * r[0];
        self.sum_prod[1] += r[0] * r[1];
        self.sum_prod[2] += r[0] * r[2];
        self.sum_prod[3] += r[1] * r[1];
        self.sum_prod[4] += r[1] * r[2];
        self.sum_prod[5] += r[2] * r[2];
    }

    pub fn centroid(&self) -> Point3<f32> {
        if self.n == 0 {
            return Point3::new(0., 0., 0.);
        }
        let n = self.n as f64;
        Point3::new(
            (self.origin[0] + self.sum[0] / n) as f32,
            (self.origin[1] + self.sum[1] / n) as f32,
            (self.origin[2] + self.sum[2] / n) as f32,
        )
    }

    /// Returns the centroid and the normal of the fitted plane, pointing upwards (+y).
    /// The normal is `None` if the points do not span a plane.
    /// Fast, and accurate to within a few degrees.
    /// see http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
    pub fn finish(&self) -> (Point3<f32>, Option<Vector3<f32>>) {
        let centroid = self.centroid();
        if self.n < 3 {
            return (centroid, None);
        }
        let n = self.n as f64;
        let m = [self.sum[0] / n, self.sum[1] / n, self.sum[2] / n];
        // covariance matrix, excluding symmetries
        let xx = (self.sum_prod[0] / n - m[0] * m[0]) as f32;
        let xy = (self.sum_prod[1] / n - m[0] * m[1]) as f32;
        let xz = (self.sum_prod[2] / n - m[0] * m[2]) as f32;
        let yy = (self.sum_prod[3] / n - m[1] * m[1]) as f32;
        let yz = (self.sum_prod[4] / n - m[1] * m[2]) as f32;
        let zz = (self.sum_prod[5] / n - m[2] * m[2]) as f32;

        let mut weighted_dir = Vector3::new(0., 0., 0.);

        {
            let det_x = yy * zz - yz * yz;
            let axis_dir = Vector3::new(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
            let mut weight = det_x * det_x;
            if weighted_dir.dot(axis_dir) < 0.0 {
                weight = -weight;
            }
            weighted_dir += axis_dir * weight;
        }
        {
            let det_y = xx * zz - xz * xz;
            let axis_dir = Vector3::new(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
            let mut weight = det_y * det_y;
            if weighted_dir.dot(axis_dir) < 0.0 {
                weight = -weight;
            }
            weighted_dir += axis_dir * weight;
        }
        {
            let det_z = xx * yy - xy * xy;
            let axis_dir = Vector3::new(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
            let mut weight = det_z * det_z;
            if weighted_dir.dot(axis_dir) < 0.0 {
                weight = -weight;
            }
            weighted_dir += axis_dir * weight;
        }

        let mut normal = weighted_dir.normalize();
        if normal.dot(Vector3::unit_y()) < 0. {
            normal = -normal;
        }
        if normal.is_finite() {
            (centroid, Some(normal))
        } else {
            (centroid, None)
        }
    }
}

/// Gaussians for rendering without compute shaders: the data is in a texture (storage buffers are
/// not available), drawn in the order of an index buffer that is sorted on the CPU in the
/// background (the sort is asynchronous, so the order is the one of an earlier camera).
pub(crate) struct CpuSortedGaussians {
    /// 8 texels (128 bytes) per gaussian, see `gaussian_vertex.wgsl`
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    points_per_row: usize,
    /// bytes of the last row of the texture if it is not full yet
    row_cache: Vec<u8>,

    /// the indices of the gaussians in draw order
    index_buffer: wgpu::Buffer,
    /// number of indices in `index_buffer` that are drawn
    visible: u32,
    /// whether the order comes from a sort, before it is the order of the points
    sorted: bool,

    sorter: CpuSorter,
    /// the view and the number of points of the last sort that was started
    requested: Option<([f32; 4], u32)>,
}

impl CpuSortedGaussians {
    const TEXELS_PER_GAUSSIAN: usize = 8;
    const BYTES_PER_TEXEL: usize = 16;

    fn new(device: &wgpu::Device, num_points: usize) -> Result<Self, anyhow::Error> {
        let max_size = device.limits().max_texture_dimension_2d as usize;
        let points_per_row = max_size.min(4096) / Self::TEXELS_PER_GAUSSIAN;
        let rows = num_points.div_ceil(points_per_row).max(1);
        if rows > max_size {
            anyhow::bail!(
                "{num_points} points do not fit into a texture of this device (max size {max_size})"
            );
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("gaussian data texture"),
            size: wgpu::Extent3d {
                width: (points_per_row * Self::TEXELS_PER_GAUSSIAN) as u32,
                height: rows as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gaussian data texture bind group"),
            layout: &PointCloud::bind_group_layout_data(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &texture.create_view(&Default::default()),
                ),
            }],
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gaussian indices"),
            size: (num_points.max(1) * mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            texture,
            bind_group,
            points_per_row,
            row_cache: Vec::new(),
            index_buffer,
            visible: 0,
            sorted: false,
            sorter: CpuSorter::new(),
            requested: None,
        })
    }

    /// uploads the points `start..start + n` of the chunk
    fn append(&mut self, queue: &wgpu::Queue, start: usize, chunk: &GaussianChunk, n: usize) {
        let row_bytes = self.points_per_row * Self::TEXELS_PER_GAUSSIAN * Self::BYTES_PER_TEXEL;
        let first_row = start / self.points_per_row;

        // the texture is written in rows, the points of the row that was not full yet are
        // written again
        let mut bytes = mem::take(&mut self.row_cache);
        bytes.reserve(n * Self::TEXELS_PER_GAUSSIAN * Self::BYTES_PER_TEXEL);
        for (gaussian, sh) in chunk.gaussians[..n].iter().zip(&chunk.sh_coefs[..n]) {
            bytes.extend_from_slice(bytemuck::bytes_of(gaussian));
            bytes.extend_from_slice(bytemuck::bytes_of(sh));
            bytes.extend_from_slice(&[0; 4]);
        }
        let full_rows = bytes.len() / row_bytes;
        let width = (self.points_per_row * Self::TEXELS_PER_GAUSSIAN) as u32;
        if full_rows > 0 {
            self.write_rows(
                queue,
                first_row,
                width,
                full_rows,
                &bytes[..full_rows * row_bytes],
            );
        }
        let rest = &bytes[full_rows * row_bytes..];
        if !rest.is_empty() {
            let width = (rest.len() / Self::BYTES_PER_TEXEL) as u32;
            self.write_rows(queue, first_row + full_rows, width, 1, rest);
        }
        self.row_cache = rest.to_vec();

        self.sorter.add_positions(
            chunk.gaussians[..n]
                .iter()
                .flat_map(|g| [g.xyz.x, g.xyz.y, g.xyz.z])
                .collect(),
        );

        if !self.sorted {
            // draw the points in their order until the first sort is there
            let indices: Vec<u32> = (start as u32..(start + n) as u32).collect();
            queue.write_buffer(
                &self.index_buffer,
                (start * mem::size_of::<u32>()) as u64,
                bytemuck::cast_slice(&indices),
            );
            self.visible = (start + n) as u32;
        }
    }

    fn write_rows(&self, queue: &wgpu::Queue, row: usize, width: u32, rows: usize, data: &[u8]) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: row as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * Self::BYTES_PER_TEXEL as u32),
                rows_per_image: Some(rows as u32),
            },
            wgpu::Extent3d {
                width,
                height: rows as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    fn update(
        &mut self,
        queue: &wgpu::Queue,
        view: [f32; 4],
        loaded_points: u32,
        complete: bool,
    ) -> bool {
        let mut changed = false;
        if let Some(indices) = self.sorter.poll() {
            queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
            self.visible = indices.len() as u32;
            self.sorted = true;
            changed = true;
        }
        // while points are streaming in, sorting all of them after every upload makes the sort
        // (and the transfer of its result) the bottleneck on slow devices. Sort again once enough
        // points arrived, and when everything is there.
        let (last_view, last_points) = self.requested.unwrap_or(([f32::NAN; 4], 0));
        let points_changed = loaded_points != last_points
            && (complete
                || self.requested.is_none()
                || loaded_points as u64 * 4 >= last_points as u64 * 5);
        if (last_view != view || points_changed)
            && self.requested != Some((view, loaded_points))
            && self.sorter.start(view)
        {
            self.requested = Some((view, loaded_points));
        }
        changed
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub(crate) fn index_buffer(&self) -> &wgpu::Buffer {
        &self.index_buffer
    }

    /// number of gaussians to draw
    pub(crate) fn visible(&self) -> u32 {
        self.visible
    }
}
