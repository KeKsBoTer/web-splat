use bytemuck::Zeroable;
use cgmath::{Point3, Vector2, Vector4};
use half::f16;
use std::fmt::Debug;
use std::io::{BufReader, Read, Seek};
use std::mem;
use wgpu::util::DeviceExt;

#[cfg(feature = "npz")]
use crate::npz::NpzReader;
use crate::ply::PlyReader;
use crate::uniform::UniformBuffer;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianSplat {
    pub xyz: Point3<f16>,
    pub opacity: i8,
    pub scale_factor: i8,
    pub geometry_idx: u32,
    pub sh_idx: u32,
}

impl Default for GaussianSplat {
    fn default() -> Self {
        GaussianSplat::zeroed()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GeometricInfo {
    pub covariance: [f16; 6],
}
impl Default for GeometricInfo {
    fn default() -> Self {
        GeometricInfo::zeroed()
    }
}

#[allow(dead_code)]
pub struct PointCloud {
    splat_2d_buffer: wgpu::Buffer,

    bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
    num_points: u32,
    sh_deg: u32,
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.num_points)
            .finish()
    }
}

impl PointCloud {
    #[cfg(feature = "npz")]
    pub fn load_npz<R: Read + Seek>(device: &wgpu::Device, f: R) -> Result<Self, anyhow::Error> {
        let mut reader = BufReader::new(f);
        let mut npz_reader = NpzReader::new(&mut reader)?;
        let (file_sh_deg, num_points) = (npz_reader.file_sh_deg()?, npz_reader.num_points()?);
        let sh_deg = file_sh_deg;
        log::info!("num_points: {num_points}, sh_deg: {sh_deg}");

        let bind_group_layout = Self::bind_group_layout(device);

        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d gaussians buffer"),
            size: (num_points * mem::size_of::<Splat2D>()) as u64,
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

        let (vertices, sh_coefs, covars, quantization) = npz_reader.read(sh_deg)?;

        let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh coefs buffer"),
            contents: bytemuck::cast_slice(sh_coefs.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let covars_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Covariances buffer"),
            contents: bytemuck::cast_slice(covars.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3d gaussians buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let quantization_uniform =
            UniformBuffer::new(device, quantization, Some("quantization uniform buffer"));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud bind group"),
            layout: &bind_group_layout,
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
                    resource: covars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: splat_2d_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: quantization_uniform.buffer().as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            splat_2d_buffer,

            bind_group,
            render_bind_group,
            num_points: num_points as u32,
            sh_deg,
        })
    }

    pub fn load_ply<R: Read + Seek>(device: &wgpu::Device, f: R) -> Result<Self, anyhow::Error> {
        let reader = BufReader::new(f);

        let mut ply_reader = PlyReader::new(reader).unwrap();

        let sh_deg = ply_reader.file_sh_deg()?;
        let num_points = ply_reader.num_points()?;
        log::info!("num_points: {num_points}, sh_deg: {sh_deg}");

        let bind_group_layout = Self::bind_group_layout_float(device);

        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d gaussians buffer"),
            size: (num_points * mem::size_of::<Splat2D>()) as u64,
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

        let vertices = ply_reader.read()?;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3d gaussians buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splat_2d_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            splat_2d_buffer,

            bind_group,
            render_bind_group,
            num_points: num_points as u32,
            sh_deg,
        })
    }

    pub fn num_points(&self) -> u32 {
        self.num_points
    }

    pub fn sh_deg(&self) -> u32 {
        self.sh_deg
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud bind group layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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

    pub fn bind_group_layout_float(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat2D {
    v: Vector4<f16>,
    pos: Vector2<f16>,
    color: Vector4<u8>,
}

pub trait PointCloudReader {
    fn read(
        &mut self,
        sh_deg: u32,
    ) -> Result<
        (
            Vec<GaussianSplat>,
            Vec<u8>,
            Vec<GeometricInfo>,
            QuantizationUniform,
        ),
        anyhow::Error,
    >;

    fn file_sh_deg(&self) -> Result<u32, anyhow::Error>;

    fn num_points(&self) -> Result<usize, anyhow::Error>;
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
pub struct QuantizationUniform {
    pub color_dc: Quantization,
    pub color_rest: Quantization,
    pub opacity: Quantization,
    pub scaling_factor: Quantization,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianSplatFloat {
    pub xyz: Point3<f32>,
    pub opacity: f32,
    pub cov: [f32; 6],
    pub sh: [[f32; 3]; 16],
    pub _pad: [u32; 2],
}
