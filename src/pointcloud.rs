use bytemuck::Zeroable;
use cgmath::{Point3, Vector2, Vector4};
use half::f16;
use std::fmt::Debug;
use std::io::{self, BufReader, Read, Seek};
use std::mem;
use wgpu::util::DeviceExt;

use crate::gpu_rs::GPURSSorter;
#[cfg(feature = "npz")]
use crate::npz::NpzReader;
use crate::ply::PlyReader;
use crate::uniform::UniformBuffer;
use crate::utils::max_supported_sh_deg;

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

    // Fields needed for data sorting
    pub sorter: GPURSSorter,
    sorter_b_a: wgpu::Buffer,              // buffer a for keyval
    sorter_b_b: wgpu::Buffer,              // buffer b for keyval
    sorter_p_a: wgpu::Buffer,              // payload buffer a
    sorter_p_b: wgpu::Buffer,              // payload buffer b
    sorter_int: wgpu::Buffer,              // internal memory storage (used for histgoram calcs)
    pub sorter_uni: wgpu::Buffer,          // uniform buffer information
    pub sorter_dis: wgpu::Buffer,          // dispatch buffer
    pub sorter_dis_bg: wgpu::BindGroup, // sorter dispatch bind group (needed mainly for the preprocess pipeline to set the correct dispatch count in shader)
    pub sorter_bg: wgpu::BindGroup,     // sorter bind group
    pub sorter_render_bg: wgpu::BindGroup, // bind group only with the sorted indices for rendering
    pub sorter_bg_pre: wgpu::BindGroup, // bind group for the preprocess (is the sorter_dis and sorter_bg merged as we only have a limited amount of bgs for the preprocessing)
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.num_points)
            .finish()
    }
}

fn ply_header_info<R: io::BufRead + io::Seek>(buf_reader: &mut R) -> (u32, usize) {
    let reader = crate::ply::PlyReader::new(buf_reader).unwrap();
    (reader.file_sh_deg().unwrap(), reader.num_points().unwrap())
}
#[cfg(feature = "npz")]
fn npz_header_info<R: io::BufRead + io::Seek>(buf_reader: &mut R) -> (u32, usize) {
    let reader = crate::npz::NpzReader::new(buf_reader).unwrap();
    (reader.file_sh_deg().unwrap(), reader.num_points().unwrap())
}

impl PointCloud {
    #[cfg(feature = "npz")]
    pub fn load_npz<R: Read + Seek>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        f: R,
        max_sh_deg: Option<u32>,
    ) -> Result<Self, anyhow::Error> {
        let mut reader = BufReader::new(f);
        let (file_sh_deg, num_points) = npz_header_info(&mut reader);
        let max_buffer_size = device
            .limits()
            .max_buffer_size
            .max(device.limits().max_storage_buffer_binding_size as u64);

        let target_sh_deg = max_sh_deg.unwrap_or(file_sh_deg);
        let sh_deg = max_supported_sh_deg(max_buffer_size, num_points as u64, target_sh_deg)
            .ok_or(anyhow::anyhow!(
                "cannot fit sh coefs into a buffer (exceeds WebGPU's max_buffer_size)"
            ))?;
        log::info!("num_points: {num_points}, sh_deg: {sh_deg}");
        if sh_deg < target_sh_deg {
            log::warn!("color sh degree (file: {file_sh_deg}) was decreased to degree {sh_deg} to fit the coefficients into memory")
        }

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

        reader.seek(std::io::SeekFrom::Start(0)).unwrap(); // have to reset reader to start of file
        let (vertices, sh_coefs, covars, quantization) =
            NpzReader::new(&mut reader).unwrap().read(sh_deg)?;

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

        let sorter = GPURSSorter::new(device, queue);
        let (sorter_b_a, sorter_b_b, sorter_p_a, sorter_p_b) =
            GPURSSorter::create_keyval_buffers(device, num_points, 4);
        let sorter_int = sorter.create_internal_mem_buffer(device, num_points);
        let (sorter_uni, sorter_dis, sorter_bg, sorter_dis_bg) = sorter.create_bind_group(
            device,
            num_points,
            &sorter_int,
            &sorter_b_a,
            &sorter_b_b,
            &sorter_p_a,
            &sorter_p_b,
        );
        let sorter_render_bg = sorter.create_bind_group_render(device, &sorter_uni, &sorter_p_a);
        let sorter_bg_pre = sorter.create_bind_group_preprocess(
            device,
            &sorter_uni,
            &sorter_dis,
            &sorter_int,
            &sorter_b_a,
            &sorter_b_b,
            &sorter_p_a,
            &sorter_p_b,
        );

        Ok(Self {
            splat_2d_buffer,

            bind_group,
            render_bind_group,
            num_points: num_points as u32,
            sh_deg,
            sorter,
            sorter_b_a,
            sorter_b_b,
            sorter_p_a,
            sorter_p_b,
            sorter_int,
            sorter_uni,
            sorter_dis,
            sorter_dis_bg,
            sorter_bg,
            sorter_render_bg,
            sorter_bg_pre,
        })
    }

    pub fn load_ply<R: Read + Seek>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        f: R,
        max_sh_deg: Option<u32>,
    ) -> Result<Self, anyhow::Error> {
        let mut reader = BufReader::new(f);
        let (file_sh_deg, num_points) = ply_header_info(&mut reader);
        let max_buffer_size = device
            .limits()
            .max_buffer_size
            .max(device.limits().max_storage_buffer_binding_size as u64);

        let target_sh_deg = max_sh_deg.unwrap_or(file_sh_deg);
        let sh_deg = max_supported_sh_deg(max_buffer_size, num_points as u64, target_sh_deg)
            .ok_or(anyhow::anyhow!(
                "cannot fit sh coefs into a buffer (exceeds WebGPU's max_buffer_size)"
            ))?;
        log::info!("num_points: {num_points}, sh_deg: {sh_deg}");
        if sh_deg < target_sh_deg {
            log::warn!("color sh degree (file: {file_sh_deg}) was decreased to degree {sh_deg} to fit the coefficients into memory")
        }

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

        reader.seek(std::io::SeekFrom::Start(0)).unwrap(); // have to reset reader to start of file
        let vertices = PlyReader::new(reader).unwrap().read(sh_deg)?;

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

        let sorter = GPURSSorter::new(device, queue);
        let (sorter_b_a, sorter_b_b, sorter_p_a, sorter_p_b) =
            GPURSSorter::create_keyval_buffers(device, num_points, 4);
        let sorter_int = sorter.create_internal_mem_buffer(device, num_points);
        let (sorter_uni, sorter_dis, sorter_bg, sorter_dis_bg) = sorter.create_bind_group(
            device,
            num_points,
            &sorter_int,
            &sorter_b_a,
            &sorter_b_b,
            &sorter_p_a,
            &sorter_p_b,
        );
        let sorter_render_bg = sorter.create_bind_group_render(device, &sorter_uni, &sorter_p_a);
        let sorter_bg_pre = sorter.create_bind_group_preprocess(
            device,
            &sorter_uni,
            &sorter_dis,
            &sorter_int,
            &sorter_b_a,
            &sorter_b_b,
            &sorter_p_a,
            &sorter_p_b,
        );

        Ok(Self {
            splat_2d_buffer,

            bind_group,
            render_bind_group,
            num_points: num_points as u32,
            sh_deg,
            sorter,
            sorter_b_a,
            sorter_b_b,
            sorter_p_a,
            sorter_p_b,
            sorter_int,
            sorter_uni,
            sorter_dis,
            sorter_dis_bg,
            sorter_bg,
            sorter_render_bg,
            sorter_bg_pre,
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
