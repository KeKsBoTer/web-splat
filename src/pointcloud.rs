use bytemuck::Zeroable;
use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::{Point3, Vector4};
use clap::ValueEnum;
use half::f16;
use std::fmt::{Debug, Display};
use std::io::{self, BufReader, Read, Seek};
use std::mem;
use wgpu::util::DeviceExt;

use crate::gpu_rs::GPURSSorter;
use crate::utils::max_supported_sh_deg;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianSplat {
    pub xyz: Point3<f16>,
    // opacity as f16 as we would need padding anyway
    pub opacity: f16,
    pub sh_idx: u32,
    pub covariance: [f16; 6],
}

impl Default for GaussianSplat {
    fn default() -> Self {
        GaussianSplat::zeroed()
    }
}

#[allow(dead_code)]
pub struct PointCloud {
    vertex_buffer: wgpu::Buffer,
    sh_coef_buffer: wgpu::Buffer,
    splat_2d_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
    points: Vec<GaussianSplat>,
    num_points: u32,
    sh_deg: u32,
    sh_dtype: SHDType,
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

impl PointCloud {
    pub fn load<R: Read + Seek>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        f: R,
        sh_dtype: SHDType,
        max_sh_deg: Option<u32>,
    ) -> Result<Self, anyhow::Error> {
        let mut reader = BufReader::new(f);
        let mut reader = crate::ply::PlyReader::new(&mut reader)?;

        let file_sh_deg = reader.file_sh_deg()?;
        let num_points = reader.num_points()?;

        let max_buffer_size = device
            .limits()
            .max_buffer_size
            .max(device.limits().max_storage_buffer_binding_size as u64);

        let target_sh_deg = max_sh_deg.unwrap_or(file_sh_deg);
        let sh_deg =
            max_supported_sh_deg(max_buffer_size, num_points as u64, sh_dtype, target_sh_deg)
                .ok_or(anyhow::anyhow!(
                    "cannot fit sh coefs into a buffer (exceeds WebGPU's max_buffer_size)"
                ))?;
        log::info!("num_points: {num_points}, sh_deg: {sh_deg}, sh_dtype: {sh_dtype}");
        if sh_deg < target_sh_deg {
            log::warn!("color sh degree (file: {file_sh_deg}) was decreased to degree {sh_deg} to fit the coefficients into memory")
        }

        let (vertices, sh_coefs) = reader.read(sh_dtype, sh_deg)?;

        let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh coefs buffer"),
            contents: bytemuck::cast_slice(sh_coefs.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3d gaussians buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d gaussians buffer"),
            size: (vertices.len() * mem::size_of::<Splat2D>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
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
                    resource: splat_2d_buffer.as_entire_binding(),
                },
            ],
        });
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud rendering bind group"),
            layout: &Self::bind_group_layout_render(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 2,
                resource: splat_2d_buffer.as_entire_binding(),
            }],
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
        let sorter_render_bg = sorter.create_bind_group_render(device, &sorter_p_a);
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
            vertex_buffer,
            sh_coef_buffer: sh_buffer,
            splat_2d_buffer,
            bind_group,
            render_bind_group,
            num_points: num_points as u32,
            points: vertices,
            sh_deg,
            sh_dtype,
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

    pub fn points(&self) -> &Vec<GaussianSplat> {
        &self.points
    }
    pub fn sh_deg(&self) -> u32 {
        self.sh_deg
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn sh_dtype(&self) -> SHDType {
        self.sh_dtype
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
                visibility: wgpu::ShaderStages::VERTEX,
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
    pos: Vector4<f16>,
    color: Vector4<u8>,
    _pad: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum SHDType {
    Float = 0,
    Half = 1,
    Byte = 2,
}

impl Display for SHDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SHDType::Float => f.write_str("f32"),
            SHDType::Half => f.write_str("f16"),
            SHDType::Byte => f.write_str("i8"),
        }
    }
}

impl SHDType {
    pub fn write_to<W: io::Write>(&self, writer: &mut W, v: f32, idx: u32) -> io::Result<()> {
        let scale = if idx == 0 { 4. } else { 0.5 };
        match self {
            SHDType::Float => writer.write_f32::<LittleEndian>(v),
            SHDType::Half => writer.write_u16::<LittleEndian>(f16::from_f32(v).to_bits()),
            SHDType::Byte => writer.write_i8((v * 127. / scale) as i8),
        }
    }

    pub fn packed_size(&self) -> usize {
        match self {
            SHDType::Float => mem::size_of::<f32>(),
            SHDType::Half => mem::size_of::<f16>(),
            SHDType::Byte => mem::size_of::<i8>(),
        }
    }
}

pub trait PointCloudReader {
    fn read(
        &mut self,
        sh_dtype: SHDType,
        sh_deg: u32,
    ) -> Result<(Vec<GaussianSplat>, Vec<u8>), anyhow::Error>;

    fn file_sh_deg(&self) -> Result<u32, anyhow::Error>;

    fn num_points(&self) -> Result<usize, anyhow::Error>;
}
