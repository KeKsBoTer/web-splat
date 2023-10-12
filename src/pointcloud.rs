use bytemuck::Zeroable;
use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::{Point3, Vector4};
use clap::ValueEnum;
use half::f16;
use std::fmt::{Debug, Display};
use std::io::{self, BufReader};
use std::{mem, path::Path};
use wgpu::util::DeviceExt;

use crate::utils::max_supported_sh_deg;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianSplat {
    pub xyz: Point3<f32>,
    pub sh_idx: u32,
    pub covariance: [f16; 6],
    pub opacity: f32,
}

impl Default for GaussianSplat {
    fn default() -> Self {
        GaussianSplat::zeroed()
    }
}

pub struct PointCloud {
    splat_2d_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    points: Vec<GaussianSplat>,
    num_points: u32,
    sh_deg: u32,
    sh_dtype: SHDType,
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.num_points)
            .finish()
    }
}

impl PointCloud {
    pub fn load<P: AsRef<Path>>(
        device: &wgpu::Device,
        path: P,
        sh_dtype: SHDType,
        max_sh_deg: Option<u32>,
    ) -> Result<Self, anyhow::Error> {
        let f = std::fs::File::open(path.as_ref())?;
        let mut reader = BufReader::new(f);
        let file_ext = path.as_ref().extension().unwrap().to_str().unwrap();
        let mut reader: Box<dyn PointCloudReader> = match file_ext {
            #[cfg(feature = "npz")]
            "npz" => Box::new(crate::npz::NpzReader::new(&mut reader)?),
            #[cfg(not(feature = "npz"))]
            "npz" => return Err(anyhow::anyhow!("viewer was compiled without npz support")),
            "ply" => Box::new(crate::ply::PlyReader::new(&mut reader)?),
            _ => return Err(anyhow::anyhow!("file extension {file_ext} not supported")),
        };

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

        Ok(Self {
            splat_2d_buffer,
            bind_group,
            num_points: vertices.len() as u32,
            points: vertices,
            sh_deg,
            sh_dtype,
        })
    }

    pub(crate) fn splats_2d_buffer(&self) -> &wgpu::Buffer {
        &self.splat_2d_buffer
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
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat2D {
    v: Vector4<f16>,
    pos: Vector4<f16>,
    color: Vector4<u8>,
    _pad: u32,
}

impl Splat2D {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        const COV_SIZE: u64 = wgpu::VertexFormat::Float16x4.size();
        const XYZ_SIZE: u64 = wgpu::VertexFormat::Float16x4.size();
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // cov_1 + cov_2
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float16x4,
                },
                // xyz
                wgpu::VertexAttribute {
                    offset: COV_SIZE,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float16x4,
                },
                // color + opacity
                wgpu::VertexAttribute {
                    offset: COV_SIZE + XYZ_SIZE,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Unorm8x4,
                },
            ],
        }
    }
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
