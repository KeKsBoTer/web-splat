use bytemuck::Zeroable;
use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use cgmath::{
    InnerSpace, Matrix, Matrix3, Point3, Quaternion, SquareMatrix, Transform, Vector3, Vector4,
};
use half::f16;
use log::{info, warn};
use ply_rs;
use std::fmt::{Debug, Display};
use std::io::{self, BufReader, Read, Seek};
use std::time::Instant;
use std::{mem, path::Path};
use wgpu::util::DeviceExt;

use crate::camera::Camera;

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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat2D {
    v: Vector4<f16>,
    pos: Vector4<f16>,
    color: Vector4<u8>,
    _pad: u32,
}

pub struct PointCloud {
    vertex_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    sh_coef_buffer: wgpu::Buffer,
    splat_2d_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    points: Vec<GaussianSplat>,
    num_points: u32,
    sh_deg: u32,
    sh_dtype: SHDtype,
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.num_points)
            .finish()
    }
}

impl PointCloud {
    pub fn load_ply<P: AsRef<Path>>(
        device: &wgpu::Device,
        path: P,
        sh_dtype: SHDtype,
    ) -> Result<Self, anyhow::Error> {
        let f = std::fs::File::open(path.as_ref())?;
        let mut reader = BufReader::new(f);

        // TODO this is very unsafe and just assumes a specific ply format
        // the format in the file is never checked

        let p = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let header = p.read_header(&mut reader).unwrap();

        let num_sh_coefs = header.elements["vertex"]
            .properties
            .keys()
            .filter(|k| k.starts_with("f_"))
            .count();

        let file_sh_deg = sh_deg_from_num_coeffs(num_sh_coefs as u32 / 3).expect(&format!(
            "number of sh coefficients {num_sh_coefs} cannot be mapped to sh degree"
        ));

        let num_points = header.elements.get("vertex").unwrap().count;

        let max_buffer_size = device
            .limits()
            .max_buffer_size
            .max(device.limits().max_storage_buffer_binding_size as u64);

        let sh_deg =
            max_supported_sh_deg(max_buffer_size, num_points as u64, sh_dtype, file_sh_deg).ok_or(
                anyhow::anyhow!(
                    "cannot fit sh coefs into a buffer (exceeds WebGPU's max_buffer_size)"
                ),
            )?;
        log::info!("num_points: {num_points}, sh_deg: {sh_deg}, sh_dtype: {sh_dtype}");
        if sh_deg < file_sh_deg {
            warn!("color sh degree ({file_sh_deg}) was decreased to degree {sh_deg} to fit the coefficients into memory")
        }

        let (vertices, sh_coef_buffer) = read_ply::<_>(
            device,
            header.encoding,
            num_points,
            &mut reader,
            sh_deg,
            sh_dtype,
        );

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
                    resource: sh_coef_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: splat_2d_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            vertex_buffer,
            sh_coef_buffer,
            splat_2d_buffer,
            bind_group,
            num_points: num_points as u32,
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

    pub fn sort(&mut self, queue: &wgpu::Queue, camera: impl Camera) {
        let view = camera.view_matrix();
        let proj = camera.proj_matrix();
        let transform = proj * view;
        self.points
            .sort_by_cached_key(|p| (-transform.transform_point(p.xyz).z * (2f32).powi(24)) as i32);
        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(self.points.as_slice()),
        );
    }

    pub fn update_points(&mut self, queue: &wgpu::Queue, new_points: Vec<GaussianSplat>) {
        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(new_points.as_slice()),
        );
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
    pub(crate) fn sh_dtype(&self) -> SHDtype {
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

/// builds a covariance matrix based on a quaterion and rotation
/// the matrix is symmetric so we only return the upper right half
/// see "3D Gaussian Splatting" Kerbel et al.
fn build_cov(rot: Quaternion<f32>, scale: Vector3<f32>) -> [f16; 6] {
    let r = Matrix3::from(rot);
    let s = Matrix3::from_diagonal(scale);

    let l = r * s;

    let m = l * l.transpose();

    return [m[0][0], m[0][1], m[0][2], m[1][1], m[1][2], m[2][2]].map(|v| f16::from_f32(v));
}

/// numerical stable sigmoid function
fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        x.exp() / (1. + x.exp())
    }
}

fn read_ply<R: Read + Seek>(
    device: &wgpu::Device,
    encoding: ply_rs::ply::Encoding,
    num_points: usize,
    reader: &mut BufReader<R>,
    sh_deg: u32,
    sh_dtype: SHDtype,
) -> (Vec<GaussianSplat>, wgpu::Buffer) {
    let start = Instant::now();
    let mut sh_coef_buffer = Vec::new();
    let vertices: Vec<GaussianSplat> = match encoding {
        ply_rs::ply::Encoding::Ascii => todo!("acsii ply format not supported"),
        ply_rs::ply::Encoding::BinaryBigEndian => (0..num_points)
            .map(|i| {
                read_line::<BigEndian, _, _>(
                    reader,
                    i as u32,
                    sh_deg,
                    sh_dtype,
                    &mut sh_coef_buffer,
                )
            })
            .collect(),
        ply_rs::ply::Encoding::BinaryLittleEndian => (0..num_points)
            .map(|i| {
                read_line::<LittleEndian, _, _>(
                    reader,
                    i as u32,
                    sh_deg,
                    sh_dtype,
                    &mut sh_coef_buffer,
                )
            })
            .collect(),
    };
    info!(
        "reading ply file took {:}ms",
        (Instant::now() - start).as_millis()
    );
    let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sh coefs buffer"),
        contents: bytemuck::cast_slice(sh_coef_buffer.as_slice()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    return (vertices, sh_buffer);
}

fn read_line<B: ByteOrder, R: io::Read + io::Seek, W: io::Write>(
    reader: &mut BufReader<R>,
    idx: u32,
    sh_deg: u32,
    sh_dtype: SHDtype,
    sh_coefs_buffer: &mut W,
) -> GaussianSplat {
    let mut pos = [0.; 3];
    reader.read_f32_into::<B>(&mut pos).unwrap();

    // skip normals
    reader
        .seek_relative(std::mem::size_of::<f32>() as i64 * 3)
        .unwrap();

    let mut sh_coefs_raw = [0.; 16 * 3];
    reader.read_f32_into::<B>(&mut sh_coefs_raw).unwrap();

    sh_dtype
        .write_to(sh_coefs_buffer, sh_coefs_raw[0], 0)
        .unwrap();
    sh_dtype
        .write_to(sh_coefs_buffer, sh_coefs_raw[1], 0)
        .unwrap();
    sh_dtype
        .write_to(sh_coefs_buffer, sh_coefs_raw[2], 0)
        .unwrap();

    // higher order coeffcients are stored with channel first (shape:[N,3,C])
    for i in 1..num_coefficients(sh_deg) {
        for j in 0..3 {
            sh_dtype
                .write_to(
                    sh_coefs_buffer,
                    sh_coefs_raw[(2 + j * 15 + i) as usize],
                    i as u32,
                )
                .unwrap();
        }
    }

    let opacity = sigmoid(reader.read_f32::<B>().unwrap());

    let scale_1 = reader.read_f32::<B>().unwrap().exp();
    let scale_2 = reader.read_f32::<B>().unwrap().exp();
    let scale_3 = reader.read_f32::<B>().unwrap().exp();
    let scale = Vector3::new(scale_1, scale_2, scale_3);

    let rot_0 = reader.read_f32::<B>().unwrap();
    let rot_1 = reader.read_f32::<B>().unwrap();
    let rot_2 = reader.read_f32::<B>().unwrap();
    let rot_3 = reader.read_f32::<B>().unwrap();
    let rot_q = Quaternion::new(rot_0, rot_1, rot_2, rot_3).normalize();

    return GaussianSplat {
        xyz: Point3::from(pos),
        opacity,
        covariance: build_cov(rot_q, scale),
        sh_idx: idx,
        ..Default::default()
    };
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

#[derive(Debug, Clone, Copy)]
pub enum SHDtype {
    Float = 0,
    Half = 1,
    Byte = 2,
}

impl Display for SHDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SHDtype::Float => f.write_str("f32"),
            SHDtype::Half => f.write_str("f16"),
            SHDtype::Byte => f.write_str("i8"),
        }
    }
}

impl SHDtype {
    pub fn write_to<W: io::Write>(&self, writer: &mut W, v: f32, idx: u32) -> io::Result<()> {
        let scale = if idx == 0 { 4. } else { 0.5 };
        match self {
            SHDtype::Float => writer.write_f32::<LittleEndian>(v),
            SHDtype::Half => writer.write_u16::<LittleEndian>(f16::from_f32(v).to_bits()),
            SHDtype::Byte => writer.write_i8((v * 127. / scale) as i8),
        }
    }

    pub fn packed_size(&self) -> usize {
        match self {
            SHDtype::Float => mem::size_of::<f32>(),
            SHDtype::Half => mem::size_of::<f16>(),
            SHDtype::Byte => mem::size_of::<i8>(),
        }
    }
}

fn num_coefficients(sh_deg: u32) -> u32 {
    (sh_deg + 1) * (sh_deg + 1)
}

fn sh_deg_from_num_coeffs(n: u32) -> Option<u32> {
    let sqrt = (n as f32).sqrt();
    if sqrt.fract() != 0. {
        return None;
    }
    return Some((sqrt as u32) - 1);
}

fn max_supported_sh_deg(
    max_buffer_size: u64,
    num_points: u64,
    sh_dtype: SHDtype,
    max_deg: u32,
) -> Option<u32> {
    for i in (0..=max_deg).rev() {
        let n_coefs = num_coefficients(i) * 3;
        let buf_size = num_points as u64 * sh_dtype.packed_size() as u64 * n_coefs as u64;
        if buf_size < max_buffer_size {
            return Some(i);
        }
    }
    return None;
}
