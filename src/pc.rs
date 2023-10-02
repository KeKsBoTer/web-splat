use anyhow::Ok;
use bytemuck::Zeroable;
use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use cgmath::{
    InnerSpace, Matrix, Matrix3, Point3, Quaternion, SquareMatrix, Transform, Vector3, Vector4,
};
use half::f16;
use log::debug;
use num_traits::{Float, Zero};
use ply_rs;
use std::fmt::Debug;
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
#[derive(Debug, Clone, Copy)]
pub struct SHCoefs<const C: usize, F: Float>([Vector3<F>; C]);

unsafe impl bytemuck::Pod for SHCoefs<16, f16> {}
unsafe impl bytemuck::Zeroable for SHCoefs<16, f16> {}
unsafe impl bytemuck::Pod for SHCoefs<9, f16> {}
unsafe impl bytemuck::Zeroable for SHCoefs<9, f16> {}
unsafe impl bytemuck::Pod for SHCoefs<4, f16> {}
unsafe impl bytemuck::Zeroable for SHCoefs<4, f16> {}
unsafe impl bytemuck::Pod for SHCoefs<1, f16> {}
unsafe impl bytemuck::Zeroable for SHCoefs<1, f16> {}

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
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.num_points)
            .finish()
    }
}

impl PointCloud {
    pub fn load_ply<P: AsRef<Path>>(device: &wgpu::Device, path: P) -> Result<Self, anyhow::Error> {
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

        let file_sh_deg = ((num_sh_coefs / 3) as f32).sqrt() as u32 - 1;

        let num_points = header.elements.get("vertex").unwrap().count;

        let splat_sizes = [
            mem::size_of::<SHCoefs<1, f16>>(),
            mem::size_of::<SHCoefs<4, f16>>(),
            mem::size_of::<SHCoefs<9, f16>>(),
            mem::size_of::<SHCoefs<16, f16>>(),
        ];

        // maximum allowed size for buffer
        let max_size = device
            .limits()
            .max_buffer_size
            .max(device.limits().max_storage_buffer_binding_size as u64);

        let mut sh_deg: u32 = file_sh_deg;
        for (i, s) in splat_sizes.iter().enumerate().rev() {
            let sh_buffer_size = *s as u64 * num_points as u64;
            if sh_buffer_size < max_size && i <= sh_deg as usize {
                sh_deg = i as u32;
                break;
            }
        }

        let (vertices, sh_coef_buffer) = match sh_deg {
            0 => read_ply::<{ (0 + 1) * (0 + 1) }, _>(
                device,
                header.encoding,
                num_points,
                &mut reader,
            ),
            1 => read_ply::<{ (1 + 1) * (1 + 1) }, _>(
                device,
                header.encoding,
                num_points,
                &mut reader,
            ),
            2 => read_ply::<{ (2 + 1) * (2 + 1) }, _>(
                device,
                header.encoding,
                num_points,
                &mut reader,
            ),
            3 => read_ply::<{ (3 + 1) * (3 + 1) }, _>(
                device,
                header.encoding,
                num_points,
                &mut reader,
            ),
            _ => unimplemented!("only up the sh degree 3 is supported"),
        };

        if sh_deg != file_sh_deg {
            let buff_size = sh_coef_buffer.size();
            log::warn!("the sh coef buffer size ({buff_size}) exceeds the maximum allowed size ({max_size}). The degree of sh coefficients was reduced from {file_sh_deg} to {sh_deg}.");
        } else {
            log::info!("sh_deg: {sh_deg}");
        }

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

    return [
        f16::from_f32(m[0][0]),
        f16::from_f32(m[0][1]),
        f16::from_f32(m[0][2]),
        f16::from_f32(m[1][1]),
        f16::from_f32(m[1][2]),
        f16::from_f32(m[2][2]),
    ];
}

/// numerical stable sigmoid function
fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        x.exp() / (1. + x.exp())
    }
}

fn read_ply<const C: usize, R: Read + Seek>(
    device: &wgpu::Device,
    encoding: ply_rs::ply::Encoding,
    num_points: usize,
    reader: &mut BufReader<R>,
) -> (Vec<GaussianSplat>, wgpu::Buffer)
where
    SHCoefs<C, f16>: bytemuck::Pod,
{
    match encoding {
        ply_rs::ply::Encoding::Ascii => todo!("acsii ply format not supported"),
        ply_rs::ply::Encoding::BinaryBigEndian => {
            let start_read = Instant::now();
            let (vertices, sh_coefs): (Vec<GaussianSplat>, Vec<SHCoefs<C, f16>>) = (0..num_points)
                .map(|i| read_line::<C, BigEndian, _>(reader, i as u32))
                .unzip();
            debug!(
                "reading ply took {}ms",
                (Instant::now() - start_read).as_millis()
            );
            let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sh coefs buffer"),
                contents: bytemuck::cast_slice(sh_coefs.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            return (vertices, sh_buffer);
        }
        ply_rs::ply::Encoding::BinaryLittleEndian => {
            let start_read = Instant::now();
            let (vertices, sh_coefs): (Vec<GaussianSplat>, Vec<SHCoefs<C, f16>>) = (0..num_points)
                .map(|i| read_line::<C, LittleEndian, _>(reader, i as u32))
                .unzip();
            debug!(
                "reading ply took {}ms",
                (Instant::now() - start_read).as_millis()
            );
            let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sh coefs buffer"),
                contents: bytemuck::cast_slice(sh_coefs.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            return (vertices, sh_buffer);
        }
    }
}

fn read_line<const C: usize, B: ByteOrder, R: io::Read + io::Seek>(
    reader: &mut BufReader<R>,
    idx: u32,
) -> (GaussianSplat, SHCoefs<C, f16>) {
    let mut pos = [0.; 3];
    reader.read_f32_into::<B>(&mut pos).unwrap();

    // skip normals
    reader
        .seek_relative(std::mem::size_of::<f32>() as i64 * 3)
        .unwrap();

    let mut sh_coefs_raw = [0.; 16 * 3];

    reader.read_f32_into::<B>(&mut sh_coefs_raw).unwrap();

    let mut sh_coefs = [Vector3::zero(); C];
    sh_coefs[0].x = f16::from_f32(sh_coefs_raw[0]);
    sh_coefs[0].y = f16::from_f32(sh_coefs_raw[1]);
    sh_coefs[0].z = f16::from_f32(sh_coefs_raw[2]);

    // higher order coeffcients are stored with channel first (shape:[N,3,C])
    for j in 0..3 {
        for i in 1..C {
            sh_coefs[i][j] = f16::from_f32(sh_coefs_raw[2 + j * 15 + i]);
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

    return (
        GaussianSplat {
            xyz: Point3::from(pos),
            opacity,
            covariance: build_cov(rot_q, scale),
            sh_idx: idx,
            ..Default::default()
        },
        SHCoefs(sh_coefs),
    );
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
