use anyhow::Ok;
use bytemuck::Zeroable;
use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use cgmath::{
    Matrix, Matrix3, Point3, Quaternion, SquareMatrix, Transform, Vector2, Vector3, Zero,
};
use ply_rs;
use std::io::{self, BufReader};
use std::{mem, path::Path};
use wgpu::util::DeviceExt;

use crate::camera::Camera;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianSplat {
    pub xyz: Point3<f32>,
    _pad0: u32,
    pub color: Vector3<f32>,
    _pad1: u32,
    pub covariance_1: Vector3<f32>,
    _pad2: u32,
    pub covariance_2: Vector3<f32>,
    // _pad3: u32,
    pub opacity: f32,
}

impl Default for GaussianSplat {
    fn default() -> Self {
        GaussianSplat::zeroed()
    }
}
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Splat2D {
    pos: Vector2<f32>,
    v1: Vector2<f32>,
    color: Vector3<f32>,
    opacity: f32,
    v2: Vector2<f32>,
}
pub struct PointCloud {
    vertex_buffer: wgpu::Buffer,
    splats_2d_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    points: Vec<GaussianSplat>,
    num_points: u32,
}

impl PointCloud {
    pub fn load_ply<P: AsRef<Path>>(device: &wgpu::Device, path: P) -> Result<Self, anyhow::Error> {
        let f = std::fs::File::open(path.as_ref())?;
        let mut reader = BufReader::new(f);

        // TODO this is very unsafe and just assumes a specific ply format
        // the format in the file is never checked

        let p = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let header = p.read_header(&mut reader).unwrap();

        let num_points = header.elements.get("vertex").unwrap().count;
        let points: Vec<GaussianSplat> = match header.encoding {
            ply_rs::ply::Encoding::Ascii => todo!("implement me"),
            ply_rs::ply::Encoding::BinaryBigEndian => (0..num_points)
                .map(|_| read_line::<BigEndian, _>(&mut reader))
                .collect(),
            ply_rs::ply::Encoding::BinaryLittleEndian => (0..num_points)
                .map(|_| read_line::<LittleEndian, _>(&mut reader))
                .collect(),
        };

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(points.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vertex buffer"),
            size: (points.len() * mem::size_of::<Splat2D>()) as u64,
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
                    resource: splat_2d_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            vertex_buffer,
            splats_2d_buffer: splat_2d_buffer,
            bind_group,
            num_points: num_points as u32,
            points,
        })
    }

    pub(crate) fn splats_2d_buffer(&self) -> &wgpu::Buffer {
        &self.splats_2d_buffer
    }
    pub fn num_points(&self) -> u32 {
        self.num_points
    }

    pub fn points(&self) -> &Vec<GaussianSplat> {
        &self.points
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
fn build_cov(rot: Quaternion<f32>, scale: Vector3<f32>) -> [f32; 6] {
    let r = Matrix3::from(rot);
    let s = Matrix3::from_diagonal(scale);

    let l = r * s;

    let m = l * l.transpose();

    return [m[0][0], m[0][1], m[0][2], m[1][1], m[1][2], m[2][2]];
}

/// numerical stable sigmoid function
fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        x.exp() / (1. + x.exp())
    }
}

fn read_line<B: ByteOrder, R: io::Read + io::Seek>(reader: &mut BufReader<R>) -> GaussianSplat {
    let x = reader.read_f32::<B>().unwrap();
    let y = reader.read_f32::<B>().unwrap();
    let z = reader.read_f32::<B>().unwrap();

    // skip normals
    reader
        .seek_relative(std::mem::size_of::<f32>() as i64 * 3)
        .unwrap();

    let mut sh_coefs = [Vector3::zero(); 16];

    sh_coefs[0].x = reader.read_f32::<B>().unwrap();
    sh_coefs[0].y = reader.read_f32::<B>().unwrap();
    sh_coefs[0].z = reader.read_f32::<B>().unwrap();

    // higher order coeffcients are stored with channel first
    for j in 0..3 {
        for i in 1..16 {
            sh_coefs[i][j] = reader.read_f32::<B>().unwrap();
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
    let rot_q = Quaternion::new(rot_0, rot_1, rot_2, rot_3);

    let cov = build_cov(rot_q, scale);
    return GaussianSplat {
        xyz: Point3::new(x, y, z),
        color: sh_coefs[0],
        covariance_1: Vector3::new(cov[0], cov[1], cov[2]),
        covariance_2: Vector3::new(cov[3], cov[4], cov[5]),
        opacity: opacity,
        ..Default::default()
    };
}

impl GaussianSplat {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        const VEC3_SIZE: u64 = wgpu::VertexFormat::Float32x3.size();
        const VEC2_SIZE: u64 = wgpu::VertexFormat::Float32x2.size();
        const FLOAT_SIZE: u64 = wgpu::VertexFormat::Float32.size();
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // xyz
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // covariance_1
                wgpu::VertexAttribute {
                    offset: VEC2_SIZE,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC2_SIZE * 2,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // opacity
                wgpu::VertexAttribute {
                    offset: VEC2_SIZE * 2 + VEC3_SIZE,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                // covariance_2
                wgpu::VertexAttribute {
                    offset: VEC2_SIZE * 2 + VEC3_SIZE + FLOAT_SIZE,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}
