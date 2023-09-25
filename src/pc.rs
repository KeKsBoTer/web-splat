use anyhow::Ok;
use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use cgmath::{Matrix, Matrix3, Point3, Quaternion, SquareMatrix, Transform, Vector3, Zero};
use ply_rs;
use std::io::{self, BufReader};
use std::{mem, path::Path};
use wgpu::util::DeviceExt;

use crate::camera::Camera;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianSplat {
    pub xyz: Point3<f32>,
    pub color: [Vector3<f32>; 16],
    pub covariance_1: Vector3<f32>,
    pub covariance_2: Vector3<f32>,
    pub opacity: f32,
}
pub struct PointCloud {
    vertex_buffer: wgpu::Buffer,
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

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(points.as_slice()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        Ok(Self {
            vertex_buffer: buffer,
            num_points: num_points as u32,
            points,
        })
    }

    pub(crate) fn vertex_buffer(&self) -> &wgpu::Buffer {
        &self.vertex_buffer
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
        color: sh_coefs,
        covariance_1: Vector3::new(cov[0], cov[1], cov[2]),
        covariance_2: Vector3::new(cov[3], cov[4], cov[5]),
        opacity: opacity,
    };
}

impl GaussianSplat {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        const VEC3_SIZE: u64 = wgpu::VertexFormat::Float32x3.size();
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // xyz
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 1,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 2,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 3,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 4,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 5,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 6,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 7,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 8,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 9,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 10,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 11,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 12,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 13,
                    shader_location: 13,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 14,
                    shader_location: 14,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 15,
                    shader_location: 15,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 16,
                    shader_location: 16,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // covariance_1
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 17,
                    shader_location: 17,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // covariance_2
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 18,
                    shader_location: 18,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // opacity
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 19,
                    shader_location: 19,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}
