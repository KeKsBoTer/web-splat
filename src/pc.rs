use anyhow::Ok;
use bytemuck::Zeroable;
use byteorder::{LittleEndian, ReadBytesExt};
use cgmath::{Matrix, Matrix3, Point3, Quaternion, SquareMatrix, Transform, Vector3};
use ply_rs;
use std::io::BufReader;
use std::{mem, path::Path};
use wgpu::util::DeviceExt;

use crate::camera::Camera;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianSplat {
    pub xyz: Point3<f32>,
    pub color: Vector3<f32>,
    pub opacity: f32,
    pub covariance_1: Vector3<f32>,
    pub covariance_2: Vector3<f32>,
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
                    offset: VEC3_SIZE,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // opacity
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 2,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
                // covariance_1
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 2 + 1,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // covariance_2
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE * 3 + 1,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
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

        let p = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let header = p.read_header(&mut reader).unwrap();

        let num_points = header.elements.get("vertex").unwrap().count;

        let mut points = Vec::with_capacity(num_points);

        for _ in 0..num_points {
            let x = reader.read_f32::<LittleEndian>().unwrap();
            let y = reader.read_f32::<LittleEndian>().unwrap();
            let z = reader.read_f32::<LittleEndian>().unwrap();
            reader
                .seek_relative(std::mem::size_of::<f32>() as i64 * 3)
                .unwrap();

            let r = reader.read_f32::<LittleEndian>().unwrap();
            let g = reader.read_f32::<LittleEndian>().unwrap();
            let b = reader.read_f32::<LittleEndian>().unwrap();

            reader
                .seek_relative(std::mem::size_of::<f32>() as i64 * 45)
                .unwrap();
            let opacity = sigmoid(reader.read_f32::<LittleEndian>().unwrap());
            let scale_1 = reader.read_f32::<LittleEndian>().unwrap();
            let scale_2 = reader.read_f32::<LittleEndian>().unwrap();
            let scale_3 = reader.read_f32::<LittleEndian>().unwrap();
            let rot_0 = reader.read_f32::<LittleEndian>().unwrap();
            let rot_1 = reader.read_f32::<LittleEndian>().unwrap();
            let rot_2 = reader.read_f32::<LittleEndian>().unwrap();
            let rot_3 = reader.read_f32::<LittleEndian>().unwrap();

            let cov = build_cov(
                Quaternion::new(rot_0, rot_1, rot_2, rot_3),
                Vector3::new(scale_1, scale_2, scale_3),
            );
            points.push(GaussianSplat {
                xyz: Point3::new(x, y, z),
                color: Vector3::new(r, g, b),
                opacity: opacity,
                covariance_1: Vector3::new(cov[0], cov[1], cov[2]),
                covariance_2: Vector3::new(cov[3], cov[4], cov[5]),
            })
        }

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
}

fn build_cov(rot: Quaternion<f32>, scale: Vector3<f32>) -> [f32; 6] {
    let r = Matrix3::from(rot);
    let s = Matrix3::from_diagonal(scale);

    let l = r * s;

    let m = l * l.transpose();

    return [m[0][0], m[0][1], m[0][2], m[1][1], m[1][2], m[2][2]];
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        x.exp() / (1. + x.exp())
    }
}
