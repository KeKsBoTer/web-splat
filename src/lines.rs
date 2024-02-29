use bytemuck::ByteHash;
use cgmath::{Point3, Vector4};

use crate::{pointcloud::Aabb, renderer::CameraUniform, uniform::UniformBuffer, PerspectiveCamera};

pub struct LineRenderer {
    pipeline: wgpu::RenderPipeline,
    line_buffer: wgpu::Buffer,
    num_lines: usize,
}

impl LineRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[&UniformBuffer::<CameraUniform>::bind_group_layout(device)],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/lines.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("line render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Line>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: wgpu::VertexFormat::Float32x4.size(),
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm8x4,
                            offset: wgpu::VertexFormat::Float32x4.size() * 2,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        let line_buffer = Self::create_line_buffer(device, 4096);

        return Self {
            pipeline,
            line_buffer,
            num_lines: 0,
        };
    }

    fn create_line_buffer(device: &wgpu::Device, num_lines: usize) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            size: (num_lines * std::mem::size_of::<Line>()) as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        return buffer;
    }

    pub fn prepare<'a>(
        &mut self,
        _encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
        lines: &Vec<&[Line]>,
    ) {
        let mut aabb = Aabb::unit();
        let mut data: Vec<u8> = Vec::new();
        for group in lines {
            for l in *group {
                aabb.grow(Point3::from_homogeneous(l.start));
                aabb.grow(Point3::from_homogeneous(l.end));
            }
            data.extend_from_slice(bytemuck::cast_slice(group));
        }
        if data.len() > self.line_buffer.size() as usize {
            self.line_buffer = Self::create_line_buffer(device, data.len());
        }
        queue.write_buffer(&self.line_buffer, 0, &data);
        self.num_lines = lines.iter().map(|v| v.len()).sum();
    }

    pub fn render<'rpass>(
        &'rpass self,
        render_pass: &mut wgpu::RenderPass<'rpass>,
        camera: &'rpass UniformBuffer<CameraUniform>,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera.bind_group(), &[]);
        render_pass.set_vertex_buffer(0, self.line_buffer.slice(..));

        render_pass.draw(0..4, 0..self.num_lines as u32);
    }
}

pub trait RendereMiddleware {
    type PrepareData;
    type FrameData;

    fn prepare(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        data: &Self::PrepareData,
    ) -> Self::FrameData;

    fn render<'rpass>(
        &'rpass self,
        render_pass: &'rpass mut wgpu::RenderPass,
        frame_data: &'rpass Self::FrameData,
    );
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, ByteHash)]
pub struct Line {
    start: Vector4<f32>,
    end: Vector4<f32>,
    color: Vector4<u8>,
}
impl Line {
    pub fn new(start: Point3<f32>, end: Point3<f32>, color: wgpu::Color) -> Line {
        Self {
            start: start.to_homogeneous(),
            end: end.to_homogeneous(),
            color: Vector4::new(
                (color.r * 255.) as u8,
                (color.g * 255.) as u8,
                (color.b * 255.) as u8,
                (color.a * 255.) as u8,
            ),
        }
    }
}
