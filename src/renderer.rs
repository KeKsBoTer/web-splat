use std::num::NonZeroU64;

use cgmath::{Matrix4, SquareMatrix, Vector2};

use crate::{
    camera::{Camera, PerspectiveCamera, OPENGL_TO_WGPU_MATRIX},
    pc::{PointCloud, SHDtype, Splat2D},
    uniform::UniformBuffer,
};

pub struct GaussianRenderer {
    pipeline: wgpu::RenderPipeline,
    camera: UniformBuffer<CameraUniform>,
    preprocess: PreprocessPointCloud,
    draw_indirect_buffer: wgpu::Buffer,
    draw_indirect: wgpu::BindGroup,
}

impl GaussianRenderer {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        sh_dtype: SHDtype,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/gaussian.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Splat2D::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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

        let draw_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect draw buffer"),
            size: std::mem::size_of::<wgpu::util::DrawIndirect>() as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let indirect_layout = Self::bind_group_layout(device);
        let draw_indirect = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw indirect buffer"),
            layout: &indirect_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: draw_indirect_buffer.as_entire_binding(),
            }],
        });

        let camera = UniformBuffer::new_default(device, Some("camera uniform buffer"));
        let preprocess = PreprocessPointCloud::new(device, sh_deg, sh_dtype);
        GaussianRenderer {
            pipeline,
            camera,
            preprocess,
            draw_indirect_buffer,
            draw_indirect,
        }
    }

    pub fn preprocess<'a>(
        &'a mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        pc: &'a PointCloud,
        camera: PerspectiveCamera,
        viewport: Vector2<u32>,
    ) {
        let uniform = self.camera.as_mut();
        uniform.set_camera(camera);
        uniform.set_focal(camera.projection.focal(viewport));
        uniform.set_viewport(viewport.cast().unwrap());
        self.camera.sync(queue);
        queue.write_buffer(
            &self.draw_indirect_buffer,
            0,
            wgpu::util::DrawIndirect {
                vertex_count: 4,
                instance_count: 0,
                base_vertex: 0,
                base_instance: 0,
            }
            .as_bytes(),
        );
        self.preprocess
            .run(encoder, pc, &self.camera, &self.draw_indirect);
    }

    pub fn render<'a>(&'a mut self, render_pass: &mut wgpu::RenderPass<'a>, pc: &'a PointCloud) {
        render_pass.set_pipeline(&self.pipeline);

        render_pass.set_vertex_buffer(0, pc.splats_2d_buffer().slice(..));
        // render_pass.draw(0..4, 0..pc.num_points()); //pc.num_points())
        render_pass.draw_indirect(&self.draw_indirect_buffer, 0);
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw indirect"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        NonZeroU64::new(std::mem::size_of::<wgpu::util::DrawIndirect>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            }],
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// the cameras view matrix
    pub(crate) view_matrix: Matrix4<f32>,
    /// inverse view matrix
    pub(crate) view_inv_matrix: Matrix4<f32>,

    // the cameras projection matrix
    pub(crate) proj_matrix: Matrix4<f32>,

    // inverse projection matrix
    pub(crate) proj_inv_matrix: Matrix4<f32>,

    pub(crate) viewport: Vector2<f32>,
    pub(crate) focal: Vector2<f32>,
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_matrix: Matrix4::identity(),
            view_inv_matrix: Matrix4::identity(),
            proj_matrix: Matrix4::identity(),
            proj_inv_matrix: Matrix4::identity(),
            viewport: Vector2::new(1., 1.),
            focal: Vector2::new(1., 1.),
        }
    }
}

impl CameraUniform {
    pub(crate) fn set_view_mat(&mut self, view_matrix: Matrix4<f32>) {
        self.view_matrix = view_matrix;
        self.view_inv_matrix = view_matrix.invert().unwrap();
    }

    pub(crate) fn set_proj_mat(&mut self, proj_matrix: Matrix4<f32>) {
        self.proj_matrix = OPENGL_TO_WGPU_MATRIX * proj_matrix;
        self.proj_inv_matrix = proj_matrix.invert().unwrap();
    }

    pub fn set_camera(&mut self, camera: impl Camera) {
        self.set_proj_mat(camera.proj_matrix());
        self.set_view_mat(camera.view_matrix());
    }

    pub fn set_viewport(&mut self, viewport: Vector2<f32>) {
        self.viewport = viewport;
    }
    pub fn set_focal(&mut self, focal: Vector2<f32>) {
        self.focal = focal
    }
}

struct PreprocessPointCloud {
    pipeline: wgpu::ComputePipeline,
    sh_deg: u32,
    sh_dtype: SHDtype,
}

impl PreprocessPointCloud {
    fn new(device: &wgpu::Device, sh_deg: u32, sh_dtype: SHDtype) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess pipeline layout"),
            bind_group_layouts: &[
                &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                &PointCloud::bind_group_layout(device),
                &GaussianRenderer::bind_group_layout(device),
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preprocess shader"),
            source: wgpu::ShaderSource::Wgsl(Self::build_shader(sh_deg, sh_dtype).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preprocess pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "preprocess",
        });
        Self {
            pipeline,
            sh_deg,
            sh_dtype,
        }
    }

    fn build_shader(sh_deg: u32, sh_dtype: SHDtype) -> String {
        const SHADER_SRC: &str = include_str!("shaders/preprocess.wgsl");
        let shader_src = format!(
            "
        const MAX_SH_DEG:u32 = {:}u;
        const SH_DTYPE:u32 = {:}u;
        {:}",
            sh_deg, sh_dtype as u32, SHADER_SRC
        );
        return shader_src;
    }

    fn run<'a>(
        &mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        pc: &PointCloud,
        camera: &UniformBuffer<CameraUniform>,
        draw_indirect: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("preprocess compute pass"),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera.bind_group(), &[]);
        pass.set_bind_group(1, pc.bind_group(), &[]);
        pass.set_bind_group(2, draw_indirect, &[]);
        let per_dim = (pc.num_points() as f32).sqrt().ceil() as u32;
        let wgs_x = (per_dim + 15) / 16;
        let wgs_y = (per_dim + 15) / 16;
        pass.dispatch_workgroups(wgs_x, wgs_y, 1);
    }
}
