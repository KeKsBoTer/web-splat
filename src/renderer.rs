use crate::gpu_rs::{GPURSSorter, PointCloudSortStuff};
use crate::{
    camera::{Camera, PerspectiveCamera, VIEWPORT_Y_FLIP},
    pointcloud::PointCloud,
    uniform::UniformBuffer,
    utils::GPUStopwatch,
};
use std::num::NonZeroU64;

#[cfg(target_arch = "wasm32")]
use instant::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
use wgpu::{include_wgsl, Extent3d, MultisampleState};

use cgmath::{Matrix4, SquareMatrix, Vector2};

pub struct GaussianRenderer {
    pipeline: wgpu::RenderPipeline,
    camera: UniformBuffer<CameraUniform>,
    preprocess: PreprocessPipeline,
    draw_indirect_buffer: wgpu::Buffer,
    draw_indirect: wgpu::BindGroup,
    color_format: wgpu::TextureFormat,
    #[cfg(not(target_arch = "wasm32"))]
    pub stopwatch: GPUStopwatch,
    sorter: GPURSSorter,
    sorter_suff: Option<PointCloudSortStuff>,
}

impl GaussianRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        float: bool,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[
                &PointCloud::bind_group_layout_render(device), // Needed for points_2d (on binding 2)
                &GPURSSorter::bind_group_layout_rendering(device), // Needed for indices   (on binding 4)
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/gaussian.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
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
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
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
        #[cfg(not(target_arch = "wasm32"))]
        let stopwatch = GPUStopwatch::new(device, Some(3));

        let sorter = GPURSSorter::new(device, queue);

        let camera = UniformBuffer::new_default(device, Some("camera uniform buffer"));
        let preprocess = PreprocessPipeline::new(device, sh_deg, float);
        GaussianRenderer {
            pipeline,
            camera,
            preprocess,
            draw_indirect_buffer,
            draw_indirect,
            color_format,
            #[cfg(not(target_arch = "wasm32"))]
            stopwatch,
            sorter,
            sorter_suff: None,
        }
    }

    fn preprocess<'a>(
        &'a mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        pc: &'a PointCloud,
        camera: PerspectiveCamera,
        viewport: Vector2<u32>,
    ) {
        let mut camera = camera;
        camera.projection.resize(viewport.x, viewport.y);
        let uniform = self.camera.as_mut();
        uniform.set_camera(camera);
        uniform.set_focal(camera.projection.focal(viewport));
        uniform.set_viewport(viewport.cast().unwrap());
        self.camera.sync(queue);
        // TODO perform this in vertex buffer after draw call
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
        let depth_buffer = &self.sorter_suff.as_ref().unwrap().sorter_bg_pre;
        self.preprocess
            .run(encoder, pc, &self.camera, &self.draw_indirect, depth_buffer);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn num_visible_points(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        let n = {
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

            wgpu::util::DownloadBuffer::read_buffer(
                device,
                queue,
                &self.draw_indirect_buffer.slice(..),
                move |b| {
                    let download = b.unwrap();
                    let data = download.as_ref();
                    let num_points = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                    tx.send(num_points).unwrap();
                },
            );
            device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap()
        };
        return n;
    }

    pub fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pc: &'a PointCloud,
        camera: PerspectiveCamera,
        viewport: Vector2<u32>,
        target: &wgpu::TextureView,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        {
            if self.sorter_suff.is_none()
                || self
                    .sorter_suff
                    .as_ref()
                    .is_some_and(|s| s.num_points != pc.num_points() as usize)
            {
                log::info!("created sort buffers for {:} points", pc.num_points());
                self.sorter_suff = Some(
                    self.sorter
                        .create_sort_stuff(device, pc.num_points() as usize),
                );
            }

            GPURSSorter::record_reset_indirect_buffer(
                &self.sorter_suff.as_ref().unwrap().sorter_dis,
                &self.sorter_suff.as_ref().unwrap().sorter_uni,
                &queue,
            );

            // convert 3D gaussian splats to 2D gaussian splats
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.start(&mut encoder, "preprocess").unwrap();
            self.preprocess(&mut encoder, queue, &pc, camera, viewport);
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.stop(&mut encoder, "preprocess").unwrap();

            // sort 2d splats
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.start(&mut encoder, "sorting").unwrap();
            self.sorter.record_sort_indirect(
                &self.sorter_suff.as_ref().unwrap().sorter_bg,
                &self.sorter_suff.as_ref().unwrap().sorter_dis,
                &mut encoder,
            );
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.stop(&mut encoder, "sorting").unwrap();

            // rasterize splats
            encoder.push_debug_group("render");
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.start(&mut encoder, "rasterization").unwrap();
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });

                render_pass.set_bind_group(0, &pc.render_bind_group, &[]);
                render_pass.set_bind_group(
                    1,
                    &self.sorter_suff.as_ref().unwrap().sorter_render_bg,
                    &[],
                );
                render_pass.set_pipeline(&self.pipeline);

                render_pass.draw_indirect(&self.draw_indirect_buffer, 0);
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.stopwatch.stop(&mut encoder, "rasterization").unwrap();
        encoder.pop_debug_group();
        #[cfg(not(target_arch = "wasm32"))]
        self.stopwatch.end(&mut encoder);
        queue.submit([encoder.finish()]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn render_stats(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> RenderStatistics {
        let durations = self.stopwatch.take_measurements(device, queue).await;
        RenderStatistics {
            preprocess_time: durations["preprocess"],
            sort_time: durations["sorting"],
            rasterization_time: durations["rasterization"],
        }
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

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
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
        self.proj_matrix = VIEWPORT_Y_FLIP * proj_matrix;
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

struct PreprocessPipeline(wgpu::ComputePipeline);

impl PreprocessPipeline {
    fn new(device: &wgpu::Device, sh_deg: u32, float: bool) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess pipeline layout"),
            bind_group_layouts: &[
                &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                &if float {
                    PointCloud::bind_group_layout_float(device)
                } else {
                    PointCloud::bind_group_layout(device)
                },
                &GaussianRenderer::bind_group_layout(device),
                &GPURSSorter::bind_group_layout_preprocess(device),
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preprocess shader"),
            source: wgpu::ShaderSource::Wgsl(Self::build_shader(sh_deg, float).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preprocess pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "preprocess",
        });
        Self(pipeline)
    }

    fn build_shader(sh_deg: u32, float: bool) -> String {
        let shader_src: &str = if float {
            include_str!("shaders/preprocess_f32.wgsl")
        } else {
            include_str!("shaders/preprocess.wgsl")
        };
        let shader = format!(
            "
        const MAX_SH_DEG:u32 = {:}u;
        {:}",
            sh_deg, shader_src
        );
        return shader;
    }

    fn run<'a>(
        &mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        pc: &PointCloud,
        camera: &UniformBuffer<CameraUniform>,
        draw_indirect: &wgpu::BindGroup,
        depth_buffer: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("preprocess compute pass"),
            ..Default::default()
        });
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, camera.bind_group(), &[]);
        pass.set_bind_group(1, pc.bind_group(), &[]);
        pass.set_bind_group(2, draw_indirect, &[]);
        pass.set_bind_group(3, &depth_buffer, &[]);

        let wgs_x = (pc.num_points() as f32 / 256.0).ceil() as u32;
        pass.dispatch_workgroups(wgs_x, 1, 1);
    }
}

pub struct RenderStatistics {
    pub preprocess_time: Duration,
    pub rasterization_time: Duration,
    pub sort_time: Duration,
}

pub struct Display {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    format: wgpu::TextureFormat,
    view: wgpu::TextureView,
}

impl Display {
    pub fn new(
        device: &wgpu::Device,
        source_format: wgpu::TextureFormat,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display pipeline layout"),
            bind_group_layouts: &[&Self::bind_group_layout(device)],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(include_wgsl!("shaders/display.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });
        let (view, bind_group) = Self::create_render_target(device, source_format, width, height);
        Self {
            pipeline,
            view,
            format: source_format,
            bind_group,
        }
    }

    pub fn texture(&self) -> &wgpu::TextureView {
        &self.view
    }

    fn create_render_target(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::TextureView, wgpu::BindGroup) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display render image"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render target bind group"),
            layout: &Display::bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        return (texture_view, bind_group);
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("disply bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (view, bind_group) = Self::create_render_target(device, self.format, width, height);
        self.bind_group = bind_group;
        self.view = view;
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_pipeline(&self.pipeline);

        render_pass.draw(0..4, 0..1);
    }
}
