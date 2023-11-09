use crate::gpu_rs::GPURSSorter;
use crate::{
    camera::{Camera, PerspectiveCamera, VIEWPORT_Y_FLIP},
    pointcloud::{PointCloud, SHDType},
    uniform::UniformBuffer,
    utils::GPUStopwatch,
};
use std::num::NonZeroU64;

use futures_intrusive::buffer;
#[cfg(target_arch = "wasm32")]
use instant::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use cgmath::{Matrix4, SquareMatrix, Vector2};

pub enum Renderer{
    Rast(GaussianRenderer),
    Comp(GaussianRendererCompute),
}

pub struct GaussianRenderer {
    pipeline: wgpu::RenderPipeline,
    camera: UniformBuffer<CameraUniform>,
    preprocess: PreprocessPipeline,
    draw_indirect_buffer: wgpu::Buffer,
    draw_indirect: wgpu::BindGroup,
    color_format: wgpu::TextureFormat,
    #[cfg(not(target_arch = "wasm32"))]
    stopwatch: GPUStopwatch,
}

impl GaussianRenderer {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        sh_dtype: SHDType,
        data_compressed: bool,
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
                buffers: &[], //&[Splat2D::desc()],
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

        let camera = UniformBuffer::new_default(device, Some("camera uniform buffer"));
        let preprocess = PreprocessPipeline::new(device, sh_deg, sh_dtype, data_compressed);
        GaussianRenderer {
            pipeline,
            camera,
            preprocess,
            draw_indirect_buffer,
            draw_indirect,
            color_format,
            #[cfg(not(target_arch = "wasm32"))]
            stopwatch,
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
        self.preprocess
            .run(encoder, pc, &self.camera, &self.draw_indirect);
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
        target: &wgpu::TextureView,
        pc: &'a PointCloud,
        camera: PerspectiveCamera,
        viewport: Vector2<u32>,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder Compare"),
        });
        {
            GPURSSorter::record_reset_indirect_buffer(&pc.sorter_dis, &pc.sorter_uni, &queue);

            // convert 3D gaussian splats to 2D gaussian splats
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.start(&mut encoder, "preprocess").unwrap();
            self.preprocess(&mut encoder, queue, &pc, camera, viewport);
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.stop(&mut encoder, "preprocess").unwrap();

            // sort 2d splats
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.start(&mut encoder, "sorting").unwrap();
            pc.sorter
                .record_sort_indirect(&pc.sorter_bg, &pc.sorter_dis, &mut encoder);
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
                        view: &target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });

                render_pass.set_bind_group(0, &pc.render_bind_group, &[]);
                render_pass.set_bind_group(1, &pc.sorter_render_bg, &[]);
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

struct ImageBuffer<T: std::default::Default> {
    w: i32,
    h: i32,
    buffer: Option<wgpu::Buffer>,
    n: T,
}
impl<T: std::default::Default> ImageBuffer<T> {
    pub fn new() -> Self{
        ImageBuffer{w: -1, h: -1, buffer: None, n: T::default()}
    }
    pub fn get_or_create_buffer(&mut self, w: u32, h: u32, device: &wgpu::Device) -> (&wgpu::Buffer, bool) {
        let mut change = false;
        if w as i32 > self.w || h as i32 > self.h {
            println!("Recreating image buffer with sizes [{w}, {h}]");
            self.w = w as i32;
            self.h = h as i32;
            self.buffer.replace(device.create_buffer(&wgpu::BufferDescriptor { label: Some("Image atomic buffer"), size: w as u64 * h as u64 * std::mem::size_of::<T>() as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
            change = true;
        }
        (self.buffer.as_ref().unwrap(), change)
    }
}
#[derive(Default)]
struct ComputeBindGroup {
    color_hash: u64,
    alpha_hash: u64,
    bind_group: Option<wgpu::BindGroup>,
}
impl ComputeBindGroup {
    fn get_or_adapt_bind_group(&mut self, 
                                device: &wgpu::Device, 
                                bind_group_layout: &wgpu::BindGroupLayout,
                                color_buffer: &wgpu::Buffer, 
                                alpha_buffer: &wgpu::Buffer, 
                                buffer_changed: bool
                                ) -> &wgpu::BindGroup {
        let color_hash = color_buffer as *const wgpu::Buffer as u64;
        let alpha_hash = alpha_buffer as *const wgpu::Buffer as u64;
        if self.bind_group.is_none() || 
            self.color_hash != color_hash || 
            self.alpha_hash != alpha_hash ||
            buffer_changed
        {
            println!("Recreating the bind_group for compute rasterization");
            self.color_hash = color_hash;
            self.alpha_hash = alpha_hash;
            self.bind_group.replace(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Compute Gaussian bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: alpha_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: color_buffer.as_entire_binding(),
                    },
                ],
            }));
        }
        self.bind_group.as_ref().unwrap()
    }
}
pub struct GaussianRendererCompute {
    pipeline: wgpu::RenderPipeline,
    pipeline_resolve: wgpu::RenderPipeline,
    camera:  UniformBuffer<CameraUniform>,
    preprocess: PreprocessPipeline,
    draw_indirect_buffer: wgpu::Buffer,
    draw_indirect: wgpu::BindGroup,
    buffer_color: ImageBuffer::<u32>,
    buffer_alpha: ImageBuffer::<u32>,
    bind_group_layout_render: wgpu::BindGroupLayout,
    bind_group_render: ComputeBindGroup,
    
    color_format: wgpu::TextureFormat,
    stopwatch: GPUStopwatch,
}

impl GaussianRendererCompute  {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        sh_dtype: SHDType,
        data_compressed: bool,
    ) -> Self {
        let bind_group_layout_render = Self::bind_group_layout_render(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Compute renderer pipeline layout"),
            bind_group_layouts: &[
                &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                &PointCloud::bind_group_layout_render(device),             // needed for points_2d (on binding 2)
                &GPURSSorter::bind_group_layout_rendering(device),  // needed for sorted indices (on binding 4)
                &bind_group_layout_render,
            ],
            push_constant_ranges: &[],
        });
        
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/gaussian_compute.wgsl"));
        
        // let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        //     label: Some("Gaussian renderer compute pipeline"),
        //     layout: Some(&pipeline_layout),
        //     module: &shader,
        //     entry_point: "draw_splat",
        // });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gaussian renderer atomic pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_splats",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_splats",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
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

        let pipeline_resolve  = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gaussian renderer resolve pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_resolve",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_resolve",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
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
            label: Some("indirect dispatch buffer rendering"),
            size: std::mem::size_of::<wgpu::util::DrawIndirect>() as u64,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let indirect_layout = Self::bind_group_layout(device);
        let draw_indirect = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dispatch indirect buffer"),
            layout: &indirect_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: draw_indirect_buffer.as_entire_binding(),
            }],
        });
        
        let stopwatch = GPUStopwatch::new(device, Some(3));

        let camera = UniformBuffer::new_default(device, Some("camera uniform buffer"));
        let preprocess = PreprocessPipeline::new(device, sh_deg, sh_dtype, data_compressed);

        let buffer_color = ImageBuffer::new();
        let buffer_alpha = ImageBuffer::new();
        let bind_group_render = ComputeBindGroup::default();
        
        GaussianRendererCompute {
            pipeline,
            pipeline_resolve,
            camera,
            preprocess,
            draw_indirect_buffer,
            draw_indirect,
            buffer_color,
            buffer_alpha,
            bind_group_layout_render,
            bind_group_render,
            color_format,
            stopwatch,
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
        self.preprocess
            .run(encoder, pc, &self.camera, &self.draw_indirect);
    }
    
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
        target: &wgpu::TextureView,
        pc: &'a PointCloud,
        camera: PerspectiveCamera,
        viewport: Vector2<u32>,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder Compare"),
        });
        {
            GPURSSorter::record_reset_indirect_buffer(&pc.sorter_dis, &pc.sorter_uni, &queue);

            // convert 3D gaussian splats to 2D gaussian splats
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.start(&mut encoder, "preprocess").unwrap();
            self.preprocess(&mut encoder, queue, &pc, camera, viewport);
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.stop(&mut encoder, "preprocess").unwrap();

            // sort 2d splats
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.start(&mut encoder, "sorting").unwrap();
            pc.sorter
                .record_sort_indirect(&pc.sorter_bg, &pc.sorter_dis, &mut encoder);
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.stop(&mut encoder, "sorting").unwrap();

            // rasterize splats
            let (buffer_color, buffer_changed) = self.buffer_color.get_or_create_buffer(viewport.x, viewport.y, device);
            let (buffer_alpha, _) = self.buffer_alpha.get_or_create_buffer(viewport.x, viewport.y, device);
            let bind_group = self.bind_group_render.get_or_adapt_bind_group(&device, &self.bind_group_layout_render, buffer_color, buffer_alpha, buffer_changed);
            encoder.push_debug_group("render");
            #[cfg(not(target_arch = "wasm32"))]
            self.stopwatch.start(&mut encoder, "rasterization").unwrap();
            encoder.clear_buffer(buffer_alpha, 0, None);
            encoder.clear_buffer(buffer_color, 0, None);
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Gaussian Renderer splat atomic rendering"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        resolve_target: None,
                        ops: wgpu::Operations{load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: true},
                    })],
                    depth_stencil_attachment: None,
                });

                // atomic rendering
                render_pass.set_bind_group(0, &self.camera.bind_group(), &[]);
                render_pass.set_bind_group(1, &pc.render_bind_group, &[]);
                render_pass.set_bind_group(2, &pc.sorter_render_bg, &[]);
                render_pass.set_bind_group(3, &bind_group, &[]);
                render_pass.set_pipeline(&self.pipeline);

                render_pass.draw_indirect(&self.draw_indirect_buffer, 0);
                
                // resolve pass
                render_pass.set_bind_group(0, &self.camera.bind_group(), &[]);
                render_pass.set_bind_group(1, &pc.render_bind_group, &[]);
                render_pass.set_bind_group(2, &pc.sorter_render_bg, &[]);
                render_pass.set_bind_group(3, &bind_group, &[]);
                render_pass.set_pipeline(&self.pipeline_resolve);
                render_pass.draw(0..4, 0..1);
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
            label: Some("Draw indirect"),
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
    
    pub fn bind_group_layout_render(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout resolve images"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                    has_dynamic_offset: false, 
                    min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Storage { read_only: false }, 
                    has_dynamic_offset: false, 
                    min_binding_size: None },
                count: None,
            },
            ],
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
    fn new(device: &wgpu::Device, sh_deg: u32, sh_dtype: SHDType, data_compressed: bool) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess pipeline layout"),
            bind_group_layouts: &[
                &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                &PointCloud::bind_group_layout(device, data_compressed),
                &GaussianRenderer::bind_group_layout(device),
                &GPURSSorter::bind_group_layout_preprocess(device),
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preprocess shader"),
            source: wgpu::ShaderSource::Wgsl(Self::build_shader(sh_deg, sh_dtype, data_compressed).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preprocess pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "preprocess",
        });
        Self(pipeline)
    }

    fn build_shader(sh_deg: u32, sh_dtype: SHDType, data_compressed: bool) -> String {
        let SHADER_SRC: &str = match data_compressed {
            false => include_str!("shaders/preprocess.wgsl"),
            true => include_str!("shaders/preprocess_compressed.wgsl"),
        } ;
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
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, camera.bind_group(), &[]);
        pass.set_bind_group(1, pc.bind_group(), &[]);
        pass.set_bind_group(2, draw_indirect, &[]);
        pass.set_bind_group(3, &pc.sorter_bg_pre, &[]);

        let wgs_x = (pc.num_points() as f32 / 256.0).ceil() as u32;
        pass.dispatch_workgroups(wgs_x, 1, 1);
    }
}

pub struct RenderStatistics {
    pub preprocess_time: Duration,
    pub rasterization_time: Duration,
    pub sort_time: Duration,
}
