use crate::gpu_rs::{GPURSSorter, PointCloudSortStuff, SORT_KEY_BITS};
use crate::pointcloud::Aabb;
use crate::utils::GPUStopwatch;
use crate::{
    camera::{Camera, PerspectiveCamera, VIEWPORT_Y_FLIP},
    pointcloud::PointCloud,
    uniform::UniformBuffer,
};

use std::num::NonZeroU64;
use std::time::Duration;

use wgpu::{Extent3d, MultisampleState, include_wgsl};

use cgmath::{EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector2, Vector4};

/// Whether the device can run the compute based renderer (preprocess and sort in compute shaders,
/// splats in storage buffers). Without (WebGL2) the splats are projected in the vertex shader
/// and drawn unsorted, see `gaussian_vertex.wgsl`.
pub fn supports_compute(device: &wgpu::Device) -> bool {
    let limits = device.limits();
    limits.max_compute_workgroups_per_dimension > 0
        && limits.max_compute_workgroup_size_x >= 256
        && limits.max_storage_buffers_per_shader_stage > 0
}

/// The resources for sorted rendering with compute shaders.
struct ComputeBackend {
    preprocess: PreprocessPipeline,
    draw_indirect_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    draw_indirect: wgpu::BindGroup,
    sorter: GPURSSorter,
    sorter_suff: Option<PointCloudSortStuff>,
    /// latest finished readback of the visible point count and whether one is in flight
    #[cfg(not(target_arch = "wasm32"))]
    visible_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
    #[cfg(not(target_arch = "wasm32"))]
    visible_pending: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl ComputeBackend {
    fn new(device: &wgpu::Device, sh_deg: u32, compressed: bool) -> Self {
        let draw_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect draw buffer"),
            size: std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let indirect_layout = GaussianRenderer::bind_group_layout(device);
        let draw_indirect = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw indirect buffer"),
            layout: &indirect_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: draw_indirect_buffer.as_entire_binding(),
            }],
        });

        Self {
            preprocess: PreprocessPipeline::new(device, sh_deg, compressed),
            draw_indirect_buffer,
            draw_indirect,
            sorter: GPURSSorter::new(device),
            sorter_suff: None,
            #[cfg(not(target_arch = "wasm32"))]
            visible_count: Default::default(),
            #[cfg(not(target_arch = "wasm32"))]
            visible_pending: Default::default(),
        }
    }
}

pub struct GaussianRenderer {
    pipeline: wgpu::RenderPipeline,
    camera: UniformBuffer<CameraUniform>,

    render_settings: UniformBuffer<SplattingArgsUniform>,
    color_format: wgpu::TextureFormat,
    /// `None` if the device has no compute shaders, then the splats are not sorted
    compute: Option<ComputeBackend>,
}

/// the index of a gaussian in the vertex shader path (`gaussian_vertex.wgsl`)
const INDEX_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Uint32];

impl GaussianRenderer {
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        compressed: bool,
    ) -> Self {
        let compute =
            supports_compute(device).then(|| ComputeBackend::new(device, sh_deg, compressed));

        let layouts: Vec<wgpu::BindGroupLayout> = if compute.is_some() {
            vec![
                PointCloud::bind_group_layout_render(device), // Needed for points_2d (on binding 2)
                GPURSSorter::bind_group_layout_rendering(device), // Needed for indices   (on binding 4)
                UniformBuffer::<SplattingArgsUniform>::bind_group_layout(device), // alpha cutoff
            ]
        } else {
            vec![
                UniformBuffer::<CameraUniform>::bind_group_layout(device),
                UniformBuffer::<SplattingArgsUniform>::bind_group_layout(device),
                PointCloud::bind_group_layout_data(device),
            ]
        };
        let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &layout_refs,
            immediate_size: 0,
        });

        let shader = if compute.is_some() {
            device.create_shader_module(wgpu::include_wgsl!("shaders/gaussian.wgsl"))
        } else {
            device.create_shader_module(wgpu::include_wgsl!("shaders/gaussian_vertex.wgsl"))
        };
        let vertex_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<u32>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INDEX_ATTRIBUTES,
        })];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: if compute.is_some() {
                    &[]
                } else {
                    &vertex_buffers
                },
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    // front to back blending
                    // this gives better visuals compared to back to front blending
                    // if used with 8 bit textures
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
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
            cache: None,
            multiview_mask: None,
        });

        GaussianRenderer {
            pipeline,
            camera: UniformBuffer::new_default(device, Some("camera uniform buffer")),
            color_format,
            compute,
            render_settings: UniformBuffer::new_default(
                device,
                Some("render settings uniform buffer"),
            ),
        }
    }

    pub(crate) fn camera(&self) -> &UniformBuffer<CameraUniform> {
        &self.camera
    }

    /// uploads the camera and the render settings
    fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        pc: &PointCloud,
        render_settings: SplattingArgs,
    ) {
        let camera = render_settings.camera;
        let uniform = self.camera.as_mut();
        uniform.set_focal(camera.projection.focal(render_settings.viewport));
        uniform.set_viewport(render_settings.viewport.cast().unwrap());
        uniform.set_camera(camera);
        self.camera.sync(queue);

        let settings_uniform = self.render_settings.as_mut();
        *settings_uniform = SplattingArgsUniform::from_args_and_pc(render_settings, pc);
        self.render_settings.sync(queue);
    }

    /// Number of visible points of the latest frame whose readback finished. The
    /// readback is asynchronous: waiting for the GPU every frame would stall the CPU
    /// until all submitted work is done and delay the input.
    /// Without compute shaders the visible points are not counted, this is 0.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn num_visible_points(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        use std::sync::atomic::Ordering;
        let Some(compute) = &self.compute else {
            return 0;
        };
        if !compute.visible_pending.swap(true, Ordering::AcqRel) {
            let (count, pending) = (
                compute.visible_count.clone(),
                compute.visible_pending.clone(),
            );
            wgpu::util::DownloadBuffer::read_buffer(
                device,
                queue,
                &compute.draw_indirect_buffer.slice(..),
                move |b| {
                    if let Ok(download) = b {
                        let data = download.as_ref();
                        count.store(
                            u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
                            Ordering::Release,
                        );
                    }
                    pending.store(false, Ordering::Release);
                },
            );
        }
        // run the callbacks of finished readbacks without waiting
        device.poll(wgpu::PollType::Poll).ok();
        compute.visible_count.load(Ordering::Acquire)
    }

    pub fn prepare(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pc: &PointCloud,
        render_settings: SplattingArgs,
        stopwatch: &mut Option<GPUStopwatch>,
    ) {
        self.update_uniforms(queue, pc, render_settings);

        // without compute shaders everything happens when the splats are drawn
        let Some(compute) = self.compute.as_mut() else {
            return;
        };

        if compute.sorter_suff.is_none()
            || compute
                .sorter_suff
                .as_ref()
                .is_some_and(|s| s.num_points != pc.num_points() as usize)
        {
            log::debug!("created sort buffers for {:} points", pc.num_points());
            compute.sorter_suff = Some(
                compute
                    .sorter
                    .create_sort_stuff(device, pc.num_points() as usize),
            );
        }

        GPURSSorter::record_reset_indirect_buffer(
            &compute.sorter_suff.as_ref().unwrap().sorter_dis,
            &compute.sorter_suff.as_ref().unwrap().sorter_uni,
            &queue,
        );

        // convert 3D gaussian splats to 2D gaussian splats
        if let Some(stopwatch) = stopwatch {
            stopwatch.start(encoder, "preprocess").unwrap();
        }

        // TODO perform this in vertex buffer after draw call
        queue.write_buffer(
            &compute.draw_indirect_buffer,
            0,
            wgpu::util::DrawIndirectArgs {
                vertex_count: 4,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }
            .as_bytes(),
        );
        let depth_buffer = &compute.sorter_suff.as_ref().unwrap().sorter_bg_pre;
        compute.preprocess.run(
            encoder,
            pc,
            &self.camera,
            &self.render_settings,
            depth_buffer,
        );
        if let Some(stopwatch) = stopwatch {
            stopwatch.stop(encoder, "preprocess").unwrap();
        }
        // sort 2d splats
        if let Some(stopwatch) = stopwatch {
            stopwatch.start(encoder, "sorting").unwrap();
        }
        compute.sorter.record_sort_indirect(
            &compute.sorter_suff.as_ref().unwrap().sorter_bg,
            &compute.sorter_suff.as_ref().unwrap().sorter_dis,
            encoder,
        );
        if let Some(stopwatch) = stopwatch {
            stopwatch.stop(encoder, "sorting").unwrap();
        }

        encoder.copy_buffer_to_buffer(
            &compute.sorter_suff.as_ref().unwrap().sorter_uni,
            0,
            &compute.draw_indirect_buffer,
            std::mem::size_of::<u32>() as u64,
            std::mem::size_of::<u32>() as u64,
        );
    }

    pub fn render<'rpass>(
        &'rpass self,
        render_pass: &mut wgpu::RenderPass<'rpass>,
        pc: &'rpass PointCloud,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        match &self.compute {
            Some(compute) => {
                render_pass.set_bind_group(0, pc.render_bind_group(), &[]);
                render_pass.set_bind_group(
                    1,
                    &compute.sorter_suff.as_ref().unwrap().sorter_render_bg,
                    &[],
                );
                render_pass.set_bind_group(2, self.render_settings.bind_group(), &[]);

                render_pass.draw_indirect(&compute.draw_indirect_buffer, 0);
            }
            None => {
                render_pass.set_bind_group(0, self.camera.bind_group(), &[]);
                render_pass.set_bind_group(1, self.render_settings.bind_group(), &[]);
                let gaussians = pc
                    .cpu_sorted()
                    .expect("point cloud without compute shaders has to be sorted on the cpu");
                if gaussians.visible() > 0 {
                    render_pass.set_bind_group(2, gaussians.bind_group(), &[]);
                    render_pass.set_vertex_buffer(0, gaussians.index_buffer().slice(..));
                    render_pass.draw(0..4, 0..gaussians.visible());
                }
            }
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
                        NonZeroU64::new(std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64)
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

    pub(crate) fn render_settings(&self) -> &UniformBuffer<SplattingArgsUniform> {
        &self.render_settings
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
    fn new(device: &wgpu::Device, sh_deg: u32, compressed: bool) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess pipeline layout"),
            bind_group_layouts: &[
                Some(&UniformBuffer::<CameraUniform>::bind_group_layout(device)),
                Some(&if !compressed {
                    PointCloud::bind_group_layout(device)
                } else {
                    PointCloud::bind_group_layout_compressed(device)
                }),
                Some(&GPURSSorter::bind_group_layout_preprocess(device)),
                Some(&UniformBuffer::<SplattingArgsUniform>::bind_group_layout(
                    device,
                )),
            ],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preprocess shader"),
            source: wgpu::ShaderSource::Wgsl(Self::build_shader(sh_deg, compressed).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preprocess pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("preprocess"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("sort_key_bits", SORT_KEY_BITS as f64)],
                ..Default::default()
            },
            cache: None,
        });
        Self(pipeline)
    }

    fn build_shader(sh_deg: u32, compressed: bool) -> String {
        let shader_src: &str = if !compressed {
            include_str!("shaders/preprocess.wgsl")
        } else {
            include_str!("shaders/preprocess_compressed.wgsl")
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
        render_settings: &UniformBuffer<SplattingArgsUniform>,
        sort_bg: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("preprocess compute pass"),
            ..Default::default()
        });
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, camera.bind_group(), &[]);
        pass.set_bind_group(1, pc.bind_group(), &[]);
        pass.set_bind_group(2, sort_bg, &[]);
        pass.set_bind_group(3, render_settings.bind_group(), &[]);

        let wgs_x = (pc.loaded_points() as f32 / 256.0).ceil() as u32;
        pass.dispatch_workgroups(wgs_x, 1, 1);
    }
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
            bind_group_layouts: &[
                Some(&Self::bind_group_layout(device)),
                Some(&UniformBuffer::<CameraUniform>::bind_group_layout(device)),
                Some(&UniformBuffer::<SplattingArgsUniform>::bind_group_layout(
                    device,
                )),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(include_wgsl!("shaders/display.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
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

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        background_color: wgpu::Color,
        camera: &UniformBuffer<CameraUniform>,
        render_settings: &UniformBuffer<SplattingArgsUniform>,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background_color),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, camera.bind_group(), &[]);
        render_pass.set_bind_group(2, render_settings.bind_group(), &[]);
        render_pass.set_pipeline(&self.pipeline);

        render_pass.draw(0..4, 0..1);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SplattingArgs {
    pub camera: PerspectiveCamera,
    pub viewport: Vector2<u32>,
    pub gaussian_scaling: f32,
    pub max_sh_deg: u32,
    pub mip_splatting: Option<bool>,
    pub kernel_size: Option<f32>,
    pub clipping_box: Option<Aabb<f32>>,
    /// splat contributions (alpha) below this value are not drawn, see gaussian.wgsl
    pub alpha_cutoff: f32,
    pub walltime: Duration,
    pub scene_center: Option<Point3<f32>>,
    pub scene_extend: Option<f32>,
    pub background_color: wgpu::Color,
}

pub const DEFAULT_KERNEL_SIZE: f32 = 0.3;
/// one step of an 8 bit color
pub const DEFAULT_ALPHA_CUTOFF: f32 = 1. / 255.;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplattingArgsUniform {
    clipping_box_min: Vector4<f32>,
    clipping_box_max: Vector4<f32>,

    gaussian_scaling: f32,
    max_sh_deg: u32,
    mip_splatting: u32,
    kernel_size: f32,

    walltime: f32,
    scene_extend: f32,
    alpha_cutoff: f32,
    /// number of points that are loaded, the rest of the buffers is not initialized
    num_points: u32,

    scene_center: Vector4<f32>,
}

impl SplattingArgsUniform {
    /// replaces values with default values for point cloud
    pub fn from_args_and_pc(args: SplattingArgs, pc: &PointCloud) -> Self {
        Self {
            gaussian_scaling: args.gaussian_scaling,
            max_sh_deg: args.max_sh_deg,
            mip_splatting: args
                .mip_splatting
                .map(|v| v as u32)
                .unwrap_or(pc.mip_splatting().unwrap_or(false) as u32),
            kernel_size: args
                .kernel_size
                .unwrap_or(pc.dilation_kernel_size().unwrap_or(DEFAULT_KERNEL_SIZE)),
            clipping_box_min: args
                .clipping_box
                .map_or(pc.bbox().min, |b| b.min)
                .to_vec()
                .extend(0.),
            clipping_box_max: args
                .clipping_box
                .map_or(pc.bbox().max, |b| b.max)
                .to_vec()
                .extend(0.),
            walltime: args.walltime.as_secs_f32(),
            alpha_cutoff: args.alpha_cutoff,
            num_points: pc.loaded_points(),
            scene_center: pc.center().to_vec().extend(0.),
            scene_extend: args
                .scene_extend
                .unwrap_or(pc.bbox().radius())
                .max(pc.bbox().radius()),
            ..Default::default()
        }
    }
}

impl Default for SplattingArgsUniform {
    fn default() -> Self {
        Self {
            gaussian_scaling: 1.0,
            max_sh_deg: 3,
            mip_splatting: false as u32,
            kernel_size: DEFAULT_KERNEL_SIZE,
            clipping_box_max: Vector4::new(f32::INFINITY, f32::INFINITY, f32::INFINITY, 0.),
            clipping_box_min: Vector4::new(
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.,
            ),
            walltime: 0.,
            scene_center: Vector4::new(0., 0., 0., 0.),
            scene_extend: 1.,
            alpha_cutoff: DEFAULT_ALPHA_CUTOFF,
            num_points: 0,
        }
    }
}
