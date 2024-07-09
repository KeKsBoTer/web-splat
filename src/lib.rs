use std::{
    collections::hash_map::DefaultHasher, hash::{Hash, Hasher}, io::{Read, Seek}, os::linux::raw::stat, path::{Path, PathBuf}, sync::Arc
};

use image::Pixel;
#[cfg(target_arch = "wasm32")]
use instant::{Duration, Instant};
use renderer::{Display, PostProcessPipeline};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use wgpu::{util::DeviceExt, Backends, Extent3d};

use cgmath::{Deg, EuclideanSpace, Point3, Quaternion, UlpsEq, Vector2, Vector3};
use egui::Color32;
use num_traits::One;

use utils::key_to_num;
#[cfg(not(target_arch = "wasm32"))]
use utils::RingBuffer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

mod animation;
mod ui;
pub use animation::{Animation, Sampler, TrackingShot, Transition};
mod camera;
pub use camera::{Camera, PerspectiveCamera, PerspectiveProjection};
mod controller;
pub use controller::CameraController;
mod pointcloud;
pub use pointcloud::PointCloud;

pub mod io;

mod renderer;
pub use renderer::{GaussianRenderer, SplattingArgs};

mod scene;
use crate::utils::GPUStopwatch;

pub use self::scene::{Scene, SceneCamera, Split};

pub mod gpu_rs;
mod ui_renderer;
mod uniform;
mod utils;

pub struct RenderConfig {
    pub no_vsync: bool,
    pub skybox: Option<PathBuf>,
    pub hdr: bool,
}

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl WGPUContext {
    pub async fn new_instance() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        return WGPUContext::new(&instance, None).await;
    }

    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();
        log::info!("using {}",adapter.get_info().name);

        #[cfg(target_arch = "wasm32")]
        let required_features = wgpu::Features::default();
        #[cfg(not(target_arch = "wasm32"))]
        let required_features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;

        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features,
                    #[cfg(not(target_arch = "wasm32"))]
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                        max_buffer_size: (1 << 30) - 1,
                        max_storage_buffers_per_shader_stage: 12,
                        max_compute_workgroup_storage_size: 1 << 15,
                        ..Default::default()
                    },

                    #[cfg(target_arch = "wasm32")]
                    required_limits: wgpu::Limits {
                        max_compute_workgroup_storage_size: 1 << 15,
                        max_texture_dimension_1d: 4096,
                        max_texture_dimension_2d: 4096,
                        max_texture_dimension_3d: 1024,
                        max_uniform_buffer_binding_size: 16384,
                        max_vertex_buffer_array_stride: 0,
                        ..Default::default()
                    },
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        Self {
            device,
            queue,
            adapter,
        }
    }
}

pub struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    window: Arc<Window>,
    scale_factor: f32,

    pc: PointCloud,
    pointcloud_file_path: Option<PathBuf>,
    renderer: GaussianRenderer,
    animation: Option<(Animation<PerspectiveCamera>, bool)>,
    controller: CameraController,
    scene: Option<Scene>,
    scene_file_path: Option<PathBuf>,
    current_view: Option<usize>,
    ui_renderer: ui_renderer::EguiWGPU,
    fps: f32,
    ui_visible: bool,

    #[cfg(not(target_arch = "wasm32"))]
    history: RingBuffer<(Duration, Duration, Duration)>,
    display: Display,
    postprocess:PostProcessPipeline,

    background_color: egui::Color32,

    /// hash for the render settings
    /// if render settings dont change we dont have to rerender
    render_settings_hash: Option<u64>,
    splatting_args: SplattingArgs,

    saved_cameras: Vec<SceneCamera>,
    #[cfg(feature = "video")]
    cameras_save_path: String,
    stopwatch: Option<GPUStopwatch>,
    playing: bool,
    playing_speed: f32,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<R: Read + Seek>(
        window: Window,
        pc_file: R,
        render_config: &RenderConfig,
    ) -> anyhow::Result<Self> {
        let mut size = window.inner_size();
        if size == PhysicalSize::new(0, 0) {
            size = PhysicalSize::new(800, 600);
        }

        let window = Arc::new(window);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        log::info!("device: {:?}", wgpu_context.adapter.get_info().name);

        let device = &wgpu_context.device;
        let queue = &wgpu_context.queue;

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0])
            .clone();


        let render_format =wgpu::TextureFormat::Rgba16Float;// if render_config.hdr{ wgpu::TextureFormat::Rgba16Float}else{wgpu::TextureFormat::Rgba8Unorm};

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency: 2,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format.remove_srgb_suffix()],
        };
        surface.configure(&device, &config);

        let pc_raw = io::GenericGaussianPointCloud::load(pc_file)?;
        let pc = PointCloud::new(&device, pc_raw)?;
        log::info!("loaded point cloud with {:} points", pc.num_points());

        let renderer =
            GaussianRenderer::new(&device, &queue, render_format, pc.sh_deg(), pc.compressed())
                .await;

        let aabb = pc.bbox();
        let aspect = size.width as f32 / size.height as f32;
        let view_camera = PerspectiveCamera::new(
            aabb.center() - Vector3::new(1., 1., 1.) * aabb.radius() * 0.5,
            Quaternion::one(),
            PerspectiveProjection::new(
                Vector2::new(size.width, size.height),
                Vector2::new(Deg(45.), Deg(45. / aspect)),
                0.01,
                1000.,
            ),
        );

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = pc.center();
        // controller.up = pc.up;
        let ui_renderer = ui_renderer::EguiWGPU::new(device, surface_format, &window);

        let display = Display::new(
            device,
            render_format,
            surface_format.remove_srgb_suffix(),
            size.width,
            size.height,
        );


        let stopwatch = if cfg!(not(target_arch = "wasm32")) {
            Some(GPUStopwatch::new(device, Some(3)))
        } else {
            None
        };

        let postprocess = PostProcessPipeline::new(device,render_format,display.format());

        Ok(Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            renderer,
            splatting_args: SplattingArgs {
                camera: view_camera,
                viewport: Vector2::new(size.width, size.height),
                gaussian_scaling: 1.,
                max_sh_deg: pc.sh_deg(),
                show_env_map: false,
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: Duration::ZERO,
                scene_center: None,
                scene_extend: None,
                time:Duration::ZERO
            },
            pc,
            // camera: view_camera,
            controller,
            ui_renderer,
            fps: 0.,
            #[cfg(not(target_arch = "wasm32"))]
            history: RingBuffer::new(512),
            ui_visible: true,
            display,
            background_color: Color32::BLACK,
            saved_cameras: Vec::new(),
            #[cfg(feature = "video")]
            cameras_save_path: "cameras_saved.json".to_string(),
            animation: None,
            scene: None,
            current_view: None,
            render_settings_hash: None,
            pointcloud_file_path: None,
            scene_file_path: None,

            stopwatch,
            playing:true,
            playing_speed:1.,
            postprocess
        })
    }

    fn reload(&mut self) -> anyhow::Result<()> {
        if let Some(file_path) = &self.pointcloud_file_path {
            log::info!("reloading volume from {:?}", file_path);
            let file = std::fs::File::open(file_path)?;
            let pc_raw = io::GenericGaussianPointCloud::load(file)?;
            self.pc = PointCloud::new(&self.wgpu_context.device, pc_raw)?;
        } else {
            return Err(anyhow::anyhow!("no pointcloud file path present"));
        }
        if let Some(scene_path) = &self.scene_file_path {
            log::info!("reloading scene from {:?}", scene_path);
            let file = std::fs::File::open(scene_path)?;

            self.set_scene(Scene::from_json(file)?);
        }
        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface
                .configure(&self.wgpu_context.device, &self.config);
            self.display
                .resize(&self.wgpu_context.device, new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
            self.splatting_args.viewport = Vector2::new(new_size.width, new_size.height);
            self.splatting_args.camera.projection
            .resize(new_size.width,new_size.height);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    fn update(&mut self, dt: Duration) {
        // ema fps update
        self.fps = (1. / dt.as_secs_f32()) * 0.05 + self.fps * 0.95;
        self.splatting_args.walltime += dt;

        // add dt and make sure it is between 0 and 1
        if self.playing{
            self.splatting_args.time = Duration::from_secs_f32((self.splatting_args.time.as_secs_f32() + dt.as_secs_f32()*self.playing_speed).fract());
        }
        if let Some((next_camera, playing)) = &mut self.animation {
            if self.controller.user_inptut {
                self.cancle_animation()
            } else {
                let dt = if *playing { dt } else { Duration::ZERO };
                self.splatting_args.camera = next_camera.update(dt);
                self.splatting_args.camera.projection.resize(self.config.width, self.config.height);
                if next_camera.done() {
                    self.animation.take();
                    self.controller.reset_to_camera(self.splatting_args.camera);
                }
            }
        } else {
            self.controller
                .update_camera(&mut self.splatting_args.camera, dt);

            // check if camera moved out of selected view
            if let Some(idx) = self.current_view {
                if let Some(scene) = &self.scene {
                    if let Some(camera) = scene.camera(idx) {
                        let scene_camera: PerspectiveCamera = camera.into();
                        if !self.splatting_args.camera.position.ulps_eq(
                            &scene_camera.position,
                            1e-4,
                            f32::default_max_ulps(),
                        ) || !self.splatting_args.camera.rotation.ulps_eq(
                            &scene_camera.rotation,
                            1e-4,
                            f32::default_max_ulps(),
                        ) {
                            self.current_view.take();
                        }
                    }
                }
            }
        }

        let aabb = self.pc.bbox();
        self.splatting_args.camera.fit_near_far(aabb);
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.stopwatch.as_mut().map(|s| s.reset());
        let window_size = self.window.inner_size();
        if window_size.width != self.config.width || window_size.height != self.config.height {
            self.resize(window_size, None);
        }

        let output = self.surface.get_current_texture()?;
        let view_rgb = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let view_srgb = output.texture.create_view(&Default::default());
        let rgba = self.background_color.to_srgba_unmultiplied();

        let mut hasher = DefaultHasher::new();
        self.splatting_args.hash(&mut hasher);
        let settings_hash = hasher.finish();

        // do prepare stuff

        let mut encoder =
            self.wgpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render command encoder"),
                });

        let redraw = self
            .render_settings_hash
            .and_then(|v| Some(v != settings_hash))
            .unwrap_or(true);

        if redraw {
            self.renderer.prepare(
                &mut encoder,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &self.pc,
                self.splatting_args,
                (&mut self.stopwatch).into(),
            );
            self.render_settings_hash.replace(settings_hash);
        }

        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.start(&mut encoder, "rasterization").unwrap();
        }
        let intermediate_textures = [0,1,2].map(|_|self.wgpu_context.device.create_texture(&wgpu::TextureDescriptor{ 
            label: Some("intermediate texture"),
            size: Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1, 
            sample_count: 1, 
            dimension: wgpu::TextureDimension::D2, 
            format: self.renderer.color_format(), 
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::STORAGE_BINDING, 
            view_formats: &[],
        }));

        let intermediate_texture_view:Vec<wgpu::TextureView> = intermediate_textures.iter().map(|t|t.create_view(&Default::default())).collect();

        if redraw {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments:&intermediate_texture_view.iter().map(|v|{
                    Some(wgpu::RenderPassColorAttachment {
                            view: &v,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: self.background_color.r() as f64 / 255.,
                                    g: self.background_color.g() as f64 / 255.,
                                    b: self.background_color.b() as f64 / 255.,
                                    a: 1.,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })
                }).collect::<Vec<Option<wgpu::RenderPassColorAttachment>>>(),
                ..Default::default()
            });
            self.renderer.render(&mut render_pass, &self.pc);
        }
        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.stop(&mut encoder, "rasterization").unwrap();
        }


        let intermediate_bind_group_layout = self.wgpu_context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("intermediate bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ], 
        });
        let intermediate_bind_group = self.wgpu_context.device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: Some("intermediate bind group"), 
            layout: &intermediate_bind_group_layout, 
            entries: &[wgpu::BindGroupEntry { 
                binding: 0, 
                resource: wgpu::BindingResource::TextureView(&intermediate_texture_view[0]), 
            },wgpu::BindGroupEntry { 
                binding: 1, 
                resource: wgpu::BindingResource::TextureView(&intermediate_texture_view[1]), 
            },wgpu::BindGroupEntry { 
                binding: 2, 
                resource: wgpu::BindingResource::TextureView(&intermediate_texture_view[2]), 
            }],
        });

        let resolve_bind_group_layout = self.wgpu_context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: self.display.format(),
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ], 
        });
        let resolve_bind_group = self.wgpu_context.device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: Some("resolve bind group"), 
            layout: &resolve_bind_group_layout, 
            entries: &[wgpu::BindGroupEntry { 
                binding: 0, 
                resource: wgpu::BindingResource::TextureView(&self.display.texture()), 
            }],
        });
        self.postprocess.run(&mut encoder,&intermediate_bind_group, &resolve_bind_group, Vector2::new(self.config.width, self.config.height));


        self.display.render(
            &mut encoder,
            &view_rgb,
            wgpu::Color {
                r: rgba[0] as f64 / 255.,
                g: rgba[1] as f64 / 255.,
                b: rgba[2] as f64 / 255.,
                a: rgba[3] as f64 / 255.,
            },
            self.renderer.camera(),
            &self.renderer.render_settings(),
        );
        self.stopwatch.as_mut().map(|s| s.end(&mut encoder));
        self.wgpu_context.queue.submit([encoder.finish()]);

        if self.ui_visible {
            // ui rendering
            self.ui_renderer.begin_frame(&self.window);
            ui::ui(self);

            let shapes = self.ui_renderer.end_frame(&self.window);

            self.ui_renderer.paint(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &view_srgb,
                shapes,
            );
        }

        output.present();
        Ok(())
    }

    fn set_scene(&mut self, scene: Scene) {
        self.splatting_args.scene_extend = Some(scene.extend());
        let mut center = Point3::origin();
        for c in scene.cameras(None) {
            let z_axis: Vector3<f32> = c.rotation[2].into();
            center += Vector3::from(c.position) + z_axis * 2.;
        }
        center /= scene.num_cameras() as f32;

        self.controller.center = center;
        self.scene.replace(scene);
        if self.saved_cameras.is_empty() {
            self.saved_cameras = self
                .scene
                .as_ref()
                .unwrap()
                .cameras(Some(Split::Test))
                .clone();
        }
    }

    fn set_env_map<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        let env_map_exr = image::open(path)?;
        let env_map_data: Vec<[f32; 4]> = env_map_exr
            .as_rgb32f()
            .ok_or(anyhow::anyhow!("env map must be rgb"))?
            .pixels()
            .map(|p| p.to_rgba().0)
            .collect();

        let env_texture = self.wgpu_context.device.create_texture_with_data(
            &self.wgpu_context.queue,
            &wgpu::TextureDescriptor {
                label: Some("env map texture"),
                size: Extent3d {
                    width: env_map_exr.width(),
                    height: env_map_exr.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&env_map_data.as_slice()),
        );
        self.display.set_env_map(
            &self.wgpu_context.device,
            Some(&env_texture.create_view(&Default::default())),
        );
        self.splatting_args.show_env_map = true;
        Ok(())
    }

    fn start_tracking_shot(&mut self) {
        if self.saved_cameras.len() > 1 {
            let shot = TrackingShot::from_cameras(self.saved_cameras.clone());
            let a = Animation::new(
                Duration::from_secs_f32(self.saved_cameras.len() as f32 * 2.),
                true,
                Box::new(shot),
            );
            self.animation = Some((a, true));
        }
    }

    fn cancle_animation(&mut self) {
        self.animation.take();
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn stop_animation(&mut self) {
        if let Some((_animation, playing)) = &mut self.animation {
            *playing = false;
        }
        self.controller.reset_to_camera(self.splatting_args.camera);
    }

    fn set_scene_camera(&mut self, i: usize) {
        if let Some(scene) = &self.scene {
            self.current_view.replace(i);
            log::info!("view moved to camera {i}");
            if let Some(camera) = scene.camera(i) {
                self.set_camera(camera, Duration::from_millis(200));
            } else {
                log::error!("camera {i} not found");
            }
        }
    }

    pub fn set_camera<C: Into<PerspectiveCamera>>(
        &mut self,
        camera: C,
        animation_duration: Duration,
    ) {
        let camera: PerspectiveCamera = camera.into();
        if animation_duration.is_zero() {
            self.update_camera(camera.into())
        } else {
            let target_camera = camera.into();
            let a = Animation::new(
                animation_duration,
                false,
                Box::new(Transition::new(
                    self.splatting_args.camera.clone(),
                    target_camera,
                    smoothstep,
                )),
            );
            self.animation = Some((a, true));
        }
    }

    fn update_camera(&mut self, camera: PerspectiveCamera) {
        self.splatting_args.camera = camera;
        self.splatting_args.camera.projection.resize(self.config.width, self.config.height);
    }

    fn save_view(&mut self) {
        let max_scene_id = if let Some(scene) = &self.scene {
            scene.cameras(None).iter().map(|c| c.id).max().unwrap_or(0)
        } else {
            0
        };
        let max_id = self.saved_cameras.iter().map(|c| c.id).max().unwrap_or(0);
        let id = max_id.max(max_scene_id) + 1;
        self.saved_cameras.push(SceneCamera::from_perspective(
            self.splatting_args.camera,
            id.to_string(),
            id,
            Vector2::new(self.config.width, self.config.height),
            Split::Test,
        ));
    }
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

pub async fn open_window<R: Read + Seek + Send + Sync + 'static>(
    file: R,
    scene_file: Option<R>,
    config: RenderConfig,
    pointcloud_file_path: Option<PathBuf>,
    scene_file_path: Option<PathBuf>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();

    let scene = scene_file.and_then(|f| match Scene::from_json(f) {
        Ok(s) => Some(s),
        Err(err) => {
            log::error!("cannot load scene: {:?}", err);
            None
        }
    });

    let window_size = if let Some(scene) = &scene {
        let camera = scene.camera(0).unwrap();
        let factor = 1200. / camera.width as f32;
        PhysicalSize::new(
            (camera.width as f32 * factor) as u32,
            (camera.height as f32 * factor) as u32,
        )
    } else {
        PhysicalSize::new(800, 600)
    };

    let window = WindowBuilder::new()
        .with_title("web-splats")
        .with_inner_size(window_size)
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                doc.get_element_by_id("loading-display")
                    .unwrap()
                    .set_text_content(Some("Unpacking"));
                doc.body()
            })
            .and_then(|body| {
                let canvas = window.canvas().unwrap();
                canvas.set_id("window-canvas");
                canvas.set_width(body.client_width() as u32);
                canvas.set_height(body.client_height() as u32);
                let elm = web_sys::Element::from(canvas);
                elm.set_attribute("style", "width: 100%; height: 100%;")
                    .unwrap();
                body.append_child(&elm).ok()
            })
            .expect("couldn't append canvas to document body");
    }

    let mut state = WindowContext::new(window, file, &config).await.unwrap();
    state.pointcloud_file_path = pointcloud_file_path;

    if let Some(scene) = scene {
        state.set_scene(scene);
        state.set_scene_camera(0);
        state.scene_file_path = scene_file_path;
    }

    if let Some(skybox) = &config.skybox {
        if let Err(e) = state.set_env_map(skybox.as_path()) {
            log::error!("failed do set skybox: {e}");
        }
    }

    #[cfg(target_arch = "wasm32")]
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            doc.get_element_by_id("spinner")
                .unwrap()
                .set_attribute("style", "display:none;")
                .unwrap();
            doc.body()
        });

    let mut last = Instant::now();

    event_loop.run(move |event,target| 
        
        match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() && !state.ui_renderer.on_event(&state.window,event) => match event {
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size, None);
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                ..
            } => {
                state.scale_factor = *scale_factor as f32;
            }
            WindowEvent::CloseRequested => {log::info!("close!");target.exit()},
            WindowEvent::ModifiersChanged(m)=>{
                state.controller.alt_pressed = m.state().alt_key();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key{
                if event.state == ElementState::Released{

                    if key == KeyCode::KeyT{
                        if state.animation.is_none(){
                            state.start_tracking_shot();
                        }else{
                            state.stop_animation()
                        }
                    }else if key == KeyCode::KeyU{
                        state.ui_visible = !state.ui_visible;
                        
                    }else if key == KeyCode::KeyC{
                        state.save_view();
                    } else  if key == KeyCode::KeyR && state.controller.alt_pressed{
                        if let Err(err) = state.reload(){
                            log::error!("failed to reload volume: {:?}", err);
                        }   
                    }else if let Some(scene) = &state.scene{

                        let new_camera = 
                        if let Some(num) = key_to_num(key){
                            Some(num as usize)
                        }
                        else if key == KeyCode::KeyR{
                            Some(rand::random::<usize>()%scene.num_cameras())
                        }else if key == KeyCode::KeyN{
                            scene.nearest_camera(state.splatting_args.camera.position,None)
                        }else if key == KeyCode::PageUp{
                            Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                        }else if key == KeyCode::KeyT{
                            Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                        }
                        else if key == KeyCode::PageDown{
                            Some(state.current_view.map_or(0, |v|v-1) % scene.num_cameras())
                        }else{None};

                        if let Some(new_camera) = new_camera{
                            state.set_scene_camera(new_camera);
                        }
                    }
                }
                state
                    .controller
                    .process_keyboard(key, event.state == ElementState::Pressed);
            }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                    state.controller.process_scroll(*dy )
                }
                winit::event::MouseScrollDelta::PixelDelta(p) => {
                    state.controller.process_scroll(p.y as f32 / 100.)
                }
            },
            WindowEvent::MouseInput { state:button_state, button, .. }=>{
                match button {
                    winit::event::MouseButton::Left =>                         state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
                    winit::event::MouseButton::Right => state.controller.right_mouse_pressed = *button_state == ElementState::Pressed,
                    _=>{}
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now-last;
                last = now;
                state.update(dt);
    
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size(), None),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) =>target.exit(),
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => println!("error: {:?}", e),
                }
            }
            _ => {}
        },
        Event::DeviceEvent {
            event: DeviceEvent::MouseMotion{ delta, },
            .. // We're not using device_id currently
        } => {
            state.controller.process_mouse(delta.0 as f32, delta.1 as f32)
        }
        
        Event::AboutToWait => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            state.window.request_redraw();
        }
        _ => {},
    }).unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn run_wasm(pc: Vec<u8>, scene: Option<Vec<u8>>,pc_file:Option<String>,scene_file:Option<String>) {
    use std::{io::Cursor, str::FromStr};

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    let pc_reader = Cursor::new(pc);
    let scene_reader = scene.map(|d: Vec<u8>| Cursor::new(d));

    wasm_bindgen_futures::spawn_local(open_window(
        pc_reader,
        scene_reader,
        RenderConfig { no_vsync: false,skybox:None,hdr:false },
        pc_file.and_then(|s|PathBuf::from_str(s.as_str()).ok()),
        scene_file.and_then(|s|PathBuf::from_str(s.as_str()).ok()),
    ));
}
