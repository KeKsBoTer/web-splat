#[cfg(target_arch = "wasm32")]
use instant::{Duration, Instant};
use renderer::{Display, FrameBuffer};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use std::{
    io::{Cursor, Read, Seek},
    path::PathBuf,
    sync::Arc,
};

use cgmath::{Deg, EuclideanSpace, Point3, Quaternion, UlpsEq, Vector2, Vector3};
use egui::FullOutput;
use num_traits::One;

use utils::key_to_num;
#[cfg(not(target_arch = "wasm32"))]
use utils::RingBuffer;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalSize},
    event::{DeviceEvent, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
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
    pub hdr: bool,
}

pub async fn new_wgpu_context(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'static>>,
) -> (wgpu::Device, wgpu::Queue, wgpu::Adapter) {
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
        .await
        .unwrap();
    log::info!("using {}", adapter.get_info().name);

    #[cfg(target_arch = "wasm32")]
    let required_features = wgpu::Features::default();
    #[cfg(not(target_arch = "wasm32"))]
    let required_features = wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
        | wgpu::Features::CLEAR_TEXTURE
        | wgpu::Features::EXPERIMENTAL_FRAGMENT_SHADER_INTERLOCK;

    let adapter_limits = adapter.limits();


    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features,
                #[cfg(not(target_arch = "wasm32"))]
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    max_storage_buffers_per_shader_stage: 12,
                    max_compute_workgroup_storage_size: 1 << 15,
                    ..adapter_limits
                },

                #[cfg(target_arch = "wasm32")]
                required_limits: wgpu::Limits {
                    max_compute_workgroup_storage_size: 1 << 15,
                    ..adapter_limits
                },
                label: None,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            },
        )
        .await
        .unwrap();

    return (device, queue, adapter);
}

pub struct Application {
    web_splat: Option<WebSplat>,
    pc: io::GenericGaussianPointCloud,
    scene: Option<Scene>,
    config: RenderConfig,
}

pub struct WebSplat {
    device: wgpu::Device,
    queue: wgpu::Queue,
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

    splatting_args: SplattingArgs,

    saved_cameras: Vec<SceneCamera>,
    #[cfg(feature = "video")]
    cameras_save_path: String,
    stopwatch: Option<GPUStopwatch>,

    last_draw: Instant,
    min_wait: Duration,

    vsync: bool,

    frame_buffer: FrameBuffer
}

impl WebSplat {
    // Creating some of the wgpu types requires async code
    async fn new(
        window: Window,
        pc: &io::GenericGaussianPointCloud,
        render_config: &RenderConfig,
    ) -> anyhow::Result<Self> {
        let window = Arc::new(window);
        let mut size = window.inner_size();
        if size == PhysicalSize::new(0, 0) {
            size = PhysicalSize::new(800, 600);
        }

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;
        let (device, queue, adapter) = new_wgpu_context(&instance, Some(&surface)).await;

        log::info!("device: {:?}", adapter.get_info().name);

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0])
            .clone();

        let render_format = if render_config.hdr {
            wgpu::TextureFormat::Rgba16Float
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            desired_maximum_frame_latency:1,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![surface_format.remove_srgb_suffix()],
        };
        surface.configure(&device, &config);

        let pc = PointCloud::new(&device, pc)?;
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
        let ui_renderer = ui_renderer::EguiWGPU::new(&device, surface_format, &window);

        let display = Display::new(
            &device,
            render_format,
            surface_format.remove_srgb_suffix(),
            size.width,
            size.height,
        );

        let stopwatch = if cfg!(not(target_arch = "wasm32")) {
            Some(GPUStopwatch::new(&device, Some(3)))
        } else {
            None
        };

        let min_wait = Duration::from_secs_f32(1000. / 60.);

        let last_draw = Instant::now();

        let frame_buffer = FrameBuffer::new(&device, size.width, size.height, render_format);

        Ok(Self {
            device,
            queue,
            scale_factor: 1.,
            window,
            surface,
            config,
            renderer,
            splatting_args: SplattingArgs {
                camera: view_camera,
                viewport: Vector2::new(size.width, size.height),
                gaussian_scaling: 1.,
                max_sh_deg: pc.sh_deg(),
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: Duration::ZERO,
                scene_center: None,
                scene_extend: None,
                background_color: wgpu::Color::BLACK,
                resolution: Vector2::new(size.width, size.height),
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
            saved_cameras: Vec::new(),
            #[cfg(feature = "video")]
            cameras_save_path: "cameras_saved.json".to_string(),
            animation: None,
            scene: None,
            current_view: None,
            pointcloud_file_path: None,
            scene_file_path: None,
            stopwatch,
            last_draw,
            min_wait,
            vsync: !render_config.no_vsync,
            frame_buffer
        })
    }

    fn reload(&mut self) -> anyhow::Result<()> {
        if let Some(file_path) = &self.pointcloud_file_path {
            log::info!("reloading volume from {:?}", file_path);
            let file = std::fs::File::open(file_path)?;
            let pc_raw = io::GenericGaussianPointCloud::load(file)?;
            self.pc = PointCloud::new(&self.device, &pc_raw)?;
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
            self.surface.configure(&self.device, &self.config);
            self.display
                .resize(&self.device, new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
            self.splatting_args.viewport = Vector2::new(new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
            self.frame_buffer.resize(&self.device, new_size.width, new_size.height);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    /// returns whether redraw is required
    fn ui(&mut self) -> (bool, egui::FullOutput) {
        self.ui_renderer.begin_frame(&self.window);
        let request_redraw = ui::ui(self);

        let shapes = self.ui_renderer.end_frame(&self.window);

        return (request_redraw, shapes);
    }

    /// returns whether the sceen changed and we need a redraw
    fn update(&mut self, dt: Duration) {
        // ema fps update

        if self.splatting_args.walltime < Duration::from_secs(5) {
            self.splatting_args.walltime += dt;
        }
        if let Some((next_camera, playing)) = &mut self.animation {
            if self.controller.user_inptut {
                self.cancle_animation()
            } else {
                let dt = if *playing { dt } else { Duration::ZERO };
                self.splatting_args.camera = next_camera.update(dt);
                self.splatting_args
                    .camera
                    .projection
                    .resize(self.config.width, self.config.height);
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

    fn render(
        &mut self,
        redraw_scene: bool,
        shapes: Option<FullOutput>,
    ) -> Result<(), wgpu::SurfaceError> {
        self.stopwatch.as_mut().map(|s| s.reset());

        let output = self.surface.get_current_texture()?;
        let view_rgb = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let view_srgb = output.texture.create_view(&Default::default());
        // do prepare stuff

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render command encoder"),
            });

        if redraw_scene {
            self.renderer.prepare(
                &mut encoder,
                &self.device,
                &self.queue,
                &self.pc,
                self.splatting_args,
                (&mut self.stopwatch).into(),
                &self.frame_buffer
            );
        }

        let ui_state = shapes.map(|shapes| {
            self.ui_renderer.prepare(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.device,
                &self.queue,
                &mut encoder,
                shapes,
            )
        });

        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.start(&mut encoder, "rasterization").unwrap();
        }
        if redraw_scene {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.display.texture(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.splatting_args.background_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.renderer.render(&mut render_pass, &self.pc,&self.frame_buffer);
        }
        if let Some(stopwatch) = &mut self.stopwatch {
            stopwatch.stop(&mut encoder, "rasterization").unwrap();
        }

        self.display.render(
            &mut encoder,
            &view_rgb,
            self.splatting_args.background_color,
            self.renderer.camera(),
            &self.renderer.render_settings(),
            &self.frame_buffer
        );
        self.stopwatch.as_mut().map(|s| s.end(&mut encoder));

        if let Some(state) = &ui_state {
            let mut render_pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass ui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view_srgb,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.ui_renderer.render(&mut render_pass, state);
        }

        if let Some(ui_state) = ui_state {
            self.ui_renderer.cleanup(ui_state)
        }
        self.queue.submit([encoder.finish()]);

        output.present();
        self.splatting_args.resolution = Vector2::new(self.config.width, self.config.height);
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
        self.splatting_args
            .camera
            .projection
            .resize(self.config.width, self.config.height);
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

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: WindowEvent,
    ) {
        if self.ui_visible && self.ui_renderer.on_event(&self.window, &event) {
            return;
        }
        match event {
            WindowEvent::Resized(physical_size) => {
                self.resize(physical_size, None);
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale_factor = scale_factor as f32;
            }
            WindowEvent::CloseRequested => {
                log::info!("close!");
                event_loop.exit()
            }
            WindowEvent::ModifiersChanged(m) => {
                self.controller.alt_pressed = m.state().alt_key();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Released {
                        if key == KeyCode::KeyT {
                            if self.animation.is_none() {
                                self.start_tracking_shot();
                            } else {
                                self.stop_animation()
                            }
                        } else if key == KeyCode::KeyU {
                            self.ui_visible = !self.ui_visible;
                        } else if key == KeyCode::KeyC {
                            self.save_view();
                        } else if key == KeyCode::KeyR && self.controller.alt_pressed {
                            if let Err(err) = self.reload() {
                                log::error!("failed to reload volume: {:?}", err);
                            }
                        } else if let Some(scene) = &self.scene {
                            let new_camera = if let Some(num) = key_to_num(key) {
                                Some(num as usize)
                            } else if key == KeyCode::KeyR {
                                Some(rand::random::<usize>() % scene.num_cameras())
                            } else if key == KeyCode::KeyN {
                                scene.nearest_camera(self.splatting_args.camera.position, None)
                            } else if key == KeyCode::PageUp {
                                Some(self.current_view.map_or(0, |v| v + 1) % scene.num_cameras())
                            } else if key == KeyCode::KeyT {
                                Some(self.current_view.map_or(0, |v| v + 1) % scene.num_cameras())
                            } else if key == KeyCode::PageDown {
                                Some(self.current_view.map_or(0, |v| v - 1) % scene.num_cameras())
                            } else {
                                None
                            };

                            if let Some(new_camera) = new_camera {
                                self.set_scene_camera(new_camera);
                            }
                        }
                    }
                    self.controller
                        .process_keyboard(key, event.state == ElementState::Pressed);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                    self.controller.process_scroll(dy)
                }
                winit::event::MouseScrollDelta::PixelDelta(p) => {
                    self.controller.process_scroll(p.y as f32 / 100.)
                }
            },

            WindowEvent::MouseInput {
                state: button_self,
                button,
                ..
            } => match button {
                winit::event::MouseButton::Left => {
                    self.controller.left_mouse_pressed = button_self == ElementState::Pressed
                }
                winit::event::MouseButton::Right => {
                    self.controller.right_mouse_pressed = button_self == ElementState::Pressed
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                // if self.vsync {
                //     // make sure the next redraw is called with a small delay
                //     event_loop.set_control_flow(ControlFlow::wait_duration(self.min_wait));
                // }
                let now = Instant::now();
                let dt = now - self.last_draw;
                self.last_draw = now;

                let old_settings = self.splatting_args.clone();
                self.update(dt);

                let (redraw_ui, shapes) = self.ui();

                let resolution_change = self.splatting_args.resolution
                    != Vector2::new(self.config.width, self.config.height);

                let request_redraw = old_settings != self.splatting_args || resolution_change;

                if request_redraw || redraw_ui {
                    self.fps = (1. / dt.as_secs_f32()) * 0.05 + self.fps * 0.95;
                    match self.render(request_redraw, self.ui_visible.then_some(shapes)) {
                        Ok(_) => {}
                        // Reconfigure the surface if lost
                        Err(wgpu::SurfaceError::Lost) => {
                            self.resize(self.window.inner_size(), None)
                        }
                        // The system is out of memory, we should probably quit
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => println!("error: {:?}", e),
                    }
                }
                // if !self.vsync {
                    self.window.request_redraw();
                // }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler<()> for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = Window::default_attributes().with_inner_size(LogicalSize::new(800, 600)).with_title(format!(
            "{} ({})",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ));
        let window = event_loop.create_window(window_attributes).unwrap();

        let mut state = pollster::block_on(WebSplat::new(
            window,
            &self.pc,
            &RenderConfig {
                no_vsync: self.config.no_vsync,
                hdr: self.config.hdr,
            },
        ))
        .unwrap();
        // state.pointcloud_file_path = pointcloud_file_path;

        if let Some(scene) = &self.scene {
            state.set_scene(scene.clone());
            state.set_scene_camera(0);
            // state.scene_file_path = scene_file_path;
        }

        self.web_splat = Some(state);

        if let Some(state) = self.web_splat.as_mut() {
            state.window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Some(state) = self.web_splat.as_mut() {
            if window_id == state.window.id() {
                state.window_event(event_loop, event);
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => self
                .web_splat
                .as_mut()
                .unwrap()
                .controller
                .process_mouse(delta.0 as f32, delta.1 as f32),
            _ => {}
        }
    }

    fn new_events(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        match cause {
            winit::event::StartCause::ResumeTimeReached { .. } => {
                if let Some(state) = self.web_splat.as_mut() {
                    state.window.request_redraw();
                }
            }
            winit::event::StartCause::WaitCancelled { .. }=>{
                if let Some(state) = self.web_splat.as_mut() {
                    state.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

pub async fn open_window(
    mut file: Box<dyn Read>,
    scene_file: Option<Box<dyn Read>>,
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

    let mut data = Vec::new();
    file.read_to_end(&mut data).unwrap();
    let pc = io::GenericGaussianPointCloud::load(Cursor::new(data)).unwrap();

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

    let mut app = Application {
        web_splat: None,
        pc,
        scene,
        config,
    };
    // limit the redraw rate to the monitor refresh rate

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

    event_loop.run_app(&mut app).unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn run_wasm(
    pc: Vec<u8>,
    scene: Option<Vec<u8>>,
    pc_file: Option<String>,
    scene_file: Option<String>,
) {
    use std::{io::Cursor, str::FromStr};

    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    let pc_reader = Cursor::new(pc);
    let scene_reader = scene.map(|d: Vec<u8>| Cursor::new(d));

    wasm_bindgen_futures::spawn_local(open_window(
        pc_reader,
        scene_reader,
        RenderConfig {
            no_vsync: false,
            skybox: None,
            hdr: false,
        },
        pc_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
        scene_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
    ));
}