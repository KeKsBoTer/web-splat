use std::io::{ Seek, Read};

#[cfg(target_arch = "wasm32")]
use instant::{Duration,Instant};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration,Instant};

use cgmath::{Deg, EuclideanSpace, Point3, Quaternion, Vector2};
use egui::{epaint::Shadow, Rounding, TextStyle, Visuals};
use egui_plot::{Legend, PlotPoints};
use num_traits::One;

use utils::{key_to_num, RingBuffer};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

mod animation;
pub use animation::{Animation, TrackingShot, Transition};
mod camera;
pub use camera::{Camera, PerspectiveCamera, PerspectiveProjection};
mod controller;
pub use controller::CameraController;
mod pointcloud;
pub use pointcloud::PointCloud;

#[cfg(feature="npz")]
mod npz;
mod ply;

mod renderer;
pub use renderer::GaussianRenderer;

mod scene;
pub use self::scene::{Scene, SceneCamera, Split};

mod ui_renderer;
mod uniform;
mod utils;
pub mod gpu_rs;

#[cfg(not(target_arch = "wasm32"))]
pub use utils::download_buffer;


pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl WGPUContext {
    pub async fn new_instance() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        return WGPUContext::new(&instance, None).await;
    }

    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface>) -> Self {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        #[cfg(target_arch="wasm32")]
        let features =  wgpu::Features::default();
        #[cfg(not(target_arch="wasm32"))]
        let features  =wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM |wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features,
                    limits: wgpu::Limits {
                        max_storage_buffer_binding_size: (1 << 31) -1,
                        max_buffer_size: (1 << 31) -1,
                        max_storage_buffers_per_shader_stage: 12,
                        max_compute_workgroup_storage_size:1<<15,
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

pub struct RenderConfig {
    pub max_sh_deg: u32,
    pub no_vsync: bool,
}

struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    window: Window,
    scale_factor: f32,

    pc: PointCloud,
    renderer: GaussianRenderer,
    camera: PerspectiveCamera,
    animation: Option<Box<dyn Animation<Animatable = PerspectiveCamera>>>,
    controller: CameraController,
    scene: Option<Scene>,
    current_view: Option<usize>,
    ui_renderer: ui_renderer::EguiWGPU,
    fps: f32,
    ui_visible:bool,

    #[cfg(not(target_arch="wasm32"))]
    history: RingBuffer<(Duration, Duration, Duration)>,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<R:Read+Seek>(
        window: Window,
        event_loop: &EventLoop<()>,
        pc_file: R,
        pc_data_type: PCDataType,
        render_config: RenderConfig,
    ) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        let surface: wgpu::Surface = unsafe { instance.create_surface(&window) }.unwrap();

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        log::info!("device: {:?}", wgpu_context.adapter.get_info().name);

        let device = &wgpu_context.device;

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| !f.is_srgb())
            .next()
            .unwrap_or(&surface_caps.formats[0])
            .clone();


        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let pc = match pc_data_type {
            PCDataType::PLY => PointCloud::load_ply(
                &device,
                &wgpu_context.queue,
                pc_file,
                Some(render_config.max_sh_deg),
            )
            .unwrap(),
            #[cfg(feature="npz")]
            PCDataType::NPZ => PointCloud::load_npz(
                &device,
                &wgpu_context.queue,
                pc_file,
                Some(render_config.max_sh_deg),
            )
            .unwrap(),
        }; 
        log::info!("loaded point cloud with {:} points", pc.num_points());

        let renderer = GaussianRenderer::new(&device, surface_format, pc.sh_deg(),pc_data_type==PCDataType::PLY);

        let aspect = size.width as f32 / size.height as f32;
        let view_camera = PerspectiveCamera::new(
            Point3::origin(),
            Quaternion::one(),
            PerspectiveProjection::new(
                Vector2::new(size.width, size.height),
                Vector2::new(Deg(45.), Deg(45. / aspect)),
                0.1,
                100.,
            ),
        );

        let controller = CameraController::new(3., 0.25);
        let ui_renderer = ui_renderer::EguiWGPU::new(event_loop, device, surface_format);
        Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            renderer,
            pc,
            camera: view_camera,
            animation: None,
            controller,
            scene: None,
            current_view: None,
            ui_renderer,
            fps: 0.,
            #[cfg(not(target_arch="wasm32"))]
            history: RingBuffer::new(512),
            ui_visible:true
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;

            self.surface
                .configure(&self.wgpu_context.device, &self.config);
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
        if let Some(next_camera) = &mut self.animation {
            self.camera = next_camera.update(dt);
            if next_camera.done() {
                self.animation.take();
            }
        } else {
            self.controller.update_camera(&mut self.camera, dt);
        }
    }

    fn ui(&mut self) {
        let ctx = &self.ui_renderer.ctx;
        #[cfg(not(target_arch="wasm32"))]
        let stats =pollster::block_on(self.renderer.render_stats(&self.wgpu_context.device, &self.wgpu_context.queue));
        #[cfg(not(target_arch="wasm32"))]
        let num_drawn = pollster::block_on(self.renderer.num_visible_points(&self.wgpu_context.device, &self.wgpu_context.queue));
        
        #[cfg(not(target_arch="wasm32"))]
        self.history.push((
            stats.preprocess_time,
            stats.sort_time,
            stats.rasterization_time,
        ));
        ctx.set_visuals(Visuals {
            window_rounding: Rounding::ZERO,
            window_shadow: Shadow::NONE,
            ..Default::default()
        });

        #[cfg(not(target_arch="wasm32"))]
        egui::Window::new("Render Stats")
            .default_width(200.)
            .default_height(100.)
            .show(ctx, |ui| {
                egui::Grid::new("timing").num_columns(2).show(ui, |ui| {
                    ui.colored_label(egui::Color32::WHITE, "FPS");
                    ui.label(format!("{:}", self.fps as u32));
                    ui.end_row();
                    ui.colored_label(egui::Color32::WHITE, "Visible points");
                    ui.label(format!(
                        "{:} ({:.2}%)",
                        num_drawn,
                        (num_drawn as f32 / self.pc.num_points() as f32) * 100.
                    ));
                });
                let history = self.history.to_vec();
                let pre: Vec<f32> = history.iter().map(|v| v.0.as_secs_f32() * 1000.).collect();
                let sort: Vec<f32> = history.iter().map(|v| v.1.as_secs_f32() * 1000.).collect();
                let rast: Vec<f32> = history.iter().map(|v| v.2.as_secs_f32() * 1000.).collect();

                ui.label("Frame times (ms):");
                egui_plot::Plot::new("frame times")
                    .allow_drag(false)
                    .allow_boxed_zoom(false)
                    .allow_zoom(false)
                    .allow_scroll(false)
                    .show_x(false)
                    .y_axis_label("ms")
                    .show_axes([false, false])
                    .legend(Legend {
                        text_style: TextStyle::Small,
                        background_alpha: 1.,
                        position: egui_plot::Corner::LeftBottom,
                    })
                    .show(ui, |ui| {
                        let line =
                            egui_plot::Line::new(PlotPoints::from_ys_f32(&pre)).name("preprocess");
                        ui.line(line);
                        let line =
                            egui_plot::Line::new(PlotPoints::from_ys_f32(&sort)).name("sorting");
                        ui.line(line);
                        let line =
                            egui_plot::Line::new(PlotPoints::from_ys_f32(&rast)).name("rasterize");
                        ui.line(line);
                    });
            });

        egui::Window::new("Controls")
            .default_width(200.)
            .resizable(false)
            .default_height(100.)
            .show(ctx, |ui| {
                egui::Grid::new("timing")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.colored_label(egui::Color32::WHITE, "Camera");
                        ui.end_row();
                        ui.label("Sideways");
                        ui.label("W/A/S/D");
                        ui.end_row();
                        ui.label("");
                        ui.label("Up/Left/Down/Right");
                        ui.end_row();
                        ui.label("Up/Down");
                        ui.label("Space/Shift");
                        ui.end_row();

                        ui.label("Rotate (Yaw/Pitch)");
                        ui.label("Mouse");
                        ui.end_row();

                        ui.label("Rotate (Roll)");
                        ui.label("Q/E");
                        ui.end_row();

                        ui.colored_label(egui::Color32::WHITE, "Scene Views");
                        ui.end_row();
                        ui.label("Views 0-9");
                        ui.label("0-9");
                        ui.end_row();
                        ui.label("Random view");
                        ui.label("R");
                        ui.end_row();
                        ui.label("Next View");
                        ui.label("Page Up");
                        ui.end_row();
                        ui.label("Previous View");
                        ui.label("Page Down");
                        ui.end_row();
                        ui.label("Snap to nearest view");
                        ui.label("N");
                        ui.end_row();
                        ui.label("Start/Stop Tracking shot");
                        ui.label("T");
                        ui.end_row();
                    });
            });

        if let Some(scene) = &self.scene {

            let mut new_camera = None;
            let mut start_tracking_shot = false;
            egui::Window::new("Scene")
                .default_width(200.)
                .resizable(false)
                .default_height(100.)
                .show(ctx, |ui| {
                    ui.horizontal(|ui|{
                    let nearest = scene.nearest_camera(self.camera.position, None);
                    if ui.button("Snap to closest").clicked() {
                        new_camera = Some(nearest);
                    }
                    if let Some(c) = &mut self.current_view {
                        let drag = ui.add(
                            egui::DragValue::new(c)
                                .clamp_range(0..=(scene.num_cameras().saturating_sub(1))),
                        );
                        if drag.changed() {
                            new_camera = Some(*c);
                        }
                        ui.label(scene.camera(*c).split.to_string());
                    }
                    });
                    if ui.button("Start tracking shot").clicked() {
                        start_tracking_shot = true;
                    }
                });

            if let Some(c) = new_camera {
                self.current_view = new_camera;
                self.set_camera(scene.camera(c), Duration::from_millis(200));
            }
            if start_tracking_shot {
                self.start_tracking_shot(Some(Split::Test))
            }
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let viewport = Vector2::new(self.config.width, self.config.height);
        self.renderer.render(
            &self.wgpu_context.device,
            &self.wgpu_context.queue,
            &view,
            &self.pc,
            self.camera,
            viewport,
        );

        if self.ui_visible{
            // ui rendering
            self.ui_renderer.begin_frame(&self.window);
            self.ui();

            let shapes = self.ui_renderer.end_frame(&self.window);

            self.ui_renderer.paint(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &view,
                shapes,
            );
        }else{
            self.renderer.stopwatch.reset();
        }
        output.present();
        Ok(())
    }

    fn set_scene(&mut self, scene: Scene) {
        self.scene.replace(scene);
    }

    fn start_tracking_shot(&mut self, split: Option<Split>) {
        if let Some(scene) = &self.scene {
            self.animation = Some(Box::new(TrackingShot::from_scene(
                scene.cameras(split),
                1.,
                Some(self.camera.clone()),
            )));
        }
    }

    pub fn set_camera<C: Into<PerspectiveCamera>>(
        &mut self,
        camera: C,
        animation_duration: Duration,
    ) {
        if animation_duration.is_zero() {
            self.update_camera(camera.into())
        } else {
            let target_camera = camera.into();
            self.animation = Some(Box::new(Transition::new(
                self.camera.clone(),
                target_camera,
                animation_duration,
                smoothstep,
            )));
        }
    }

    fn update_camera(&mut self, camera: PerspectiveCamera) {
        self.camera = camera;
    }
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

#[derive(PartialEq)]
pub enum PCDataType {
    PLY,
    #[cfg(feature="npz")]
    NPZ,
}

pub async fn open_window<R: Read + Seek + Send + Sync + 'static>(file: R, pc_data_type: PCDataType, scene_file: Option<R>
    ,config: RenderConfig) {
        #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let event_loop = EventLoop::new();

    let scene = scene_file.map(|f| Scene::from_json(f).unwrap());

    let window_size = if let Some(scene) = &scene {
        let camera = scene.camera(0);
        let factor = 1200. / camera.width as f32;
        PhysicalSize::new(
            (camera.width as f32 * factor) as u32,
            (camera.height as f32 * factor) as u32,
        )
    } else {
        PhysicalSize::new(800, 600)
    };
    log::info!("rendering at resolution {}x{}px",window_size.width,window_size.height);

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
                .and_then(|doc| doc.body())
                .and_then(|body| {
                    let canvas = window.canvas();
                    canvas.set_width(body.client_width() as u32);
                    canvas.set_height(body.client_height() as u32);
                    let elm = web_sys::Element::from(canvas);
                    body.append_child(&elm).ok()
                })
                .expect("couldn't append canvas to document body");
        }

    let mut state = WindowContext::new(window, &event_loop, file, pc_data_type, config).await;

    if let Some(scene) = scene {
        let init_camera = scene.camera(0);
        state.set_scene(scene);
        state.set_camera(init_camera, Duration::ZERO);
        state.start_tracking_shot(Some(Split::Test));
    }

    let mut last = Instant::now();

    event_loop.run(move |event, _, control_flow| 
        match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() && !state.ui_renderer.on_event(event) => match event {
            WindowEvent::Resized(physical_size) => {
                state.resize(*physical_size, None);
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                new_inner_size,
                ..
            } => {
                state.resize(**new_inner_size, Some(*scale_factor as f32));
            }
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::KeyboardInput { input, .. } => {
                if let Some(key) = input.virtual_keycode {
                    if input.state == ElementState::Released{
                        if key == VirtualKeyCode::T{
                            state.start_tracking_shot(Some(Split::Test));
                            
                        }else if key == VirtualKeyCode::U{
                            state.ui_visible = !state.ui_visible;
                            
                        }else
                        if let Some(scene) = &state.scene{

                            let new_camera = 
                            if let Some(num) = key_to_num(key){
                                Some(num as usize)
                            }
                            else if key == VirtualKeyCode::R{
                                Some(rand::random::<usize>()%scene.num_cameras())
                            }else if key == VirtualKeyCode::N{
                                Some(scene.nearest_camera(state.camera.position,None))
                            }else if key == VirtualKeyCode::PageUp{
                                Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                            }else if key == VirtualKeyCode::T{
                                Some(state.current_view.map_or(0, |v|v+1) % scene.num_cameras())
                            }
                            else if key == VirtualKeyCode::PageDown{
                                Some(state.current_view.map_or(0, |v|v-1) % scene.num_cameras())
                            }else{None};

                            if let Some(new_camera) = new_camera{
                                state.current_view.replace(new_camera as usize);
                                log::info!("view moved to camera {new_camera}");
                                state.set_camera(scene.camera(new_camera as usize),Duration::from_millis(500));
                            }
                        }
                    }
                    state
                        .controller
                        .process_keyboard(key, input.state == ElementState::Pressed);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => match delta {
                winit::event::MouseScrollDelta::LineDelta(_, dy) => {
                    state.controller.process_scroll(*dy)
                }
                winit::event::MouseScrollDelta::PixelDelta(p) => {
                    state.controller.process_scroll(p.y as f32)
                }
            },
            WindowEvent::MouseInput { state:button_state, button, .. }=>{
                match button {
                    winit::event::MouseButton::Left => state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
                    winit::event::MouseButton::Right => state.controller.right_mouse_pressed = *button_state == ElementState::Pressed,
                    _=>{}
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
        Event::RedrawRequested(window_id) if window_id == state.window.id() => {
            let now = Instant::now();
            let dt = now-last;
            last = now;
            state.update(dt);

            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => state.resize(state.window.inner_size(), None),
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => println!("error: {:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            state.window.request_redraw();
        }
        _ => {}
    });
}


#[cfg(target_arch="wasm32")]
#[wasm_bindgen]
pub fn run_wasm(pc: Vec<u8>, scene: Option<Vec<u8>>) {
    use std::io::Cursor;
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    let pc_reader = Cursor::new(pc);
    let scene_reader = scene.map(|d: Vec<u8>| Cursor::new(d));

    wasm_bindgen_futures::spawn_local(open_window(pc_reader, scene_reader,RenderConfig { max_sh_deg: 3, sh_dtype: SHDType::Byte, no_vsync: false }));
}
