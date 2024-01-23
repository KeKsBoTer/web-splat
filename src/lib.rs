use std::io::{Read, Seek};

#[cfg(target_arch = "wasm32")]
use instant::{Duration, Instant};
use renderer::Display;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use wgpu::Backends;

use cgmath::{Deg, EuclideanSpace, MetricSpace, Point3, Quaternion, Vector2, Vector3};
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
    keyboard::{ PhysicalKey, KeyCode},
    window::{Window, WindowBuilder},
};

mod animation;
mod ui;
pub use animation::{Sampler, TrackingShot, Transition,Animation};
mod camera;
pub use camera::{Camera, PerspectiveCamera, PerspectiveProjection};
mod controller;
pub use controller::CameraController;
mod pointcloud;
pub use pointcloud::PointCloud;

#[cfg(feature = "npz")]
mod npz;
mod ply;

mod renderer;
pub use renderer::GaussianRenderer;

mod scene;
pub use self::scene::{Scene, SceneCamera, Split};

pub mod gpu_rs;
mod ui_renderer;
mod uniform;
mod utils;

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

    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();

        #[cfg(target_arch = "wasm32")]
        let features = wgpu::Features::default();
        #[cfg(not(target_arch = "wasm32"))]
        let features = wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features,
                    #[cfg(not(target_arch = "wasm32"))]
                    limits: wgpu::Limits {
                        max_storage_buffer_binding_size: (1 << 30) - 1,
                        max_buffer_size: (1 << 30) - 1,
                        max_storage_buffers_per_shader_stage: 12,
                        max_compute_workgroup_storage_size: 1 << 15,
                        ..Default::default()
                    },
                    #[cfg(target_arch = "wasm32")]
                    limits: wgpu::Limits {
                        max_compute_workgroup_storage_size: 1 << 15,
                        max_texture_dimension_1d: 4096,
                        max_texture_dimension_2d: 4096,
                        max_texture_dimension_3d: 1024,
                        max_uniform_buffer_binding_size: 16384,
                        max_vertex_buffer_array_stride:0,
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
    pub no_vsync: bool,
}

pub struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    window: Window,
    scale_factor: f32,

    pc: PointCloud,
    renderer: GaussianRenderer,
    camera: PerspectiveCamera,
    animation: Option<(Animation<PerspectiveCamera>,bool)>,
    controller: CameraController,
    scene: Option<Scene>,
    current_view: Option<usize>,
    ui_renderer: ui_renderer::EguiWGPU,
    fps: f32,
    ui_visible: bool,

    #[cfg(not(target_arch = "wasm32"))]
    history: RingBuffer<(Duration, Duration, Duration)>,
    display: Display,

    background_color: egui::Color32,

    saved_cameras: Vec<SceneCamera>,
    cameras_save_path: String,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<R: Read + Seek>(window: Window, pc_file: R, render_config: RenderConfig) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let surface: wgpu::Surface = unsafe { instance.create_surface(&window) }.unwrap();

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

        let render_format = wgpu::TextureFormat::Rgba16Float;

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
            view_formats: vec![surface_format.remove_srgb_suffix()],
        };
        surface.configure(&device, &config);

        let pc = PointCloud::load(&device, pc_file).unwrap();
        log::info!("loaded point cloud with {:} points", pc.num_points());

        let renderer = GaussianRenderer::new(
            &device,
            &queue,
            render_format,
            pc.sh_deg(),
            pc.compressed(),
        )
        .await;

        let aabb = pc.bbox();
        let aspect = size.width as f32 / size.height as f32;
        let view_camera = PerspectiveCamera::new(
            aabb.center() - Vector3::new(1., 1., 1.) * aabb.sphere() * 0.5,
            Quaternion::one(),
            PerspectiveProjection::new(
                Vector2::new(size.width, size.height),
                Vector2::new(Deg(45.), Deg(45. / aspect)),
                0.01,
                1000.,
            ),
        );

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = aabb.center();
        let ui_renderer = ui_renderer::EguiWGPU::new(device, surface_format, &window);
        let display = Display::new(
            device,
            render_format,
            surface_format.remove_srgb_suffix(),
            size.width,
            size.height,
        );
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
            #[cfg(not(target_arch = "wasm32"))]
            history: RingBuffer::new(512),
            ui_visible: true,
            display,
            background_color: Color32::BLACK,
            saved_cameras: Vec::new(),
            cameras_save_path: "cameras_saved.json".to_string(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;

            self.surface
                .configure(&self.wgpu_context.device, &self.config);
            self.display
                .resize(&self.wgpu_context.device, new_size.width, new_size.height);
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
        if let Some((next_camera,playing)) = &mut self.animation {
            if self.controller.user_inptut {
                self.cancle_animation()
            } else{
                let dt = if *playing {
                    dt
                } else {
                    Duration::ZERO
                };
                self.camera = next_camera.update(dt);
                if next_camera.done() {
                    self.animation.take();
                    self.controller.reset_to_camera(self.camera);
                }
            }
        } else {
            self.controller.update_camera(&mut self.camera, dt);
        }

        // set camera near and far plane
        let center = self.pc.bbox().center();
        let radius = self.pc.bbox().sphere();
        let distance = self.camera.position.distance(center);
        let zfar = distance + radius;
        let znear = (distance - radius).max(zfar / 1000.);
        self.camera.projection.zfar = zfar;
        self.camera.projection.znear = znear;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
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
        let viewport = Vector2::new(output.texture.size().width, output.texture.size().height);
        let rgba = self.background_color.to_srgba_unmultiplied();
        self.renderer.render(
            &self.wgpu_context.device,
            &self.wgpu_context.queue,
            &self.pc,
            self.camera,
            viewport,
            self.display.texture(),
            wgpu::Color {
                r: rgba[0] as f64 / 255.,
                g: rgba[1] as f64 / 255.,
                b: rgba[2] as f64 / 255.,
                a: rgba[3] as f64 / 255.,
            },
        );

        let mut encoder =
            self.wgpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("display"),
                });

        self.display.render(&mut encoder, &view_rgb);
        self.wgpu_context.queue.submit([encoder.finish()]);

        if self.ui_visible {
            // ui rendering
            self.ui_renderer.begin_frame(&self.window);
            ui::ui( self);

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
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            self.renderer.stopwatch.reset();
        }

        output.present();
        Ok(())
    }

    fn set_scene(&mut self, scene: Scene) {
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
            let a = Animation::new(Duration::from_secs_f32(self.saved_cameras.len() as f32*2.), true, Box::new(TrackingShot::from_scene(
                self.saved_cameras.clone(),
            )));
            self.animation = Some((a,true));
        }
    }
    
    fn cancle_animation(&mut self) {
        self.animation.take();
        self.controller.reset_to_camera(self.camera);
    }

    fn stop_animation(&mut self) {
        if let Some((_animation,playing)) = &mut self.animation{
            *playing = false;
        }
        self.controller.reset_to_camera(self.camera);
    }

    fn set_scene_camera(&mut self, i:usize) {
        if let Some(scene) =&self.scene{
            self.current_view.replace(i);
            log::info!("view moved to camera {i}");
            self.set_camera(scene.camera(i).unwrap(),Duration::from_millis(200));
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
            let a = Animation::new(animation_duration, false, Box::new(Transition::new(
                self.camera.clone(),
                target_camera,
                smoothstep,
            )));
            self.animation = Some((a,true));
        }
    }

    fn update_camera(&mut self, camera: PerspectiveCamera) {
        self.camera = camera;
    }

    fn save_view(&mut self) {
        let max_scene_id = if let Some(scene) = &self.scene {
            scene.cameras(None).iter().map(|c| c.id).max().unwrap_or(0)
        } else {
            0
        };
        let max_id = self.saved_cameras.iter().map(|c| c.id).max().unwrap_or(0);
        let id = max_id.max(max_scene_id) +1;
        self.saved_cameras.push(SceneCamera::from_perspective(
            self.camera,
            id.to_string(),
            id ,
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
                let canvas = window.canvas();
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

    let mut state = WindowContext::new(window, file, config).await;

    if let Some(scene) = scene {
        let init_camera = scene.camera(0).unwrap();
        state.set_scene(scene);
        state.set_camera(init_camera, Duration::ZERO);
        state.start_tracking_shot();
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
                    }else
                    if let Some(scene) = &state.scene{

                        let new_camera = 
                        if let Some(num) = key_to_num(key){
                            Some(num as usize)
                        }
                        else if key == KeyCode::KeyR{
                            Some(rand::random::<usize>()%scene.num_cameras())
                        }else if key == KeyCode::KeyN{
                            scene.nearest_camera(state.camera.position,None)
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
                    winit::event::MouseButton::Left => state.controller.left_mouse_pressed = *button_state == ElementState::Pressed,
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
pub fn run_wasm(pc: Vec<u8>, scene: Option<Vec<u8>>) {
    use std::io::Cursor;
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    let pc_reader = Cursor::new(pc);
    let scene_reader = scene.map(|d: Vec<u8>| Cursor::new(d));

    wasm_bindgen_futures::spawn_local(open_window(
        pc_reader,
        scene_reader,
        RenderConfig { no_vsync: false },
    ));
}
