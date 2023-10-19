use instant::{Duration, Instant};
use std::{
    path::Path,
    sync::{Arc, RwLock},
    thread,
};

use cgmath::{Deg, EuclideanSpace, Point3, Quaternion, Transform, Vector2};
use log::{debug, info};
use num_traits::One;
use rayon::slice::ParallelSliceMut;
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{
    animation::{smoothstep, Animation, TrackingShot, Transition},
    camera::{Camera, PerspectiveCamera, PerspectiveProjection},
    controller::CameraController,
    pc,
    renderer::Renderer,
    renderer::GaussianRendererCompute,
    renderer::GaussianRenderer,

    utils, PointCloud, Scene, WGPUContext,
};

struct WindowContext {
    wgpu_context: WGPUContext,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    window: Window,
    scale_factor: f32,

    pc: Arc<RwLock<PointCloud>>,
    renderer: Renderer,
    camera: Arc<RwLock<PerspectiveCamera>>,
    animation: Option<Box<dyn Animation<Animatable = PerspectiveCamera>>>,
    controller: CameraController,
    scene: Option<Scene>,
    pause_sort: Arc<RwLock<bool>>,
    current_view: Option<usize>,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<P: AsRef<Path>>(window: Window, pc_file: P) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        let surface: wgpu::Surface = unsafe { instance.create_surface(&window) }.unwrap();

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;
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
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let pc = PointCloud::load_ply(&device, pc_file, pc::SHDtype::Byte).unwrap();
        log::info!("loaded point cloud with {:} points", pc.num_points());

        println!("render_config has: {window.render_config.renderer}");
        let renderer = match window.render_config.renderer {
            "rast" => GaussianRenderer::new(&device, surface_format, pc.sh_deg(), pc.sh_dtype())
            "comp" => GaussianRendererCompute::new(&device, surface_format, pc.sh_deg(), pc.sh_dtype())
            _ => println!("Renderer {} not supported, using \"comp\" as default", window.render_config.renderer);
                GaussianRendererCompute::new(&device, surface_format, pc.sh_deg(), pc.sh_dtype())
        }

        let aspect = size.width as f32 / size.height as f32;
        let view_camera = PerspectiveCamera::new(
            Point3::origin(),
            Quaternion::one(),
            PerspectiveProjection::new(Vector2::new(Deg(45.), Deg(45. * aspect)), 0.1, 100.),
        );

        let controller = CameraController::new(3., 0.25);
        Self {
            wgpu_context,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            renderer,
            pc: Arc::new(RwLock::new(pc)),
            camera: Arc::new(RwLock::new(view_camera)),
            animation: None,
            controller,
            scene: None,
            pause_sort: Arc::new(RwLock::new(false)),
            current_view: None,
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
        if let Some(next_camera) = &mut self.animation {
            let mut curr_camera = self.camera.write().unwrap();
            *curr_camera = next_camera.update(dt);
            if next_camera.done() {
                *self.pause_sort.write().unwrap() = false;
                self.animation.take();
            }
        } else {
            self.controller
                .update_camera(&mut self.camera.write().unwrap(), dt);
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
            &self.pc.read().unwrap(),
            self.camera.read().unwrap().clone(),
            viewport,
        );

        output.present();

        Ok(())
    }

    fn set_scene(&mut self, scene: Scene) {
        self.scene.replace(scene);
    }

    fn start_tracking_shot(&mut self) {
        if let Some(scene) = &self.scene {
            self.animation = Some(Box::new(TrackingShot::from_scene(
                scene,
                1.,
                Some(self.camera.read().unwrap().clone()),
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
            *self.pause_sort.write().unwrap() = true;
            let target_camera = camera.into();
            self.animation = Some(Box::new(Transition::new(
                self.camera.read().unwrap().clone(),
                target_camera,
                animation_duration,
                smoothstep,
            )));
        }
    }

    fn update_camera(&mut self, camera: PerspectiveCamera) {
        let mut curr_camera = self.camera.write().unwrap();
        *curr_camera = camera;
    }
}

pub async fn open_window<P: AsRef<Path> + Clone + Send + Sync + 'static>(
    file: P,
    scene_file: Option<P>,
) {
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

    let window = WindowBuilder::new()
        .with_title("web-splats")
        .with_inner_size(window_size)
        .build(&event_loop)
        .unwrap();

    let mut state = WindowContext::new(window, file).await;

    if let Some(scene) = scene {
        let init_camera = scene.camera(0);
        state.set_scene(scene);
        state.set_camera(init_camera, Duration::ZERO);
        state.start_tracking_shot();
    }

    let mut last = Instant::now();

    let queue = state.wgpu_context.queue.clone();
    let camera = state.camera.clone();
    let pause_sort = state.pause_sort.clone();

    let pc = state.pc.clone();
    // sorting thread
    // this thread copies the point cloud, sorts it and copies the result
    // back to the point cloud that is rendered
    // TODO do this on the GPU!
    thread::spawn(move || {
        let mut last_camera = <_>::default();
        loop {
            let perform_sort = !{ *pause_sort.read().unwrap() };
            if perform_sort {
                let curr_camera = { *camera.read().unwrap() };
                if last_camera != curr_camera {
                    let mut curr_pc = { pc.read().unwrap().points().clone() };
                    let view = curr_camera.view_matrix();
                    let proj = curr_camera.proj_matrix();
                    let transform = proj * view;

                    let start = Instant::now();
                    // par_sort_unstable_by_key is faster than stable cached sort
                    // source: trust me bro
                    curr_pc.par_sort_unstable_by_key(|p| {
                        (-transform.transform_point(p.xyz).z * (2f32).powi(24)) as i32
                    });
                    debug!("sorting took: {:}ms", (Instant::now() - start).as_millis());

                    pc.write().unwrap().update_points(&queue, curr_pc);
                    last_camera = curr_camera;
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    });

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => match event {
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
                            state.start_tracking_shot();
                            
                        }else
                        if let Some(scene) = &state.scene{

                            let new_camera = 
                            if let Some(num) = utils::key_to_num(key){
                                Some(num as usize)
                            }
                            else if key == VirtualKeyCode::R{
                                Some(rand::random::<usize>()%scene.num_cameras())
                            }else if key == VirtualKeyCode::N{
                                Some(scene.nearest_camera(state.camera.read().unwrap().position))
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
                                info!("view moved to camera {new_camera}");
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
