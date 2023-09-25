use std::{
    path::Path,
    time::{Duration, Instant}, thread, sync::{RwLock, Arc},
};

use camera::{PerspectiveCamera, PerspectiveProjection};
use cgmath::{Deg, EuclideanSpace, One, Point3, Quaternion, Vector2, Transform};
use controller::CameraController;
use pc::PointCloud;
use renderer::GaussianRenderer;
use scene::Scene;
use utils::smoothstep;
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::camera::Camera;

mod camera;
mod controller;
pub mod pc;
mod renderer;
mod scene;
mod uniform;
mod utils;

struct WindowContext {
    device: wgpu::Device,
    queue: Arc<wgpu::Queue>,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    window: Window,
    scale_factor: f32,

    pc: Option<Arc<RwLock<PointCloud>>>,
    renderer: GaussianRenderer,
    camera: Arc<RwLock<PerspectiveCamera>>,
    next_camera: Option<((Duration,Duration),(PerspectiveCamera,PerspectiveCamera))>,
    controller: CameraController,
    scene: Option<Scene>,
    pause_sort:Arc<RwLock<bool>>
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        let surface: wgpu::Surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits {
                        max_vertex_attributes: 20,
                        max_buffer_size: 2 << 29,
                        ..Default::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);

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

        let renderer = GaussianRenderer::new(&device, surface_format);

        let aspect = size.width as f32 / size.height as f32;
        let view_camera = PerspectiveCamera::new(
            Point3::origin(),
            Quaternion::one(),
            PerspectiveProjection::new(Vector2::new(Deg(45.), Deg(45. * aspect)), 0.1, 100.),
        );

        let controller = CameraController::new(1., 1.);
        Self {
            device,
            queue:Arc::new(queue),
            adapter,
            scale_factor: window.scale_factor() as f32,
            window,
            surface,
            config,
            renderer,
            pc: None,
            camera: Arc::new(RwLock::new(view_camera)),
            next_camera:None,
            controller,
            scene: None,
            pause_sort:Arc::new(RwLock::new(false))
        }
    }

    pub fn set_point_cloud(&mut self, pc: Arc<RwLock<PointCloud>>) {
        self.pc = Some(pc);
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, scale_factor: Option<f32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.camera.write().unwrap()
                .projection
                .resize(new_size.width, new_size.height);

            self.surface.configure(&self.device, &self.config);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    fn update(&mut self, dt: Duration) {
        if let Some(((time_left,duration),(start_camera,target_camera)))=self.next_camera{
            match time_left.checked_sub(dt){
                Some(new_left) => {
                    // set time left 
                    if let Some(c) = &mut self.next_camera{
                        c.0.0 = new_left;
                    }  
                    let elapsed = 1.-new_left.as_secs_f32()/duration.as_secs_f32();
                    let amount = smoothstep(elapsed);
                    *self.camera.write().unwrap() = start_camera.lerp(&target_camera, amount)
                },
                None => {
                    let mut camera = self.camera.write().unwrap();
                    *camera = target_camera.clone();
                    camera
                        .projection
                        .resize(self.config.width, self.config.height);
                    self.next_camera.take();
                    *self.pause_sort.write().unwrap() = false;
                },
            }
        }else{
            self.controller.update_camera(&mut self.camera.write().unwrap(), dt);
        }
        
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder Compare"),
            });
        {
           

            if let Some(pc) = &self.pc {
                let viewport = Vector2::new(self.config.width, self.config.height);
                let pc = pc.read().unwrap();

                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    })],
                    depth_stencil_attachment: None,
                });
                self.renderer.render(
                    &mut render_pass,
                    &self.queue,
                    &pc,
                    self.camera.read().unwrap().clone(),
                    viewport
                )
            }
        }

        self.queue.submit([encoder.finish()]);

        output.present();

        Ok(())
    }

    fn set_scene(&mut self, scene: Scene) {
        self.scene.replace(scene);
    }

    pub fn set_camera<C: Into<PerspectiveCamera>>(&mut self, camera: C,animation_duration:Duration) {
        if animation_duration.is_zero(){
            self.update_camera(camera.into())
        }else{
            *self.pause_sort.write().unwrap() = true;
            self.next_camera = Some(((animation_duration,animation_duration),(self.camera.read().unwrap().clone(),camera.into())));
        }
    }

    fn update_camera(&mut self, camera: PerspectiveCamera) {
        let mut curr_camera = self.camera.write().unwrap();
        *curr_camera = camera;
        curr_camera
            .projection
            .resize(self.config.width, self.config.height);
    }
}

pub async fn open_window<P: AsRef<Path> + Clone + Send + Sync + 'static>(
    file: P,
    scene_file: Option<P>,
) {
    let event_loop = EventLoop::new();

    let scene = scene_file.map(|f| Scene::from_json(f).unwrap());

    let window_size = if let Some(scene) = &scene {
        let camera = scene.camera(0);
        let factor = 1200. /camera.width as f32;
        PhysicalSize::new((camera.width as f32*factor) as u32, (camera.height as f32*factor) as u32)
    } else {
        PhysicalSize::new(800, 600)
    };

    let window = WindowBuilder::new()
        .with_title("web-splats")
        .with_inner_size(window_size)
        .build(&event_loop)
        .unwrap();

    let mut state = WindowContext::new(window).await;

    let pc = Arc::new(RwLock::new(PointCloud::load_ply(&state.device, file).unwrap()));

    if let Some(scene) = scene {
        state.set_scene(scene);
    }

    let mut last = Instant::now();

    state.set_point_cloud(pc.clone());

    let queue = state.queue.clone();
    let camera = state.camera.clone();
    let pause_sort = state.pause_sort.clone();

    // sorting thread
    // this thread copies the point cloud, sorts it and copies the result
    // back to the point cloud that is rendered
    // TODO do this on the GPU!
    thread::spawn(move ||{
        let mut last_camera = camera.read().unwrap().clone();
        loop {
            let perform_sort = !{*pause_sort.read().unwrap()};
            if perform_sort{
                let curr_camera = {*camera.read().unwrap()};
                if last_camera != curr_camera{
                    let mut curr_pc = {pc.read().unwrap().points().clone()};
                    let view = curr_camera.view_matrix();
                    let proj = curr_camera.proj_matrix();
                    let transform = proj * view;
                
                    curr_pc.sort_by_cached_key(|p|(-transform.transform_point(p.xyz).z * (2f32).powi(24)) as i32);
                    
                    pc.write().unwrap().update_points(&queue,curr_pc);
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
                    if let Some(num) = utils::key_to_num(key){
                        if let Some(scene) = &state.scene{
                            state.set_camera(scene.camera(num as usize),Duration::from_millis(500));
                        }
                    }
                    else if key == VirtualKeyCode::R{
                        if let Some(scene) = &state.scene{
                            let rnd_idx = rand::random::<usize>();
                            state.set_camera(scene.camera(rnd_idx % scene.num_cameras()),Duration::from_millis(500));
                        }   
                    }}
                
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
