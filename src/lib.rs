use std::{
    fmt::format,
    io::{Read, Seek},
};

use egui_plot::{format_number, Bar};
#[cfg(target_arch = "wasm32")]
use instant::{Duration, Instant};
use renderer::Display;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
use wgpu::Backends;

use cgmath::{
    Deg, EuclideanSpace, MetricSpace, Point3, Quaternion, Rotation, Vector2, Vector3, Vector4, VectorSpace
};
use egui::{epaint::Shadow, Align2, Color32, Stroke, Vec2, Vec2b, Visuals};
#[cfg(not(target_arch = "wasm32"))]
use egui_plot::{Legend, PlotPoints};
use num_traits::One;

use utils::key_to_num;
#[cfg(not(target_arch = "wasm32"))]
use utils::RingBuffer;

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

mod io;
#[cfg(feature = "npz")]
use io::npz;
use io::ply;

mod renderer;
pub use renderer::GaussianRenderer;

mod scene;
use crate::renderer::ColorSchema;

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
                        max_bind_groups: 5,
                        ..Default::default()
                    },
                    #[cfg(target_arch = "wasm32")]
                    limits: wgpu::Limits {
                        max_compute_workgroup_storage_size: 1 << 15,
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
    ui_visible: bool,

    #[cfg(not(target_arch = "wasm32"))]
    history: RingBuffer<(Duration, Duration, Duration)>,
    display: Display,

    background_color: egui::Color32,
    vis_settings: renderer::VisSettings,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new<R: Read + Seek>(
        window: Window,
        event_loop: &EventLoop<()>,
        pc_file: R,
        render_config: RenderConfig,
    ) -> Self {
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

        let a = Vector4::new(1., 0., 0., 1.);
        let b = Vector4::new(1., 1., 1., 0.1);
        let c = Vector4::new(0., 0., 1., 1.);
        let colors: Vec<Vector4<f32>> = (0..5).map(|i| a.lerp(b, i as f32 / 4.)).chain((0..5).map(|i| b.lerp(c, i as f32 / 4.))).collect();
        let color_schema = ColorSchema::new(device, queue, colors);

        let renderer = GaussianRenderer::new(
            &device,
            &queue,
            render_format,
            pc.sh_deg(),
            !pc.compressed(),
            color_schema,
        )
        .await;

        let aspect = size.width as f32 / size.height as f32;
        let aabb = pc.bbox();
        println!("aabb: {:?}", aabb);
        let cam_pos = aabb.center()-aabb.size()/2.;
        let look_at = aabb.center();
        let view_camera = PerspectiveCamera::new(
            cam_pos,
            Quaternion::look_at(look_at-cam_pos, Vector3::unit_y()),
            PerspectiveProjection::new(
                Vector2::new(size.width, size.height),
                Vector2::new(Deg(45.), Deg(45. / aspect)),
                0.01,
                1000.,
            ),
        );

        let mut controller = CameraController::new(0.1, 0.05);
        controller.center = look_at;
        let ui_renderer = ui_renderer::EguiWGPU::new(event_loop, device, surface_format);
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
            vis_settings: Default::default(),
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
        if let Some(next_camera) = &mut self.animation {
            if self.controller.user_inptut {
                self.stop_animation()
            } else {
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

    fn ui(&mut self) {
        let ctx = &self.ui_renderer.ctx;
        #[cfg(not(target_arch = "wasm32"))]
        let stats = pollster::block_on(
            self.renderer
                .render_stats(&self.wgpu_context.device, &self.wgpu_context.queue),
        );
        #[cfg(not(target_arch = "wasm32"))]
        let num_drawn = pollster::block_on(
            self.renderer
                .num_visible_points(&self.wgpu_context.device, &self.wgpu_context.queue),
        );

        #[cfg(not(target_arch = "wasm32"))]
        self.history.push((
            stats.preprocess_time,
            stats.sort_time,
            stats.rasterization_time,
        ));
        ctx.set_visuals(Visuals {
            window_shadow: Shadow::small_light(),
            ..Default::default()
        });

        #[cfg(not(target_arch = "wasm32"))]
        egui::Window::new("Render Stats")
            .default_width(200.)
            .default_height(100.)
            .show(ctx, |ui| {
                use egui::TextStyle;
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
                    .y_axis_width(1)
                    .y_axis_label("ms")
                    .auto_bounds_y()
                    .auto_bounds_x()
                    .show_axes([false, true])
                    .legend(Legend {
                        text_style: TextStyle::Body,
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

        egui::Window::new("🎮")
            .default_width(200.)
            .resizable(false)
            .default_height(100.)
            .default_open(false)
            .movable(false)
            .anchor(Align2::LEFT_BOTTOM, Vec2::new(10., -10.))
            .show(ctx, |ui| {
                egui::Grid::new("controls")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Camera");
                        ui.end_row();
                        ui.label("Rotate Camera");
                        ui.label("Left click + drag");
                        ui.end_row();

                        ui.label("Move Target/Center");
                        ui.label("Right click + drag");
                        ui.end_row();

                        ui.label("Tilt Camera");
                        ui.label("Alt + drag mouse");
                        ui.end_row();

                        ui.label("Zoom");
                        ui.label("Mouse wheel");
                        ui.end_row();

                        ui.label("Toggle UI");
                        ui.label("U");
                        ui.end_row();

                        ui.strong("Scene Views");
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

        egui::Window::new("⚙ Render Settings").show(ctx, |ui| {
            egui::Grid::new("render_settings")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Background Color");
                    egui::color_picker::color_edit_button_srgba(
                        ui,
                        &mut self.background_color,
                        egui::color_picker::Alpha::BlendOrAdditive,
                    );
                });
        });

        let mut new_camera = None;
        let mut toggle_tracking_shot = false;
        egui::Window::new("ℹ Scene")
            .default_width(200.)
            .resizable(false)
            .default_height(100.)
            .show(ctx, |ui| {
                egui::Grid::new("scene info")
                    .num_columns(2)
                    .striped(false)
                    .show(ui, |ui| {
                        ui.strong("Gaussians:");
                        ui.label(self.pc.num_points().to_string());
                        ui.end_row();
                        ui.strong("SH Degree:");
                        ui.label(self.pc.sh_deg().to_string());
                        ui.end_row();
                        ui.strong("Compressed:");
                        ui.label(self.pc.compressed().to_string());
                        ui.end_row();
                    });

                if let Some(scene) = &self.scene {
                    let nearest = scene.nearest_camera(self.camera.position, None);
                    ui.separator();
                    ui.heading("Training Images");
                    egui::Grid::new("image info")
                        .num_columns(2)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.strong("Images");
                            ui.label(scene.num_cameras().to_string());
                            ui.end_row();
                            ui.strong("Current");

                            if let Some(c) = &mut self.current_view {
                                ui.horizontal(|ui| {
                                    let drag =
                                        ui.add(egui::DragValue::new(c).clamp_range(
                                            0..=(scene.num_cameras().saturating_sub(1)),
                                        ));
                                    if drag.changed() {
                                        new_camera = Some(*c);
                                    }
                                    ui.label(scene.camera(*c).split.to_string());
                                });
                            }
                        });
                    if ui.button(format!("Snap to closest ({nearest})")).clicked() {
                        new_camera = Some(nearest);
                    }
                    let text = if self.animation.is_some() {
                        "Stop tracking shot"
                    } else {
                        "Start tracking shot"
                    };
                    if ui.button(text).clicked() {
                        toggle_tracking_shot = true;
                    }
                }
            });

        egui::Window::new("Visualization")
            .max_height(400.)
            .scroll2(Vec2b::new(false, true))
            .show(ctx, |ui| {
                egui::Grid::new("image info")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("scale");
                        ui.add(
                            egui::DragValue::new(&mut self.vis_settings.scale)
                                .speed(1e-3)
                                .clamp_range(1e-5..=10.),
                        );
                        ui.end_row();
                        ui.strong("Gaussian");
                        let mut gaussian = self.vis_settings.gaussian != 0;
                        ui.checkbox(&mut gaussian, "");
                        self.vis_settings.gaussian = gaussian as u32;
                        ui.end_row();
                        let n_colors = self.renderer.color_schema.colors().len();
                        for (i, c) in self.renderer.color_schema.colors().iter_mut().enumerate() {
                            ui.label(format!("Color {}", i as f32 / n_colors as f32));
                            let mut color = [c.x, c.y, c.z, c.w];
                            ui.color_edit_button_rgba_unmultiplied(&mut color);
                            c.x = color[0];
                            c.y = color[1];
                            c.z = color[2];
                            c.w = color[3];
                            ui.end_row();
                        }
                    });

                if let Some(histogram) = &self.pc.histogram {
                    ui.heading("Histogram");
                    egui_plot::Plot::new("histogram")
                        .auto_bounds_x()
                        .auto_bounds_y()
                        .y_axis_formatter(  |v,n,_|format_number(v.exp().round(), n))
                        .y_axis_label("num")
                        .show(ui, |ui| {
                            let bar_chart = egui_plot::BarChart::new(
                                histogram
                                    .iter()
                                    .map(|(x, y)| {
                                        let color = self.renderer.color_schema.sample(*x as f32);
                                        let y = if *y != 0 { (*y as f64).ln() } else { 0. };
                                        Bar::new(*x as f64, y)
                                            .stroke(Stroke::new(
                                                1.,
                                                egui::Rgba::from_rgba_unmultiplied(
                                                    color.x, color.y, color.z, color.w,
                                                ),
                                            ))
                                            .fill(egui::Rgba::from_rgba_unmultiplied(
                                                color.x, color.y, color.z, color.w,
                                            ))
                                            .width(1. / histogram.len() as f64)
                                    })
                                    .collect(),
                            )
                            .vertical();
                            ui.bar_chart(bar_chart);
                        });
                }
            });

        if let Some(c) = new_camera {
            self.current_view = new_camera;
            let c = self.scene.as_ref().unwrap().camera(c);
            self.set_camera(c, Duration::from_millis(200));
        }
        if toggle_tracking_shot {
            if self.animation.is_none() {
                self.start_tracking_shot(Some(Split::Test))
            } else {
                self.stop_animation();
            }
        }
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
            self.vis_settings,
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
    fn stop_animation(&mut self) {
        self.animation.take();
        self.controller.reset_to_camera(self.camera);
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

pub async fn open_window<R: Read + Seek + Send + Sync + 'static>(
    file: R,
    scene_file: Option<R>,
    config: RenderConfig,
) {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let event_loop = EventLoop::new();

    let scene = scene_file.and_then(|f| match Scene::from_json(f) {
        Ok(s) => Some(s),
        Err(err) => {
            log::error!("cannot load scene: {:?}", err);
            None
        }
    });

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

    let mut state = WindowContext::new(window, &event_loop, file, config).await;

    if let Some(scene) = scene {
        let init_camera = scene.camera(0);
        state.set_scene(scene);
        state.set_camera(init_camera, Duration::ZERO);
        state.start_tracking_shot(Some(Split::Test));
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
            WindowEvent::CloseRequested => {log::info!("close!");*control_flow = ControlFlow::Exit},
            WindowEvent::ModifiersChanged(m)=>{
                state.controller.alt_pressed = m.ctrl();
            }
            WindowEvent::KeyboardInput { input, .. } => {
                if let Some(key) = input.virtual_keycode {
                    if input.state == ElementState::Released{

                        if key == VirtualKeyCode::T{
                            if state.animation.is_none(){
                                state.start_tracking_shot(Some(Split::Test));
                            }else{
                                state.stop_animation()
                            }
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
        _ => {},
    });
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
