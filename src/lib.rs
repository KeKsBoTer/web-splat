use std::{
    io::{Read, Seek},
    path::PathBuf,
    sync::Arc,
};

use egui_wgpu::EguiDisplayHandle;
use renderer::Display;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

use cgmath::{
    AbsDiffEq, Deg, EuclideanSpace, InnerSpace, Point3, Quaternion, UlpsEq, Vector2, Vector3,
};
use egui::FullOutput;
use num_traits::One;

#[cfg(not(target_arch = "wasm32"))]
use utils::RingBuffer;
use utils::key_to_num;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event::{DeviceEvent, ElementState, Event, TouchPhase as WinitTouchPhase, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
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
mod cpu_sort;
mod pointcloud;
mod sort;
#[cfg(target_arch = "wasm32")]
mod web_backend;
pub use pointcloud::PointCloud;

pub mod io;

mod renderer;
pub use renderer::{DEFAULT_ALPHA_CUTOFF, GaussianRenderer, SplattingArgs};

mod scene;
use crate::utils::GPUStopwatch;

pub use self::scene::{Scene, SceneCamera, Split};

pub mod gpu_rs;
mod ui_renderer;
mod uniform;
mod utils;

/// Where the points of the viewed point cloud come from.
pub enum PointCloudSource {
    /// everything is loaded, ready to be uploaded
    Full(io::GenericGaussianPointCloud),
    /// points arrive while the point cloud is rendered, the header must be available already
    Stream(io::PointCloudStream),
}

/// number of streamed points that are uploaded to the GPU per frame
const STREAM_UPLOAD_POINTS_PER_FRAME: usize = 1 << 19;

/// duration of the camera movement when the point cloud grows while it is loading
const FRAME_TRANSITION: Duration = Duration::from_millis(600);
/// the camera only moves to a new view of a growing point cloud when the view moved by more
/// than this fraction of the radius of the point cloud
const FRAME_RETARGET_DISTANCE: f32 = 0.05;

/// The camera that looks at the whole point cloud.
fn default_camera(aabb: &pointcloud::Aabb<f32>, size: PhysicalSize<u32>) -> PerspectiveCamera {
    let aspect = size.width as f32 / size.height as f32;
    PerspectiveCamera::new(
        aabb.center() - Vector3::new(1., 1., 1.) * aabb.radius() * 0.5,
        Quaternion::one(),
        PerspectiveProjection::new(
            Vector2::new(size.width, size.height),
            Vector2::new(Deg(45.), Deg(45. / aspect)),
            0.01,
            1000.,
        ),
    )
}

pub struct RenderConfig {
    pub no_vsync: bool,
    pub hdr: bool,
}

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
}

impl WGPUContext {
    pub async fn new(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(instance, surface)
            .await
            .unwrap();
        log::info!("using apdater \"{}\"", adapter.get_info().name);

        #[cfg(target_arch = "wasm32")]
        let required_features = wgpu::Features::default();
        #[cfg(not(target_arch = "wasm32"))]
        let required_features = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            // GPU timing is optional, not every backend supports it
            | (adapter.features()
                & (wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS));

        let adapter_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
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
                    // (there are no compute shaders with WebGL2)
                    max_compute_workgroup_storage_size: adapter_limits
                        .max_compute_workgroup_storage_size
                        .min(1 << 15),
                    ..adapter_limits
                },
                label: None,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
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
    /// points that are still being loaded
    stream: Option<io::PointCloudStream>,
    load_error: Option<String>,
    /// While the point cloud is loading and the camera was not moved, it keeps looking at
    /// the whole point cloud.
    auto_frame: bool,
    /// timing of loading a streamed point cloud, to find out what is slow
    load_started: Instant,
    /// time spent to upload the points
    upload_time: Duration,
    /// frames drawn since the loading started
    load_frames: u32,
    /// the view the camera moved (or is moving) to by auto framing
    frame_target: Option<PerspectiveCamera>,
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
    /// the render target was recreated and the scene has to be drawn again
    scene_dirty: bool,

    #[cfg(not(target_arch = "wasm32"))]
    history: RingBuffer<(Duration, Duration, Duration)>,
    display: Display,

    splatting_args: SplattingArgs,

    saved_cameras: Vec<SceneCamera>,
    #[cfg(feature = "video")]
    cameras_save_path: String,
    stopwatch: Option<GPUStopwatch>,
}

impl WindowContext {
    // Creating some of the wgpu types requires async code
    async fn new(
        window: Window,
        source: PointCloudSource,
        render_config: &RenderConfig,
    ) -> anyhow::Result<Self> {
        let mut size = window.inner_size();
        if size == PhysicalSize::new(0, 0) {
            size = PhysicalSize::new(800, 600);
        }

        let window = Arc::new(window);

        #[allow(unused_mut)]
        let mut instance_desc =
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(window.clone_for_wgpu());
        // The url can force a backend (?backend=webgpu or ?backend=webgl). Otherwise: browsers
        // can have `navigator.gpu` without being able to create an adapter, this checks for
        // a working WebGPU and uses WebGL2 if there is none.
        #[cfg(target_arch = "wasm32")]
        let instance = match web_backend::requested_backend() {
            Some(web_backend::WebBackend::WebGpu) => {
                instance_desc.backends = wgpu::Backends::BROWSER_WEBGPU;
                wgpu::Instance::new(instance_desc)
            }
            Some(web_backend::WebBackend::WebGl) => {
                instance_desc.backends = wgpu::Backends::GL;
                wgpu::Instance::new(instance_desc)
            }
            None => wgpu::util::new_instance_with_webgpu_detection(instance_desc).await,
        };
        #[cfg(not(target_arch = "wasm32"))]
        let instance = wgpu::Instance::new(instance_desc);

        let surface: wgpu::Surface = instance.create_surface(window.clone())?;

        let wgpu_context = WGPUContext::new(&instance, Some(&surface)).await;

        let device = &wgpu_context.device;

        let surface_caps = surface.get_capabilities(&wgpu_context.adapter);

        // The scene is drawn to a view of the surface without the srgb suffix. Where surfaces
        // can't have other view formats (WebGL2) the surface itself must not be srgb.
        let surface_view_formats = wgpu_context
            .adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::SURFACE_VIEW_FORMATS);
        let surface_format = surface_caps
            .formats
            .iter()
            .filter(|f| f.is_srgb() == surface_view_formats)
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
            // one queued frame: less delay between input and display
            desired_maximum_frame_latency: 1,
            present_mode: if render_config.no_vsync {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            color_space: wgpu::SurfaceColorSpace::Auto,
            view_formats: if surface_view_formats {
                vec![surface_format.remove_srgb_suffix()]
            } else {
                vec![]
            },
        };
        surface.configure(&device, &config);

        let (pc, stream) = match source {
            PointCloudSource::Full(pc_raw) => {
                (PointCloud::new(&device, &wgpu_context.queue, pc_raw)?, None)
            }
            PointCloudSource::Stream(stream) => {
                let info = stream
                    .info()
                    .ok_or_else(|| anyhow::anyhow!("ply header is not available"))?;
                (PointCloud::new_streaming(&device, &info)?, Some(stream))
            }
        };
        log::info!("point cloud with {:} points", pc.num_points());

        let renderer = GaussianRenderer::new(&device, render_format, pc.sh_deg(), pc.compressed());

        let view_camera = default_camera(pc.bbox(), size);

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

        let stopwatch = if cfg!(not(target_arch = "wasm32"))
            && device.features().contains(
                wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            ) {
            Some(GPUStopwatch::new(device, Some(3)))
        } else {
            None
        };

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
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                alpha_cutoff: DEFAULT_ALPHA_CUTOFF,
                // the splats grow in over the first seconds (see preprocess), a high value
                // shows them fully grown from the start (like the offline renderers)
                walltime: Duration::from_secs(100),
                scene_center: None,
                scene_extend: None,
                background_color: wgpu::Color::BLACK,
            },
            auto_frame: stream.is_some(),
            load_started: Instant::now(),
            upload_time: Duration::ZERO,
            load_frames: 0,
            frame_target: None,
            stream,
            load_error: None,
            pc,
            // camera: view_camera,
            controller,
            ui_renderer,
            fps: 0.,
            #[cfg(not(target_arch = "wasm32"))]
            history: RingBuffer::new(512),
            ui_visible: true,
            scene_dirty: false,
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
        })
    }

    fn reload(&mut self) -> anyhow::Result<()> {
        if let Some(file_path) = &self.pointcloud_file_path {
            log::info!("reloading volume from {:?}", file_path);
            let file = std::fs::File::open(file_path)?;
            let pc_raw = io::GenericGaussianPointCloud::load(file)?;
            self.pc = PointCloud::new(&self.wgpu_context.device, &self.wgpu_context.queue, pc_raw)?;
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
            // the new render target is empty
            self.scene_dirty = true;
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
            self.splatting_args.viewport = Vector2::new(new_size.width, new_size.height);
            self.splatting_args
                .camera
                .projection
                .resize(new_size.width, new_size.height);
        }
        if let Some(scale_factor) = scale_factor {
            if scale_factor > 0. {
                self.scale_factor = scale_factor;
            }
        }
    }

    /// returns whether redraw is required
    fn ui(&mut self) -> (bool, egui::FullOutput) {
        let raw_input = self.ui_renderer.winit.take_egui_input(&self.window);
        // run_ui (egui >= 0.35) records the root ui's area (the whole window, the
        // scene behind the egui windows). egui needs it to tell pointer input over
        // the scene from input over its windows: with begin_pass / end_pass every
        // pointer event counts as consumed by egui and never reaches the camera
        // controller. (The context is reference counted; the clone avoids
        // borrowing it from self inside the closure.)
        let ctx = self.ui_renderer.winit.egui_ctx().clone();
        let mut request_redraw = false;
        let ui_visible = self.ui_visible;
        let output = ctx.run_ui(raw_input, |_root_ui| {
            // a hidden UI must not exist for egui, or its (invisible) windows
            // would still take pointer input
            if ui_visible {
                request_redraw = ui::ui(self);
            }
        });
        self.ui_renderer
            .winit
            .handle_platform_output(&self.window, output.platform_output.clone());
        (request_redraw, output)
    }

    /// Updates the draw order of the splats when the sorting happens on the CPU (without compute
    /// shaders). Returns whether the scene changed and we need a redraw.
    fn poll_sort(&mut self) -> bool {
        let view = self.splatting_args.camera.view_matrix();
        self.pc.update_sort(
            &self.wgpu_context.queue,
            [view.x.z, view.y.z, view.z.z, view.w.z],
        )
    }

    /// Uploads the points of a streamed point cloud that arrived since the last call.
    /// Returns whether the scene changed and we need a redraw.
    fn poll_stream(&mut self) -> bool {
        let Some(stream) = &self.stream else {
            return false;
        };
        if let Some(err) = stream.error() {
            log::error!("loading the point cloud failed: {err}");
            self.load_error = Some(err);
            self.stream = None;
            return true;
        }
        let mut changed = false;
        let upload_start = Instant::now();
        for chunk in stream.take(STREAM_UPLOAD_POINTS_PER_FRAME) {
            changed |= self.pc.append(&self.wgpu_context.queue, &chunk) > 0;
        }
        self.upload_time += upload_start.elapsed();
        self.load_frames += 1;
        let complete = self.pc.is_complete();
        if complete {
            let total = self.load_started.elapsed();
            log::info!(
                "loaded point cloud with {:} points in {:.1} s ({:.1} s uploading to the gpu, {} frames, {:.1} fps while loading)",
                self.pc.num_points(),
                total.as_secs_f32(),
                self.upload_time.as_secs_f32(),
                self.load_frames,
                self.load_frames as f32 / total.as_secs_f32().max(1e-3),
            );
            self.stream = None;
        }
        if changed {
            self.reframe_camera(complete);
        }
        changed
    }

    /// Looks at the whole point cloud as long as the camera was not moved. The first time the
    /// camera jumps to the point cloud, after that it moves there smoothly.
    /// A transition that is running is not interrupted (restarting it every frame would make
    /// the camera crawl), unless this is the final view (`last`).
    fn reframe_camera(&mut self, last: bool) {
        if !self.auto_frame || self.scene.is_some() {
            return;
        }
        let size = PhysicalSize::new(self.config.width, self.config.height);
        let target = default_camera(self.pc.bbox(), size);
        self.controller.center = self.pc.center();

        let Some(previous) = self.frame_target else {
            // nothing to move from yet
            self.splatting_args.camera = target;
            self.frame_target = Some(target);
            return;
        };
        let moved = (target.position - previous.position).magnitude();
        let significant = moved > FRAME_RETARGET_DISTANCE * self.pc.bbox().radius();
        if last || (significant && self.animation.is_none()) {
            self.frame_target = Some(target);
            self.set_camera(target, FRAME_TRANSITION);
        }
    }

    /// returns whether the sceen changed and we need a redraw
    fn update(&mut self, dt: Duration) {
        // ema fps update

        if self.splatting_args.walltime < Duration::from_secs(5) {
            self.splatting_args.walltime += dt;
        }
        if let Some((next_camera, playing)) = &mut self.animation {
            if self.controller.user_inptut {
                self.auto_frame = false;
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
            if self.controller.user_inptut {
                self.auto_frame = false;
            }
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
    ) -> Result<(), wgpu::CurrentSurfaceTexture> {
        self.stopwatch.as_mut().map(|s| s.reset());

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => surface_texture,
            err => {
                if let Some(shapes) = shapes {
                    self.ui_renderer.skip(shapes);
                }
                return Err(err);
            }
        };
        let view_rgb = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.config.format.remove_srgb_suffix()),
            ..Default::default()
        });
        let view_srgb = output.texture.create_view(&Default::default());
        // do prepare stuff

        let mut encoder =
            self.wgpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render command encoder"),
                });

        if redraw_scene {
            self.renderer.prepare(
                &mut encoder,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
                &self.pc,
                self.splatting_args,
                (&mut self.stopwatch).into(),
            );
        }

        let ui_state = shapes.map(|shapes| {
            self.ui_renderer.prepare(
                PhysicalSize {
                    width: output.texture.size().width,
                    height: output.texture.size().height,
                },
                self.scale_factor,
                &self.wgpu_context.device,
                &self.wgpu_context.queue,
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
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            self.renderer.render(&mut render_pass, &self.pc);
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
                        depth_slice: None,
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.ui_renderer.render(&mut render_pass, state);
        }

        if let Some(ui_state) = ui_state {
            self.ui_renderer.cleanup(ui_state)
        }
        self.wgpu_context.queue.submit([encoder.finish()]);

        self.wgpu_context.queue.present(output);
        self.splatting_args.viewport = Vector2::new(self.config.width, self.config.height);
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
        self.auto_frame = false;
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
            let a: Animation<PerspectiveCamera> = Animation::new(
                Duration::from_secs_f32(self.saved_cameras.len() as f32 * 4.),
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
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

/// Opens the viewer for a point cloud file. Ply files are streamed on native, the viewer opens as
/// soon as the header is read and the points appear while the file is read.
pub async fn open_window<R: Read + Seek + Send + Sync + 'static>(
    file: R,
    scene_file: Option<R>,
    config: RenderConfig,
    pointcloud_file_path: Option<PathBuf>,
    scene_file_path: Option<PathBuf>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let source = load_source(file).unwrap();
    run_window(
        source,
        scene_file,
        config,
        pointcloud_file_path,
        scene_file_path,
    )
    .await
}

fn load_source<R: Read + Seek + Send + 'static>(mut file: R) -> anyhow::Result<PointCloudSource> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut signature = [0u8; 3];
        file.read_exact(&mut signature)?;
        file.rewind()?;
        if signature == *b"ply" {
            return Ok(PointCloudSource::Stream(io::stream::spawn_reader(file)?));
        }
    }
    Ok(PointCloudSource::Full(io::GenericGaussianPointCloud::load(
        &mut file,
    )?))
}

async fn run_window<S: Read + Seek>(
    source: PointCloudSource,
    scene_file: Option<S>,
    config: RenderConfig,
    pointcloud_file_path: Option<PathBuf>,
    scene_file_path: Option<PathBuf>,
) {
    let event_loop = EventLoop::new().unwrap();

    let scene = scene_file.and_then(|f| match Scene::from_json(f) {
        Ok(s) => Some(s),
        Err(err) => {
            log::error!("cannot load scene: {:?}", err);
            None
        }
    });

    // let window_size = if let Some(scene) = &scene {
    //     let camera = scene.camera(0).unwrap();
    //     let factor = 1200. / camera.width as f32;
    //     LogicalSize::new(
    //         (camera.width as f32 * factor) as u32,
    //         (camera.height as f32 * factor) as u32,
    //     )
    // } else {
    //     LogicalSize::new(800, 600)
    // };
    let window_size = LogicalSize::new(800, 600);
    let window_attributes = Window::default_attributes()
        .with_inner_size(window_size)
        .with_title(format!(
            "{} ({})",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ));

    #[allow(deprecated)]
    let window = event_loop.create_window(window_attributes).unwrap();

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

    // limit the redraw rate to the monitor refresh rate
    let min_wait = window
        .current_monitor()
        .map(|m| {
            let hz = m.refresh_rate_millihertz().unwrap_or(60_000);
            Duration::from_millis(1000000 / hz as u64)
        })
        .unwrap_or(Duration::from_millis(17));

    let mut state = WindowContext::new(window, source, &config).await.unwrap();
    state.pointcloud_file_path = pointcloud_file_path;

    if let Some(scene) = scene {
        state.set_scene(scene);
        state.set_scene_camera(0);
        state.scene_file_path = scene_file_path;
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

    #[allow(deprecated)]
    event_loop.run(move |event,target|
        match event {
            Event::NewEvents(e) =>  match e{
                winit::event::StartCause::ResumeTimeReached { .. }=>{
                    state.window.request_redraw();
                }
                _=>{}
            },
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
                            Some((rand::random::<u32>() as usize)%scene.num_cameras())
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
            WindowEvent::Touch(touch) => {
                let touch_phase = match touch.phase {
                    WinitTouchPhase::Started => controller::TouchPhase::Started,
                    WinitTouchPhase::Moved => controller::TouchPhase::Moved,
                    WinitTouchPhase::Ended => controller::TouchPhase::Ended,
                    WinitTouchPhase::Cancelled => controller::TouchPhase::Cancelled,
                };
                let controller_touch = controller::Touch {
                    id: touch.id,
                    position: (touch.location.x as f32, touch.location.y as f32),
                    phase: touch_phase,
                };
                state.controller.process_touch(controller_touch);
            }
            WindowEvent::RedrawRequested => {
                if !config.no_vsync{
                    // make sure the next redraw is called with a small delay
                    target.set_control_flow(ControlFlow::wait_duration(min_wait));
                }
                let now = Instant::now();
                let dt = now-last;
                last = now;

                let old_settings = state.splatting_args.clone();
                state.update(dt);

                // The controller recomputes the camera every frame. Without input only
                // rounding differs, which counted as a change: the scene was rendered
                // every frame and input and UI waited for it. Keep the old camera then.
                {
                    let (old_camera, camera) = (&old_settings.camera, &state.splatting_args.camera);
                    if old_camera.position.abs_diff_eq(&camera.position, 1e-6)
                        && old_camera.rotation.abs_diff_eq(&camera.rotation, 1e-6)
                        && (old_camera.projection.fovx.0 - camera.projection.fovx.0).abs() <= 1e-6
                        && (old_camera.projection.fovy.0 - camera.projection.fovy.0).abs() <= 1e-6
                    {
                        state.splatting_args.camera = old_settings.camera;
                    }
                }

                let stream_changed = state.poll_stream();
                let sort_changed = state.poll_sort();

                let (redraw_ui,shapes) = state.ui();

                let resolution_change = state.splatting_args.viewport != Vector2::new(state.config.width, state.config.height);

                let request_redraw = old_settings != state.splatting_args || resolution_change || state.scene_dirty || stream_changed || sort_changed;

                if request_redraw || redraw_ui{
                    state.fps = (1. / dt.as_secs_f32()) * 0.05 + state.fps * 0.95;
                    let shapes = if state.ui_visible {
                        Some(shapes)
                    } else {
                        state.ui_renderer.skip(shapes);
                        None
                    };
                    match state.render(request_redraw,shapes) {
                        Ok(_) => state.scene_dirty = false,
                        // Reconfigure the surface if lost
                        Err(wgpu::CurrentSurfaceTexture::Suboptimal(_)) => state.resize(state.window.inner_size(), None),
                        Err(wgpu::CurrentSurfaceTexture::Lost) => state.resize(state.window.inner_size(), None),
                        // the window is hidden or minimized, nothing to draw
                        Err(wgpu::CurrentSurfaceTexture::Occluded) => {}
                        // The system is out of memory, we should probably quit
                        // All other errors (Outdated, Timeout) should be resolved by the next frame
                        Err(e) => println!("error: {:?}", e),
                    }
                } else {
                    state.ui_renderer.skip(shapes);
                }
                if config.no_vsync{
                    state.window.request_redraw();
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
        _ => {},
    }).unwrap();
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
            hdr: false,
        },
        pc_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
        scene_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
    ));
}

/// Streams a ply file into the viewer on the web. The page reads the response body of a `fetch`
/// and passes the pieces to [`Self::push`]. After the header arrived ([`Self::is_ready`]) the
/// viewer can be opened with [`Self::start_viewer`], the points appear while the rest of the file
/// is still downloading.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct PointCloudLoader {
    feeder: io::PlyFeeder,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl PointCloudLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().ok();
        Self {
            feeder: io::PlyFeeder::new(),
        }
    }

    /// the next bytes of the file
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), wasm_bindgen::JsError> {
        self.feeder
            .push(bytes)
            .map_err(|e| wasm_bindgen::JsError::new(&format!("{e:#}")))
    }

    /// to be called when the whole file was pushed
    pub fn finish(&mut self) -> Result<(), wasm_bindgen::JsError> {
        self.feeder
            .finish()
            .map_err(|e| wasm_bindgen::JsError::new(&format!("{e:#}")))
    }

    /// whether the header was received, which is required to open the viewer
    pub fn is_ready(&self) -> bool {
        self.feeder.is_ready()
    }

    pub fn start_viewer(
        &self,
        scene: Option<Vec<u8>>,
        pc_file: Option<String>,
        scene_file: Option<String>,
    ) -> Result<(), wasm_bindgen::JsError> {
        use std::{io::Cursor, str::FromStr};

        if !self.feeder.is_ready() {
            return Err(wasm_bindgen::JsError::new(
                "the ply header was not received",
            ));
        }
        wasm_bindgen_futures::spawn_local(run_window(
            PointCloudSource::Stream(self.feeder.stream()),
            scene.map(Cursor::new),
            RenderConfig {
                no_vsync: false,
                hdr: false,
            },
            pc_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
            scene_file.and_then(|s| PathBuf::from_str(s.as_str()).ok()),
        ));
        Ok(())
    }
}

/// The sort of the splats for devices without compute shaders, runs in a web worker
/// (see `public/sort_worker.js` and `cpu_sort.rs`).
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct SplatSorter(sort::DepthSorter);

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl SplatSorter {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self(sort::DepthSorter::new())
    }

    /// adds points, three values (xyz) per point
    pub fn add_positions(&mut self, xyz: &[f32]) {
        self.0.add_positions(xyz);
    }

    /// the indices of the points in front of the camera, the nearest first, `view` is the third
    /// row of the view matrix
    pub fn sort(&self, view: &[f32]) -> Vec<u32> {
        self.0.sort([view[0], view[1], view[2], view[3]])
    }
}
