use std::sync::Arc;
mod animation;
pub use animation::{Animation, TrackingShot, Transition};
mod camera;
pub use camera::{Camera, PerspectiveCamera, PerspectiveProjection};
mod controller;
pub use controller::CameraController;
mod pc;
pub use pc::{PointCloud, SHDtype};

mod renderer;
pub use renderer::GaussianRenderer;

mod scene;
pub use self::scene::{Scene, SceneCamera};

mod uniform;
// mod window;
// pub use window::open_window;

pub struct WGPUContext {
    pub device: wgpu::Device,
    pub queue: Arc<wgpu::Queue>,
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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 1 << 30,
                        max_buffer_size: 1 << 30,
                        ..Default::default()
                    },
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        Self {
            device,
            queue: Arc::new(queue),
            adapter,
        }
    }
}
