use cgmath::Vector2;
use clap::Parser;
use std::{fs::File, path::PathBuf, time::Instant};
use web_splats::{
    GaussianRenderer, PointCloud, Scene, SceneCamera, SplattingArgs, Split, WGPUContext,
};

#[derive(Debug, Parser)]
#[command(author, version)]
#[command(about = "Dataset offline renderer. Renders to PNG files", long_about = None)]
struct Opt {
    /// input file
    input: PathBuf,

    /// scene json file
    scene: PathBuf,
}

async fn render_views(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut GaussianRenderer,
    pc: &PointCloud,
    cameras: Vec<SceneCamera>,
) {
    let resolution: Vector2<u32> = Vector2::new(1920, 1080);

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: resolution.x,
            height: resolution.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: renderer.color_format(),
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let start = Instant::now();
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // first render to lazy init sorter stuff
    renderer.render(
        device,
        queue,
        &pc,
        &target_view,
        SplattingArgs {
            camera: cameras[0].clone().into(),
            viewport: resolution,
            gaussian_scaling: 1.,
            max_sh_deg: pc.sh_deg(),
            show_env_map: false,
            mip_splatting: None,
            kernel_size: None,
        },
    );
    device.poll(wgpu::MaintainBase::Wait);

    let num_samples = 50;
    for s in cameras.iter() {
        for _ in 0..num_samples {
            renderer.render(
                device,
                queue,
                &pc,
                &target_view,
                SplattingArgs {
                    camera: s.clone().into(),
                    viewport: resolution,
                    gaussian_scaling: 1.,
                    max_sh_deg: pc.sh_deg(),
                    show_env_map: false,
                    mip_splatting: None,
                    kernel_size: None,
                },
            );
        }
    }
    device.poll(wgpu::MaintainBase::Wait);
    let end = Instant::now();
    let duration = end - start;
    println!(
        "average FPS: {:}",
        1. / (duration.as_secs_f32() / (cameras.len() as f32 * num_samples as f32))
    );
}

#[pollster::main]
async fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let opt = Opt::parse();

    println!("reading scene file '{}'", opt.scene.to_string_lossy());

    let scene_file = File::open(opt.scene).unwrap();

    let scene = Scene::from_json(scene_file).unwrap();

    let wgpu_context = WGPUContext::new_instance().await;
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    println!("reading point cloud file '{}'", opt.input.to_string_lossy());

    let file = File::open(&opt.input).unwrap();
    let mut reader = std::io::BufReader::new(file);

    let pc = PointCloud::load(device, &mut reader).unwrap();

    let mut renderer = GaussianRenderer::new(
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        pc.sh_deg(),
        pc.compressed(),
    )
    .await;

    render_views(
        device,
        queue,
        &mut renderer,
        &pc,
        scene.cameras(Some(Split::Train)),
    )
    .await;
}
