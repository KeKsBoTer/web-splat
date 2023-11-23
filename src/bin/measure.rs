use cgmath::Vector2;
use clap::Parser;
use std::{
    fs::File,
    path::PathBuf,
    time::{Duration, Instant},
};
use web_splats::{
    GaussianRenderer, PCDataType, PointCloud, Scene, SceneCamera, Split, WGPUContext,
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
    pc: &mut PointCloud,
    cameras: Vec<SceneCamera>,
) {
    let resolution: Vector2<u32> = Vector2::new(1920, 1080);

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render texture"),
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
    let num_samples = 50;
    let mut preprocess_time = Duration::ZERO;
    let mut sorting_time = Duration::ZERO;
    let mut rasterization_time = Duration::ZERO;
    for (i, s) in cameras.iter().enumerate() {
        for _ in 0..num_samples {
            renderer.render(
                device,
                queue,
                &target_view,
                &pc,
                s.clone().into(),
                resolution,
            );
            let timing = renderer.stopwatch.take_measurements(&device, &queue).await;
            preprocess_time += *timing.get("preprocess").unwrap();
            sorting_time += *timing.get("sorting").unwrap();
            rasterization_time += *timing.get("rasterization").unwrap();
        }
    }
    device.poll(wgpu::MaintainBase::Wait);
    let end = Instant::now();
    let duration = end - start;
    println!(
        "average FPS: {:}",
        1. / (duration.as_secs_f32() / (cameras.len() as f32 * num_samples as f32))
    );
    println!(
        "preprocess: {:.2}, sorting: {:.2}, rasterization: {:.2}, total {:.2}",
        preprocess_time.as_secs_f32() * 1000. / (num_samples as f32 * cameras.len() as f32),
        sorting_time.as_secs_f32() * 1000. / (num_samples as f32 * cameras.len() as f32),
        rasterization_time.as_secs_f32() * 1000. / (num_samples as f32 * cameras.len() as f32),
        (preprocess_time + sorting_time + rasterization_time).as_secs_f32() * 1000.
            / (num_samples as f32 * cameras.len() as f32),
    );
}

#[pollster::main]
async fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let opt = Opt::parse();

    println!("reading scene file '{}'", opt.scene.to_string_lossy());

    // TODO this is suboptimal as it is never closed
    let ply_file = File::open(&opt.input).unwrap();
    let scene_file = File::open(opt.scene).unwrap();

    let scene = Scene::from_json(scene_file).unwrap();

    let wgpu_context = WGPUContext::new_instance().await;
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    println!("reading point cloud file '{}'", opt.input.to_string_lossy());
    let pc_data_type = match opt
        .input
        .extension()
        .expect("file has no extension!")
        .to_str()
        .unwrap()
    {
        "ply" => PCDataType::PLY,
        #[cfg(feature = "npz")]
        "npz" => PCDataType::NPZ,
        ext => panic!("unsupported file type '{ext}"),
    };
    let mut pc = match pc_data_type {
        PCDataType::PLY => {
            PointCloud::load_ply(&wgpu_context.device, &wgpu_context.queue, ply_file, None).unwrap()
        }
        #[cfg(feature = "npz")]
        PCDataType::NPZ => {
            PointCloud::load_npz(&wgpu_context.device, &wgpu_context.queue, ply_file, None).unwrap()
        }
    };

    let mut renderer = GaussianRenderer::new(
        device,
        wgpu::TextureFormat::Rgba8Unorm,
        pc.sh_deg(),
        pc_data_type == PCDataType::PLY,
    );

    render_views(
        device,
        queue,
        &mut renderer,
        &mut pc,
        scene.cameras(Some(Split::Train)),
    )
    .await;
}
