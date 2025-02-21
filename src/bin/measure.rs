use cgmath::Vector2;
use clap::Parser;
#[allow(unused_imports)]
use std::{
    fs::File,
    path::PathBuf,
    time::{Duration, Instant},
};
#[allow(unused_imports)]
use web_splats::{
    io, GaussianRenderer, PerspectiveCamera, PointCloud, Scene, SceneCamera, SplattingArgs, Split,
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

#[allow(unused)]
async fn render_views(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut GaussianRenderer,
    pc: &PointCloud,
    cameras: Vec<SceneCamera>,
) {
    let resolution: Vector2<u32> = Vector2::new(2048, 2048);

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

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("render encoder"),
    });
    let mut camera: PerspectiveCamera = cameras[0].clone().into();
    camera.fit_near_far(pc.bbox());
    // first render to lazy init sorter stuff
    renderer.prepare(
        &mut encoder,
        device,
        queue,
        &pc,
        SplattingArgs {
            camera: camera,
            viewport: resolution,
            gaussian_scaling: 1.,
            max_sh_deg: pc.sh_deg(),
            mip_splatting: None,
            kernel_size: None,
            clipping_box: None,
            walltime: Duration::from_secs(100),
            scene_center: None,
            scene_extend: None,
            background_color: wgpu::Color::BLACK,
            resolution,
        },
        &mut None,
    );
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        renderer.render(&mut render_pass, &pc);
    }
    queue.submit(std::iter::once(encoder.finish()));

    let num_samples = 10;
    for (i, s) in cameras.iter().enumerate() {
        for _ in 0..num_samples {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("render encoder"),
            });
            let mut camera: PerspectiveCamera = s.clone().into();
            camera.fit_near_far(pc.bbox());
            // first render to lazy init sorter stuff
            renderer.prepare(
                &mut encoder,
                device,
                queue,
                &pc,
                SplattingArgs {
                    camera: camera,
                    viewport: resolution,
                    gaussian_scaling: 1.,
                    max_sh_deg: pc.sh_deg(),
                    mip_splatting: None,
                    kernel_size: None,
                    clipping_box: None,
                    walltime: Duration::from_secs(100),
                    scene_center: None,
                    scene_extend: None,
                    background_color: wgpu::Color::BLACK,
                    resolution,
                },
                &mut None,
            );
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                renderer.render(&mut render_pass, &pc);
            }
            queue.submit(std::iter::once(encoder.finish()));
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

#[cfg(not(target_arch = "wasm32"))]
#[pollster::main]
async fn main() {
    use web_splats::new_wgpu_context;

    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let opt = Opt::parse();

    println!("reading scene file '{}'", opt.scene.to_string_lossy());

    let scene_file = File::open(opt.scene).unwrap();

    let scene = Scene::from_json(scene_file).unwrap();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let (device,queue,_) =new_wgpu_context(&instance, None).await;

    println!("reading point cloud file '{}'", opt.input.to_string_lossy());

    let file = File::open(&opt.input).unwrap();
    let mut reader = std::io::BufReader::new(file);

    let pc_raw = io::GenericGaussianPointCloud::load(&mut reader).unwrap();
    let pc = PointCloud::new(&device, &pc_raw).unwrap();

    let mut renderer = GaussianRenderer::new(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba8Unorm,
        pc.sh_deg(),
        pc.compressed(),
    )
    .await;

    render_views(
        &device,
        &queue,
        &mut renderer,
        &pc,
        scene.cameras(Some(Split::Train)),
    )
    .await;
}

#[cfg(target_arch = "wasm32")]
fn main() {
    todo!("not implemented")
}
