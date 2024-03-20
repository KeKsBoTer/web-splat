use cgmath::Vector2;
use clap::Parser;
use half::f16;
use image::{ImageBuffer, Rgba};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use std::{fs::File, path::PathBuf, time::Duration};
use web_splats::{
    io::GenericGaussianPointCloud, GaussianRenderer, PerspectiveCamera, PointCloud, Scene,
    SceneCamera, SplattingArgs, Split, WGPUContext,
};

#[derive(Debug, Parser)]
#[command(author, version)]
#[command(about = "Dataset offline renderer. Renders to PNG files", long_about = None)]
struct Opt {
    /// input file
    input: PathBuf,

    /// scene json file
    scene: PathBuf,

    /// image output directory
    img_out: PathBuf,

    /// maximum allowed Spherical Harmonics (SH) degree
    #[arg(long, default_value_t = 3)]
    max_sh_deg: u32,
}

async fn render_views(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut GaussianRenderer,
    pc: &mut PointCloud,
    cameras: Vec<SceneCamera>,
    img_out: &PathBuf,
    split: &str,
) {
    let img_out = img_out.join(&split);
    println!("saving images to '{}'", img_out.to_string_lossy());
    std::fs::create_dir_all(img_out.clone()).unwrap();

    let pb = ProgressBar::new(cameras.len() as u64);
    let pb_style = ProgressStyle::with_template(
        "{msg} {spinner:.green} [{bar:.cyan/blue}] {pos}/{len} [{elapsed}/{duration}]",
    )
    .unwrap()
    .progress_chars("#>-");
    pb.set_style(pb_style);
    pb.set_message(format!("rendering {split}"));

    for (i, s) in cameras.iter().enumerate().progress_with(pb) {
        let mut resolution: Vector2<u32> = Vector2::new(s.width, s.height);

        if resolution.x > 1600 {
            let s = resolution.x as f32 / 1600.;
            resolution.x = 1600;
            resolution.y = (resolution.y as f32 / s) as u32;
        }

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

        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

        let mut camera: PerspectiveCamera = s.clone().into();
        camera.fit_near_far(pc.bbox());
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
                show_env_map: false,
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: Duration::from_secs(100),
                scene_center: None,
                scene_extend: None,
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
        let img = download_texture(&target, device, queue).await;
        img.save(img_out.join(format!("{i:0>5}.png"))).unwrap();
    }
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

    let pc_raw = GenericGaussianPointCloud::load(ply_file).unwrap();
    let mut pc = PointCloud::new(&device, pc_raw).unwrap();

    let render_format = wgpu::TextureFormat::Rgba16Float;

    let mut renderer =
        GaussianRenderer::new(&device, &queue, render_format, pc.sh_deg(), pc.compressed()).await;

    render_views(
        device,
        queue,
        &mut renderer,
        &mut pc,
        scene.cameras(Some(Split::Test)),
        &opt.img_out,
        "test",
    )
    .await;
    render_views(
        device,
        queue,
        &mut renderer,
        &mut pc,
        scene.cameras(Some(Split::Train)),
        &opt.img_out,
        "train",
    )
    .await;

    println!("done!");
}

pub async fn download_texture(
    texture: &wgpu::Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let texture_format = texture.format();

    let texel_size: u32 = texture_format.block_copy_size(None).unwrap();
    let fb_size = texture.size();
    let align: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1;
    let bytes_per_row = (texel_size * fb_size.width) + align & !align;

    let output_buffer_size = (bytes_per_row * fb_size.height) as wgpu::BufferAddress;

    let output_buffer_desc = wgpu::BufferDescriptor {
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: Some("texture download buffer"),
        mapped_at_creation: false,
    };
    let staging_buffer = device.create_buffer(&output_buffer_desc);

    let mut encoder: wgpu::CommandEncoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download frame buffer encoder"),
        });

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::ImageCopyBufferBase {
            buffer: &staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(fb_size.height),
            },
        },
        fb_size,
    );
    let sub_idx = queue.submit(std::iter::once(encoder.finish()));

    let mut image = {
        let data: wgpu::BufferView<'_> =
            download_buffer(device, &staging_buffer, Some(sub_idx)).await;

        ImageBuffer::<Rgba<u8>, _>::from_raw(
            bytes_per_row / texel_size,
            fb_size.height,
            data.to_vec()
                .chunks(2)
                .map(|c| (f16::from_le_bytes([c[0], c[1]]).to_f32().clamp(0., 1.) * 255.) as u8)
                .collect::<Vec<u8>>(),
        )
        .unwrap()
    };

    staging_buffer.unmap();

    return image::imageops::crop(&mut image, 0, 0, fb_size.width, fb_size.height).to_image();
}

async fn download_buffer<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
    wait_idx: Option<wgpu::SubmissionIndex>,
) -> wgpu::BufferView<'a> {
    let slice = buffer.slice(..);

    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(match wait_idx {
        Some(idx) => wgpu::Maintain::WaitForSubmissionIndex(idx),
        None => wgpu::Maintain::Wait,
    });
    rx.receive().await.unwrap().unwrap();

    let view = slice.get_mapped_range();
    return view;
}
