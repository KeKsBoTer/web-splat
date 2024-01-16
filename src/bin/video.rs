use cgmath::{InnerSpace, Point3, Vector2, Vector3};
use clap::Parser;
use image::{ImageBuffer, Rgb, Rgba};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use minimp4::Mp4Muxer;
use openh264::{
    encoder::{Encoder, EncoderConfig},
    OpenH264API,
};
use rand::{random, seq::IteratorRandom};
use std::{
    f32::consts::PI,
    fs::File,
    io::{BufWriter, Cursor, Read, Seek, SeekFrom},
    path::PathBuf,
    time::Duration,
};
use web_splats::{
    Animation, DiscAnimation, GaussianRenderer, PerspectiveCamera, PointCloud, Scene, SceneCamera,
    Split, TrackingShot, WGPUContext,
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
    video_out: PathBuf,

    /// maximum allowed Spherical Harmonics (SH) degree
    #[arg(long, default_value_t = 3)]
    max_sh_deg: u32,
}

async fn render_tracking_shot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut GaussianRenderer,
    pc: &mut PointCloud,
    cameras: Vec<SceneCamera>,
    video_out: &PathBuf,
) {
    println!("saving video to '{}'", video_out.to_string_lossy());

    let resolution: Vector2<u32> = Vector2::new(1600, 1060);

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

    let center = Point3::new(0., 3.5, 0.);
    let radius = 6.;
    let up: Vector3<f32> = Vector3::new(0.008746162, 0.74420416, 0.6678949).normalize();

    let first_cam: PerspectiveCamera = cameras[0].clone().into();
    // let rng = rand::SeedableRng::from_seed(0);
    // let mut animation = TrackingShot::from_scene(
    //     cameras
    //         .into_iter()
    //         .choose_multiple(&mut rand::thread_rng(), 10),
    //     10.,
    //     None,
    // );
    let mut animation = DiscAnimation::new(center, up, radius, 1. / 5., first_cam.projection);

    let animation_duration = animation.duration();
    println!("{:?}", animation_duration);
    let fps = 30;

    let config = EncoderConfig::new(resolution.x, resolution.y);
    let api = OpenH264API::from_source();
    let mut encoder = Encoder::with_config(api, config).unwrap();

    let bg = wgpu::Color::BLACK;

    let mut buf = Vec::new();

    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let num_frames = (animation_duration.as_secs_f32() * fps as f32).ceil() as u32;
    for i in (0..num_frames).progress() {
        let cam = animation.update(Duration::from_secs_f32(1. / fps as f32));
        // let mut resolution: Vector2<u32> = Vector2::new(s.width, s.height);

        renderer.render(device, queue, &pc, cam, resolution, &target_view, bg);

        renderer.stopwatch.reset();

        let img = download_texture(&target, device, queue).await;

        let yuv = openh264::formats::YUVBuffer::with_rgb(
            img.width() as usize,
            img.height() as usize,
            img.as_raw(),
        );

        // Encode YUV into H.264.
        let bitstream = encoder.encode(&yuv).unwrap();
        bitstream.write_vec(&mut buf);
    }

    let mut out_file = std::fs::File::create(video_out).unwrap();
    let mut writer = BufWriter::new(&mut out_file);
    let mut mp4muxer = Mp4Muxer::new(&mut writer);
    mp4muxer.init_video(
        resolution.x as i32,
        resolution.y as i32,
        false,
        "scene tracking shot",
    );
    mp4muxer.write_video_with_fps(&buf, fps);
    mp4muxer.close();
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
    let mut pc = PointCloud::load(&wgpu_context.device, ply_file).unwrap();

    let mut renderer = GaussianRenderer::new(
        device,
        queue,
        wgpu::TextureFormat::Rgba32Float,
        pc.sh_deg(),
        pc.compressed(),
    )
    .await;

    render_tracking_shot(
        device,
        queue,
        &mut renderer,
        &mut pc,
        scene.cameras(Some(Split::Test)),
        &opt.video_out,
    )
    .await;

    println!("done!");
}

pub async fn download_texture(
    texture: &wgpu::Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let texture_format = texture.format();

    let texel_size: u32 = texture_format.block_size(None).unwrap();
    let fb_size = texture.size();
    let align: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1;
    let bytes_per_row = (texel_size * fb_size.width) + align & !align;

    let output_buffer_size = (bytes_per_row * fb_size.height) as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        label: Some("texture download buffer"),
        mapped_at_creation: false,
    });

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
    queue.submit(std::iter::once(encoder.finish()));

    let mut image = {
        let data = download_buffer(&staging_buffer, device, queue).await;

        let buf: Vec<u8> = data
            .to_vec()
            .chunks_exact(16)
            .flat_map(|c| {
                let r = f32::from_le_bytes(c[0..4].try_into().unwrap()).clamp(0., 1.);
                let g = f32::from_le_bytes(c[4..8].try_into().unwrap()).clamp(0., 1.);
                let b = f32::from_le_bytes(c[8..12].try_into().unwrap()).clamp(0., 1.);
                let a = f32::from_le_bytes(c[12..16].try_into().unwrap()).clamp(0., 1.);
                [(r * 255.) as u8, (g * 255.) as u8, (b * 255.) as u8]
            })
            .collect();
        ImageBuffer::<Rgb<u8>, _>::from_raw(fb_size.width, fb_size.height, buf).unwrap()
    };

    return image::imageops::crop(&mut image, 0, 0, fb_size.width, fb_size.height).to_image();
}

async fn download_buffer<T: Clone>(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<T> {
    // copy buffer data
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Download buffer"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &download_buffer, 0, buffer.size());
    queue.submit([encoder.finish()]);

    // download buffer
    let buffer_slice = download_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.receive().await.unwrap().unwrap();
    let data = buffer_slice.get_mapped_range();
    let r;

    unsafe {
        let (_, d, _) = data.align_to::<T>();
        r = d.to_vec();
    }

    return r;
}
