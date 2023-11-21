use cgmath::Vector2;
use clap::Parser;
use image::{ImageBuffer, Rgb, Rgba};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use minimp4::Mp4Muxer;
use openh264::encoder::{Encoder, EncoderConfig};
use std::{
    fs::File,
    io::{BufWriter, Cursor, Read, Seek, SeekFrom},
    path::PathBuf,
    time::Duration,
};
use web_splats::{
    Animation, GaussianRenderer, PCDataType, PointCloud, Scene, SceneCamera, Split, TrackingShot,
    WGPUContext,
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
    let mut animation = TrackingShot::from_scene(cameras.clone(), 2., None);

    let animation_duration = animation.duration();
    let fps = 30.;

    let config = EncoderConfig::new(resolution.x, resolution.y);
    let mut encoder = Encoder::with_config(config).unwrap();

    let mut buf = Vec::new();

    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let num_frames = (animation_duration.as_secs_f32() * fps).ceil() as u32;
    for i in (0..num_frames).progress() {
        let cam = animation.update(Duration::from_secs_f32(1. / fps));
        // let mut resolution: Vector2<u32> = Vector2::new(s.width, s.height);

        renderer.render(device, queue, &target_view, &pc, cam, resolution);

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
    mp4muxer.write_video(&buf);
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
        PCDataType::PLY => PointCloud::load_ply(
            &wgpu_context.device,
            &wgpu_context.queue,
            ply_file,
            Some(opt.max_sh_deg),
        )
        .unwrap(),
        #[cfg(feature = "npz")]
        PCDataType::NPZ => PointCloud::load_npz(
            &wgpu_context.device,
            &wgpu_context.queue,
            ply_file,
            Some(opt.max_sh_deg),
        )
        .unwrap(),
    };

    let mut renderer = GaussianRenderer::new(
        device,
        wgpu::TextureFormat::Rgba32Float,
        pc.sh_deg(),
        pc_data_type == PCDataType::PLY,
    );

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

    let output_buffer_desc = wgpu::BufferDescriptor {
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: Some("texture download buffer"),
        mapped_at_creation: false,
    };
    let download_buffer = device.create_buffer(&output_buffer_desc);

    let mut encoder: wgpu::CommandEncoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download frame buffer encoder"),
        });

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::ImageCopyBufferBase {
            buffer: &download_buffer,
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
        let data = web_splats::download_buffer(device, &download_buffer, Some(sub_idx)).await;

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

        ImageBuffer::<Rgb<_>, _>::from_raw(bytes_per_row / texel_size, fb_size.height, buf).unwrap()
    };

    download_buffer.unmap();

    return image::imageops::crop(&mut image, 0, 0, fb_size.width, fb_size.height).to_image();
}
