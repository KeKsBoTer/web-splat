use cgmath::Vector2;
use clap::Parser;

use image::{ImageBuffer, Rgb};
use indicatif::{ProgressIterator, ProgressStyle};
use std::{
    fs::{create_dir_all, File},
    path::PathBuf,
    time::Duration,
};
use web_splats::{
    io, new_wgpu_context, smoothstep, Animation, GaussianRenderer, PointCloud, Scene, SceneCamera, SplattingArgs, TrackingShot
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

    /// duration of animation
    /// if not set, the duration is determined by the number of cameras in the scene (1 sec per camera)
    #[arg(long)]
    duration: Option<f32>,

    #[arg(long, default_value_t = 30)]
    fps: u32,
}

async fn render_tracking_shot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut GaussianRenderer,
    pc: PointCloud,
    cameras: Vec<SceneCamera>,
    video_out: &PathBuf,
    duration: Option<Duration>,
    fps: u32,
) {
    println!("saving video to '{}'", video_out.to_string_lossy());

    let resolution: Vector2<u32> = Vector2::new(1024, 1024) * 2;

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

    let trackshot_duration = duration.unwrap_or(Duration::from_secs_f32(cameras.len() as f32 * 3.));

    let mut animation = Animation::new(
        trackshot_duration,
        true,
        Box::new(TrackingShot::from_cameras(cameras)),
    );

    let init_duration = Duration::from_secs_f32(4.);
    let video_duration = trackshot_duration + init_duration;

    println!("video duration: {:?}", video_duration);

    let bg = wgpu::Color::BLACK;

    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let num_frames = (video_duration.as_secs_f32() * fps as f32).ceil() as u32;

    let pb_style =
        ProgressStyle::with_template("rendering {bar:40.cyan/blue} {pos:>7}/{len:7} {eta_precise}")
            .unwrap();

    create_dir_all(video_out).unwrap();

    let mut state_time = Duration::ZERO;
    for i in (0..num_frames).progress_with_style(pb_style) {
        let dt = Duration::from_secs_f32(1. / (fps as f32));
        state_time += dt;
        // let mut cam = animation.update(if state_time >= init_duration {
        //     dt
        // } else {
        //     Duration::ZERO
        //     // dt.mul_f32()
        // });
        animation.set_progress(smoothstep(
            state_time.as_secs_f32() / video_duration.as_secs_f32(),
        ));
        let mut cam = animation.update(Duration::ZERO);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });
        cam.fit_near_far(pc.bbox());
        // first render to lazy init sorter stuff
        renderer.prepare(
            &mut encoder,
            device,
            queue,
            &pc,
            SplattingArgs {
                camera: cam,
                viewport: resolution,
                gaussian_scaling: 1.,
                max_sh_deg: pc.sh_deg(),
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: state_time,
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
                        load: wgpu::LoadOp::Clear(bg),
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

        img.save(&video_out.join(format!("frame_{:04}.png", i)))
            .unwrap();
    }
}

#[pollster::main]
async fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let opt = Opt::parse();

    println!("reading scene file '{}'", opt.scene.to_string_lossy());

    // TODO this is suboptimal as it is never closed
    let mut ply_file = File::open(&opt.input).unwrap();
    let scene_file = File::open(opt.scene).unwrap();

    let scene = Scene::from_json(scene_file).unwrap();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let (device,queue,_) = new_wgpu_context(&instance, None).await;

    println!("reading point cloud file '{}'", opt.input.to_string_lossy());
    let pc_raw = io::GenericGaussianPointCloud::load(&mut ply_file).unwrap();
    let pc = PointCloud::new(&device, &pc_raw).unwrap();

    let mut renderer = GaussianRenderer::new(
        &device,
        &queue,
        wgpu::TextureFormat::Rgba32Float,
        pc.sh_deg(),
        pc.compressed(),
    )
    .await;

    render_tracking_shot(
        &device,
        &queue,
        &mut renderer,
        pc,
        scene.cameras(None),
        &opt.video_out,
        opt.duration.map(Duration::from_secs_f32),
        opt.fps,
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

    let texel_size: u32 = texture_format.block_copy_size(None).unwrap();
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
                let _a = f32::from_le_bytes(c[12..16].try_into().unwrap()).clamp(0., 1.);
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
    device.poll(wgpu::PollType::Wait).unwrap();
    rx.receive().await.unwrap().unwrap();
    let data = buffer_slice.get_mapped_range();
    let r;

    unsafe {
        let (_, d, _) = data.align_to::<T>();
        r = d.to_vec();
    }

    return r;
}
