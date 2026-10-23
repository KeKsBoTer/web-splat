use cgmath::Vector2;
use clap::Parser;

use half::f16;
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressIterator, ProgressStyle};
use std::{
    fs::{File, create_dir_all},
    path::PathBuf,
    time::Duration,
};
use web_splats::{
    Animation, DEFAULT_ALPHA_CUTOFF, GaussianRenderer, PointCloud, Scene, SceneCamera,
    SplattingArgs, TrackingShot, WGPUContext, io, smoothstep,
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
                alpha_cutoff: DEFAULT_ALPHA_CUTOFF,
                walltime: state_time,
                scene_center: None,
                scene_extend: None,
                background_color: wgpu::Color::TRANSPARENT,
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
                        // (front to back blending needs the alpha to start at 0, the
                        // background is added when the image is saved)
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            renderer.render(&mut render_pass, &pc);
        }
        queue.submit(std::iter::once(encoder.finish()));

        let img = download_texture(&target, device, queue, bg).await;

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

    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let wgpu_context = WGPUContext::new(&instance, None).await;
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    println!("reading point cloud file '{}'", opt.input.to_string_lossy());
    let pc_raw = io::GenericGaussianPointCloud::load(&mut ply_file).unwrap();
    let pc = PointCloud::new(device, queue, pc_raw).unwrap();

    let mut renderer = GaussianRenderer::new(
        device,
        wgpu::TextureFormat::Rgba16Float,
        pc.sh_deg(),
        pc.compressed(),
    );

    render_tracking_shot(
        device,
        queue,
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
    background: wgpu::Color,
) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let texture_format = texture.format();

    let texel_size: u32 = texture_format.block_copy_size(None).unwrap();
    let fb_size = texture.size();
    let align: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1;
    let bytes_per_row = (texel_size * fb_size.width) + align & !align;

    let output_buffer_size = (bytes_per_row * fb_size.height) as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: Some("texture download buffer"),
        mapped_at_creation: false,
    });

    let mut encoder: wgpu::CommandEncoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download frame buffer encoder"),
        });

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfoBase {
            buffer: &staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(fb_size.height),
            },
        },
        fb_size,
    );
    let sub_idx = queue.submit(std::iter::once(encoder.finish()));

    let mut image = {
        let data = download_buffer(device, &staging_buffer, Some(sub_idx)).await;

        // rgba, 16 bit floats (the rows can be padded). The color is premultiplied with the
        // alpha: the background shows through where the alpha is below 1
        let bg = [
            background.r as f32,
            background.g as f32,
            background.b as f32,
        ];
        let channel = |c: &[u8], i: usize| f16::from_le_bytes([c[2 * i], c[2 * i + 1]]).to_f32();
        let buf: Vec<u8> = data
            .chunks_exact(texel_size as usize)
            .flat_map(|c| {
                let alpha = channel(c, 3).clamp(0., 1.);
                [0, 1, 2]
                    .map(|i| ((channel(c, i) + (1. - alpha) * bg[i]).clamp(0., 1.) * 255.) as u8)
            })
            .collect();
        ImageBuffer::<Rgb<u8>, _>::from_raw(bytes_per_row / texel_size, fb_size.height, buf)
            .unwrap()
    };

    staging_buffer.unmap();

    return image::imageops::crop(&mut image, 0, 0, fb_size.width, fb_size.height).to_image();
}

async fn download_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    wait_idx: Option<wgpu::SubmissionIndex>,
) -> wgpu::BufferView {
    let slice = buffer.slice(..);

    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: wait_idx,
            timeout: None,
        })
        .unwrap();
    rx.receive().await.unwrap().unwrap();

    let view = slice.get_mapped_range();
    return view.unwrap();
}
