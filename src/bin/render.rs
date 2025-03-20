use cgmath::Vector2;
use clap::Parser;
use half::f16;
use image::{ImageBuffer, Rgba};
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use npyz::WriterBuilder;
use url::Url;
use web_splats::{Camera, FrameBuffer, VisChannel};
use wgpu::Color;
use std::str::FromStr;
#[allow(unused_imports)]
use std::{fs::File, path::PathBuf, time::Duration};
#[allow(unused_imports)]
use web_splats::{
    io::GenericGaussianPointCloud, GaussianRenderer, PerspectiveCamera, PointCloud, Scene,
    SceneCamera, SplattingArgs, Split, 
};

#[derive(Debug, Parser)]
#[command(author, version)]
#[command(about = "Dataset offline renderer. Renders to PNG files", long_about = None)]
struct Opt {
    /// input file
    input: Url,

    /// scene json file
    scene: Url,

    /// image output directory
    img_out: PathBuf,

    /// maximum allowed Spherical Harmonics (SH) degree
    #[arg(long, default_value_t = 3)]
    max_sh_deg: u32,
}

#[allow(unused)]
async fn render_views(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut GaussianRenderer,
    pc: &mut PointCloud,
    mut cameras: Vec<SceneCamera>,
    img_out: &PathBuf,
    split: &str,
) {

    if cameras.is_empty() {
        return;
    }

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

    println!("cameras: {:?}",cameras);
    // first render is always wrong...TODO fix this
    cameras.insert(0,cameras[0].clone() );
    cameras.insert(0,cameras[0].clone() );
    cameras.insert(0,cameras[0].clone() );
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

        let frame_buffer = FrameBuffer::new(&device, resolution.x,resolution.y, renderer.color_format());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render encoder"),
        });

        let mut camera: PerspectiveCamera = s.clone().into();
        camera.fit_near_far(&pc.bbox());

        let args = SplattingArgs {
            camera: camera,
            viewport: resolution,
            gaussian_scaling: 1.,
            max_sh_deg: pc.sh_deg(),
            mip_splatting: None,
            kernel_size: None,
            clipping_box: None,
            walltime: Duration::from_secs(100000),
            scene_center: None,
            scene_extend: Some(0.),
            background_color: Color::TRANSPARENT,
            resolution,
            selected_channel: VisChannel::Color,
            
        };
        renderer.prepare(
            &mut encoder,
            device,
            queue,
            &pc,
            args,
            &mut None,
            &frame_buffer
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
            renderer.render(&mut render_pass, &pc, &frame_buffer);
        }
        queue.submit(std::iter::once(encoder.finish()));

        if i < 3{
            continue;
        }

        println!("{:#?}",renderer.sorted_key_value(device, queue).await);

        let id = i - 3;
        save_texture(frame_buffer.color(), device, queue, &img_out.join(format!("{id:0>5}_color.npy"))).await;
        save_texture(frame_buffer.grad_x(), device, queue, &img_out.join(format!("{id:0>5}_dx.npy"))).await;
        save_texture(frame_buffer.grad_y(), device, queue, &img_out.join(format!("{id:0>5}_dy.npy"))).await;
        save_texture(frame_buffer.grad_xy(), device, queue, &img_out.join(format!("{id:0>5}_dxy.npy"))).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[pollster::main]
async fn main() {
    use std::io::Cursor;

    use web_splats::{io, new_wgpu_context};

    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();
    let opt = Opt::parse();

    println!("reading scene file '{}'", opt.scene);

    // TODO this is suboptimal as it is never closed
    let mut ply_file = io::read_from_url(&opt.input).await.unwrap();
    let scene_file = io::read_from_url(&opt.scene).await.unwrap();

    let scene = Scene::from_json(scene_file).unwrap();

    let instance = wgpu::Instance::new(&Default::default());
    let (device,queue,_) = new_wgpu_context(&instance,None).await;
    println!("reading point cloud file '{}'", opt.input);

    let mut data = Vec::new();
    ply_file.read_to_end(&mut data).unwrap();
    let pc_raw = io::GenericGaussianPointCloud::load(Cursor::new(data)).unwrap();
    

    let mut pc = PointCloud::new(&device, &pc_raw).unwrap();
    println!("loaded point cloud with {} points", pc.num_points());

    let render_format = wgpu::TextureFormat::Rgba16Float;

    let mut renderer =
        GaussianRenderer::new(&device, &queue, render_format, pc.sh_deg(), pc.compressed()).await;

    render_views(
        &device,
        &queue,
        &mut renderer,
        &mut pc,
        scene.cameras(Some(Split::Test)),
        &opt.img_out,
        "test",
    )
    .await;
    render_views(
        &device,
        &queue,
        &mut renderer,
        &mut pc,
        scene.cameras(Some(Split::Train)),
        &opt.img_out,
        "train",
    )
    .await;

    println!("done!");
}
#[cfg(target_arch = "wasm32")]
fn main() {
    todo!("not implemented")
}

pub async fn download_texture(
    texture: &wgpu::Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<f16> {
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
        wgpu::TexelCopyBufferInfo {
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

    let image = {
        let data: wgpu::BufferView<'_> =
            download_buffer(device, &staging_buffer, Some(sub_idx)).await;
        data.to_vec()
        .chunks(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]))
        .collect::<Vec<f16>>()
    };

    staging_buffer.unmap();

    return image;
}

async fn save_texture(
    texture: &wgpu::Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    out_file: &PathBuf) {

    let img = download_texture(texture, device, queue).await;
    let resolution = texture.size();
    let mut out_file = File::create(out_file).unwrap();
    let mut writer =npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[resolution.height as u64, resolution.width as u64,4 as u64])
            .writer(&mut out_file)
            .begin_nd().unwrap();
    writer.extend(img).unwrap();
    writer.finish().unwrap();
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
        Some(idx) => wgpu::PollType::WaitForSubmissionIndex(idx),
        None => wgpu::PollType::Wait,
    }).unwrap();
    rx.receive().await.unwrap().unwrap();

    let view = slice.get_mapped_range();
    return view;
}
