use cgmath::Vector2;
use image::{ImageBuffer, Rgba};
use std::path::PathBuf;
use structopt::StructOpt;
use web_splats::{GaussianRenderer, PerspectiveCamera, PointCloud, SHDtype, Scene, WGPUContext};

#[derive(Debug, StructOpt)]
#[structopt(name = "viewer", about = "3D gaussian splats renderer")]
struct Opt {
    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Scene json file
    #[structopt(parse(from_os_str))]
    scene: PathBuf,

    /// Scene json file
    #[structopt(parse(from_os_str))]
    img_out: PathBuf,
}

fn main() {
    let opt = Opt::from_args();

    let scene = Scene::from_json(opt.scene).unwrap();

    let wgpu_context = pollster::block_on(WGPUContext::new_instance());
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    let mut pc = PointCloud::load_ply(&wgpu_context.device, opt.input, SHDtype::Float).unwrap();

    let mut renderer = GaussianRenderer::new(
        &wgpu_context.device,
        wgpu::TextureFormat::Rgba8Unorm,
        pc.sh_deg(),
        pc.sh_dtype(),
    );

    let resolution: Vector2<u32> = Vector2::new(scene.camera(0).width, scene.camera(0).height);

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
        view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
    });

    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    std::fs::create_dir_all(opt.img_out.clone()).unwrap();

    for (i, s) in scene.cameras().iter().enumerate() {
        let c: PerspectiveCamera = s.clone().into();
        pc.sort(queue, c);
        renderer.render(
            device,
            queue,
            &target_view,
            &pc,
            s.clone().into(),
            resolution,
        );
        let img = pollster::block_on(download_texture(&target, device, queue));
        img.save(opt.img_out.join(format!("{i:0>5}.png"))).unwrap();
    }
}

pub async fn download_texture(
    texture: &wgpu::Texture,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let texture_format = texture.format();

    let texel_size: u32 = texture_format.block_size(None).unwrap();
    let fb_size = texture.size();
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1;
    let bytes_per_row = (texel_size * fb_size.width) + align & !align;

    let output_buffer_size = (bytes_per_row * fb_size.height) as wgpu::BufferAddress;

    let output_buffer_desc = wgpu::BufferDescriptor {
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: None,
        mapped_at_creation: false,
    };
    let download_buffer = device.create_buffer(&output_buffer_desc);

    let mut encoder: wgpu::CommandEncoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("download frame buffer encoder"),
        });

    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1;
    let bytes_per_row = (texel_size * fb_size.width) + align & !align;

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
    let buffer_slice = download_buffer.slice(..);

    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(sub_idx));
    rx.receive().await.unwrap().unwrap();

    let mut image = {
        // unmap can only happen im BufferView is droped before
        let data = buffer_slice.get_mapped_range();

        ImageBuffer::<Rgba<u8>, _>::from_raw(
            bytes_per_row / texel_size,
            fb_size.height,
            data.to_vec(),
        )
        .unwrap()
    };

    download_buffer.unmap();

    return image::imageops::crop(&mut image, 0, 0, fb_size.width, fb_size.height).to_image();
}
