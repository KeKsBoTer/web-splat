// dont want to check out how to exactly setup tests...

use web_splats::gpu_rs::{self, GPURSSorter};
use wgpu::util::DeviceExt;

use crate::rs_ref::{calculate_histogram, compare_slice_beginning, prefix_sum_histogram};
mod rs_ref;

async fn download_buffer<T: Clone>(buffer: &wgpu::Buffer, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<T>{
    // copy buffer data
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Download buffer"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder= device.create_command_encoder(&wgpu::CommandEncoderDescriptor {label: Some("Copy encoder")});
    encoder.copy_buffer_to_buffer(buffer, 0, &download_buffer, 0, buffer.size());
    queue.submit([encoder.finish()]);
    
    // download buffer
    let buffer_slice = download_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.receive().await.unwrap().unwrap();
    
    let data = buffer_slice.get_mapped_range();
    let mut r;
    
    unsafe {
        let (prefix, d, suffix) = data.align_to::<T>();
        r = d.to_vec();
    }
    
    download_buffer.destroy();
    
    return r;
}

fn upload_to_buffer<T: bytemuck::Pod>(buffer: &wgpu::Buffer, device : &wgpu::Device, queue: &wgpu::Queue, values: &[T]){
    let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Staging buffer"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {label: Some("Copye endoder")});
    encoder.copy_buffer_to_buffer(&staging_buffer, 0, buffer, 0, staging_buffer.size());
    queue.submit([encoder.finish()]);
    
    device.poll(wgpu::Maintain::Wait);
    staging_buffer.destroy();
}

fn print_first_n<T : std::fmt::Debug + Clone>(v: &Vec<T>, n: usize) {
    println!("{:?}", v[0..n].to_vec());
}

fn main() {
    // creating the context
    
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        dx12_shader_compiler: Default::default(),
    });

    let adapter = pollster::block_on(instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .unwrap();

    let (device, queue) = pollster::block_on(adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits {
                    max_vertex_attributes: 20,
                    max_buffer_size: 2 << 29,
                    max_storage_buffer_binding_size: 2<<29,
                    ..Default::default()
                },
                label: None,
            },
            None, // Trace path
        ))
        .unwrap();

    // testing the histogram counting
    let n = 10000;
    let test_data: Vec<f32> = (0..n).rev().map(|n| n as f32).collect();
    // print_first_n(&test_data, 100);
    
    // creating the gpu buffers
    let histograms = gpu_rs::GPURSSorter::create_internal_mem_buffer(&device, test_data.len());
    let (keyval_a, keyval_b) = GPURSSorter::create_keyval_buffers(&device, test_data.len());
    let mut compute_pipeline = gpu_rs::GPURSSorter::new(&device);
    let (uniform_infos, bind_group) = compute_pipeline.create_bind_group(&device, test_data.len(), &histograms, &keyval_a, &keyval_b);
    
    upload_to_buffer(&keyval_a, &device, &queue, test_data.as_slice());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label: Some("auf gehts")});
    
    // test histogram calculation ----------------------------------------------------------------------------------------------------------
    compute_pipeline.record_calculate_histogram(&bind_group, test_data.len(), &mut encoder);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
    let gpu_hist = pollster::block_on(download_buffer::<u32>(&histograms, &device, &queue));
    
    // println!("Histogramm data: {:?}", gpu_hist);
    
    // println!("size of keyvals: {}", keyval_a.size() / 4);
    let ref_hist = calculate_histogram(test_data.as_slice(), keyval_a.size() as usize);
    // println!("Ref Histogram: {:?}", ref_hist);
    println!("Checking histograms...");
    compare_slice_beginning(gpu_hist.as_slice(), ref_hist.as_slice());
    
    // test prefix calculation ----------------------------------------------------------------------------------------------------------
    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {label: Some("auf ein neues")});
    compute_pipeline.record_prefix_histogram(&bind_group, 4, &mut encoder);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
    let gpu_prefix = pollster::block_on(download_buffer::<u32>(&histograms, &device, &queue));
    // println!("Prefixed histogram data: \n{:?}", gpu_prefix);
    
    let prefix_histogram = prefix_sum_histogram(ref_hist.as_slice());
    // println!("Reference histogram: \n{:?}", prefix_histogram);
    println!("Checking prefixed histograms...");
    compare_slice_beginning(gpu_prefix.as_slice(), prefix_histogram.as_slice());

    // test key scattering ----------------------------------------------------------------------------------------------------------
    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {label: Some("tes sort")});
    compute_pipeline.record_scatter_keys(&bind_group, 4, test_data.len(), &mut encoder);
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
    let gpu_sort = pollster::block_on(download_buffer::<f32>(&keyval_a, &device, &queue));
    println!("keval_b: \n {:?}", gpu_sort);
    
    // tests done ----------------------------------------------------------------------------------------------------------
    println!("Kinda works...");
}
