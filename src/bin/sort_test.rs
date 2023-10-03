// dont want to check out how to exactly setup tests...

use web_splats::gpu_rs;
use wgpu::util::DeviceExt;

use crate::rs_ref::{calculate_histogram, compare_slice_beginning};
mod rs_ref;

async fn download_buffer<T: Clone>(buffer: &wgpu::Buffer, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<T>{
    // copy buffer data
    let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Download bffer"),
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

fn print_first_n<T : std::fmt::Debug + Clone>(v: &Vec<T>, n: usize) {
    println!("{:?}", v[0..n].to_vec());
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>(),)
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
    let test_data: Vec<f32> = (0..10000).rev().map(|n| n as f32).collect();
    print_first_n(&test_data, 100);
    let uniform_infos= gpu_rs::GeneralInfo{histogram_size: 0, keys_size: test_data.len() as u32, passes: 4};
    
    let gpu_data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("data buffer"),
        contents: bytemuck::cast_slice(test_data.as_slice()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let histograms = gpu_rs::GPURSSorter::create_internal_mem_buffer(&device, test_data.len());
    let gpu_uniform_infos = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniform infos"),
        contents: unsafe{any_as_u8_slice(&uniform_infos)},
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
    });
    
    let mut compute_pipeline = gpu_rs::GPURSSorter::new(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label: Some("auf gehts")});
    let mut bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("main bind group"),
        layout: &compute_pipeline.bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_data.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: histograms.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 2,
            resource: gpu_uniform_infos.as_entire_binding(),
        },
        ],
    });

    compute_pipeline.record_calculate_histogram(&bind_group, &histograms, test_data.len(), &mut encoder);
    
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
    let res = pollster::block_on(download_buffer::<u32>(&histograms, &device, &queue));
    
    println!("Histogramm data: {:?}", res);
    
    let ref_hist = calculate_histogram(test_data.as_slice());
    compare_slice_beginning(res.as_slice(), ref_hist.as_slice());
    
    println!("Kinda works...");
}