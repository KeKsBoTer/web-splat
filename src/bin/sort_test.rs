// dont want to check out how to exactly setup tests...

use web_splats::gpu_rs;
use wgpu::util::DeviceExt;

async fn download_buffer<T: Clone>(buffer: &wgpu::Buffer, device: &wgpu::Device) -> Vec<T>{
    let buffer_slice = buffer.slice(..);
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
    
    return r;
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
    let test_data: Vec<f32> = (0..10000).rev().map(|n| n as f32).collect();
    print_first_n(&test_data, 100);
    
    let gpu_data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("data buffer"),
        contents: bytemuck::cast_slice(test_data.as_slice()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let histograms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("histogram data"),
        size: 100,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
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
        }
        ],
    });

    compute_pipeline.fill_histogram(&bind_group, test_data.len(), &mut encoder);
    
    println!("Kinda works...");
}