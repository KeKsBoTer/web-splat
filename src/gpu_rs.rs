/*
    This file implements a gpu version of radix sort. A good introduction to general purpose radix sort can
    be found here: http://www.codercorner.com/RadixSortRevisited.htm

    The gpu radix sort implemented here is a reimplementation of the vulkan radix sort found in the fuchsia repos: https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/
    Currently only the sorting for floating point key-value pairs is implemented, as only this is needed for this project

    All shaders can be found in shaders/radix_sort.wgsl
*/

use wgpu::ComputePassDescriptor;

use crate::{
    camera::{Camera},
    uniform::UniformBuffer,
};

// IMPORTANT: the following constants have to be synced with the numbers in radix_sort.wgsl
const histogram_wg_size: usize = 256;

pub struct GPURSSorter {
    pub bind_group_layout: wgpu::BindGroupLayout,
    zero_p: wgpu::ComputePipeline,
    histogram_p: wgpu::ComputePipeline,
}

impl GPURSSorter{
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout =device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
                    label: Some("Radix bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage {read_only: true},
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer { 
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false, 
                                min_binding_size: None 
                            },
                            count: None,
                        }
                    ]
                });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess pipeline layout"),
            bind_group_layouts: &[ &bind_group_layout ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/radix_sort.wgsl"));
        let zero_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Zero the histograms"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "zero_histograms",
        });
        let histogram_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("calculate_histogram"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "calculate_histogram",
        });
        Self { bind_group_layout, zero_p, histogram_p } 
    }
    
    pub fn fill_histogram(&mut self, bind_group: &wgpu::BindGroup, keysize: usize, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {label:Some("fill histogram")});
        pass.set_pipeline(&self.histogram_p);
        pass.set_bind_group(0, bind_group, &[]);
        let dispatch = ((keysize as f32) / (histogram_wg_size as f32)).ceil() as u32;
        pass.dispatch_workgroups(dispatch, 1, 1);
    }
}