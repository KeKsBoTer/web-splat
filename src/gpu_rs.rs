/*
    This file implements a gpu version of radix sort. A good introduction to general purpose radix sort can
    be found here: http://www.codercorner.com/RadixSortRevisited.htm

    The gpu radix sort implemented here is a reimplementation of the vulkan radix sort found in the fuchsia repos: https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/
    Currently only the sorting for floating point key-value pairs is implemented, as only this is needed for this project

    All shaders can be found in shaders/radix_sort.wgsl
*/

use wgpu::{ComputePassDescriptor, util::DeviceExt};

use crate::{
    camera::{Camera},
    uniform::UniformBuffer,
};

// IMPORTANT: the following constants have to be synced with the numbers in radix_sort.wgsl
const histogram_sg_size: usize = 32;            // depends on the platform, 32 for nvida, 64 for amd
const histogram_wg_size: usize = 256;
const rs_radix_log2: usize = 8;                 // 8 bit radices
const rs_radix_size: usize = 1 << rs_radix_log2;// 256 entries into the radix table
const rs_keyval_size: usize = 32 / rs_radix_log2;
const rs_histogram_block_rows : usize = 15;
const rs_scatter_block_rows : usize = 15;

pub struct GPURSSorter {
    pub bind_group_layout: wgpu::BindGroupLayout,
    zero_p: wgpu::ComputePipeline,
    histogram_p: wgpu::ComputePipeline,
}

pub struct GeneralInfo{
    pub histogram_size: u32,
    pub keys_size: u32,
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
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer { 
                                ty: wgpu::BufferBindingType::Uniform {}, 
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

        const raw_shader : &str = include_str!("shaders/radix_sort.wgsl");
        let shader_w_const = format!("const histogram_sg_size: u32 = {:}u;\n\
                                            const histogram_wg_size: u32 = {:}u;\n\
                                            const rs_radix_log2: u32 = {:}u;\n\
                                            const rs_radix_size: u32 = {:}u;\n\
                                            const rs_keyval_size: u32 = {:}u;\n\
                                            const rs_histogram_block_rows: u32 = {:}u;\n\
                                            const rs_scatter_block_rows: u32 = {:}u;\n{:}", histogram_sg_size, histogram_wg_size, rs_radix_log2, rs_radix_size, rs_keyval_size, rs_histogram_block_rows, rs_scatter_block_rows, raw_shader);
        let shader_code = shader_w_const.replace("{histogram_wg_size}", histogram_wg_size.to_string().as_str());
        println!("{}", shader_code);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Radix sort shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
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
    
    // caclulates and allocates a buffer that is sufficient for holding all needed information for
    // sorting. This includes the histograms and the temporary scatter buffer
    pub fn create_internal_mem_buffer(device: &wgpu::Device, keysize: usize) -> wgpu::Buffer {
        // currently only a few different key bits are supported, maybe has to be extended
        // assert!(key_bits == 32 || key_bits == 64 || key_bits == 16);
        
        // subgroup and workgroup sizes
        const histo_sg_size : usize = histogram_sg_size;
        const histo_wg_size : usize = histogram_wg_size;
        const prefix_sg_size : usize = histo_sg_size;
        const scatter_wg_size : usize = histo_wg_size;
        const internal_sg_size : usize = histo_sg_size;

        // The "internal" memory map looks like this:
        //
        //   +---------------------------------+ <-- 0
        //   | histograms[keyval_size]         |
        //   +---------------------------------+ <-- keyval_size                           * histo_size
        //   | partitions[scatter_blocks_ru-1] |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size
        //   | workgroup_ids[keyval_size]      |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size + workgroup_ids_size
        
        let scatter_block_kvs = scatter_wg_size * rs_scatter_block_rows;
        let scatter_blocks_ru = (keysize + scatter_block_kvs - 1) / scatter_block_kvs;
        let count_ru_scatter = scatter_blocks_ru * scatter_block_kvs;
        
        let histo_block_kvs = histo_wg_size * rs_histogram_block_rows;
        let histo_blocks_ru = (count_ru_scatter + histo_block_kvs - 1) / histo_block_kvs;
        let count_ru_histo = histo_blocks_ru * scatter_block_kvs;

        let mr_keyval_size = rs_keyval_size * count_ru_histo;
        let mr_keyvals_align = rs_keyval_size * histo_sg_size;
        
        let histo_size = rs_radix_size * std::mem::size_of::<u32>();

        let mut internal_size= (rs_keyval_size + scatter_blocks_ru - 1) * histo_size;
        let internal_alignment = internal_sg_size * std::mem::size_of::<u32>();
        
        println!("Created buffer for {keysize} keys, count_ru_scatter {count_ru_scatter}, count_ru_histo {count_ru_histo}, mr_keyval_size {mr_keyval_size}, histo_size {histo_size}");
        println!("internal_size {internal_size}");
        
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Internal radix sort buffer"),
            size: internal_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false
        });
        return buffer;
    }
    
    pub fn record_calculate_histogram(&mut self, bind_group: &wgpu::BindGroup, histogram_buffer: &wgpu::Buffer, keysize: usize, encoder: &mut wgpu::CommandEncoder) {
        // histogram has to be zeroed out such that counts that might have been done in the past are erased and do not interfere with the new count
        // encoder.clear_buffer(histogram_buffer, 0, None);
        
        // as we only deal with 32 bit float values always 4 passes are conducted
        const passes: u32 = 4;

        const scatter_wg_size: usize = histogram_wg_size;
        const scatter_block_kvs: usize = scatter_wg_size * rs_scatter_block_rows;
        let scatter_blocks_ru: usize = (keysize + scatter_block_kvs - 1) / scatter_block_kvs;
        // let count_ru_scatter: usize = scatter_blocks_ru * scatter_block_kvs;

        // const histo_block_kvs: usize = histogram_wg_size * rs_histogram_block_rows;
        // let histo_blocks_ru = (count_ru_scatter + histo_block_kvs - 1) / histo_block_kvs;
        // let count_ru_histo = histo_blocks_ru * histo_block_kvs;
        
        let histo_size = rs_radix_size;
        
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {label: Some("zeroing the histogram")});
            
            pass.set_pipeline(&self.zero_p);
            pass.set_bind_group(0, bind_group, &[]);
            let n = (rs_keyval_size + scatter_blocks_ru - 1) * histo_size;
            let dispatch = ((n as f32 / histogram_wg_size as f32)).ceil() as u32;
            pass.dispatch_workgroups(dispatch, 1, 1);
            println!("Having to clear {n} fields in the histogram buffer");
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {label:Some("calculate histogram")});

            pass.set_pipeline(&self.histogram_p);
            pass.set_bind_group(0, bind_group, &[]);
            let dispatch = ((keysize as f32) / (histogram_wg_size as f32)).ceil() as u32;
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
    }
}