/*
    This file implements a gpu version of radix sort. A good introduction to general purpose radix sort can
    be found here: http://www.codercorner.com/RadixSortRevisited.htm

    The gpu radix sort implemented here is a reimplementation of the vulkan radix sort found in the fuchsia repos: https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/
    Currently only the sorting for floating point key-value pairs is implemented, as only this is needed for this project

    All shaders can be found in shaders/radix_sort.wgsl
*/

use wgpu::{util::DeviceExt, ComputePassDescriptor};

// IMPORTANT: the following constants have to be synced with the numbers in radix_sort.wgsl
pub const HISTOGRAM_WG_SIZE: usize = 256;
const RS_RADIX_LOG2: usize = 8; // 8 bit radices
const RS_RADIX_SIZE: usize = 1 << RS_RADIX_LOG2; // 256 entries into the radix table
const RS_KEYVAL_SIZE: usize = 32 / RS_RADIX_LOG2;
pub const RS_HISTOGRAM_BLOCK_ROWS: usize = 15;
const RS_SCATTER_BLOCK_ROWS: usize = RS_HISTOGRAM_BLOCK_ROWS; // DO NOT CHANGE, shader assume this!!!
const PREFIX_WG_SIZE: usize = 1 << 7; // one thread operates on 2 prefixes at the same time
const SCATTER_WG_SIZE: usize = 1 << 8;

pub struct GPURSSorter {
    bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_layout: wgpu::BindGroupLayout,
    preprocess_bind_group_layout: wgpu::BindGroupLayout,
    zero_p: wgpu::ComputePipeline,
    histogram_p: wgpu::ComputePipeline,
    prefix_p: wgpu::ComputePipeline,
    scatter_even_p: wgpu::ComputePipeline,
    scatter_odd_p: wgpu::ComputePipeline,
    subgroup_size: usize,
}

pub struct PointCloudSortStuff {
    pub num_points: usize,
    pub(crate) sorter_uni: wgpu::Buffer, // uniform buffer information
    pub(crate) sorter_dis: wgpu::Buffer, // dispatch buffer
    pub(crate) sorter_bg: wgpu::BindGroup, // sorter bind group
    pub(crate) sorter_render_bg: wgpu::BindGroup, // bind group only with the sorted indices for rendering
    pub(crate) sorter_bg_pre: wgpu::BindGroup, // bind group for the preprocess (is the sorter_dis and sorter_bg merged as we only have a limited amount of bgs for the preprocessing)
}

#[allow(dead_code)]
pub struct IndirectDispatch {
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
}

pub struct GeneralInfo {
    pub keys_size: u32,
    pub padded_size: u32,
    pub passes: u32,
    pub even_pass: u32,
    pub odd_pass: u32,
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}

impl GPURSSorter {
    // The new call also needs the queue to be able to determine the maximum subgroup size (Does so by running test runs)
    pub async fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {

        
        let mut cur_sorter: GPURSSorter;

        let mut biggest_that_worked = 0;
        if device.limits().min_subgroup_size > 0 && device.limits().min_subgroup_size < 256{
            biggest_that_worked = device.limits().min_subgroup_size as i32;
        }else{

            log::debug!("Searching for the maximum subgroup size (wgpu currently does not allow to query subgroup sizes)");
            let sizes = vec![1, 8, 16, 32];
            let mut cur_size = 2;
            enum State {
                Init,
                Increasing,
                Decreasing,
            }
            let mut s = State::Init;
            loop {
                if cur_size >= sizes.len() {
                    break;
                }
                log::debug!("Checking sorting with subgroupsize {}", sizes[cur_size]);
                cur_sorter = Self::new_with_sg_size(device, sizes[cur_size]);
                let sort_success = cur_sorter.test_sort(device, queue).await;
                log::debug!("{} worked: {}", sizes[cur_size], sort_success);
                match s {
                    State::Init => {
                        if sort_success {
                            biggest_that_worked = sizes[cur_size];
                            s = State::Increasing;
                            cur_size += 1;
                        } else {
                            s = State::Decreasing;
                            cur_size -= 1;
                        }
                    }
                    State::Increasing => {
                        if sort_success {
                            if sizes[cur_size] > biggest_that_worked {
                                biggest_that_worked = sizes[cur_size];
                            }
                            cur_size += 1;
                        } else {
                            break;
                        }
                    }
                    State::Decreasing => {
                        if sort_success {
                            if sizes[cur_size] > biggest_that_worked {
                                biggest_that_worked = sizes[cur_size];
                            }
                            break;
                        } else {
                            cur_size -= 1;
                        }
                    }
                }
            }
            if biggest_that_worked == 0 {
                panic!(
                    "GPURSSorter::new() No workgroup size that works was found. Unable to use sorter"
                );
            }

        }
        cur_sorter = Self::new_with_sg_size(device, biggest_that_worked);
        log::info!(
            "Created a sorter with subgroup size {}",
            cur_sorter.subgroup_size
        );
        return cur_sorter;
    }

    pub fn create_sort_stuff(
        &self,
        device: &wgpu::Device,
        num_points: usize,
    ) -> PointCloudSortStuff {
        let (sorter_b_a, sorter_b_b, sorter_p_a, sorter_p_b) =
            GPURSSorter::create_keyval_buffers(device, num_points, 4);
        let sorter_int = self.create_internal_mem_buffer(device, num_points);
        let (sorter_uni, sorter_dis, sorter_bg) = self.create_bind_group(
            device,
            num_points,
            &sorter_int,
            &sorter_b_a,
            &sorter_b_b,
            &sorter_p_a,
            &sorter_p_b,
        );
        let sorter_render_bg = self.create_bind_group_render(device, &sorter_uni, &sorter_p_a);
        let sorter_bg_pre = self.create_bind_group_preprocess(
            device,
            &sorter_uni,
            &sorter_dis,
            &sorter_b_a,
            &sorter_p_a,
        );

        PointCloudSortStuff {
            num_points,
            sorter_uni,
            sorter_dis,
            sorter_bg,
            sorter_render_bg,
            sorter_bg_pre,
        }
    }

    fn new_with_sg_size(device: &wgpu::Device, sg_size: i32) -> Self {
        // special variables for scatter shade
        let histogram_sg_size: usize = sg_size as usize;
        let rs_sweep_0_size: usize = RS_RADIX_SIZE / histogram_sg_size;
        let rs_sweep_1_size: usize = rs_sweep_0_size / histogram_sg_size;
        let rs_sweep_2_size: usize = rs_sweep_1_size / histogram_sg_size;
        let rs_sweep_size: usize = rs_sweep_0_size + rs_sweep_1_size + rs_sweep_2_size;
        let _rs_smem_phase_1: usize = RS_RADIX_SIZE + RS_RADIX_SIZE + rs_sweep_size;
        let rs_smem_phase_2: usize = RS_RADIX_SIZE + RS_SCATTER_BLOCK_ROWS * SCATTER_WG_SIZE;
        // rs_smem_phase_2 will always be larger, so always use phase2
        let rs_mem_dwords: usize = rs_smem_phase_2;
        let rs_mem_sweep_0_offset: usize = 0;
        let rs_mem_sweep_1_offset: usize = rs_mem_sweep_0_offset + rs_sweep_0_size;
        let rs_mem_sweep_2_offset: usize = rs_mem_sweep_1_offset + rs_sweep_1_size;

        let bind_group_layout = Self::bind_group_layouts(device);
        let render_bind_group_layout = Self::bind_group_layout_rendering(device);
        let preprocess_bind_group_layout = Self::bind_group_layout_preprocess(device);

        let pipeline_layout: wgpu::PipelineLayout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("radix sort pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let raw_shader: &str = include_str!("shaders/radix_sort.wgsl");
        let shader_w_const = format!(
            "const histogram_sg_size: u32 = {:}u;\n\
            const histogram_wg_size: u32 = {:}u;\n\
            const rs_radix_log2: u32 = {:}u;\n\
            const rs_radix_size: u32 = {:}u;\n\
            const rs_keyval_size: u32 = {:}u;\n\
            const rs_histogram_block_rows: u32 = {:}u;\n\
            const rs_scatter_block_rows: u32 = {:}u;\n\
            const rs_mem_dwords: u32 = {:}u;\n\
            const rs_mem_sweep_0_offset: u32 = {:}u;\n\
            const rs_mem_sweep_1_offset: u32 = {:}u;\n\
            const rs_mem_sweep_2_offset: u32 = {:}u;\n{:}",
            histogram_sg_size,
            HISTOGRAM_WG_SIZE,
            RS_RADIX_LOG2,
            RS_RADIX_SIZE,
            RS_KEYVAL_SIZE,
            RS_HISTOGRAM_BLOCK_ROWS,
            RS_SCATTER_BLOCK_ROWS,
            rs_mem_dwords,
            rs_mem_sweep_0_offset,
            rs_mem_sweep_1_offset,
            rs_mem_sweep_2_offset,
            raw_shader
        );
        let shader_code = shader_w_const
            .replace(
                "{histogram_wg_size}",
                HISTOGRAM_WG_SIZE.to_string().as_str(),
            )
            .replace("{prefix_wg_size}", PREFIX_WG_SIZE.to_string().as_str())
            .replace("{scatter_wg_size}", SCATTER_WG_SIZE.to_string().as_str());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Radix sort shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        let zero_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Zero the histograms"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "zero_histograms",
            compilation_options: Default::default(),
            cache: None,
        });
        let histogram_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("calculate_histogram"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "calculate_histogram",
            compilation_options: Default::default(),
            cache: None,
        });
        let prefix_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefix_histogram"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "prefix_histogram",
            compilation_options: Default::default(),
            cache: None,
        });
        let scatter_even_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter_even"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "scatter_even",
            compilation_options: Default::default(),
            cache: None,
        });
        let scatter_odd_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter_odd"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "scatter_odd",
            compilation_options: Default::default(),
            cache: None,
        });

        return Self {
            bind_group_layout,
            render_bind_group_layout,
            preprocess_bind_group_layout,
            zero_p,
            histogram_p,
            prefix_p,
            scatter_even_p,
            scatter_odd_p,
            subgroup_size: histogram_sg_size,
        };
    }

    async fn test_sort(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        // smiply runs a small sort and check if the sorting result is correct
        let n = 8192; // means that 2 workgroups are needed for sorting
        let scrambled_data: Vec<f32> = (0..n).rev().map(|x| x as f32).collect();
        let sorted_data: Vec<f32> = (0..n).map(|x| x as f32).collect();

        let internal_mem_buffer = Self::create_internal_mem_buffer(self, device, n);
        let (keyval_a, keyval_b, payload_a, payload_b) = Self::create_keyval_buffers(device, n, 4);
        let (_uniform_buffer, _dispatch_buffer, bind_group) = self.create_bind_group(
            device,
            n,
            &internal_mem_buffer,
            &keyval_a,
            &keyval_b,
            &payload_a,
            &payload_b,
        );

        upload_to_buffer(&keyval_a, device, queue, scrambled_data.as_slice());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPURSSorter test_sort"),
        });
        self.record_sort(&bind_group, n, &mut encoder);
        let idx = queue.submit([encoder.finish()]);
        device.poll(wgpu::Maintain::WaitForSubmissionIndex(idx));

        let sorted = download_buffer::<f32>(&keyval_a, device, queue).await;
        for i in 0..n {
            if sorted[i] != sorted_data[i] {
                return false;
            }
        }
        return true;
    }

    // layouts used by the sorting pipeline, as the dispatch buffer has to be in separate bind group
    pub fn bind_group_layouts(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Radix bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    }
    // is used by the preprocess pipeline as the limitation of bind groups forces us to only use 1 bind group for the sort infos
    pub fn bind_group_layout_preprocess(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Radix bind group layout for preprocess pipeline"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    }
    // used by the renderer, as read_only : false is not allowed without an extension
    pub fn bind_group_layout_rendering(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Radix bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    }

    fn get_scatter_histogram_sizes(keysize: usize) -> (usize, usize, usize, usize, usize, usize) {
        // as a general rule of thumb, scater_blocks_ru is equal to histo_blocks_ru, except the amount of elements in these two stages is different

        let scatter_block_kvs = HISTOGRAM_WG_SIZE * RS_SCATTER_BLOCK_ROWS;
        let scatter_blocks_ru = (keysize + scatter_block_kvs - 1) / scatter_block_kvs;
        let count_ru_scatter = scatter_blocks_ru * scatter_block_kvs;

        let histo_block_kvs = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;
        let histo_blocks_ru = (count_ru_scatter + histo_block_kvs - 1) / histo_block_kvs;
        let count_ru_histo = histo_blocks_ru * histo_block_kvs;

        return (
            scatter_block_kvs,
            scatter_blocks_ru,
            count_ru_scatter,
            histo_block_kvs,
            histo_blocks_ru,
            count_ru_histo,
        );
    }

    pub fn create_keyval_buffers(
        device: &wgpu::Device,
        keysize: usize,
        bytes_per_payload_elem: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        //let (_, _, _, _, _, count_ru_histo) = Self::get_scatter_histogram_sizes(keysize);
        let keys_per_workgroup = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;
        let count_ru_histo =
            ((keysize + keys_per_workgroup) / keys_per_workgroup + 1) * keys_per_workgroup;

        // creating the two needed buffers for sorting
        let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix data buffer a"),
            size: (count_ru_histo * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix data buffer a"),
            size: (count_ru_histo * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        assert!(bytes_per_payload_elem == 4); // currently only 4 byte values are allowed
        let payload_size = (keysize * bytes_per_payload_elem).max(1); // make sure that we have at least 1 byte of data;
        let payload_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix data buffer a"),
            size: payload_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let payload_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix data buffer a"),
            size: payload_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        return (buffer_a, buffer_b, payload_a, payload_b);
    }

    // caclulates and allocates a buffer that is sufficient for holding all needed information for
    // sorting. This includes the histograms and the temporary scatter buffer
    // @return: tuple containing [internal memory buffer (should be bound at shader binding 1, count_ru_histo (padded size needed for the keyval buffer)]
    pub fn create_internal_mem_buffer(
        &self,
        device: &wgpu::Device,
        keysize: usize,
    ) -> wgpu::Buffer {
        // currently only a few different key bits are supported, maybe has to be extended
        // assert!(key_bits == 32 || key_bits == 64 || key_bits == 16);

        // subgroup and workgroup sizes
        let histo_sg_size: usize = self.subgroup_size;
        let _histo_wg_size: usize = HISTOGRAM_WG_SIZE;
        let _prefix_sg_size: usize = histo_sg_size;
        let _internal_sg_size: usize = histo_sg_size;

        // The "internal" memory map looks like this:
        //   +---------------------------------+ <-- 0
        //   | histograms[keyval_size]         |
        //   +---------------------------------+ <-- keyval_size                           * histo_size
        //   | partitions[scatter_blocks_ru-1] |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size
        //   | workgroup_ids[keyval_size]      |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size + workgroup_ids_size

        let (_, scatter_blocks_ru, _, _, _, _) = Self::get_scatter_histogram_sizes(keysize);

        let histo_size = RS_RADIX_SIZE * std::mem::size_of::<u32>();

        let internal_size = (RS_KEYVAL_SIZE + scatter_blocks_ru - 1 + 1) * histo_size; // +1 safety

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Internal radix sort buffer"),
            size: internal_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        return buffer;
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        keysize: usize,
        internal_mem_buffer: &wgpu::Buffer,
        keyval_a: &wgpu::Buffer,
        keyval_b: &wgpu::Buffer,
        payload_a: &wgpu::Buffer,
        payload_b: &wgpu::Buffer,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
        let (_, scatter_blocks_ru, _, _, _, count_ru_histo) =
            Self::get_scatter_histogram_sizes(keysize);
        // if keyval_a.size() as usize != count_ru_histo * std::mem::size_of::<f32>()
        //     || keyval_b.size() as usize != count_ru_histo * std::mem::size_of::<f32>()
        // {
        //     panic!("Keyval buffers are not padded correctly. Were they created with GPURSSorter::create_keyval_buffers()");
        // }
        let dispatch_infos = IndirectDispatch {
            dispatch_x: scatter_blocks_ru as u32,
            dispatch_y: 1,
            dispatch_z: 1,
        };
        let uniform_infos = GeneralInfo {
            keys_size: keysize as u32,
            padded_size: count_ru_histo as u32,
            passes: 4,
            even_pass: 0,
            odd_pass: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Radix uniform buffer"),
            contents: unsafe { any_as_u8_slice(&uniform_infos) },
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let dispatch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dispatch indirect buffer"),
            contents: unsafe { any_as_u8_slice(&dispatch_infos) },
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::INDIRECT,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Radix bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: internal_mem_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: keyval_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: keyval_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: payload_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: payload_b.as_entire_binding(),
                },
            ],
        });
        return (uniform_buffer, dispatch_buffer, bind_group);
    }
    pub fn create_bind_group_render(
        &self,
        device: &wgpu::Device,
        general_infos: &wgpu::Buffer,
        payload_a: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let rendering_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render bind group"),
            layout: &self.render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: general_infos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: payload_a.as_entire_binding(),
                },
            ],
        });
        return rendering_bind_group;
    }
    pub fn create_bind_group_preprocess(
        &self,
        device: &wgpu::Device,
        uniform_buffer: &wgpu::Buffer,
        dispatch_buffer: &wgpu::Buffer,
        keyval_a: &wgpu::Buffer,
        payload_a: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Preprocess bind group"),
            layout: &self.preprocess_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: keyval_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: payload_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dispatch_buffer.as_entire_binding(),
                },
            ],
        });
        return bind_group;
    }

    pub fn record_reset_indirect_buffer(
        indirect_buffer: &wgpu::Buffer,
        uniform_buffer: &wgpu::Buffer,
        queue: &wgpu::Queue,
    ) {
        queue.write_buffer(indirect_buffer, 0, &[0u8, 0u8, 0u8, 0u8]); // nulling dispatch x
        queue.write_buffer(uniform_buffer, 0, &[0u8, 0u8, 0u8, 0u8]); // nulling keysize
    }

    pub fn record_calculate_histogram(
        &self,
        bind_group: &wgpu::BindGroup,
        keysize: usize,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // histogram has to be zeroed out such that counts that might have been done in the past are erased and do not interfere with the new count
        // encoder.clear_buffer(histogram_buffer, 0, None);

        // as we only deal with 32 bit float values always 4 passes are conducted
        let (_, _, _, _, hist_blocks_ru, _) = Self::get_scatter_histogram_sizes(keysize);
        const _PASSES: u32 = 4;

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("zeroing the histogram"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.zero_p);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(hist_blocks_ru as u32, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("calculate histogram"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.histogram_p);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(hist_blocks_ru as u32, 1, 1);
        }
    }
    pub fn record_calculate_histogram_indirect(
        &self,
        bind_group: &wgpu::BindGroup,
        dispatch_buffer: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("zeroing the histogram"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.zero_p);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups_indirect(dispatch_buffer, 0);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("calculate histogram"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.histogram_p);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups_indirect(dispatch_buffer, 0);
        }
    }

    // There does not exist an indirect histogram dispatch as the number of prefixes is determined by the amount of passes
    pub fn record_prefix_histogram(
        &self,
        bind_group: &wgpu::BindGroup,
        passes: usize,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prefix histogram"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.prefix_p);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(passes as u32, 1, 1);
    }

    pub fn record_scatter_keys(
        &self,
        bind_group: &wgpu::BindGroup,
        passes: usize,
        keysize: usize,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert!(passes == 4); // currently the amount of passes is hardcoded in the shader
        let (_, scatter_blocks_ru, _, _, _, _) = Self::get_scatter_histogram_sizes(keysize);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scatter keyvals"),
            timestamp_writes: None,
        });

        pass.set_bind_group(0, bind_group, &[]);
        pass.set_pipeline(&self.scatter_even_p);
        pass.dispatch_workgroups(scatter_blocks_ru as u32, 1, 1);

        pass.set_pipeline(&self.scatter_odd_p);
        pass.dispatch_workgroups(scatter_blocks_ru as u32, 1, 1);

        pass.set_pipeline(&self.scatter_even_p);
        pass.dispatch_workgroups(scatter_blocks_ru as u32, 1, 1);

        pass.set_pipeline(&self.scatter_odd_p);
        pass.dispatch_workgroups(scatter_blocks_ru as u32, 1, 1);
    }
    pub fn record_scatter_keys_indirect(
        &self,
        bind_group: &wgpu::BindGroup,
        passes: usize,
        dispatch_buffer: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        assert!(passes == 4);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scatter keyvals"),
            timestamp_writes: None,
        });

        pass.set_bind_group(0, bind_group, &[]);
        pass.set_pipeline(&self.scatter_even_p);
        pass.dispatch_workgroups_indirect(dispatch_buffer, 0);

        pass.set_pipeline(&self.scatter_odd_p);
        pass.dispatch_workgroups_indirect(dispatch_buffer, 0);

        pass.set_pipeline(&self.scatter_even_p);
        pass.dispatch_workgroups_indirect(dispatch_buffer, 0);

        pass.set_pipeline(&self.scatter_odd_p);
        pass.dispatch_workgroups_indirect(dispatch_buffer, 0);
    }

    pub fn record_sort(
        &self,
        bind_group: &wgpu::BindGroup,
        keysize: usize,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.record_calculate_histogram(&bind_group, keysize, encoder);
        self.record_prefix_histogram(&bind_group, 4, encoder);
        self.record_scatter_keys(&bind_group, 4, keysize, encoder);
    }
    pub fn record_sort_indirect(
        &self,
        bind_group: &wgpu::BindGroup,
        dispatch_buffer: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.record_calculate_histogram_indirect(bind_group, dispatch_buffer, encoder);
        self.record_prefix_histogram(bind_group, 4, encoder);
        self.record_scatter_keys_indirect(bind_group, 4, dispatch_buffer, encoder);
    }
}

fn upload_to_buffer<T: bytemuck::Pod>(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[T],
) {
    let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Staging buffer"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copye endoder"),
    });
    encoder.copy_buffer_to_buffer(&staging_buffer, 0, buffer, 0, staging_buffer.size());
    queue.submit([encoder.finish()]);

    device.poll(wgpu::Maintain::Wait);
    staging_buffer.destroy();
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
    device.poll(wgpu::Maintain::Wait);
    rx.receive().await.unwrap().unwrap();
    let data = buffer_slice.get_mapped_range();
    let r;

    unsafe {
        let (_, d, _) = data.align_to::<T>();
        r = d.to_vec();
    }

    return r;
}
