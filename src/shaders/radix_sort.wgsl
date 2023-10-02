// shader implementing gpu radix sort. More information in the beginning of gpu_rs.rs

// also the workgroup sizes are added in these prepasses
// before the pipeline is started the following constant definitionis are prepended to this shadercode

// const histogram_sg_size
// const histogram_wg_size
// const rs_radix_log2
// const rs_radix_size
// const rs_keyval_size
// const rs_histogram_block_rows
// const rs_scatter_block_rows

struct GeneralInfo{
    histogram_size: u32,
    keys_size: u32,
};

@group(0) @binding(0)
var<storage, read> keys : array<f32>;
@group(0) @binding(1)
var<storage, write> histograms : array<u32>;
@group(0) @binding(2)
var<uniform> infos: GeneralInfo;

@compute @workgroup_size({histogram_wg_size})
fn zero_histograms(@builtin(global_invocation_id) gid : vec3<u32>) {
    // here the histograms are set to zero and the partitions are set to 0xfffffffff to avoid sorting problems
    let scatter_wg_size = histogram_wg_size;
    let scatter_block_kvs = scatter_wg_size * rs_scatter_block_rows;
    let scatter_blocks_ru = (infos.keys_size + scatter_block_kvs - 1u) / scatter_block_kvs;
    
    let histo_size = rs_radix_size;
    let n = (rs_keyval_size + scatter_blocks_ru - 1u) * histo_size;
    
    if gid.x >= n {
        return;
    }
        
    if gid.x < rs_keyval_size * histo_size {
        histograms[gid.x] = 0u;
    }
    else {
        histograms[gid.x] = 0xFFFFFFFFu;
    }
}

// the workgrpu_size can be gotten on the cpu by by calling pipeline.get_bind_group_layout(0).unwrap().get_local_workgroup_size();
@compute @workgroup_size({histogram_wg_size})
fn calculate_histogram(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) wgs : vec3<u32>) {
    
}