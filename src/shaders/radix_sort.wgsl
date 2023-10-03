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
    padded_size: u32,
    passes: u32,
};

@group(0) @binding(0)
var<uniform> infos: GeneralInfo;
@group(0) @binding(1)
var<storage, write> histograms : array<atomic<u32>>;
@group(0) @binding(2)
var<storage, read_write> keys : array<f32>;
@group(0) @binding(3)
var<storage, read_write> keys_b : array<f32>;

@compute @workgroup_size({histogram_wg_size})
fn zero_histograms(@builtin(global_invocation_id) gid : vec3<u32>) {
    // here the histograms are set to zero and the partitions are set to 0xfffffffff to avoid sorting problems
    let scatter_wg_size = histogram_wg_size;
    let scatter_block_kvs = scatter_wg_size * rs_scatter_block_rows;
    let scatter_blocks_ru = (infos.keys_size + scatter_block_kvs - 1u) / scatter_block_kvs;
    
    let histo_size = rs_radix_size;
    var n = (rs_keyval_size + scatter_blocks_ru - 1u) * histo_size;
    let b = n;
    if infos.keys_size < infos.padded_size {
        n += infos.padded_size - infos.keys_size;
    }
    
    if gid.x >= n {
        return;
    }
        
    if gid.x < rs_keyval_size * histo_size {
        histograms[gid.x] = 0u;
    }
    else if gid.x < b {
        histograms[gid.x] = 0xFFFFFFFFu;
    }
    else {
        keys[infos.keys_size + gid.x - b] = bitcast<f32>(0xFFFFFFFFu);
    }
}

var<workgroup> smem : array<atomic<u32>, rs_radix_size>;
var<private> kv : array<f32, rs_histogram_block_rows>;
fn zero_smem(lid: u32) {
    if lid < rs_radix_size {
        smem[lid] = 0u;
    }
}
fn histogram_pass(pass_: u32, lid: u32) {
    zero_smem(lid);
    workgroupBarrier();
    
    for (var j = 0u; j < rs_histogram_block_rows; j++) {
        let u_val = bitcast<u32>(kv[j]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        atomicAdd(&smem[digit], 1u);
    }
    
    workgroupBarrier();
    let histogram_offset = rs_radix_size * pass_ + lid;
    if lid < rs_radix_size && smem[lid] >= 0u {
        atomicAdd(&histograms[histogram_offset], smem[lid]);
    }
}

// the workgrpu_size can be gotten on the cpu by by calling pipeline.get_bind_group_layout(0).unwrap().get_local_workgroup_size();
@compute @workgroup_size({histogram_wg_size})
fn calculate_histogram(@builtin(workgroup_id) gid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
    // efficient loading of multiple values
    let rs_block_keyvals : u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = gid.x * rs_block_keyvals + lid.x;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys[pos];
    }
    
    // Accumulate and store histograms for passes
    histogram_pass(3u, lid.x);
    histogram_pass(2u, lid.x);
    if infos.passes > 2u {
        histogram_pass(1u, lid.x);
    }
    if infos.passes > 3u {
        histogram_pass(0u, lid.x);
    }
}