// shader implementing gpu radix sort. More information in the beginning of gpu_rs.rs
// info: 

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
    even_pass: u32,
    odd_pass: u32,
};

@group(0) @binding(0)
var<storage, read_write> infos: GeneralInfo;
@group(0) @binding(1)
var<storage, read_write> histograms : array<atomic<u32>>;
@group(0) @binding(2)
var<storage, read_write> keys : array<f32>;
@group(0) @binding(3)
var<storage, read_write> keys_b : array<f32>;

// --------------------------------------------------------------------------------------------------------------
// Filling histograms and keys with default values (also resets the pass infos for odd and even scattering)
// --------------------------------------------------------------------------------------------------------------
@compute @workgroup_size({histogram_wg_size})
fn zero_histograms(@builtin(global_invocation_id) gid : vec3<u32>) {
    if gid.x == 0u {
        infos.even_pass = 0u;
        infos.odd_pass = 1u;    // has to be one, as on the first call to even pass + 1 % 2 is calculated
    }
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

// --------------------------------------------------------------------------------------------------------------
// Calculating the histograms
// --------------------------------------------------------------------------------------------------------------
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
fn fill_kv(wid: u32, lid: u32) {
    let rs_block_keyvals : u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + lid;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys[pos];
    }
}
@compute @workgroup_size({histogram_wg_size})
fn calculate_histogram(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
    // efficient loading of multiple values
    fill_kv(wid.x, lid.x);
    
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

// --------------------------------------------------------------------------------------------------------------
// Prefix sum over histogram
// --------------------------------------------------------------------------------------------------------------
fn prefix_reduce_smem(lid: u32) {
    if lid >= rs_radix_size / 2u {
        return;
    }
    var offset = 1u;
    for (var d = rs_radix_size >> 1u; d > 0u; d = d >> 1u) { // sum in place tree
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            smem[bi] += smem[ai];
        }
        offset = offset << 1u;
    }
    
    if lid == 0u { smem[rs_radix_size - 1u] = 0u; } // clear the last element
        
    for (var d = 1u; d < rs_radix_size; d = d << 1u) {
        offset = offset >> 1u;
        workgroupBarrier();
        if lid < d {
            let ai = offset * (2u * lid + 1u) - 1u;
            let bi = offset * (2u * lid + 2u) - 1u;
            
            let t     = smem[ai];
            smem[ai]  = smem[bi];
            smem[bi] += t;
        }
    }
}
@compute @workgroup_size({prefix_wg_size})
fn prefix_histogram(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid : vec3<u32>) {
    // the work group  id is the pass, and is inverted in the next line, such that pass 3 is at the first position in the histogram buffer
    let histogram_base = (rs_keyval_size - 1u - wid.x) * rs_radix_size;
    let histogram_offset = histogram_base + lid.x;
    
    // the following coode now corresponds to the prefix calc code in fuchsia/../shaders/prefix.h
    // however the implementation is taken from https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf listing 2 (better overview, nw subgroup arithmetic)
    // this also means that only half the amount of workgroups is spawned (one workgroup calculates for 2 positioons)
    // the smemory is used from the previous section
    smem[lid.x] = histograms[histogram_offset];
    smem[lid.x + {prefix_wg_size}u] = histograms[histogram_offset + {prefix_wg_size}u];

    prefix_reduce_smem(lid.x);
    workgroupBarrier();
    
    histograms[histogram_offset] = smem[lid.x];
    histograms[histogram_offset + {prefix_wg_size}u] = smem[lid.x + {prefix_wg_size}u];
}

// --------------------------------------------------------------------------------------------------------------
// Scattering the keys
// --------------------------------------------------------------------------------------------------------------
// General note: Only 2 sweeps needed here
var<workgroup> scatter_smem: array<u32, rs_mem_dwords>; // note: rs_mem_dwords is caclulated in the beginngin of gpu_rs.rs
//            | Dwords                                    | Bytes
//  ----------+-------------------------------------------+--------
//  Lookback  | 256                                       | 1 KB
//  Histogram | 256                                       | 1 KB
//  Prefix    | 4-84                                      | 16-336
//  Reorder   | RS_WORKGROUP_SIZE * RS_SCATTER_BLOCK_ROWS | 2-8 KB
fn partitions_offset() -> u32 { return rs_keyval_size * rs_radix_size;}
fn smem_prefix_offset() -> u32 { return rs_radix_size + rs_radix_size;}
fn rs_prefix_sweep_0(idx: u32) -> u32 { return scatter_smem[smem_prefix_offset() + rs_mem_sweep_0_offset + idx];}
fn rs_prefix_sweep_1(idx: u32) -> u32 { return scatter_smem[smem_prefix_offset() + rs_mem_sweep_1_offset + idx];}
fn rs_prefix_sweep_2(idx: u32) -> u32 { return scatter_smem[smem_prefix_offset() + rs_mem_sweep_2_offset + idx];}
fn rs_prefix_load(lid: u32, idx: u32) -> u32 { return scatter_smem[rs_radix_size + lid + idx];}
fn rs_prefix_store(lid: u32, idx: u32, val: u32) { scatter_smem[rs_radix_size + lid + idx] = val;}
fn is_first_local_invocation(lid: u32) -> bool { return lid == 0u;}

@compute @workgroup_size({scatter_wg_size})
fn scatter_even(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    infos.odd_pass = (infos.odd_pass + 1u) % 2u; // for this to work correctly the odd_pass has to start 1
    fill_kv(wid.x, lid.x);
}
@compute @workgroup_size({scatter_wg_size})
fn scatter_odd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) { 
    infos.even_pass = (infos.even_pass + 1u) % 2u; // for this to work correctly the even_pass has to start at 0
    fill_kv(wid.x, lid.x);

}
