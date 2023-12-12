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
@group(0) @binding(4)
var<storage, read_write> payload_a : array<u32>;
@group(0) @binding(5)
var<storage, read_write> payload_b : array<u32>;

// layout of the histograms buffer
//   +---------------------------------+ <-- 0
//   | histograms[keyval_size]         |
//   +---------------------------------+ <-- keyval_size                           * histo_size
//   | partitions[scatter_blocks_ru-1] |
//   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size
//   | workgroup_ids[keyval_size]      |
//   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size + workgroup_ids_size

// --------------------------------------------------------------------------------------------------------------
// Filling histograms and keys with default values (also resets the pass infos for odd and even scattering)
// --------------------------------------------------------------------------------------------------------------
@compute @workgroup_size({histogram_wg_size})
fn zero_histograms(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
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
    
    let line_size = nwg.x * {histogram_wg_size}u;
    for (var cur_index = gid.x; cur_index < n; cur_index += line_size){
        if cur_index >= n {
            return;
        }
            
        if cur_index  < rs_keyval_size * histo_size {
            histograms[cur_index] = 0u;
        }
        else if cur_index < b {
            histograms[cur_index] = 0u;//0xFFFFFFFFu;
        }
        else {
            keys[infos.keys_size + cur_index - b] = bitcast<f32>(0xFFFFFFFFu);
        }
    }
}

// --------------------------------------------------------------------------------------------------------------
// Calculating the histograms
// --------------------------------------------------------------------------------------------------------------
var<workgroup> smem : array<atomic<u32>, rs_radix_size>;
var<private> kv : array<f32, rs_histogram_block_rows>;
fn zero_smem(lid: u32) {
    for (var i = lid; i < rs_radix_size; i += histogram_wg_size) {
        smem[i] = 0u;
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
    for (var i = lid; i < rs_radix_size; i += histogram_wg_size) {
        if smem[i] >= 0u {
            atomicAdd(&histograms[histogram_offset], smem[i]);
        }
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
fn fill_kv_keys_b(wid: u32, lid: u32) {
    let rs_block_keyvals : u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + lid;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_wg_size;
        kv[i] = keys_b[pos];
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
    //const_assert(rs_radix_size <= 2 * histogram_wg_size);   // the radix size must not be larger than double the worgroup size, otherwise this function will not work as expected
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
    
    // the following coode now corresponds to the prefix calc code in fuchsia/../shaders/prefix.h
    // however the implementation is taken from https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf listing 2 (better overview, nw subgroup arithmetic)
    // this also means that only half the amount of workgroups is spawned (one workgroup calculates for 2 positioons)
    // the smemory is used from the previous section
    for (var i = lid.x; i < rs_radix_size; i += histogram_wg_size) {
        let histogram_offset = histogram_base + i;
        smem[i] = histograms[histogram_offset];
        smem[i + {prefix_wg_size}u] = histograms[histogram_offset + {prefix_wg_size}u];
    }

    prefix_reduce_smem(lid.x);
    workgroupBarrier();
    
    for (var i = lid.x; i < rs_radix_size; i += histogram_wg_size) {
        let histogram_offset = histogram_base + i;
        histograms[histogram_offset] = smem[i];
        histograms[histogram_offset + {prefix_wg_size}u] = smem[i + {prefix_wg_size}u];
    }
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
fn partitions_base_offset() -> u32 { return rs_keyval_size * rs_radix_size;}
fn smem_prefix_offset() -> u32 { return rs_radix_size + rs_radix_size;}
fn rs_prefix_sweep_0(idx: u32) -> u32 { return scatter_smem[smem_prefix_offset() + rs_mem_sweep_0_offset + idx];}
fn rs_prefix_sweep_1(idx: u32) -> u32 { return scatter_smem[smem_prefix_offset() + rs_mem_sweep_1_offset + idx];}
fn rs_prefix_sweep_2(idx: u32) -> u32 { return scatter_smem[smem_prefix_offset() + rs_mem_sweep_2_offset + idx];}
fn rs_prefix_load(lid: u32, idx: u32) -> u32 { return scatter_smem[rs_radix_size + lid + idx];}
fn rs_prefix_store(lid: u32, idx: u32, val: u32) { scatter_smem[rs_radix_size + lid + idx] = val;}
fn is_first_local_invocation(lid: u32) -> bool { return lid == 0u;}
fn histogram_load(digit: u32) -> u32 { return smem[digit];}// scatter_smem[rs_radix_size + digit];}
fn histogram_store(digit: u32, count: u32) { smem[digit] = count;} // scatter_smem[rs_radix_size + digit] = count; }
const rs_partition_mask_status : u32 = 0xC0000000u;
const rs_partition_mask_count : u32 = 0x3FFFFFFFu;
var<private> kr : array<u32, rs_scatter_block_rows>;
var<private> pv : array<u32, rs_scatter_block_rows>;

fn fill_kv_even(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals : u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_a[pos];
    }
}
fn fill_kv_odd(wid: u32, lid: u32) {
    let subgroup_id = lid / histogram_sg_size;
    let subgroup_invoc_id = lid - subgroup_id * histogram_sg_size;
    let subgroup_keyvals = rs_scatter_block_rows * histogram_sg_size;
    let rs_block_keyvals : u32 = rs_histogram_block_rows * histogram_wg_size;
    let kv_in_offset = wid * rs_block_keyvals + subgroup_id * subgroup_keyvals + subgroup_invoc_id;
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        kv[i] = keys_b[pos];
    }
    for (var i = 0u; i < rs_histogram_block_rows; i++) {
        let pos = kv_in_offset + i * histogram_sg_size;
        pv[i] = payload_b[pos];
    }
}
fn scatter(pass_: u32, lid: vec3<u32>, gid: vec3<u32>, wid: vec3<u32>, nwg: vec3<u32>, partition_status_invalid: u32, partition_status_reduction: u32, partition_status_prefix: u32) {
    let partition_mask_invalid = partition_status_invalid << 30u;
    let partition_mask_reduction = partition_status_reduction << 30u;
    let partition_mask_prefix = partition_status_prefix << 30u;
    // kv_filling is done in the scatter_even and scatter_odd functions to account for front and backbuffer switch
    // in the reference there is a nulling of the smmem here, was moved to line 251 as smem is not used in the code until then

    // The following implements conceptually the same as the
    // Emulate a "match" operation with broadcasts for small subgroup sizes (line 665 ff in scatter.glsl)
    // The difference however is, that instead of using subrgoupBroadcast each thread stores
    // its current number in the smem at lid.x, and then looks up their neighbouring values of the subgroup
    let subgroup_id = lid.x / histogram_sg_size;
    let subgroup_offset = subgroup_id * histogram_sg_size;
    let subgroup_tid = lid.x - subgroup_offset;
    let subgroup_count = {scatter_wg_size}u / histogram_sg_size;
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let u_val = bitcast<u32>(kv[i]);
        let digit = extractBits(u_val, pass_ * rs_radix_log2, rs_radix_log2);
        smem[lid.x] = digit;
        var count = 0u;
        var rank = 0u;
        
        for (var j = 0u; j < histogram_sg_size; j++) {
            if smem[subgroup_offset + j] == digit {
                count += 1u;
                if j <= subgroup_tid {
                    rank += 1u;
                }
            }
        }
        
        kr[i] = (count << 16u) | rank;
    }
    
    zero_smem(lid.x);   // now zeroing the smmem as we are now accumulating the histogram there
    workgroupBarrier();

    // The final histogram is stored in the smem buffer
    for (var i = 0u; i < subgroup_count; i++) {
        if subgroup_id == i {
            for (var j = 0u; j < rs_scatter_block_rows; j++) {
                let v = bitcast<u32>(kv[j]);
                let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
                let prev = histogram_load(digit);
                let rank = kr[j] & 0xFFFFu;
                let count = kr[j] >> 16u;
                kr[j] = prev + rank;

                if rank == count {
                    histogram_store(digit, (prev + count));
                }
                
                // TODO: check if the barrier here is needed
            }            
        }
        workgroupBarrier();
    }
    // kr filling is now done and contains the total offset for each value to be able to 
    // move the values into order without having any collisions
    
    // we do not check for single work groups (is currently not assumed to occur very often)
    let partition_base = wid.x * rs_radix_size;
    if wid.x == 0u {
        // special treating for the first workgroup as the data might be read back by later workgroups
        // corresponds to rs_first_prefix_store
        for (var i = lid.x; i < rs_radix_size; i += {scatter_wg_size}u) {
            let partition_offset = i + partitions_base_offset();    // is correct, the partitions pointer does not change
            let hist_offset = pass_ * rs_radix_size + i;
            let exc = histograms[hist_offset];
            let red = histogram_load(i);// scatter_smem[rs_keyval_size + lid.x];
            
            scatter_smem[i] = exc;
            
            let inc = exc + red;

            atomicStore(&histograms[partition_offset], inc | partition_mask_prefix);
        }
    }
    else {
        // standard case for the "inbetween" workgroups
        
        // rs_reduction_store, only for inbetween workgroups
        for (var i = lid.x; i < rs_radix_size; i += {scatter_wg_size}u) {
            if wid.x < nwg.x - 1u {
                let partition_offset = i + partitions_base_offset();    // is correct, the partitions pointer does not change
                let red = histogram_load(i);
                atomicStore(&histograms[partition_offset + partition_base], red | partition_mask_reduction);
            }
        }
        
        // rs_loopback_store
        for (var i = lid.x; i < rs_radix_size; i += {scatter_wg_size}u) {
            let partition_offset = i + partitions_base_offset();    // is correct, the partitions pointer does not change
            var partition_base_prev = partition_base - rs_radix_size;
            var exc                 = 0u;

            // Note: Each workgroup invocation can proceed independently.
            // Subgroups and workgroups do NOT have to coordinate.
            while true {
                //let prev = atomicLoad(&histograms[partition_offset]);// histograms[partition_offset + partition_base_prev];
                let prev = atomicLoad(&histograms[partition_base_prev + partition_offset]);// histograms[partition_offset + partition_base_prev];
                if (prev & rs_partition_mask_status) == partition_mask_invalid {
                    continue;
                }
                exc += prev & rs_partition_mask_count;
                if (prev & rs_partition_mask_status) != partition_mask_prefix {
                    // continue accumulating reduction
                    partition_base_prev -= rs_radix_size;
                    continue;
                }

                // otherwise save the exclusive scan and atomically transform the
                // reduction into an inclusive prefix status math: reduction + 1 = prefix
                scatter_smem[i] = exc;

                if wid.x < nwg.x - 1u { // only store when inbetween, skip for last workgrup
                    atomicAdd(&histograms[partition_offset + partition_base], exc | (1u << 30u));
                }
                break;
            }
        }
    }
    // speial case for last workgroup is also done in the "inbetween" case
    
    // compute exclusive prefix scan of histogram
    // corresponds to rs_prefix
    // TODO make shure that the data is put into smem
    prefix_reduce_smem(lid.x);
    workgroupBarrier();

    // convert keyval rank to local index, corresponds to rs_rank_to_local
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = bitcast<u32>(kv[i]);
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let exc   = histogram_load(digit);
        let idx   = exc + kr[i];
        
        kr[i] |= (idx << 16u);
    }
    workgroupBarrier();
    
    // reorder kv[] and kr[], corresponds to rs_reorder
    let smem_reorder_offset = rs_radix_size;
    let smem_base = smem_reorder_offset + lid.x;  // as we are in smem, the radix_size offset is not needed
  
        // keyvalues ----------------------------------------------
        // store keyval to sorted location
        for (var j = 0u; j < rs_scatter_block_rows; j++) {
            let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
            
            scatter_smem[smem_idx] = bitcast<u32>(kv[j]);
        }
        workgroupBarrier();

        // Load keyval dword from sorted location
        for (var j = 0u; j < rs_scatter_block_rows; j++) {
            kv[j] = bitcast<f32>(scatter_smem[smem_base + j * {scatter_wg_size}u]);
        }
        workgroupBarrier();
        // payload ----------------------------------------------
        // store payload to sorted location
        for (var j = 0u; j < rs_scatter_block_rows; j++) {
            let smem_idx = smem_reorder_offset + (kr[j] >> 16u) - 1u;
            
            scatter_smem[smem_idx] = pv[j];
        }
        workgroupBarrier();

        // Load payload dword from sorted location
        for (var j = 0u; j < rs_scatter_block_rows; j++) {
            pv[j] = scatter_smem[smem_base + j * {scatter_wg_size}u];
        }
        workgroupBarrier();
    //}
    
    // store the digit-index to sorted location
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let smem_idx = smem_reorder_offset + (kr[i] >> 16u) - 1u;
        scatter_smem[smem_idx] = kr[i];
    }
    workgroupBarrier();

    // Load kr[] from sorted location -- we only need the rank
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        kr[i] = scatter_smem[smem_base + i * {scatter_wg_size}u] & 0xFFFFu;
    }
    
    // convert local index to a global index, corresponds to rs_local_to_global
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        let v = bitcast<u32>(kv[i]);
        let digit = extractBits(v, pass_ * rs_radix_log2, rs_radix_log2);
        let exc   = scatter_smem[digit];

        kr[i] += exc - 1u;
    }
    
    // the storing is done in the scatter_even and scatter_odd functions as the front and back buffer changes
}
@compute @workgroup_size({scatter_wg_size})
fn scatter_even(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.odd_pass = (infos.odd_pass + 1u) % 2u; // for this to work correctly the odd_pass has to start 1
    }
    let cur_pass = infos.even_pass * 2u;
    
    // load from keys, store to keys_b
    fill_kv_even(wid.x, lid.x);
    
    let partition_status_invalid = 0u;
    let partition_status_reduction = 1u;
    let partition_status_prefix = 2u;
    scatter(cur_pass, lid, gid, wid, nwg, partition_status_invalid, partition_status_reduction, partition_status_prefix);

    // store keyvals to their new locations, corresponds to rs_store
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys_b[kr[i]] = kv[i];
    }
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        payload_b[kr[i]] = pv[i];
    }
}
@compute @workgroup_size({scatter_wg_size})
fn scatter_odd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    if gid.x == 0u {
        infos.even_pass = (infos.even_pass + 1u) % 2u; // for this to work correctly the even_pass has to start at 0
    }
    let cur_pass = infos.odd_pass * 2u + 1u;

    // load from keys_b, store to keys
    fill_kv_odd(wid.x, lid.x);

    let partition_status_invalid = 2u;
    let partition_status_reduction = 3u;
    let partition_status_prefix = 0u;
    scatter(cur_pass, lid, gid, wid, nwg, partition_status_invalid, partition_status_reduction, partition_status_prefix);

    // store keyvals to their new locations, corresponds to rs_store
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        keys[kr[i]] = kv[i];
    }
    for (var i = 0u; i < rs_scatter_block_rows; i++) {
        payload_a[kr[i]] = pv[i];
    }

    // the indirect buffer is reset after scattering via write buffer, see record_scatter_indirect for details
}
