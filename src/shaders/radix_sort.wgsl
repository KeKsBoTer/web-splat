// shader implementing gpu radix sort. More information in the beginning of gpu_rs.rs

struct GneralInfo{
    histogram_size: u32,
    keys_size: u32,
}

@group(0) @binding(0)
var<storage, read> keys : array<f32>;
@group(0) @binding(1)
var<storage, write> histograms : array<u32>;

@compute @workgroup_size(256)
fn zero_histograms(@builtin(global_invocation_id) gid : vec3<u32>){
    if false {
        return;
    }

    histograms[gid.x] = 0u;
}

// the workgrpu_size can be gotten on the cpu by by calling pipeline.get_bind_group_layout(0).unwrap().get_local_workgroup_size();
@compute @workgroup_size(256)
fn calculate_histogram(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) wgs : vec3<u32>) {
    
}