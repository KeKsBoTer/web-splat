pub fn compare_slice_beginning<T: std::cmp::PartialEq>(a: &[T], b: &[T]) {
    let len = a.len().min(b.len());
    for i in 0..len {
        if a[i] != b[i] {
            println!("Comparison failed");
        }
    }
    println!("Comparison successful");
}

pub fn calculate_histogram(v: &[f32], gpu_buffer_size: usize) -> Vec<u32> {
    let total_length = 4 /*rs_keyval */ * 256 /*radix_size*/;
    let gpu_elements = gpu_buffer_size / std::mem::size_of::<f32>();
    let uv : &[u32] = bytemuck::cast_slice(v);
    let mut res = vec![0; total_length];
    for i in 0..4 {
        for val in uv {
            let bucket = (val >> (i * 8)) & 0xFFu32;
            let place = bucket + i * 256;
            res[place as usize] += 1u32;
        }
        res[(i * 256 + 255) as usize] += (gpu_elements - v.len()) as u32;
    }
    return res;
}

pub fn prefix_sum_histogram(v: &[u32]) -> Vec<u32> {
    let mut ret = vec![0; v.len()];
    let mut cur = 0u32;
    for i in 0..v.len() {
        ret[i] = cur;
        cur += v[i];
        if i % 256 == 255 {
            cur = 0u32;
        }
    }
    return ret;
}