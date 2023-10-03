pub fn compare_slice_beginning<T: std::cmp::PartialEq>(a: &[T], b: &[T]) {
    let len = a.len().min(b.len());
    for i in 0..len {
        assert!(a[i] == b[i]);
    }
    println!("Comparison successful");
}

pub fn calculate_histogram(v: &[f32]) -> Vec<u32> {
    let total_length = 4 /*rs_keyval */ * 256 /*radix_size*/;
    let uv : &[u32] = bytemuck::cast_slice(v);
    let mut res = vec![0; total_length];
    for i in 0..4 {
        for val in uv {
            let bucket = (val >> (i * 8)) & 0xFFu32;
            let place = bucket + i * 256;
            res[place as usize] += 1;
        }
    }
    return res;
}