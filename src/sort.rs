//! Sorting the splats on the CPU, for devices without compute shaders (WebGL2).
//!
//! The sort runs on another thread (a web worker on the web), see [`crate::cpu_sort`].

/// Keeps the positions of the points and sorts them by their depth.
#[derive(Default)]
pub struct DepthSorter {
    /// xyz of all points
    positions: Vec<f32>,
}

impl DepthSorter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends points, `xyz` has three values per point.
    pub fn add_positions(&mut self, xyz: &[f32]) {
        debug_assert_eq!(xyz.len() % 3, 0);
        self.positions.extend_from_slice(xyz);
    }

    pub fn len(&self) -> usize {
        self.positions.len() / 3
    }

    /// Returns the indices of the points in front of the camera, the nearest first. The depth of
    /// a point is `view[0] * x + view[1] * y + view[2] * z + view[3]`, which is the z coordinate
    /// in view space (the third row of the view matrix).
    pub fn sort(&self, view: [f32; 4]) -> Vec<u32> {
        let [a, b, c, d] = view;
        let mut keys = Vec::with_capacity(self.len());
        let mut indices = Vec::with_capacity(self.len());
        for (i, p) in self.positions.chunks_exact(3).enumerate() {
            let depth = a * p[0] + b * p[1] + c * p[2] + d;
            // (false for NaN)
            if depth > 0. {
                // the bits of positive floats are ordered like the floats
                keys.push(depth.to_bits());
                indices.push(i as u32);
            }
        }

        // least significant digit radix sort with two 16 bit digits
        let mut keys_tmp = vec![0u32; keys.len()];
        let mut indices_tmp = vec![0u32; keys.len()];
        let mut counts = vec![0u32; 1 << 16];
        for shift in [0, 16] {
            counts.fill(0);
            for &k in &keys {
                counts[((k >> shift) & 0xFFFF) as usize] += 1;
            }
            let mut offset = 0;
            for count in counts.iter_mut() {
                let c = *count;
                *count = offset;
                offset += c;
            }
            for (&k, &i) in keys.iter().zip(&indices) {
                let slot = &mut counts[((k >> shift) & 0xFFFF) as usize];
                keys_tmp[*slot as usize] = k;
                indices_tmp[*slot as usize] = i;
                *slot += 1;
            }
            std::mem::swap(&mut keys, &mut keys_tmp);
            std::mem::swap(&mut indices, &mut indices_tmp);
        }
        indices
    }
}
