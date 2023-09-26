use std::{mem::swap, time::Instant};

use num_traits::{int::PrimInt, Unsigned};
use rand::{self, seq::SliceRandom};

fn bucket_count<T: PrimInt + Unsigned, const B: usize, const C: usize>(
    data: &[T],
    offset: usize,
) -> [usize; C] {
    let mut count = [0; C];
    let mask = C - 1;
    for d in data {
        let cmp = (*d >> (offset * B)).to_usize().unwrap();
        let idx = cmp & mask;
        count[idx as usize] += 1;
    }
    return count;
}
fn prefix_sum<T: PrimInt + Unsigned, const B: usize>(counts: &[T; B]) -> [T; B] {
    let mut cumsum = [T::zero(); B];
    let mut acc = T::zero();
    for (i, c) in counts.iter().enumerate() {
        cumsum[i] = acc;
        acc = acc + *c;
    }
    return cumsum;
}

fn randix_sort<T: PrimInt + Unsigned>(data: &mut [T]) {
    // number of bits beeing sorted per sweep
    const B: usize = 8;
    // number of counting buckets (2**B)
    const B2: usize = (2usize).pow(B as u32);

    // number of sweeps needed for sorting
    let sweeps: usize = std::mem::size_of::<T>() * 8 / B;
    let mut copy: Vec<T> = data.to_vec();

    let mut source: &mut [T] = data;
    let mut target: &mut [T] = copy.as_mut();

    for s in 0..sweeps {
        let count = bucket_count::<T, B, B2>(&source, s);
        let mut offset = prefix_sum(&count);

        for j in source.iter() {
            let cmp = (*j >> (s * B)).to_usize().unwrap();
            let idx = cmp & (B2 - 1);
            target[offset[idx]] = *j;
            offset[idx] += 1;
        }
        // swap pointers for next sweep
        swap(&mut source, &mut target);
    }
}
#[allow(dead_code)]
fn is_sorted(data: &[usize]) -> bool {
    let mut last = 0;
    for d in data {
        if *d < last {
            return false;
        }
        last = *d;
    }
    return true;
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut nums: Vec<u32> = (0..5000000).map(|_| rand::random()).collect();
    nums.shuffle(&mut rng);

    println!("start");
    let start = Instant::now();
    randix_sort(&mut nums);
    println!("done {:?}", Instant::now() - start);
}
