//! Correctness and speed of the GPU radix sort (needs a GPU):
//! `cargo test --release --test gpu_sort -- --nocapture`

use web_splats::{
    WGPUContext,
    gpu_rs::{GPURSSorter, SORT_KEY_BITS},
};

/// xorshift64*, deterministic test data
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
    }
}

/// key distributions: (name, generator)
fn distributions() -> Vec<(&'static str, Box<dyn Fn(&mut Rng, usize) -> u32>)> {
    vec![
        ("random", Box::new(|r, _| r.next())),
        // 16 bit depth keys in a narrow range, like the renderers' keys
        (
            "depths",
            Box::new(|r, _| {
                let depth = 0.2 + 50. * (r.next() as f32 / u32::MAX as f32);
                (depth.to_bits() - 0.2f32.to_bits()) >> 11
            }),
        ),
        ("duplicates", Box::new(|r, _| r.next() % 7)),
        // few different keys in the low 16 bits, random upper bits
        (
            "duplicates, random upper bits",
            Box::new(|r, _| (r.next() % 7) | (r.next() & 0xFFFF_0000)),
        ),
        ("sorted", Box::new(|_, i| i as u32)),
        ("reversed", Box::new(|_, i| u32::MAX - i as u32)),
        ("max", Box::new(|_, _| u32::MAX)),
    ]
}

/// The sort keys are the low SORT_KEY_BITS (feature `sort-keys-32`: 32, else 16) bits of the
/// u32 keys, the upper bits are ignored (the tests use random upper bits to check that).
fn cpu_sort(keys: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mask = u32::MAX >> (32 - SORT_KEY_BITS);
    let mut idx: Vec<u32> = (0..keys.len() as u32).collect();
    idx.sort_by_key(|&i| keys[i as usize] & mask); // stable
    (idx.iter().map(|&i| keys[i as usize]).collect(), idx)
}

#[test]
fn gpu_sort_matches_cpu_stable_sort() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let ctx = WGPUContext::new(&instance, None).await;
        let (device, queue) = (&ctx.device, &ctx.queue);
        let info = ctx.adapter.get_info();
        println!("adapter: {} ({:?})", info.name, info.backend);
        let sorter = GPURSSorter::new(device);
        let mut rng = Rng(0x9E3779B97F4A7C15);

        // SORT_CHECK_SIZES=5000000,4999999: other numbers of keys
        let sizes: Vec<usize> = match std::env::var("SORT_CHECK_SIZES") {
            Ok(sizes) => sizes
                .split(',')
                .map(|n| n.trim().parse().unwrap())
                .collect(),
            Err(_) => vec![
                1, 2, 255, 256, 3839, 3840, 3841, 7680, 10_000, 160_000, 119_281, 50_000, 999_999,
                1_000_000, 3841, 160_000,
            ],
        };
        // buffers are reused across sorts of different sizes, as in the renderers
        let buffers = sorter.create_sort_buffers(device, *sizes.iter().max().unwrap());
        for (name, generate) in distributions() {
            for &n in &sizes {
                let keys: Vec<u32> = (0..n).map(|i| generate(&mut rng, i)).collect();
                let values: Vec<u32> = (0..n as u32).collect();
                let (gpu_keys, gpu_values) = sorter
                    .sort_key_values(&buffers, device, queue, &keys, &values)
                    .await;
                let (cpu_keys, cpu_values) = cpu_sort(&keys);
                assert!(gpu_keys == cpu_keys, "{name}, n = {n}: keys differ");
                assert!(
                    gpu_values == cpu_values,
                    "{name}, n = {n}: values differ (not stable)"
                );
            }
        }
    });
}

/// Heavy on the GPU, only runs on request:
/// `cargo test --release --test gpu_sort -- --ignored --nocapture gpu_sort_speed`
#[test]
#[ignore = "GPU benchmark, run with --ignored"]
fn gpu_sort_speed() {
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let ctx = WGPUContext::new(&instance, None).await;
        let (device, queue) = (&ctx.device, &ctx.queue);
        let info = ctx.adapter.get_info();
        println!("adapter: {} ({:?})", info.name, info.backend);
        let sorter = GPURSSorter::new(device);
        let mut rng = Rng(1);
        const ITERATIONS: u32 = 11;
        // SORT_SIZES=1000000,5000000: the numbers of keys to sort
        let sizes: Vec<usize> = match std::env::var("SORT_SIZES") {
            Ok(sizes) => sizes
                .split(',')
                .map(|n| n.trim().parse().unwrap())
                .collect(),
            Err(_) => vec![120_000, 160_000, 1_000_000],
        };
        for n in sizes {
            let buffers = sorter.create_sort_buffers(device, n);
            let keys: Vec<u32> = (0..n)
                .map(|_| (0.2 + 50. * (rng.next() as f32 / u32::MAX as f32)).to_bits())
                .collect();
            let values: Vec<u32> = (0..n as u32).collect();
            let mut times: Vec<f64> = sorter
                .bench_sort(&buffers, device, queue, &keys, &values, ITERATIONS)
                .await
                .iter()
                .map(|t| t.as_secs_f64() * 1e3)
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "sort {n:>9} keys: {:.3} ms (median of {ITERATIONS}, wall clock)",
                times[times.len() / 2]
            );
        }
    });
}
