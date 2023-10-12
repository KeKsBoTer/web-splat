use cgmath::{Matrix, Matrix3, Quaternion, SquareMatrix, Vector3};
use half::f16;
use instant::Duration;
use std::{fmt::Debug, mem::MaybeUninit};
use winit::event::VirtualKeyCode;

use std::{collections::HashMap, mem::size_of, ops::Deref};

use crate::SHDType;
pub fn key_to_num(key: VirtualKeyCode) -> Option<u32> {
    match key {
        VirtualKeyCode::Key0 => Some(0),
        VirtualKeyCode::Key1 => Some(1),
        VirtualKeyCode::Key2 => Some(2),
        VirtualKeyCode::Key3 => Some(3),
        VirtualKeyCode::Key4 => Some(4),
        VirtualKeyCode::Key5 => Some(5),
        VirtualKeyCode::Key6 => Some(6),
        VirtualKeyCode::Key7 => Some(7),
        VirtualKeyCode::Key8 => Some(8),
        VirtualKeyCode::Key9 => Some(9),
        _ => None,
    }
}

pub struct GPUStopwatch {
    query_set: wgpu::QuerySet,
    query_buffer: wgpu::Buffer,
    query_set_capacity: u32,
    index: u32,
    labels: HashMap<String, u32>,
}

impl GPUStopwatch {
    pub fn new(device: &wgpu::Device, capacity: Option<u32>) -> Self {
        let capacity = capacity.unwrap_or(wgpu::QUERY_SET_MAX_QUERIES / 2);
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("time stamp query set"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity * 2,
        });

        let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("query set buffer"),
            size: capacity as u64 * 2 * size_of::<u64>() as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let labels = HashMap::with_capacity(capacity as usize);

        Self {
            query_set,
            query_buffer,
            query_set_capacity: capacity * 2,
            index: 0,
            labels,
        }
    }

    pub fn start(&mut self, encoder: &mut wgpu::CommandEncoder, label: &str) -> Result<(), String> {
        if self.labels.contains_key(label) {
            return Err("cannot start measurement for same label twice".to_string());
        }
        if self.labels.len() * 2 >= self.query_set_capacity as usize {
            return Err(format!(
                "query set capacity ({:})reached",
                self.query_set_capacity
            ));
        }
        self.labels.insert(label.to_string(), self.index);
        encoder.write_timestamp(&self.query_set, self.index * 2);
        self.index += 1;
        Ok(())
    }

    pub fn stop(&mut self, encoder: &mut wgpu::CommandEncoder, label: &str) -> Result<(), String> {
        match self.labels.get(label) {
            Some(idx) => {
                encoder.write_timestamp(&self.query_set, *idx * 2 + 1);
                Ok(())
            }
            None => Err(format!("start was not yet called for label {label}")),
        }
    }

    pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.query_set,
            0..self.query_set_capacity,
            &self.query_buffer,
            0,
        );
        self.index = 0;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn take_measurements(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> HashMap<String, Duration> {
        let period = queue.get_timestamp_period();

        let labels: Vec<(String, u32)> = self.labels.drain().collect();

        let mut durations = HashMap::new();
        {
            let view = download_buffer(device, &self.query_buffer, None).await;
            let data_raw: &[u8] = view.deref();
            let timestamps: &[u64] = bytemuck::cast_slice(data_raw);
            for (label, index) in labels {
                let diff_ticks =
                    timestamps[(index * 2 + 1) as usize] - timestamps[(index * 2) as usize];
                let diff_time = Duration::from_nanos((diff_ticks as f32 * period) as u64);
                durations.insert(label, diff_time);
            }
        }
        self.query_buffer.unmap();
        return durations;
    }
}

#[derive(Debug)]
pub struct RingBuffer<T: Copy> {
    index: usize,
    size: usize,
    container: Box<[MaybeUninit<T>]>,
}

impl<T> RingBuffer<T>
where
    T: Copy + Debug,
{
    pub fn new(size: usize) -> Self {
        Self {
            index: 0,
            size: 0,
            container: vec![MaybeUninit::uninit(); size].into_boxed_slice(),
        }
    }

    pub fn push(&mut self, item: T) {
        self.container[self.index] = MaybeUninit::new(item);
        self.index = (self.index + 1) % self.container.len();
        self.size = (self.size + 1).clamp(0, self.container.len());
    }

    pub fn to_vec(&self) -> Vec<T> {
        let start = self
            .index
            .checked_sub(self.size)
            .unwrap_or(self.container.len() - (self.size - self.index));
        self.container
            .iter()
            .cycle()
            .skip(start)
            .take(self.size)
            .map(|c| unsafe { c.assume_init() })
            .collect()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn download_buffer<'a>(
    device: &wgpu::Device,
    buffer: &'a wgpu::Buffer,
    wait_idx: Option<wgpu::SubmissionIndex>,
) -> wgpu::BufferView<'a> {
    let slice = buffer.slice(..);

    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(match wait_idx {
        Some(idx) => wgpu::Maintain::WaitForSubmissionIndex(idx),
        None => wgpu::Maintain::Wait,
    });
    rx.receive().await.unwrap().unwrap();

    let view = slice.get_mapped_range();
    return view;
}

pub fn sh_num_coefficients(sh_deg: u32) -> u32 {
    (sh_deg + 1) * (sh_deg + 1)
}

pub fn sh_deg_from_num_coefs(n: u32) -> Option<u32> {
    let sqrt = (n as f32).sqrt();
    if sqrt.fract() != 0. {
        return None;
    }
    return Some((sqrt as u32) - 1);
}

/// calculates the maximum sh degree that will fit into
/// the max_buffer_size
pub fn max_supported_sh_deg(
    max_buffer_size: u64,
    num_points: u64,
    sh_dtype: SHDType,
    max_deg: u32,
) -> Option<u32> {
    for i in (0..=max_deg).rev() {
        let n_coefs = sh_num_coefficients(i) * 3;
        let buf_size = num_points as u64 * sh_dtype.packed_size() as u64 * n_coefs as u64;
        if buf_size < max_buffer_size {
            return Some(i);
        }
    }
    return None;
}

/// builds a covariance matrix based on a quaterion and rotation
/// the matrix is symmetric so we only return the upper right half
/// see "3D Gaussian Splatting" Kerbel et al.
pub fn build_cov(rot: Quaternion<f32>, scale: Vector3<f32>) -> [f16; 6] {
    let r = Matrix3::from(rot);
    let s = Matrix3::from_diagonal(scale);

    let l = r * s;

    let m = l * l.transpose();

    return [m[0][0], m[0][1], m[0][2], m[1][1], m[1][2], m[2][2]].map(|v| f16::from_f32(v));
}

/// numerical stable sigmoid function
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        x.exp() / (1. + x.exp())
    }
}
