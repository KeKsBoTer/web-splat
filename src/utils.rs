use std::mem::MaybeUninit;
use winit::event::VirtualKeyCode;

use std::{collections::HashMap, mem::size_of, ops::Deref};
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

    pub async fn take_measurements(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> HashMap<String, std::time::Duration> {
        let period = queue.get_timestamp_period();

        let labels: Vec<(String, u32)> = self.labels.drain().collect();

        let slice = self.query_buffer.slice(..);

        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap().unwrap();

        let mut durations = HashMap::new();
        {
            let view = slice.get_mapped_range();
            let data_raw: &[u8] = view.deref();
            let timestamps: &[u64] = bytemuck::cast_slice(data_raw);
            for (label, index) in labels {
                let diff_ticks =
                    timestamps[(index * 2 + 1) as usize] - timestamps[(index * 2) as usize];
                let diff_time =
                    std::time::Duration::from_nanos((diff_ticks as f32 * period) as u64);
                durations.insert(label, diff_time);
            }
        }
        self.query_buffer.unmap();
        return durations;
    }
}

pub struct RingBuffer<T: Copy, const N: usize> {
    index: usize,
    size: usize,
    container: [MaybeUninit<T>; N],
}

impl<T, const N: usize> RingBuffer<T, N>
where
    T: Copy,
{
    pub fn new() -> Self {
        Self {
            index: 0,
            size: 0,
            container: [MaybeUninit::uninit(); N],
        }
    }

    pub fn push(&mut self, item: T) {
        self.container[self.index] = MaybeUninit::new(item);
        self.index = (self.index + 1) % self.container.len();
        self.size = (self.size + 1).clamp(0, N)
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.container
            .iter()
            .cycle()
            .skip(self.index)
            .take(self.size)
            .map(|c| unsafe { c.assume_init() })
            .collect()
    }
}
