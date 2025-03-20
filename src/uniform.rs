use std::{mem, num::NonZeroU64};

use bytemuck::{NoUninit, Pod, Zeroable};
use wgpu::{util::DeviceExt, Device};

#[derive(Debug)]
pub struct UniformBuffer<T: NoUninit + Pod> {
    buffer: wgpu::Buffer,
    data: T,
    label: Option<String>,
    bind_group: wgpu::BindGroup,
}

impl<T> UniformBuffer<T>
where
    T: NoUninit + Pod + Default,
{
    pub fn new_default(device: &wgpu::Device, label: Option<&str>) -> Self {
        let data = T::default();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label,
            contents: bytemuck::cast_slice(&[data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_label = label.map(|l| format!("{l} bind group"));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: bg_label.as_ref().map(|s| s.as_str()),
            layout: &Self::bind_group_layout(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer,
            data,
            label: label.map(|a| a.to_string()),
            bind_group,
        }
    }
}


impl<T> UniformBuffer<T>
where
    T: NoUninit + Zeroable + Pod,
{
    pub fn new_zeored(device: &wgpu::Device, label: Option<&str>) -> Self {
        let data = T::zeroed();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label,
            contents: bytemuck::cast_slice(&[data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_label = label.map(|l| format!("{l} bind group"));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: bg_label.as_ref().map(|s| s.as_str()),
            layout: &Self::bind_group_layout(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer,
            data,
            label: label.map(|a| a.to_string()),
            bind_group,
        }
    }
}

impl<T> UniformBuffer<T>
where
    T: NoUninit + Pod,
{
    #[allow(dead_code)]
    pub fn new(device: &wgpu::Device, data: T, label: Option<&str>) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label,
            contents: bytemuck::cast_slice(&[data]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_label = label.map(|l| format!("{l} bind group"));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: bg_label.as_ref().map(|s| s.as_str()),
            layout: &Self::bind_group_layout(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer,
            data,
            label: label.map(|a| a.to_string()),
            bind_group,
        }
    }

    #[allow(dead_code)]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    #[allow(dead_code)]
    pub fn data(&self) -> &T {
        &self.data
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::all(),
                ty: Self::binding_type(),
                count: None,
            }],
        })
    }

    /// uploads data from cpu to gpu if necesarry
    pub fn sync(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.data]));
    }

    pub fn binding_type() -> wgpu::BindingType {
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(NonZeroU64::new(mem::size_of::<T>() as u64).unwrap()),
        }
    }

    #[allow(dead_code)]
    pub fn clone(&self, device: &Device, queue: &wgpu::Queue) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: self.label.as_deref(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: self.buffer.size(),
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy uniform buffer encode"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, buffer.size());

        queue.submit([encoder.finish()]);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &Self::bind_group_layout(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        return Self {
            buffer,
            data: self.data.clone(),
            label: self.label.clone(),
            bind_group: bind_group,
        };
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

impl<T: ?Sized + Pod> AsMut<T> for UniformBuffer<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.data
    }
}
