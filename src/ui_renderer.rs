use egui::{FullOutput, ViewportId};
// adapted from https://github.com/niklaskorz/linon/blob/main/src/egui_wgpu.rs
use winit::{dpi::PhysicalSize, event_loop::EventLoop};

pub struct EguiWGPU {
    pub ctx: egui::Context,
    pub winit: egui_winit::State,
    pub renderer: egui_wgpu::Renderer,
}

impl EguiWGPU {
    pub fn new(
        event_loop: &EventLoop<()>,
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            ctx: Default::default(),
            winit: egui_winit::State::new(ViewportId::ROOT, event_loop, None, None),
            renderer: egui_wgpu::Renderer::new(device, output_format, None, 1),
        }
    }

    /// Returns `true` if egui wants exclusive use of this event
    /// (e.g. a mouse click on an egui window, or entering text into a text field).
    /// For instance, if you use egui for a game, you want to first call this
    /// and only when this returns `false` pass on the events to your game.
    ///
    /// Note that egui uses `tab` to move focus between elements, so this will always return `true` for tabs.
    pub fn on_event(&mut self, event: &winit::event::WindowEvent<'_>) -> bool {
        self.winit.on_window_event(&self.ctx, event).consumed
    }

    pub fn begin_frame(&mut self, window: &winit::window::Window) {
        let raw_input = self.winit.take_egui_input(window);
        self.ctx.begin_frame(raw_input);
    }

    /// Returns `needs_repaint` and shapes to draw.
    pub fn end_frame(&mut self, window: &winit::window::Window) -> FullOutput {
        let output = self.ctx.end_frame();
        self.winit
            .handle_platform_output(window, &self.ctx, output.platform_output.clone());
        output
    }

    pub fn paint(
        &mut self,
        size: PhysicalSize<u32>,
        scale_factor: f32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_attachment: &wgpu::TextureView,
        output: FullOutput,
    ) {
        let clipped_meshes = self.ctx.tessellate(output.shapes, scale_factor);

        // let size = window.inner_size();l
        let screen_descriptor = egui_wgpu::renderer::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: self.ctx.pixels_per_point(),
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("egui_wgpu_encoder"),
        });

        for (id, delta) in output.textures_delta.set {
            self.renderer.update_texture(device, queue, id, &delta);
        }

        self.renderer.update_buffers(
            device,
            queue,
            &mut encoder,
            &clipped_meshes,
            &screen_descriptor,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_attachment,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                label: Some("egui_render"),
                ..Default::default()
            });
            self.renderer
                .render(&mut render_pass, &clipped_meshes, &screen_descriptor);
        }

        for id in output.textures_delta.free {
            self.renderer.free_texture(&id);
        }

        queue.submit(Some(encoder.finish()));
    }
}
