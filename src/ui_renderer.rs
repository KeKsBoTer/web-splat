use egui::{FullOutput, ViewportId};
// adapted from https://github.com/niklaskorz/linon/blob/main/src/egui_wgpu.rs
use winit::dpi::PhysicalSize;

pub struct EguiWGPU {
    pub winit: egui_winit::State,
    pub renderer: egui_wgpu::Renderer,
}

impl EguiWGPU {
    pub fn new(
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        window: &winit::window::Window,
    ) -> Self {
        let ctx = Default::default();
        Self {
            winit: egui_winit::State::new(ctx, ViewportId::ROOT, window, None, None,None),
            renderer: egui_wgpu::Renderer::new(device, output_format, None, 1,false),
        }
    }

    /// Returns `true` if egui wants exclusive use of this event
    /// (e.g. a mouse click on an egui window, or entering text into a text field).
    /// For instance, if you use egui for a game, you want to first call this
    /// and only when this returns `false` pass on the events to your game.
    ///
    /// Note that egui uses `tab` to move focus between elements, so this will always return `true` for tabs.
    pub fn on_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let resp = self.winit.on_window_event(window, event);
        return resp.consumed;
    }

    pub fn begin_frame(&mut self, window: &winit::window::Window) {
        let raw_input = self.winit.take_egui_input(window);
        self.winit.egui_ctx().begin_pass(raw_input);
    }

    /// Returns `needs_repaint` and shapes to draw.
    pub fn end_frame(&mut self, window: &winit::window::Window) -> FullOutput {
        let output = self.winit.egui_ctx().end_pass();
        self.winit
            .handle_platform_output(window, output.platform_output.clone());
        output
    }

    pub fn prepare(
        &mut self,
        size: PhysicalSize<u32>,
        scale_factor: f32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: FullOutput,
    ) -> UIRenderState {
        let clipped_meshes = self
            .winit
            .egui_ctx()
            .tessellate(output.shapes.clone(), scale_factor);

        // let size = window.inner_size();l
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: self.winit.egui_ctx().pixels_per_point(),
        };

        for (id, delta) in &output.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, &delta);
        }

        self.renderer
            .update_buffers(device, queue, encoder, &clipped_meshes, &screen_descriptor);

        UIRenderState {
            clipped_meshes,
            screen_descriptor,
            output,
        }
    }

    pub fn render(
        &mut self,
        render_pass: &mut wgpu::RenderPass<'static>,
        state: & UIRenderState,
    ) {
        self.renderer
            .render(render_pass, &state.clipped_meshes, &state.screen_descriptor);
    }

    pub fn cleanup(&mut self, state: UIRenderState) {
        for id in &state.output.textures_delta.free {
            self.renderer.free_texture(&id);
        }
    }
}

pub struct UIRenderState {
    clipped_meshes: Vec<egui::ClippedPrimitive>,
    screen_descriptor: egui_wgpu::ScreenDescriptor,
    output: FullOutput,
}
