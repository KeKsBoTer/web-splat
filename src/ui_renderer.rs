use egui::{FullOutput, TexturesDelta, ViewportId};
// adapted from https://github.com/niklaskorz/linon/blob/main/src/egui_wgpu.rs
use winit::dpi::PhysicalSize;

pub struct EguiWGPU {
    pub winit: egui_winit::State,
    pub renderer: egui_wgpu::Renderer,
    pending_textures: TexturesDelta,
    /// latest cursor position (points), see [`Self::on_event`]
    cursor_pos: Option<egui::Pos2>,
}

impl EguiWGPU {
    pub fn new(
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        window: &winit::window::Window,
    ) -> Self {
        let ctx = Default::default();
        Self {
            winit: egui_winit::State::new(ctx, ViewportId::ROOT, window, None, None, None),
            renderer: egui_wgpu::Renderer::new(
                device,
                output_format,
                egui_wgpu::RendererOptions::default(),
            ),
            pending_textures: TexturesDelta::default(),
            cursor_pos: None,
        }
    }

    /// Returns `true` if egui wants exclusive use of this event
    /// (e.g. a mouse click on an egui window, or entering text into a text field).
    /// For instance, if you use egui for a game, you want to first call this
    /// and only when this returns `false` pass on the events to your game.
    ///
    /// Note that egui uses `tab` to move focus between elements, so this will always return `true` for tabs.
    ///
    /// Mouse buttons are decided here: egui judges them by the pointer position of
    /// its last pass, which is a whole frame old in a slow frame, so a press just
    /// after leaving a UI window was taken by egui and the camera drag did not start.
    /// A press belongs to egui if it is using the pointer (e.g. dragging a slider)
    /// or the cursor is over one of its windows now; releases always reach the
    /// application, so that a drag cannot get stuck.
    pub fn on_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        use winit::event::{ElementState, WindowEvent};
        let resp = self.winit.on_window_event(window, event);
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let pixels_per_point =
                    egui_winit::pixels_per_point(self.winit.egui_ctx(), window);
                self.cursor_pos = Some(egui::pos2(
                    position.x as f32 / pixels_per_point,
                    position.y as f32 / pixels_per_point,
                ));
                resp.consumed
            }
            WindowEvent::CursorLeft { .. } => {
                self.cursor_pos = None;
                resp.consumed
            }
            WindowEvent::MouseInput { state, .. } => {
                *state == ElementState::Pressed && self.pointer_over_ui()
            }
            _ => resp.consumed,
        }
    }

    /// whether egui uses the pointer or the cursor is over an egui window (layers
    /// in front of the background, which shows the scene)
    fn pointer_over_ui(&self) -> bool {
        let ctx = self.winit.egui_ctx();
        ctx.egui_is_using_pointer()
            || self.cursor_pos.is_some_and(|pos| {
                ctx.layer_id_at(pos)
                    .is_some_and(|layer| layer.order != egui::Order::Background)
            })
    }

    pub fn skip(&mut self, output: FullOutput) {
        self.pending_textures.append(output.textures_delta);
    }

    pub fn prepare(
        &mut self,
        size: PhysicalSize<u32>,
        scale_factor: f32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        mut output: FullOutput,
    ) -> UIRenderState {
        let mut textures = std::mem::take(&mut self.pending_textures);
        textures.append(std::mem::take(&mut output.textures_delta));
        output.textures_delta = textures;

        let clipped_meshes = self
            .winit
            .egui_ctx()
            .tessellate(output.shapes.clone(), scale_factor);

        // let size = window.inner_size();l
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: self.winit.egui_ctx().pixels_per_point(),
        };

        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                self.renderer.update_texture(device, queue, *id, delta);
            }
        }

        self.renderer
            .update_buffers(device, queue, encoder, &clipped_meshes, &screen_descriptor);

        UIRenderState {
            clipped_meshes,
            screen_descriptor,
            output,
        }
    }

    pub fn render(&mut self, render_pass: &mut wgpu::RenderPass<'static>, state: &UIRenderState) {
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
