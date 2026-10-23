use std::ops::RangeInclusive;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;
#[cfg(target_arch = "wasm32")]
use web_time::Duration;

use crate::renderer::DEFAULT_ALPHA_CUTOFF;
#[cfg(not(target_arch = "wasm32"))]
use crate::renderer::DEFAULT_KERNEL_SIZE;
use crate::{SceneCamera, Split, WindowContext};
use cgmath::{Euler, Matrix3, Quaternion};
#[cfg(not(target_arch = "wasm32"))]
use egui::Vec2b;

use egui::{Color32, RichText, emath::Numeric};

#[cfg(not(target_arch = "wasm32"))]
use egui_plot::{Legend, PlotPoints};

pub(crate) fn ui(state: &mut WindowContext) -> bool {
    let ctx = state.ui_renderer.winit.egui_ctx();
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(stopwatch) = state.stopwatch.as_mut() {
        let durations = pollster::block_on(
            stopwatch.take_measurements(&state.wgpu_context.device, &state.wgpu_context.queue),
        );
        state.history.push((
            *durations.get("preprocess").unwrap_or(&Duration::ZERO),
            *durations.get("sorting").unwrap_or(&Duration::ZERO),
            *durations.get("rasterization").unwrap_or(&Duration::ZERO),
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    let num_drawn = state
        .renderer
        .num_visible_points(&state.wgpu_context.device, &state.wgpu_context.queue);

    if !state.pc.is_complete() && state.load_error.is_none() {
        loading_bar(
            ctx,
            state.pc.loaded_points(),
            state.pc.num_points(),
            is_compact_screen(ctx),
        );
    }

    // the windows would cover most of a phone screen, they start collapsed there
    let default_open = !is_compact_screen(ctx);

    let mut new_camera: Option<SetCamera> = None;
    #[allow(unused_mut)]
    let mut toggle_tracking_shot = false;
    // One window in the top left corner, it starts collapsed on phones where it would cover
    // most of the screen. Its size is the one of its content (it can be resized), but it is not
    // higher than the screen:
    // the content scrolls if it is higher. (The scroll area of a window itself would always
    // take the maximum height.)
    const WINDOW_MARGIN: f32 = 10.;
    // title bar and frame of the window
    const WINDOW_DECORATION_HEIGHT: f32 = 40.;
    let max_content_height =
        (ctx.content_rect().height() - 2. * WINDOW_MARGIN - WINDOW_DECORATION_HEIGHT).max(100.);
    egui::Window::new("Viewer")
        .default_open(default_open)
        .anchor(
            egui::Align2::LEFT_TOP,
            egui::Vec2::new(WINDOW_MARGIN, WINDOW_MARGIN),
        )
        .movable(false)
        .resizable(true)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                // (smaller when the window was made smaller)
                .max_height(max_content_height.min(ui.available_height()))
                .auto_shrink([true, true])
                .show(ui, |ui| {
    #[cfg(not(target_arch = "wasm32"))]
    egui::CollapsingHeader::new("Render Stats")
        .default_open(false)
        .show(ui, |ui| {
            use egui::TextStyle;
            egui::Grid::new("timing").num_columns(2).show(ui, |ui| {
                ui.colored_label(egui::Color32::WHITE, "FPS");
                ui.label(format!("{:}", state.fps as u32));
                ui.end_row();
                ui.colored_label(egui::Color32::WHITE, "Visible points");
                ui.label(format!(
                    "{:} ({:.2}%)",
                    format_thousands(num_drawn),
                    (num_drawn as f32 / state.pc.loaded_points().max(1) as f32) * 100.
                ));
                ui.end_row();
                if !state.pc.is_complete() {
                    ui.colored_label(egui::Color32::WHITE, "Loading");
                    ui.label(format!(
                        "{:.0}%",
                        state.pc.loaded_points() as f32 / state.pc.num_points() as f32 * 100.
                    ));
                    ui.end_row();
                }
            });
            let history = state.history.to_vec();
            let pre: Vec<f32> = history.iter().map(|v| v.0.as_secs_f32() * 1000.).collect();
            let sort: Vec<f32> = history.iter().map(|v| v.1.as_secs_f32() * 1000.).collect();
            let rast: Vec<f32> = history.iter().map(|v| v.2.as_secs_f32() * 1000.).collect();

            // no timing data without GPU timestamp support
            if history.is_empty() {
                return;
            }
            ui.label("Frame times (ms):");
            egui_plot::Plot::new("frame times")
                .height(120.)
                .allow_drag(false)
                .allow_boxed_zoom(false)
                .allow_zoom(false)
                .allow_scroll(false)
                .y_axis_min_width(1.0)
                .y_axis_label("ms")
                .auto_bounds(Vec2b::TRUE)
                .show_axes([false, true])
                .legend(
                    Legend::default()
                        .text_style(TextStyle::Body)
                        .background_alpha(1.)
                        .position(egui_plot::Corner::LeftBottom),
                )
                .show(ui, |ui| {
                    let line = egui_plot::Line::new("preprocess", PlotPoints::from_ys_f32(&pre));
                    ui.line(line);
                    let line = egui_plot::Line::new("sorting", PlotPoints::from_ys_f32(&sort));
                    ui.line(line);
                    let line = egui_plot::Line::new("rasterize", PlotPoints::from_ys_f32(&rast));
                    ui.line(line);
                });
        });

    egui::CollapsingHeader::new("⚙ Render Settings")
        .default_open(true)
        .show(ui, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                    ui.strong("Backend:");
                    let info = state.wgpu_context.adapter.get_info();
                    ui.label(match info.backend {
                        wgpu::Backend::BrowserWebGpu => "WebGPU".to_string(),
                        #[cfg(target_arch = "wasm32")]
                        wgpu::Backend::Gl => "WebGL2".to_string(),
                        backend => format!("{backend:?}"),
                    })
                    .on_hover_text(info.name);
                    ui.end_row();
                    // reloads the page with the other backend (url parameter `backend`)
                    #[cfg(target_arch = "wasm32")]
                    {
                        use crate::web_backend::{WebBackend, switch_backend};
                        let (text, target) = if info.backend == wgpu::Backend::BrowserWebGpu {
                            ("Switch to WebGL", WebBackend::WebGl)
                        } else {
                            ("Switch to WebGPU", WebBackend::WebGpu)
                        };
                        ui.label("");
                        if ui
                            .button(text)
                            .on_hover_text("reloads the page with the other graphics backend")
                            .clicked()
                        {
                            switch_backend(target);
                        }
                        ui.end_row();
                    }
                    ui.strong("Sorting:");
                    if crate::renderer::supports_compute(&state.wgpu_context.device) {
                        ui.label("GPU");
                    } else {
                        ui.label("CPU (background thread)")
                            .on_hover_text("no compute shaders: the splats are sorted on the CPU");
                    }
                    ui.end_row();
                ui.label("Gaussian Scaling");
                ui.add(
                    egui::Slider::new(&mut state.splatting_args.gaussian_scaling, (1e-4)..=1.)
                        .logarithmic(true)
                        .clamping(egui::SliderClamping::Always)
                        .max_decimals(4),
                );
                ui.end_row();
                ui.label("Opacity Cutoff").on_hover_text(
                    "Splat contributions (opacity * gaussian falloff) below this alpha are not \
                     drawn. 1/255 (0.0039, the smallest step of an 8 bit color) is the minimum; higher \
                     values draw smaller splats (faster, but they fade out earlier).",
                );
                ui.add(
                    egui::Slider::new(&mut state.splatting_args.alpha_cutoff, DEFAULT_ALPHA_CUTOFF..=0.2)
                        .logarithmic(true)
                        .clamping(egui::SliderClamping::Always)
                        .custom_formatter(|v, _| format!("{v:.2e}"))
                        .custom_parser(|s| s.trim().parse::<f64>().ok()),
                );
                ui.end_row();
                ui.label("Directional Color");
                let mut dir_color = state.splatting_args.max_sh_deg > 0;
                ui.add_enabled(
                    state.pc.sh_deg() > 0,
                    egui::Checkbox::new(&mut dir_color, ""),
                );
                state.splatting_args.max_sh_deg = if dir_color { state.pc.sh_deg() } else { 0 };

                ui.end_row();
                ui.add(egui::Label::new("Background Color"));
                let mut color = egui::Color32::from_rgba_premultiplied(
                    (state.splatting_args.background_color.r * 255.) as u8,
                    (state.splatting_args.background_color.g * 255.) as u8,
                    (state.splatting_args.background_color.b * 255.) as u8,
                    (state.splatting_args.background_color.a * 255.) as u8,
                );
                egui::color_picker::color_edit_button_srgba(
                    ui,
                    &mut color,
                    egui::color_picker::Alpha::BlendOrAdditive,
                );

                let color32 = color.to_normalized_gamma_f32();
                state.splatting_args.background_color.r = color32[0] as f64;
                state.splatting_args.background_color.g = color32[1] as f64;
                state.splatting_args.background_color.b = color32[2] as f64;
                state.splatting_args.background_color.a = color32[3] as f64;

                ui.end_row();
                #[cfg(not(target_arch = "wasm32"))]
                {
                    ui.label("Dilation Kernel Size");
                    optional_drag(
                        ui,
                        &mut state.splatting_args.kernel_size,
                        Some(0.0..=10.0),
                        Some(0.1),
                        Some(
                            state
                                .pc
                                .dilation_kernel_size()
                                .unwrap_or(DEFAULT_KERNEL_SIZE),
                        ),
                    );
                    ui.end_row();
                    ui.label("Mip Splatting");
                    optional_checkbox(
                        ui,
                        &mut state.splatting_args.mip_splatting,
                        state.pc.mip_splatting().unwrap_or(false),
                    );
                    ui.end_row();
                }
            });
    });

    egui::CollapsingHeader::new("ℹ Scene")
        .default_open(true)
        .show(ui, |ui| {
            egui::Grid::new("scene info")
                .num_columns(2)
                .striped(false)
                .show(ui, |ui| {
                    ui.strong("Gaussians:");
                    if state.pc.is_complete() {
                        ui.label(format_thousands(state.pc.num_points()));
                    } else {
                        ui.label(format!(
                            "{} / {}",
                            format_thousands(state.pc.loaded_points()),
                            format_thousands(state.pc.num_points())
                        ));
                    }
                    ui.end_row();
                    if let Some(err) = &state.load_error {
                        ui.strong("Error:");
                        ui.colored_label(Color32::RED, err);
                        ui.end_row();
                    }
                    ui.strong("SH Degree:");
                    ui.label(state.pc.sh_deg().to_string());
                    ui.end_row();
                    ui.strong("Compressed:");
                    ui.label(state.pc.compressed().to_string());
                    ui.end_row();
                    ui.strong("Mip Splatting:");
                    ui.label(
                        state
                            .pc
                            .mip_splatting()
                            .map(|v| v.to_string())
                            .unwrap_or("-".to_string()),
                    );
                    ui.end_row();
                    ui.strong("Dilation Kernel Size:");
                    ui.label(
                        state
                            .pc
                            .dilation_kernel_size()
                            .map(|v| v.to_string())
                            .unwrap_or("-".to_string()),
                    );
                    ui.end_row();
                    if let Some(path) = &state.pointcloud_file_path {
                        ui.strong("File:");
                        let text = path.to_string_lossy().to_string();

                        ui.add(egui::Label::new(
                            path.file_name().unwrap().to_string_lossy().to_string(),
                        ))
                        .on_hover_text(text);
                        ui.end_row();
                    }
                    ui.end_row();
                });
        });

    if let Some(scene) = &state.scene {
        let nearest = scene.nearest_camera(state.splatting_args.camera.position, None);
        egui::CollapsingHeader::new("🖼 Dataset Images")
            .default_open(true)
            .show(ui, |ui| {
                    // everything uses the whole width of the window
                    let width = ui.available_width();
                    let spacing = ui.spacing().item_spacing.x;
                    egui::Grid::new("image info")
                        .num_columns(2)
                        .striped(true)
                        .min_col_width((width - spacing) / 2.)
                        .show(ui, |ui| {
                            ui.strong("Images");
                            ui.label(scene.num_cameras().to_string());
                            ui.end_row();

                            ui.strong("Current View");

                            if let Some(c) = &mut state.current_view {
                                ui.horizontal(|ui| {
                                    let drag = ui.add(
                                        egui::DragValue::new(c)
                                            .range(0..=(scene.num_cameras().saturating_sub(1)))
                                            .clamp_existing_to_range(true),
                                    );
                                    if drag.changed() {
                                        new_camera = Some(SetCamera::ID(*c));
                                    }
                                    ui.label(scene.camera(*c as usize).unwrap().split.to_string());
                                });
                            } else {
                                ui.label("-");
                            }
                            if let Some(path) = &state.scene_file_path {
                                ui.end_row();
                                ui.strong("File:");
                                let text = path.to_string_lossy().to_string();

                                ui.add(egui::Label::new(
                                    path.file_name().unwrap().to_string_lossy().to_string(),
                                ))
                                .on_hover_text(text);
                            }
                        });

                    let label = match &state.animation {
                        Some((_, true)) => "⏸ Pause tracking shot",
                        Some((_, false)) => "▶ Resume tracking shot",
                        None => "▶ Start tracking shot",
                    };
                    if ui
                        .add_sized([width, ui.spacing().interact_size.y], egui::Button::new(label))
                        .clicked()
                    {
                        toggle_tracking_shot = true;
                    }

                    egui::ScrollArea::vertical()
                        .max_height(300.)
                        .auto_shrink([false, true])
                        .show(ui, |ui| {
                            let cameras = scene.cameras(None);
                            let cameras2 = cameras.clone();
                            let curr_view = state.current_view;
                            egui::Grid::new("scene views grid")
                                .num_columns(4)
                                .striped(true)
                                .with_row_color(move |idx, _| {
                                    if let Some(view_id) = curr_view {
                                        if idx < cameras.len() && (&cameras)[idx].id == view_id {
                                            return Some(Color32::from_gray(64));
                                        }
                                    }
                                    return None;
                                })
                                .min_col_width((ui.available_width() - 3. * spacing) / 4.)
                                .show(ui, |ui| {
                                    let style = ui.style().clone();
                                    for c in cameras2 {
                                        ui.colored_label(
                                            style.visuals.strong_text_color(),
                                            c.id.to_string(),
                                        );
                                        ui.colored_label(
                                            match c.split {
                                                Split::Train => Color32::DARK_GREEN,
                                                Split::Test => Color32::LIGHT_GREEN,
                                            },
                                            c.split.to_string(),
                                        )
                                        .on_hover_text(
                                            RichText::new(format!(
                                                "{:#?}",
                                                Euler::from(Quaternion::from(Matrix3::from(
                                                    c.rotation
                                                )))
                                            )),
                                        );

                                        let resp =
                                            ui.add(egui::Label::new(c.img_name.clone()).truncate());
                                        if let Some(view_id) = curr_view {
                                            if c.id == view_id {
                                                resp.scroll_to_me(None);
                                            }
                                        }
                                        if ui.button("🎥").clicked() {
                                            new_camera = Some(SetCamera::ID(c.id));
                                        }
                                        ui.end_row();
                                    }
                                });
                        });
                    if let Some(nearest) = nearest {
                        ui.separator();
                        if ui
                            .add_sized(
                                [width, ui.spacing().interact_size.y],
                                egui::Button::new(format!("Snap to closest ({nearest})")),
                            )
                            .clicked()
                        {
                            new_camera = Some(SetCamera::ID(nearest));
                        }
                    }
            });
    }

                });
        });

    #[cfg(target_arch = "wasm32")]
    egui::Window::new("🎮")
        .default_width(200.)
        .resizable(false)
        .default_height(100.)
        .default_open(false)
        .movable(false)
        .anchor(egui::Align2::LEFT_BOTTOM, egui::Vec2::new(10., -10.))
        .show(ctx, |ui| {
            egui::Grid::new("controls")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    // (mouse and keyboard are not there on touch devices)
                    let touch = is_touch_device();
                    if !touch {
                        ui.strong("Camera Controls");
                        ui.end_row();

                        // Desktop controls
                        ui.label("Rotate Camera");
                        ui.label("Left click + drag / Touch + drag");
                        ui.end_row();

                        ui.label("Move Target/Center");
                        ui.label("Right click + drag / Two finger drag");
                        ui.end_row();

                        ui.label("Tilt Camera");
                        ui.label("Alt + drag mouse");
                        ui.end_row();

                        ui.label("Zoom");
                        ui.label("Mouse wheel / Pinch gesture");
                        ui.end_row();

                        ui.separator();
                        ui.end_row();
                    }

                    ui.strong("Mobile Touch Controls");
                    ui.end_row();
                    ui.label("Rotate");
                    ui.label("Single finger drag");
                    ui.end_row();
                    ui.label("Pan/Move");
                    ui.label("Two finger drag");
                    ui.end_row();
                    ui.label("Zoom");
                    ui.label("Pinch to zoom");
                    ui.end_row();

                    if !touch {
                        ui.separator();
                        ui.end_row();

                        ui.strong("Scene Navigation");
                        ui.end_row();
                        ui.label("Toggle UI");
                        ui.label("U");
                        ui.end_row();
                        ui.label("Views 0-9");
                        ui.label("0-9");
                        ui.end_row();
                        ui.label("Random view");
                        ui.label("R");
                        ui.end_row();
                        ui.label("Next View");
                        ui.label("Page Up");
                        ui.end_row();
                        ui.label("Previous View");
                        ui.label("Page Down");
                        ui.end_row();
                        ui.label("Snap to nearest view");
                        ui.label("N");
                        ui.end_row();
                        ui.label("Start/Pause Tracking shot");
                        ui.label("T");
                        ui.end_row();
                    }
                });
        });

    let requested_repaint = ctx.has_requested_repaint();

    if let Some(c) = new_camera {
        match c {
            SetCamera::ID(id) => state.set_scene_camera(id),
            SetCamera::Camera(c) => state.set_camera(c, Duration::from_millis(200)),
        }
    }
    if toggle_tracking_shot {
        if let Some((_animation, playing)) = &mut state.animation {
            *playing = !*playing;
        } else {
            state.start_tracking_shot();
        }
    }
    return requested_repaint;
}

enum SetCamera {
    ID(usize),
    #[allow(dead_code)]
    Camera(SceneCamera),
}

/// 212312321 -> 212.312.321
fn format_thousands(n: u32) -> String {
    let mut n = n;
    let mut result = String::new();
    while n > 0 {
        let rem = n % 1000;
        n /= 1000;
        if n > 0 {
            result = format!(".{:03}", rem) + &result;
        } else {
            result = rem.to_string() + &result;
        }
    }
    result
}

/// whether the main pointer of the device is a touch screen (a phone or tablet)
#[cfg(target_arch = "wasm32")]
fn is_touch_device() -> bool {
    static TOUCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TOUCH.get_or_init(|| {
        web_sys::window()
            .and_then(|w| w.match_media("(pointer: coarse)").ok().flatten())
            .is_some_and(|m| m.matches())
    })
}

/// progress bar at the bottom center while the point cloud is still loading
fn loading_bar(ctx: &egui::Context, loaded: u32, total: u32, compact: bool) {
    egui::Area::new(egui::Id::unique("loading bar"))
        // (phones have the home indicator and browser bars at the bottom)
        .anchor(
            egui::Align2::CENTER_BOTTOM,
            egui::Vec2::new(0., if compact { -64. } else { -20. }),
        )
        // does not take the pointer: the camera can be moved while loading
        .interactable(false)
        .show(ctx, |ui| {
            let progress = loaded as f32 / total.max(1) as f32;
            ui.add(
                egui::ProgressBar::new(progress)
                    .desired_width(120.)
                    .text(format!("{:.0}%", progress * 100.)),
            );
        });
}

/// whether the screen is small like the one of a phone (on the web, native windows are not
/// collapsed): the short side is less than 600 points
fn is_compact_screen(ctx: &egui::Context) -> bool {
    cfg!(target_arch = "wasm32") && {
        let size = ctx.content_rect().size();
        size.x.min(size.y) < 600.
    }
}

#[allow(unused)]
fn optional_drag<T: Numeric>(
    ui: &mut egui::Ui,
    opt: &mut Option<T>,
    range: Option<RangeInclusive<T>>,
    speed: Option<impl Into<f64>>,
    default: Option<T>,
) {
    let mut placeholder = default.unwrap_or(T::from_f64(0.));
    let mut drag = if let Some(val) = opt {
        egui_winit::egui::DragValue::new(val)
    } else {
        egui_winit::egui::DragValue::new(&mut placeholder).custom_formatter(|_, _| {
            if let Some(v) = default {
                format!("{:.2}", v.to_f64())
            } else {
                "—".into()
            }
        })
    };
    if let Some(range) = range {
        drag = drag.range(range).clamp_existing_to_range(true);
    }
    if let Some(speed) = speed {
        drag = drag.speed(speed);
    }
    let drag = ui.add(drag);
    let changed = drag.changed();
    if ui
        .add_enabled(opt.is_some(), egui::Button::new("↺"))
        .on_hover_text("Reset to default")
        .clicked()
    {
        *opt = None;
    }
    if changed && opt.is_none() {
        *opt = Some(placeholder);
    }
}

#[allow(unused)]
fn optional_checkbox(ui: &mut egui::Ui, opt: &mut Option<bool>, default: bool) {
    let mut val = default;
    let checkbox = if let Some(val) = opt {
        egui::Checkbox::new(val, "")
    } else {
        egui::Checkbox::new(&mut val, "")
    };
    let changed = ui.add(checkbox).changed();
    if ui
        .add_enabled(opt.is_some(), egui::Button::new("↺"))
        .on_hover_text("Reset to default")
        .clicked()
    {
        *opt = None;
    }
    if changed && opt.is_none() {
        *opt = Some(val);
    }
}
