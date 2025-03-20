#[cfg(target_arch = "wasm32")]
use instant::Duration;
use std::ops::RangeInclusive;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use crate::renderer::DEFAULT_KERNEL_SIZE;
use crate::{ SceneCamera, Split, WebSplat};
use cgmath::{Euler, Matrix3, Quaternion};
#[cfg(not(target_arch = "wasm32"))]
use egui::Vec2b;

#[cfg(target_arch = "wasm32")]
use egui::{Align2, Vec2};

use egui::{emath::Numeric, Color32, RichText};

#[cfg(not(target_arch = "wasm32"))]
use egui_plot::{Legend, PlotPoints};

pub(crate) fn ui(state: &mut WebSplat) -> bool {
    let ctx = state.ui_renderer.winit.egui_ctx();
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(stopwatch) = state.stopwatch.as_mut() {
        let durations =
            pollster::block_on(stopwatch.take_measurements(&state.device, &state.queue));
        state.history.push((
            *durations.get("preprocess").unwrap_or(&Duration::ZERO),
            *durations.get("sorting").unwrap_or(&Duration::ZERO),
            *durations.get("rasterization").unwrap_or(&Duration::ZERO),
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    let num_drawn = pollster::block_on(
        state
            .renderer
            .num_visible_points(&state.device, &state.queue),
    );
    let old_upscale_factor = state.upscale_factor;
    let old_upscaling_method = state.splatting_args.upscaling_method;
    #[cfg(not(target_arch = "wasm32"))]
    egui::Window::new("Render Stats")
        .default_width(200.)
        .default_height(100.)
        .show(ctx, |ui| {
            use egui::TextStyle;
            egui::Grid::new("timing").num_columns(2).show(ui, |ui| {
                ui.colored_label(egui::Color32::WHITE, "FPS");
                ui.label(format!("{:}", state.fps as u32));
                ui.end_row();
                ui.colored_label(egui::Color32::WHITE, "Visible points");
                ui.label(format!(
                    "{:} ({:.2}%)",
                    format_thousands(num_drawn),
                    (num_drawn as f32 / state.pc.num_points() as f32) * 100.
                ));
            });
            let history = state.history.to_vec();
            let pre: Vec<f32> = history.iter().map(|v| v.0.as_secs_f32() * 1000.).collect();
            let sort: Vec<f32> = history.iter().map(|v| v.1.as_secs_f32() * 1000.).collect();
            let rast: Vec<f32> = history.iter().map(|v| v.2.as_secs_f32() * 1000.).collect();

            ui.label("Frame times (ms):");
            egui_plot::Plot::new("frame times")
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
                    let line =
                        egui_plot::Line::new(PlotPoints::from_ys_f32(&pre)).name("preprocess");
                    ui.line(line);
                    let line = egui_plot::Line::new(PlotPoints::from_ys_f32(&sort)).name("sorting");
                    ui.line(line);
                    let line =
                        egui_plot::Line::new(PlotPoints::from_ys_f32(&rast)).name("rasterize");
                    ui.line(line);
                });
        });

    egui::Window::new("⚙ Render Settings").show(ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Gaussian Scaling");
                ui.add(
                    egui::DragValue::new(&mut state.splatting_args.gaussian_scaling)
                        .range((1e-4)..=2.)
                        .clamp_existing_to_range(true)
                        .speed(1e-2),
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
                ui.label("Background Color");
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
                ui.heading("Upscaling");
                ui.end_row();

                ui.label("Method");

                egui::ComboBox::new("upscaling_method", "")
                    .selected_text(state.splatting_args.upscaling_method.to_string())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.splatting_args.upscaling_method,
                            crate::renderer::UpscalingMethod::Nearest,
                            crate::renderer::UpscalingMethod::Nearest.to_string(),
                        );
                        ui.selectable_value(
                            &mut state.splatting_args.upscaling_method,
                            crate::renderer::UpscalingMethod::Bilinear,
                            crate::renderer::UpscalingMethod::Bilinear.to_string(),
                        );
                        ui.selectable_value(
                            &mut state.splatting_args.upscaling_method,
                            crate::renderer::UpscalingMethod::Bicubic,
                            crate::renderer::UpscalingMethod::Bicubic.to_string(),
                        );
                        ui.selectable_value(
                            &mut state.splatting_args.upscaling_method,
                            crate::renderer::UpscalingMethod::Spline,
                            crate::renderer::UpscalingMethod::Spline.to_string(),
                        );
                    });
                ui.end_row();

                ui.label("Upscale Factor");
                ui.add(egui::Slider::new(&mut state.upscale_factor, 1.0..=8.0));
                ui.end_row();
                if state.splatting_args.upscaling_method == crate::renderer::UpscalingMethod::Spline
                {
                    ui.label("Channel");
                    egui::ComboBox::new("select_channel", "")
                        .selected_text(state.splatting_args.selected_channel.to_string())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut state.splatting_args.selected_channel,
                                crate::renderer::VisChannel::Color,
                                crate::renderer::VisChannel::Color.to_string(),
                            );
                            ui.selectable_value(
                                &mut state.splatting_args.selected_channel,
                                crate::renderer::VisChannel::GradX,
                                crate::renderer::VisChannel::GradX.to_string(),
                            );
                            ui.selectable_value(
                                &mut state.splatting_args.selected_channel,
                                crate::renderer::VisChannel::GradY,
                                crate::renderer::VisChannel::GradY.to_string(),
                            );
                            ui.selectable_value(
                                &mut state.splatting_args.selected_channel,
                                crate::renderer::VisChannel::GradXY,
                                crate::renderer::VisChannel::GradXY.to_string(),
                            );
                        });
                    ui.end_row();
                } else {
                    state.splatting_args.selected_channel = crate::renderer::VisChannel::Color;
                }
            });
    });

    let mut new_camera: Option<SetCamera> = None;
    #[allow(unused_mut)]
    let mut toggle_tracking_shot = false;
    egui::Window::new("ℹ Scene")
        .default_width(200.)
        .resizable(true)
        .default_height(100.)
        .show(ctx, |ui| {
            egui::Grid::new("scene info")
                .num_columns(2)
                .striped(false)
                .show(ui, |ui| {
                    ui.strong("Gaussians:");
                    ui.label(format_thousands(state.pc.num_points()));
                    ui.end_row();
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

            if let Some(scene) = &state.scene {
                let nearest = scene.nearest_camera(state.splatting_args.camera.position, None);
                ui.separator();
                ui.collapsing("Dataset Images", |ui| {
                    egui::Grid::new("image info")
                        .num_columns(2)
                        .striped(true)
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

                    egui::ScrollArea::vertical()
                        .max_height(300.)
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
                                .min_col_width(50.)
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
                        if ui.button(format!("Snap to closest ({nearest})")).clicked() {
                            new_camera = Some(SetCamera::ID(nearest));
                        }
                    }
                });
            }
        });

    #[cfg(target_arch = "wasm32")]
    egui::Window::new("🎮")
        .default_width(200.)
        .resizable(false)
        .default_height(100.)
        .default_open(false)
        .movable(false)
        .anchor(Align2::LEFT_BOTTOM, Vec2::new(10., -10.))
        .show(ctx, |ui| {
            egui::Grid::new("controls")
                .num_columns(2)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Camera");
                    ui.end_row();
                    ui.label("Rotate Camera");
                    ui.label("Left click + drag");
                    ui.end_row();

                    ui.label("Move Target/Center");
                    ui.label("Right click + drag");
                    ui.end_row();

                    ui.label("Tilt Camera");
                    ui.label("Alt + drag mouse");
                    ui.end_row();

                    ui.label("Zoom");
                    ui.label("Mouse wheel");
                    ui.end_row();

                    ui.label("Toggle UI");
                    ui.label("U");
                    ui.end_row();

                    ui.strong("Scene Views");
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

    if old_upscale_factor != state.upscale_factor {
        state.resize_framebuffer(state.config.width, state.config.height);
        if state.splatting_args.upscaling_method == crate::renderer::UpscalingMethod::Spline {
            state.renderer.recreate_pipeline(
                &state.device,
                state.upscale_factor > 1. 
            );
        }
    }

    if old_upscaling_method != state.splatting_args.upscaling_method  {
        state.renderer.recreate_pipeline(
            &state.device,
            state.splatting_args.upscaling_method == crate::renderer::UpscalingMethod::Spline,
        );
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

#[allow(unused)]
fn optional_drag<T: Numeric>(
    ui: &mut egui::Ui,
    opt: &mut Option<T>,
    range: Option<RangeInclusive<T>>,
    speed: Option<impl Into<f64>>,
    default: Option<T>,
) {
    let mut placeholder = default.unwrap_or(T::from_f64(0.));
    let mut drag = if let Some(ref mut val) = opt {
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
    let changed = ui.add(drag).changed();
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
    let checkbox = if let Some(ref mut val) = opt {
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
