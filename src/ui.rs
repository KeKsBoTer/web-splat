use std::{
    fs::{create_dir_all, File},
    path::Path,
    time::Duration,
};

use egui::{epaint::Shadow, Align2, Color32, Margin, Vec2, Vec2b, Visuals};
use egui_dnd::dnd;
use egui_plot::{Legend, PlotPoints};

use crate::{SceneCamera, Split, WindowContext};

pub(crate) fn ui(state: &mut WindowContext) {
    let ctx = state.ui_renderer.winit.egui_ctx();
    #[cfg(not(target_arch = "wasm32"))]
    let stats = pollster::block_on(
        state
            .renderer
            .render_stats(&state.wgpu_context.device, &state.wgpu_context.queue),
    );
    #[cfg(not(target_arch = "wasm32"))]
    let num_drawn = pollster::block_on(
        state
            .renderer
            .num_visible_points(&state.wgpu_context.device, &state.wgpu_context.queue),
    );

    #[cfg(not(target_arch = "wasm32"))]
    state.history.push((
        stats.preprocess_time,
        stats.sort_time,
        stats.rasterization_time,
    ));
    ctx.set_visuals(Visuals {
        window_shadow: Shadow::small_light(),
        ..Default::default()
    });

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
                    num_drawn,
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
                .y_axis_width(1)
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

    egui::Window::new("⚙ Render Settings").show(ctx, |ui| {
        egui::Grid::new("render_settings")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                ui.label("Background Color");
                egui::color_picker::color_edit_button_srgba(
                    ui,
                    &mut state.background_color,
                    egui::color_picker::Alpha::BlendOrAdditive,
                );
            });
    });

    let mut new_camera: Option<SetCamera> = None;
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
                    ui.label(state.pc.num_points().to_string());
                    ui.end_row();
                    ui.strong("SH Degree:");
                    ui.label(state.pc.sh_deg().to_string());
                    ui.end_row();
                    ui.strong("Compressed:");
                    ui.label(state.pc.compressed().to_string());
                    ui.end_row();
                });

            if let Some(scene) = &state.scene {
                let nearest = scene.nearest_camera(state.camera.position, None);
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
                                    let drag =
                                        ui.add(egui::DragValue::new(c).clamp_range(
                                            0..=(scene.num_cameras().saturating_sub(1)),
                                        ));
                                    if drag.changed() {
                                        new_camera = Some(SetCamera::ID(*c));
                                    }
                                    ui.label(scene.camera(*c as usize).unwrap().split.to_string());
                                });
                            }
                        });

                    egui::ScrollArea::vertical()
                        .max_height(300.)
                        .show(ui, |ui| {
                            egui::Grid::new("scene views grid")
                                .num_columns(4)
                                .striped(true)
                                .show(ui, |ui| {
                                    let style = ui.style().clone();
                                    for c in scene.cameras(None) {
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
                                        );
                                        ui.label(c.img_name.clone());
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

    let mut save_view = false;
    let mut cancle_animation = false;
    egui::Window::new("Tracking Shot")
        .default_width(200.)
        .resizable(false)
        .default_height(100.)
        .max_height(500.)
        .default_open(false)
        .show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.horizontal_wrapped(|ui| {
                    if let Some(scene) = &state.scene {
                        ui.menu_button("Scene Cameras", |ui| {
                            if ui.button("Train").clicked() {
                                state.saved_cameras = scene.cameras(Some(Split::Train)).clone();
                                cancle_animation = true;
                                ui.close_menu();
                            }
                            if ui.button("Test").clicked() {
                                state.saved_cameras = scene.cameras(Some(Split::Test)).clone();
                                cancle_animation = true;
                                ui.close_menu();
                            }
                            if ui.button("All").clicked() {
                                state.saved_cameras = scene.cameras(None).clone();
                                cancle_animation = true;
                                ui.close_menu();
                            }
                        });
                    }
                    if ui.button("Clear").clicked() {
                        state.saved_cameras.clear();
                        cancle_animation = true;
                    }
                    if ui.button("Add Current View").clicked() {
                        save_view = true;
                        cancle_animation = true;
                    }
                });
                ui.heading("Cameras");
                egui::ScrollArea::new(Vec2b::new(false, true))
                    .max_height(300.)
                    .show(ui, |ui| {
                        let mut trash = Vec::new();
                        let playing = state.animation.as_ref().map(|(_, p)| *p).unwrap_or(false);
                        ui.add_enabled_ui(!playing, |ui| {
                            let resp = dnd(ui, "dnd_cameras").show_vec(
                                &mut state.saved_cameras,
                                |ui, c, handle, state| {
                                    let style = (*ctx.style()).clone();
                                    let bg = if state.dragged {
                                        style.visuals.extreme_bg_color
                                    } else {
                                        style.visuals.panel_fill
                                    };
                                    egui::Frame::default()
                                        .fill(bg)
                                        .inner_margin(Margin::same(3.))
                                        .show(ui, |ui| {
                                            ui.horizontal(|ui| {
                                                handle.ui(ui, |ui| {
                                                    ui.label("|||");
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.set_width(120.);

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
                                                    );
                                                    ui.label(c.img_name.clone());
                                                });
                                                if ui.button("🗑").clicked() {
                                                    trash.push(state.index);
                                                }
                                                if ui.button("🎥").clicked() {
                                                    new_camera = Some(SetCamera::Camera(c.clone()));
                                                }
                                            });
                                        });
                                },
                            );

                            if resp.final_update().is_some() {
                                cancle_animation = true;
                            }
                        });
                        for t in trash {
                            state.saved_cameras.remove(t);
                        }
                    });
                ui.separator();

                ui.heading("Animation");
                ui.add_enabled_ui(state.saved_cameras.len() > 1, |ui| {
                    let text = if state
                        .animation
                        .as_ref()
                        .and_then(|(_, playing)| Some(!*playing))
                        .unwrap_or(true)
                    {
                        "Play"
                    } else {
                        "Pause"
                    };
                    ui.horizontal(|ui| {
                        if ui.add(egui::Button::new(text).shortcut_text("T")).clicked() {
                            toggle_tracking_shot = true;
                        }

                        if ui
                            .add_enabled(state.animation.is_some(), egui::Button::new("Cancel"))
                            .clicked()
                        {
                            cancle_animation = true;
                        }
                    });
                    if let Some((animation, playing)) = &mut state.animation {
                        egui::Grid::new("animation grid")
                            .num_columns(2)
                            .show(ui, |ui| {
                                ui.strong("Progress");
                                let mut progress = animation.progress();
                                if ui
                                    .add(egui::Slider::new(&mut progress, (0.)..=(1.)))
                                    .changed()
                                {
                                    *playing = false;
                                }
                                animation.set_progress(progress);
                                ui.end_row();

                                let mut duration = animation.duration().as_secs_f32();
                                ui.strong("Duration");
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut duration)
                                            .clamp_range(0.1..=1e4)
                                            .speed(0.5)
                                            .suffix("s"),
                                    )
                                    .changed()
                                {
                                    animation.set_duration(Duration::from_secs_f32(duration));
                                }
                            });
                    }
                    ui.heading("Save Animation");
                    ui.horizontal_wrapped(|ui| {
                        ui.text_edit_singleline(&mut state.cameras_save_path);
                        if ui.button("Save").clicked() {
                            let path = Path::new(&state.cameras_save_path);
                            if let Some(parent) = path.parent() {
                                create_dir_all(parent).unwrap();
                            }
                            let mut file = File::create(path).unwrap();
                            serde_json::to_writer_pretty(&mut file, &state.saved_cameras).unwrap();
                        }
                    });
                });
            });
        });

    if let Some(c) = new_camera {
        match c {
            SetCamera::ID(id) => state.set_scene_camera(id),
            SetCamera::Camera(c) => state.set_camera(c, Duration::from_millis(200)),
        }
    }
    if save_view {
        state.save_view();
    }
    if toggle_tracking_shot {
        if let Some((_animation, playing)) = &mut state.animation {
            *playing = !*playing;
        } else {
            state.start_tracking_shot();
        }
    }
    if cancle_animation {
        state.animation.take();
    }
}

enum SetCamera {
    ID(usize),
    Camera(SceneCamera),
}
