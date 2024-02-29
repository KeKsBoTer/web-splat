use cgmath::{EuclideanSpace, Matrix4, Point3, Quaternion, SquareMatrix, Transform, Vector3};
use wgpu::Color;

use crate::{
    lines::Line, Camera, PerspectiveCamera, PointCloud, Sampler, Scene, Split, TrackingShot,
};

pub struct DebugLines {
    pub cameras: Option<Vec<Line>>,
    pub show_cameras: bool,

    pub volume_aabb: Vec<Line>,
    pub show_volume_aabb: bool,

    pub origin: Vec<Line>,
    pub show_origin: bool,

    pub center_up: Vec<Line>,
    pub show_center_up: bool,

    pub camera_path: Option<Vec<Line>>,
    pub show_camera_path: bool,
}

impl DebugLines {
    pub fn new(pc: &PointCloud) -> Self {
        let aabb = pc.bbox();
        let pc_size = aabb.size() * 0.5;
        let center = aabb.center();
        let t =
            Matrix4::from_translation(center.to_vec()) * Matrix4::from_diagonal(pc_size.extend(1.));
        let volume_aabb = box_lines(t, wgpu::Color::BLUE).to_vec();

        let origin = axes_lines(Matrix4::identity()).to_vec();

        let t = if let Some(up) = pc.up() {
            let rot = Quaternion::from_arc(Vector3::unit_y(), up, None);
            Matrix4::from_translation(pc.center().to_vec()) * Matrix4::from(rot)
        } else {
            Matrix4::from_translation(pc.center().to_vec())
        };

        let center_up = axes_lines(t).to_vec();

        Self {
            cameras: None,
            show_cameras: false,
            volume_aabb,
            show_volume_aabb: false,
            origin,
            show_origin: false,
            center_up,
            show_center_up: false,
            camera_path: None,
            show_camera_path: false,
        }
    }

    pub fn set_scene(&mut self, scene: &Scene) {
        self.cameras = Some(
            scene
                .cameras(None)
                .iter()
                .flat_map(|c| {
                    let cam: PerspectiveCamera = c.clone().into();
                    let t =
                        cam.view_matrix().inverse_transform().unwrap() * Matrix4::from_scale(0.1);
                    camera_lines(
                        t,
                        match c.split {
                            Split::Train => Color::WHITE,
                            Split::Test => Color::RED,
                        },
                    )
                })
                .collect(),
        );
    }

    pub fn visible_lines(&self) -> Vec<&[Line]> {
        let mut lines = vec![];
        if self.show_cameras {
            if let Some(cameras) = &self.cameras {
                lines.push(cameras.as_slice());
            }
        }
        if self.show_camera_path {
            if let Some(camera_path) = &self.camera_path {
                lines.push(camera_path.as_slice());
            }
        }
        if self.show_volume_aabb {
            lines.push(&self.volume_aabb);
        }
        if self.show_origin {
            lines.push(&self.origin);
        }
        if self.show_center_up {
            lines.push(&self.center_up);
        }
        lines
    }

    pub fn set_tracking_shot(&mut self, tracking_shot: &TrackingShot) {
        self.camera_path = Some(tracking_shot_lines(tracking_shot, Color::GREEN, Color::RED));
    }
}

fn camera_lines(t: Matrix4<f32>, color: wgpu::Color) -> [Line; 9] {
    let a = t.transform_point(Point3::new(0., 0., 0.));
    let b = t.transform_point(Point3::new(1., 1., 1.));
    let c = t.transform_point(Point3::new(1., -1., 1.));
    let d = t.transform_point(Point3::new(-1., 1., 1.));
    let e = t.transform_point(Point3::new(-1., -1., 1.));
    let f = t.transform_point(Point3::new(0., 1., 1.));
    let g = t.transform_point(Point3::new(0., 1.1, 1.));
    [
        Line::new(a, b, color),
        Line::new(a, c, color),
        Line::new(a, d, color),
        Line::new(a, e, color),
        Line::new(b, c, color),
        Line::new(c, e, color),
        Line::new(e, d, color),
        Line::new(d, b, color),
        Line::new(f, g, color),
    ]
}

fn box_lines(t: Matrix4<f32>, color: wgpu::Color) -> [Line; 12] {
    let p = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| {
        t.transform_point(Point3::new(
            (i & 1) as f32 * 2. - 1.,
            ((i >> 1) & 1) as f32 * 2. - 1.,
            ((i >> 2) & 1) as f32 * 2. - 1.,
        ))
    });
    [
        // front
        Line::new(p[0], p[1], color),
        Line::new(p[1], p[3], color),
        Line::new(p[2], p[0], color),
        Line::new(p[3], p[2], color),
        // back
        Line::new(p[4], p[5], color),
        Line::new(p[5], p[7], color),
        Line::new(p[6], p[4], color),
        Line::new(p[7], p[6], color),
        // sides
        Line::new(p[0], p[4], color),
        Line::new(p[1], p[5], color),
        Line::new(p[2], p[6], color),
        Line::new(p[3], p[7], color),
    ]
}

fn axes_lines(t: Matrix4<f32>) -> [Line; 3] {
    let c = t.transform_point(Point3::origin());
    let x = t.transform_point(Point3::new(1., 0., 0.));
    let y = t.transform_point(Point3::new(0., 1., 0.));
    let z = t.transform_point(Point3::new(0., 0., 1.));
    [
        Line::new(c, x, Color::RED),
        Line::new(c, y, Color::GREEN),
        Line::new(c, z, Color::BLUE),
    ]
}

fn tracking_shot_lines(
    tracking_shot: &TrackingShot,
    color_start: wgpu::Color,
    color_end: wgpu::Color,
) -> Vec<Line> {
    let n = tracking_shot.num_control_points() * 20;
    let cameras: Vec<Point3<f32>> = (0..n)
        .map(|i| tracking_shot.sample(i as f32 / n as f32).position)
        .collect();

    cameras
        .iter()
        .cycle()
        .skip(1)
        .zip(cameras.iter())
        .enumerate()
        .map(|(i, (a, b))| {
            Line::new(
                *a,
                *b,
                blend(color_start, color_end, i as f64 / (n + 1) as f64),
            )
        })
        .collect()
}

fn blend(a: wgpu::Color, b: wgpu::Color, t: f64) -> wgpu::Color {
    wgpu::Color {
        r: a.r * (1. - t) + b.r * t,
        g: a.g * (1. - t) + b.g * t,
        b: a.b * (1. - t) + b.b * t,
        a: a.a * (1. - t) + b.a * t,
    }
}
