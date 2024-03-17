use std::collections::HashMap;

use bytemuck::Zeroable;
use cgmath::{
    EuclideanSpace, Matrix, Matrix4, Point3, Quaternion, SquareMatrix, Transform, Vector3, Vector4,
};
use rayon::str::MatchIndices;
use wgpu::Color;

use crate::{
    camera, lines::Line, pointcloud::Aabb, Camera, PerspectiveCamera, PointCloud, Sampler, Scene,
    Split, TrackingShot,
};

pub struct LineGroup {
    pub lines: Vec<Line>,
    pub visible: bool,
}

impl LineGroup {
    fn new(lines: impl Into<Vec<Line>>, visible: bool) -> Self {
        Self {
            lines: lines.into(),
            visible,
        }
    }

    fn aabb(&self) -> Aabb<f32> {
        let mut aabb: Aabb<f32> = Aabb::zeroed();
        for line in &self.lines {
            aabb.grow(&line.start);
            aabb.grow(&line.end);
        }
        aabb
    }
}

pub struct DebugLines {
    line_groups: HashMap<String, LineGroup>,
    aabb: Aabb<f32>,
}

impl DebugLines {
    pub fn new(pc: &PointCloud) -> Self {
        let mut lines = HashMap::new();

        let aabb = pc.bbox();
        let pc_size = aabb.size() * 0.5;
        let center = aabb.center();
        let t =
            Matrix4::from_translation(center.to_vec()) * Matrix4::from_diagonal(pc_size.extend(1.));
        let volume_aabb = box_lines(t, wgpu::Color::BLUE).to_vec();

        lines.insert(
            "volume_aabb".to_string(),
            LineGroup::new(volume_aabb.clone(), false),
        );

        let origin = axes_lines(Matrix4::identity()).to_vec();
        lines.insert("origin".to_string(), LineGroup::new(origin, false));

        let t = if let Some(up) = pc.up() {
            let rot = Quaternion::from_arc(Vector3::unit_y(), up, None);
            Matrix4::from_translation(pc.center().to_vec()) * Matrix4::from(rot)
        } else {
            Matrix4::from_translation(pc.center().to_vec())
        };

        let center_up = axes_lines(t).to_vec();
        lines.insert("center_up".to_string(), LineGroup::new(center_up, false));
        let clipping_box: Vec<Line> = volume_aabb
            .iter()
            .map(|l| {
                let mut l2 = l.clone();
                l2.color = Vector4::new(255, 255, 0, 255);
                l2
            })
            .collect();

        lines.insert(
            "clipping_box".to_string(),
            LineGroup::new(clipping_box, false),
        );

        let mut me = Self {
            line_groups: lines,
            aabb: Aabb::zeroed(),
        };
        me.update_aabb();
        return me;
    }

    pub fn all_lines(&mut self) -> Vec<(&String, &mut LineGroup)> {
        self.line_groups.iter_mut().collect()
    }

    fn update_aabb(&mut self) {
        let mut aabb = Aabb::zeroed();
        for lg in self.line_groups.values() {
            aabb.grow_union(&lg.aabb());
        }
        self.aabb = aabb;
    }

    pub fn any_visible(&self) -> bool {
        self.line_groups.values().any(|lg| lg.visible)
    }

    pub fn update_clipping_box(&mut self, clipping_box: &Aabb<f32>) {
        let t = Matrix4::from_translation(clipping_box.center().to_vec())
            * Matrix4::from_diagonal((clipping_box.size() * 0.5).extend(1.));
        self.update_lines(
            "clipping_box",
            box_lines(
                t,
                wgpu::Color {
                    r: 1.,
                    g: 1.,
                    b: 0.,
                    a: 1.,
                },
            )
            .to_vec(),
        );
    }

    pub fn set_scene(&mut self, scene: &Scene) {
        self.update_lines(
            "cameras",
            scene
                .cameras(None)
                .iter()
                .flat_map(|c| {
                    let cam: PerspectiveCamera = c.clone().into();
                    camera_lines(
                        cam,
                        match c.split {
                            Split::Train => Color::WHITE,
                            Split::Test => Color::RED,
                        },
                        scene.extend() * 2e-4,
                    )
                })
                .collect::<Vec<_>>(),
        );
    }

    pub fn visible_lines(&self) -> Vec<Line> {
        let mut lines = vec![];
        for set in self.line_groups.values() {
            if set.visible {
                lines.extend_from_slice(&set.lines);
            }
        }
        lines
    }

    fn update_lines(&mut self, name: &str, lines: Vec<Line>) {
        if let Some(lg) = self.line_groups.get_mut(name) {
            lg.lines = lines;
        } else {
            self.line_groups
                .insert(name.to_string(), LineGroup::new(lines, false));
            self.update_aabb();
        }
    }

    pub fn set_tracking_shot(&mut self, tracking_shot: &TrackingShot) {
        self.update_lines(
            "camera_path",
            tracking_shot_lines(tracking_shot, Color::GREEN, Color::RED),
        );
    }

    pub fn bbox(&self) -> &Aabb<f32> {
        &self.aabb
    }
}

fn camera_lines(camera: impl Camera, color: wgpu::Color, scale: f32) -> [Line; 12] {
    let frustum_planes = camera.frustum_planes();
    let a_l =
        three_plane_intersection(frustum_planes.near, frustum_planes.top, frustum_planes.left);
    let b_l = three_plane_intersection(
        frustum_planes.near,
        frustum_planes.bottom,
        frustum_planes.left,
    );
    let c_l = three_plane_intersection(frustum_planes.far, frustum_planes.top, frustum_planes.left);
    let d_l = three_plane_intersection(
        frustum_planes.far,
        frustum_planes.bottom,
        frustum_planes.left,
    );
    // right side
    let a_r = three_plane_intersection(
        frustum_planes.near,
        frustum_planes.top,
        frustum_planes.right,
    );
    let b_r = three_plane_intersection(
        frustum_planes.near,
        frustum_planes.bottom,
        frustum_planes.right,
    );
    let c_r =
        three_plane_intersection(frustum_planes.far, frustum_planes.top, frustum_planes.right);
    let d_r = three_plane_intersection(
        frustum_planes.far,
        frustum_planes.bottom,
        frustum_planes.right,
    );
    let campos = camera.position();

    let t = Matrix4::from_translation(campos.to_vec())
        * Matrix4::from_scale(scale)
        * Matrix4::from_translation(-campos.to_vec());

    let mut lines = [
        // right
        Line::new(a_r, b_r, color),
        Line::new(b_r, d_r, color),
        Line::new(d_r, c_r, color),
        Line::new(c_r, a_r, color),
        // left
        Line::new(a_l, b_l, color),
        Line::new(b_l, d_l, color),
        Line::new(d_l, c_l, color),
        Line::new(c_l, a_l, color),
        // rest
        Line::new(a_l, a_r, color),
        Line::new(b_l, b_r, color),
        Line::new(d_l, d_r, color),
        Line::new(c_l, c_r, color),
    ];
    for l in &mut lines {
        l.start = t.transform_point(l.start);
        l.end = t.transform_point(l.end);
    }
    return lines;
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

fn three_plane_intersection(a: Vector4<f32>, b: Vector4<f32>, c: Vector4<f32>) -> Point3<f32> {
    let m = Matrix4::from_cols(a, b, c, Vector4::new(0., 0., 0., 1.)).transpose();

    let intersection = m.inverse_transform().unwrap() * Vector4::new(0., 0., 0., 1.);
    return Point3::from_homogeneous(intersection);
}
