use std::time::Duration;

use cgmath::{EuclideanSpace, InnerSpace, Matrix4, MetricSpace, Point3, Vector4};

use crate::{camera::PerspectiveCamera, scene::Scene};

pub trait Lerp {
    fn lerp(&self, other: &Self, amount: f32) -> Self;
}

pub trait Animation {
    type Animatable: Lerp;

    fn update(&mut self, dt: Duration) -> Self::Animatable;

    fn done(&self) -> bool;
}

pub struct Transition<T: Lerp> {
    from: T,
    to: T,
    time_left: Duration,
    duration: Duration,
    interp_fn: fn(f32) -> f32,
}
impl<T: Lerp + Clone> Transition<T> {
    pub fn new(from: T, to: T, duration: Duration, interp_fn: fn(f32) -> f32) -> Self {
        Self {
            from,
            to,
            time_left: duration,
            duration: duration,
            interp_fn,
        }
    }
}

impl<T: Lerp + Clone> Animation for Transition<T> {
    type Animatable = T;
    fn update(&mut self, dt: Duration) -> T {
        match self.time_left.checked_sub(dt) {
            Some(new_left) => {
                // set time left
                self.time_left = new_left;
                let elapsed = 1. - new_left.as_secs_f32() / self.duration.as_secs_f32();
                let amount = (self.interp_fn)(elapsed);
                let new_value = self.from.lerp(&self.to, amount);
                return new_value;
            }
            None => {
                self.time_left = Duration::ZERO;
                return self.to.clone();
            }
        }
    }

    fn done(&self) -> bool {
        self.time_left.is_zero()
    }
}

pub struct TrackingShot {
    cameras: Vec<PerspectiveCamera>,
    current_idx: usize,
    transiton: Transition<PerspectiveCamera>,
    speed: f32,
}

impl TrackingShot {
    pub fn from_scene(scene: &Scene, speed: f32, start: Option<PerspectiveCamera>) -> Self {
        let cameras: Vec<PerspectiveCamera> =
            scene.cameras().iter().map(|c| c.clone().into()).collect();
        let n = cameras.len();
        let first = start.unwrap_or(cameras[0]);
        let second = if start.is_none() {
            cameras[1]
        } else {
            cameras[0]
        };
        Self {
            cameras,
            current_idx: if start.is_none() { 0 } else { n - 1 },
            transiton: Self::create_transition(first, second, speed),
            speed,
        }
    }

    fn create_transition(
        from: PerspectiveCamera,
        to: PerspectiveCamera,
        speed: f32,
    ) -> Transition<PerspectiveCamera> {
        let d = from.position.distance(to.position);
        let mut arc = from.rotation.angle(to.rotation);
        if arc.0.is_nan() {
            arc.0 = 0.;
        }
        Transition::new(
            from,
            to,
            Duration::from_secs_f32((d + 2. * arc.0) / speed),
            linear,
        )
    }
}

impl Animation for TrackingShot {
    type Animatable = PerspectiveCamera;
    fn update(&mut self, dt: Duration) -> PerspectiveCamera {
        let curr_camera = self.transiton.update(dt);
        if self.transiton.done() {
            self.current_idx = (self.current_idx + 1) % self.cameras.len();
            let next: usize = (self.current_idx + 1) % self.cameras.len();
            self.transiton = Self::create_transition(
                self.cameras[self.current_idx].clone(),
                self.cameras[next].clone(),
                self.speed,
            )
        }
        curr_camera
    }

    fn done(&self) -> bool {
        // never ends
        false
    }
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}

pub fn linear(x: f32) -> f32 {
    x
}

fn spline_coefs(pos: Vec<Point3<f32>>) -> Vec<Matrix4<f32>> {
    let mut spline = Vec::new();
    let n = spline.len();

    let mut d = Vec::new();
    for i in 0..n {
        let prev = pos[((i as i32 - 1) % n as i32) as usize];
        let curr = pos[i];
        let next = pos[(i + 1) % n];
        d.push(((next - curr) + (curr - prev)).normalize());
    }
    for i in 0..n {
        let curr = i;
        let next = (i + 1) % n;

        let x0 = pos[curr].to_vec();
        let x1 = d[i];
        let x2 = 3. * (pos[next] - pos[curr]) - (d[next] + 2. * d[curr]);
        let x3 = 2. * (pos[curr] - pos[next]) + d[next] + d[curr];

        spline.push(Matrix4::from_cols(
            Vector4::new(x0.x, x0.y, x0.z, 0.),
            Vector4::new(x1.x, x1.y, x1.z, 0.),
            Vector4::new(x2.x, x2.y, x2.z, 0.),
            Vector4::new(x3.x, x3.y, x3.z, 0.),
        ))
    }
    return spline;
}
