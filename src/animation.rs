#[cfg(target_arch = "wasm32")]
use instant::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use cgmath::{InnerSpace, MetricSpace, Point3};

use crate::camera::PerspectiveCamera;

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
    pub fn from_scene<C>(cameras: Vec<C>, speed: f32, start: Option<PerspectiveCamera>) -> Self
    where
        C: Into<PerspectiveCamera>,
    {
        let cameras: Vec<PerspectiveCamera> = cameras.into_iter().map(|c| c.into()).collect();
        let n = cameras.len();
        let (first, second, idx) = if let Some(start) = start {
            let idx = cameras
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| {
                    (Point3::from(c.position).distance2(start.position) * 1e6) as u32
                })
                .unwrap()
                .0;
            (start, cameras[idx], ((idx as i32 + n as i32 - 1) % n as i32) as usize)
        } else {
            (cameras[0], cameras[1], 0)
        };
        Self {
            cameras,
            current_idx: idx,
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

        // find shortest possible angle
        let mut arc = from.rotation.angle(if from.rotation.dot(to.rotation) < 0. {
            -to.rotation
        } else {
            to.rotation
        });

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
        let curr_camera: PerspectiveCamera = self.transiton.update(dt);
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

pub fn linear(x: f32) -> f32 {
    x
}
