#[cfg(target_arch = "wasm32")]
use instant::Duration;
use splines::{Interpolate, Key};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Quaternion, Rad, VectorSpace};

use crate::{camera::PerspectiveCamera, PerspectiveProjection};

pub trait Lerp {
    fn lerp(&self, other: &Self, amount: f32) -> Self;
}

pub trait Sampler {
    type Sample;

    fn sample(&self, v: f32) -> Self::Sample;
}

pub struct Transition<T> {
    from: T,
    to: T,
    interp_fn: fn(f32) -> f32,
}
impl<T: Lerp + Clone> Transition<T> {
    pub fn new(from: T, to: T, interp_fn: fn(f32) -> f32) -> Self {
        Self {
            from,
            to,
            interp_fn,
        }
    }
}

impl<T: Lerp + Clone> Sampler for Transition<T> {
    type Sample = T;
    fn sample(&self, v: f32) -> Self::Sample {
        self.from.lerp(&self.to, (self.interp_fn)(v))
    }
}

pub struct TrackingShot {
    spline: splines::Spline<f32, PerspectiveCamera>,
}

impl TrackingShot {
    pub fn from_scene<C>(cameras: Vec<C>) -> Self
    where
        C: Into<PerspectiveCamera>,
    {
        let cameras: Vec<PerspectiveCamera> = cameras.into_iter().map(|c| c.into()).collect();

        let last_two = cameras.iter().skip(cameras.len() - 2).take(2);
        let first_two = cameras.iter().take(2);
        let spline = splines::Spline::from_iter(
            last_two
                .chain(cameras.iter())
                .chain(first_two)
                .enumerate()
                .map(|(i, c)| {
                    let v = (i as f32 - 1.) / (cameras.len()) as f32;
                    Key::new(v, c.clone(), splines::Interpolation::CatmullRom)
                }),
        );

        Self { spline }
    }
}

impl Sampler for TrackingShot {
    type Sample = PerspectiveCamera;
    fn sample(&self, v: f32) -> Self::Sample {
        match self.spline.sample(v) {
            Some(p) => p,
            None => panic!("spline sample failed at {}", v),
        }
    }
}

impl Interpolate<f32> for PerspectiveCamera {
    fn step(t: f32, threshold: f32, a: Self, b: Self) -> Self {
        if t < threshold {
            a
        } else {
            b
        }
    }

    fn lerp(t: f32, a: Self, b: Self) -> Self {
        Self {
            position: Point3::from_vec(a.position.to_vec().lerp(b.position.to_vec(), t)),
            rotation: a.rotation.slerp(b.rotation, t),
            projection: a.projection.lerp(&b.projection, t),
        }
    }

    fn cosine(t: f32, a: Self, b: Self) -> Self {
        todo!()
    }

    fn cubic_hermite(
        t: f32,
        x: (f32, Self),
        a: (f32, Self),
        b: (f32, Self),
        y: (f32, Self),
    ) -> Self {
        // unroll quaternion rotations so that the animation always takes the shortest path
        // this is just a hack...
        let q_unrolled = unroll([x.1.rotation, a.1.rotation, b.1.rotation, y.1.rotation]);
        Self {
            position: Point3::from_vec(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.position.to_vec()),
                (a.0, a.1.position.to_vec()),
                (b.0, b.1.position.to_vec()),
                (y.0, y.1.position.to_vec()),
            )),
            rotation: Interpolate::cubic_hermite(
                t,
                (x.0, q_unrolled[0]),
                (a.0, q_unrolled[1]),
                (b.0, q_unrolled[2]),
                (y.0, q_unrolled[3]),
            )
            .normalize(),
            projection: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.projection),
                (a.0, a.1.projection),
                (b.0, b.1.projection),
                (y.0, y.1.projection),
            ),
        }
    }

    fn quadratic_bezier(t: f32, a: Self, u: Self, b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier(t: f32, a: Self, u: Self, v: Self, b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier_mirrored(t: f32, a: Self, u: Self, v: Self, b: Self) -> Self {
        todo!()
    }
}

impl Interpolate<f32> for PerspectiveProjection {
    fn step(t: f32, threshold: f32, a: Self, b: Self) -> Self {
        if t < threshold {
            a
        } else {
            b
        }
    }

    fn lerp(t: f32, a: Self, b: Self) -> Self {
        return a.lerp(&b, t);
    }

    fn cosine(t: f32, a: Self, b: Self) -> Self {
        todo!()
    }

    fn cubic_hermite(
        t: f32,
        x: (f32, Self),
        a: (f32, Self),
        b: (f32, Self),
        y: (f32, Self),
    ) -> Self {
        Self {
            fovx: Rad(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fovx.0),
                (a.0, a.1.fovx.0),
                (b.0, b.1.fovx.0),
                (y.0, y.1.fovx.0),
            )),
            fovy: Rad(Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fovy.0),
                (a.0, a.1.fovy.0),
                (b.0, b.1.fovy.0),
                (y.0, y.1.fovy.0),
            )),
            znear: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.znear),
                (a.0, a.1.znear),
                (b.0, b.1.znear),
                (y.0, y.1.znear),
            ),
            zfar: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.zfar),
                (a.0, a.1.zfar),
                (b.0, b.1.zfar),
                (y.0, y.1.zfar),
            ),
            fov2view_ratio: Interpolate::cubic_hermite(
                t,
                (x.0, x.1.fov2view_ratio),
                (a.0, a.1.fov2view_ratio),
                (b.0, b.1.fov2view_ratio),
                (y.0, y.1.fov2view_ratio),
            ),
        }
    }

    fn quadratic_bezier(t: f32, a: Self, u: Self, b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier(t: f32, a: Self, u: Self, v: Self, b: Self) -> Self {
        todo!()
    }

    fn cubic_bezier_mirrored(t: f32, a: Self, u: Self, v: Self, b: Self) -> Self {
        todo!()
    }
}

pub struct Animation<T> {
    duration: Duration,
    time_left: Duration,
    looping: bool,
    sampler: Box<dyn Sampler<Sample = T>>,
}

impl<T> Animation<T> {
    pub fn new(duration: Duration, looping: bool, sampler: Box<dyn Sampler<Sample = T>>) -> Self {
        Self {
            duration,
            time_left: duration,
            looping,
            sampler,
        }
    }

    pub fn done(&self) -> bool {
        if self.looping {
            false
        } else {
            self.time_left.is_zero()
        }
    }

    pub fn update(&mut self, dt: Duration) -> T {
        match self.time_left.checked_sub(dt) {
            Some(new_left) => {
                // set time left
                self.time_left = new_left;
            }
            None => {
                if self.looping {
                    self.time_left = self.duration + self.time_left - dt;
                } else {
                    self.time_left = Duration::ZERO;
                }
            }
        }
        return self.sampler.sample(self.progress());
    }

    pub fn progress(&self) -> f32 {
        return 1. - self.time_left.as_secs_f32() / self.duration.as_secs_f32();
    }

    pub fn set_progress(&mut self, v: f32) {
        self.time_left = self.duration.mul_f32(1. - v);
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }
}

/// unroll quaternion rotations so that the animation always takes the shortest path
fn unroll(rot: [Quaternion<f32>; 4]) -> [Quaternion<f32>; 4] {
    let mut rot = rot;
    if rot[0].s < 0. {
        rot[0] = -rot[0];
    }
    for i in 1..4 {
        if rot[i].dot(rot[i - 1]) < 0. {
            rot[i] = -rot[i];
        }
    }
    return rot;
}
