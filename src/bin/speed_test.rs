use std::time;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Vector3};
use rand;
use rayon::prelude::*;

fn ray_point_distance(orig: Point3<f32>, dir: Vector3<f32>, point: Point3<f32>) -> (f32, f32) {
    let t = 2.0 * dir.dot(orig - point);
    let diff: Vector3<f32> = point - (orig + t * dir);
    return (t, diff.magnitude());
}

fn main() {
    let points: Vec<Point3<f32>> = (0..5_000_000)
        .map(|_| Point3::new(rand::random(), rand::random(), rand::random()))
        .collect();

    let orig = Point3::origin();
    let dir = Vector3::unit_z();
    let start = time::Instant::now();
    points
        .par_iter()
        .map(|p| (ray_point_distance(orig, dir, *p).0 * 1000.) as u32)
        .min()
        .unwrap();
    let duration = start.elapsed();
    println!("took {:} ms", duration.as_millis())
}
