use std::{
    hash::Hash,
    io::{self, BufReader},
};

use cgmath::{Matrix3, MetricSpace, Point3, SquareMatrix, Vector2};
use serde::{Deserialize, Serialize};

use crate::camera::{ focal2fov, fov2focal, PerspectiveCamera, PerspectiveProjection};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SceneCamera {
    pub id: usize,
    pub img_name: String,
    pub width: u32,
    pub height: u32,
    pub position: [f32; 3],
    pub rotation: [[f32; 3]; 3],
    pub fx: f32,
    pub fy: f32,
    #[serde(skip_deserializing, skip_serializing)]
    pub split: Split,
}

impl PartialEq for SceneCamera {
    fn eq(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height && self.position == other.position && self.rotation == other.rotation && self.fx == other.fx && self.fy == other.fy
    }
}

impl SceneCamera {
    pub fn from_perspective(
        cam: PerspectiveCamera,
        name: String,
        id: usize,
        viewport: Vector2<u32>,
        split: Split,
    ) -> Self {
        let fx = fov2focal(cam.projection.fovx, viewport.x as f32);
        let fy = fov2focal(cam.projection.fovy, viewport.y as f32);
        let rot: Matrix3<f32> = cam.rotation.into();
        Self {
            id,
            img_name: name,
            width: viewport.x,
            height: viewport.y,
            position: cam.position.into(),
            rotation: rot.into(),
            fx,
            fy,
            split,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Hash)]
pub enum Split {
    Train,
    Test,
}

impl Default for Split {
    fn default() -> Self {
        Split::Train
    }
}

impl ToString for Split {
    fn to_string(&self) -> String {
        match self {
            Split::Train => "train",
            Split::Test => "test",
        }
        .to_string()
    }
}

impl Into<PerspectiveCamera> for SceneCamera {
    fn into(self) -> PerspectiveCamera {
        let fovx = focal2fov(self.fx, self.width as f32);
        let fovy = focal2fov(self.fy, self.height as f32);
        let mut rot = Matrix3::from(self.rotation);
        if rot.determinant() < 0. {
            // make sure determinant is 1
            // flip y axis if determinant is -1
            rot.x[1] = -rot.x[1];
            rot.y[1] = -rot.y[1];
            rot.z[1] = -rot.z[1];
        }
        PerspectiveCamera {
            position: self.position.into(),
            rotation: rot.into(),
            projection: PerspectiveProjection::new(
                Vector2::new(self.width, self.height),
                Vector2::new(fovx, fovy),
                0.01,
                100.,
            ),
        }
    }
}

#[derive(Debug)]
pub struct Scene {
    cameras: Vec<SceneCamera>,
    /// maximum distance between two cameras
    extend: f32,
}

impl Scene {
    pub fn from_cameras(mut cameras:Vec<SceneCamera>) -> Self {
        let extend = max_distance(cameras.iter().map(|c| Point3::from(c.position)).collect());
        let mut list = Vec::new();
        cameras.sort_by_key(|c|c.img_name.clone());

        let mut last: Option<SceneCamera> = None;
        for c in cameras{
            if let Some(l) = last{
                if l != c{
                    list.push(l);
                }
            }
            last = Some(c);
        
        }
        Self {
            cameras:list,
            extend,
        }
    }

    pub fn from_json<R: io::Read>(file: R) -> Result<Self, anyhow::Error> {
        let mut reader = BufReader::new(file);
        let mut cameras: Vec<SceneCamera> = serde_json::from_reader(&mut reader)?;
        for (i, c) in cameras.iter_mut().enumerate() {
            // according to Kerbl et al "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
            // 7 out of 8 cameras are taken as training images
            c.split = if i % 8 == 0 {
                Split::Test
            } else {
                Split::Train
            }
        }
        log::info!("loaded scene file with {} views", cameras.len());
        Ok(Self::from_cameras(cameras))
    }

    pub fn camera(&self, i: usize) -> Option<SceneCamera> {
        self.cameras.get(i).cloned()
    }
    pub fn camera_by_id(&self, id: usize) -> Option<&SceneCamera> {
        self.cameras.iter().find(|c| c.id == id)
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    pub fn cameras(&self, split: Option<Split>) -> Vec<SceneCamera> {
        return if let Some(split) = split {
            self.cameras
                .iter()
                .filter_map(|c| (c.split == split).then_some(c.clone()))
                .collect()
        } else {
            self.cameras.iter().cloned().collect()
        };
    }

    pub fn extend(&self) -> f32 {
        self.extend
    }

    pub fn next_camera(&self,i: usize) -> Option<usize> {
        Some((i+1)%(self.num_cameras()-1))
    }

    pub fn prev_camera(&self,i: usize) -> Option<usize> {
        Some((i-1)%(self.num_cameras()-1))
    }

    /// index of nearest camera
    pub fn nearest_camera(&self, pos: Point3<f32>, split: Option<Split>) -> Option<usize> {
        self.cameras
            .iter()
            .enumerate()
            .filter(|(_,c)| match split {
                Some(s) => s == c.split,
                None => true,
            })
            .min_by_key(|(_,c)| (Point3::from(c.position).distance2(pos) * 1e6) as u32)
            .map(|(i,_)| i)
    }
    
}

/// calculate the maximum distance between any two points
/// naive implementation with O(n^2)
fn max_distance(points: Vec<Point3<f32>>) -> f32 {
    let mut max_distance: f32 = 0.;
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            max_distance = max_distance.max(points[i].distance2(points[j]));
        }
    }

    max_distance.sqrt()
}
