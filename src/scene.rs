use std::{fs::File, io::BufReader, path::Path};

use cgmath::{Matrix3, MetricSpace, Point3, Vector2};
use serde::{Deserialize, Serialize};

use crate::camera::{focal2fov, PerspectiveCamera, PerspectiveProjection};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneCamera {
    pub img_name: String,
    pub id: u32,
    pub width: u32,
    pub height: u32,
    pub position: [f32; 3],
    pub rotation: [[f32; 3]; 3],
    pub fx: f32,
    pub fy: f32,
}

impl Into<PerspectiveCamera> for SceneCamera {
    fn into(self) -> PerspectiveCamera {
        let fovx = focal2fov(self.fx, self.width as f32);
        let fovy = focal2fov(self.fy, self.height as f32);
        PerspectiveCamera {
            position: self.position.into(),
            rotation: Matrix3::from(self.rotation).into(),
            projection: PerspectiveProjection::new(Vector2::new(fovx, fovy), 0.01, 100.),
        }
    }
}

#[derive(Debug)]
pub struct Scene {
    cameras: Vec<SceneCamera>,
}

impl Scene {
    pub fn from_json<P: AsRef<Path>>(file: P) -> Result<Self, anyhow::Error> {
        let f = File::open(file)?;
        let mut reader = BufReader::new(f);
        let mut cameras: Vec<SceneCamera> = serde_json::from_reader(&mut reader)?;
        cameras.sort_by_key(|c| c.img_name.clone());
        log::info!("loaded scene file with {} views", cameras.len());
        Ok(Scene { cameras })
    }

    pub fn camera(&self, i: usize) -> SceneCamera {
        self.cameras[i].clone()
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    pub fn cameras(&self) -> &Vec<SceneCamera> {
        &self.cameras
    }

    /// index of nearest camera
    pub fn nearest_camera(&self, pos: Point3<f32>) -> usize {
        self.cameras
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| (Point3::from(c.position).distance2(pos) * 1e6) as u32)
            .unwrap()
            .clone()
            .0
    }

    /// according to Kerbl et al "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    /// 7 out of 8 cameras are taken as training images
    pub fn train_cameras(&self) -> Vec<SceneCamera> {
        self.cameras
            .iter()
            .enumerate()
            .filter_map(|(i, c)| if i % 8 != 0 { Some(c.clone()) } else { None })
            .collect()
    }

    /// according to Kerbl et al "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    /// every 8th camera is used as test camera
    pub fn test_cameras(&self) -> Vec<SceneCamera> {
        self.cameras
            .iter()
            .enumerate()
            .filter_map(|(i, c)| if i % 8 == 0 { Some(c.clone()) } else { None })
            .collect()
    }
}
