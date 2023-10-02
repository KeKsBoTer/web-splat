use std::{fs::File, io::BufReader, path::Path};

use cgmath::{Matrix3, Vector2};
use serde::{Deserialize, Serialize};

use crate::camera::{focal2fov, PerspectiveCamera, PerspectiveProjection};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SceneCamera {
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
        cameras.sort_by_key(|c| c.id);
        Ok(Scene { cameras })
    }

    pub fn camera(&self, i: usize) -> SceneCamera {
        self.cameras[i].into()
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }
}
