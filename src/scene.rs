use std::{fs::File, io::BufReader, path::Path};

use cgmath::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

use crate::camera::{build_proj, world2view, SimpleCamera};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SceneCamera {
    id: u32,
    width: u32,
    height: u32,
    position: [f32; 3],
    rotation: [[f32; 3]; 3],
    fx: f32,
    fy: f32,
}

impl Into<SimpleCamera> for SceneCamera {
    fn into(self) -> SimpleCamera {
        let pos: Vector3<f32> = self.position.into();
        let rot: Matrix3<f32> = self.rotation.into();
        let view_matrix = world2view(rot, pos);
        let fovx = focal2fov(self.fx, self.width as f32);
        let fovy = focal2fov(self.fy, self.height as f32);
        let proj_matrix = build_proj(0.01, 100., fovx, fovy);
        SimpleCamera::new(view_matrix, proj_matrix)
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
        let cameras: Vec<SceneCamera> = serde_json::from_reader(&mut reader)?;
        Ok(Scene { cameras })
    }

    pub fn camera(&self, i: usize) -> SimpleCamera {
        self.cameras[i].into()
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }
}

fn focal2fov(focal: f32, pixels: f32) -> f32 {
    return 2. * (pixels / (2. * focal)).atan();
}
