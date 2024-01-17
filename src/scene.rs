use std::{
    collections::HashMap,
    hash::Hash,
    io::{self, BufReader},
};

use cgmath::{Matrix3, MetricSpace, Point3, Vector2};
use serde::{Deserialize, Serialize};

use crate::camera::{focal2fov, fov2focal, PerspectiveCamera, PerspectiveProjection};

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

impl std::hash::Hash for SceneCamera {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.img_name.hash(state);
        self.width.hash(state);
        self.height.hash(state);
        bytemuck::cast_slice::<_, u8>(&self.position).hash(state);
        bytemuck::cast_slice::<_, u8>(&self.rotation).hash(state);
        bytemuck::cast_slice::<_, u8>(&[self.fx, self.fy]).hash(state);
        self.split.hash(state);
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
        PerspectiveCamera {
            position: self.position.into(),
            rotation: Matrix3::from(self.rotation).into(),
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
    cameras: HashMap<usize, SceneCamera>,
}

impl Scene {
    pub fn from_cameras(cameras: Vec<SceneCamera>) -> Self {
        let mut map = HashMap::with_capacity(cameras.len());
        for c in cameras {
            let id = c.id;
            if map.insert(c.id, c).is_some() {
                log::warn!(
                    "duplicate camera id {:?} in scene (duplicates were removed)",
                    id,
                );
            }
        }
        Self { cameras: map }
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
        self.cameras.get(&i).cloned()
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    pub fn cameras(&self, split: Option<Split>) -> Vec<SceneCamera> {
        let mut c: Vec<SceneCamera> = if let Some(split) = split {
            self.cameras
                .iter()
                .filter_map(|(_, c)| (c.split == split).then_some(c.clone()))
                .collect()
        } else {
            self.cameras.iter().map(|(_, c)| c.clone()).collect()
        };
        c.sort_by_key(|c| c.id);
        return c;
    }

    /// index of nearest camera
    pub fn nearest_camera(&self, pos: Point3<f32>, split: Option<Split>) -> Option<usize> {
        self.cameras
            .iter()
            .filter_map(|(_, c)| match split {
                Some(s) => (s == c.split).then_some(c),
                None => Some(c),
            })
            .min_by_key(|c| (Point3::from(c.position).distance2(pos) * 1e6) as u32)
            .map(|c| c.id)
    }
}
