// based on https://sotrh.github.io/learn-wgpu/intermediate/tutorial12-camera/#cleaning-up-lib-rs
// (camera controller code mostly)

use cgmath::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveCamera {
    pub position: Point3<f32>,
    pub rotation: Quaternion<f32>,
    pub projection: PerspectiveProjection,
}

impl PerspectiveCamera {
    pub fn new(
        position: Point3<f32>,
        rotation: Quaternion<f32>,
        projection: PerspectiveProjection,
    ) -> Self {
        PerspectiveCamera {
            position,
            rotation,
            projection: projection,
        }
    }

    pub fn lerp(&self, other: &PerspectiveCamera, amount: f32) -> PerspectiveCamera {
        PerspectiveCamera {
            position: Point3::from_vec(
                self.position.to_vec().lerp(other.position.to_vec(), amount),
            ),
            rotation: self.rotation.slerp(other.rotation, amount),
            projection: self.projection.lerp(&other.projection, amount),
        }
    }
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            position: Point3::new(0., 0., -1.),
            rotation: Quaternion::new(1., 0., 0., 0.),
            projection: PerspectiveProjection {
                fov: Vector2::new(Deg(45.).into(), Deg(45.).into()),
                znear: 0.1,
                zfar: 100.,
            },
        }
    }
}

impl Camera for PerspectiveCamera {
    fn view_matrix(&self) -> Matrix4<f32> {
        world2view(Matrix3::from(self.rotation), self.position.to_vec())
    }

    fn proj_matrix(&self) -> Matrix4<f32> {
        self.projection.projection_matrix()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveProjection {
    pub fov: Vector2<Rad<f32>>,
    pub znear: f32,
    pub zfar: f32,
}

/// see https://sotrh.github.io/learn-wgpu/intermediate/tutorial12-camera/#the-camera
#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

impl PerspectiveProjection {
    pub fn new<F: Into<Rad<f32>>>(fov: Vector2<F>, znear: f32, zfar: f32) -> Self {
        Self {
            fov: fov.map(|v| v.into()),
            znear,
            zfar,
        }
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        self.fov.x = self.fov.y * width as f32 / height as f32;
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        build_proj(self.znear, self.zfar, self.fov.x, self.fov.y)
    }

    pub(crate) fn focal(&self, viewport: Vector2<u32>) -> Vector2<f32> {
        let viewport: Vector2<f32> = viewport.cast().unwrap();
        return Vector2::new(
            fov2focal(self.fov.x, viewport.x),
            fov2focal(self.fov.y, viewport.y),
        );
    }

    pub fn lerp(&self, other: &PerspectiveProjection, amount: f32) -> PerspectiveProjection {
        PerspectiveProjection {
            fov: self
                .fov
                .map(|v| v.0)
                .lerp(other.fov.map(|v| v.0), amount)
                .map(|v| Rad(v)),
            znear: self.znear * (1. - amount) + other.znear * amount,
            zfar: self.zfar * (1. - amount) + other.zfar * amount,
        }
    }
}

pub trait Camera {
    fn view_matrix(&self) -> Matrix4<f32>;
    fn proj_matrix(&self) -> Matrix4<f32>;
}

pub fn world2view(r: Matrix3<f32>, t: Vector3<f32>) -> Matrix4<f32> {
    let mut rt = Matrix4::from(r);
    rt[0].w = t.x;
    rt[1].w = t.y;
    rt[2].w = t.z;
    rt[3].w = 1.;
    return rt.invert().unwrap().transpose();
}

pub fn build_proj(znear: f32, zfar: f32, fov_x: Rad<f32>, fov_y: Rad<f32>) -> Matrix4<f32> {
    let tan_half_fov_y = (fov_y / 2.).tan();
    let tan_half_fov_x = (fov_x / 2.).tan();

    let top = tan_half_fov_y * znear;
    let bottom = -top;
    let right = tan_half_fov_x * znear;
    let left = -right;

    let mut p = Matrix4::zero();

    let z_sign = 1.0;

    p[0][0] = 2.0 * znear / (right - left);
    p[1][1] = 2.0 * znear / (top - bottom);
    p[0][2] = (right + left) / (right - left);
    p[1][2] = (top + bottom) / (top - bottom);
    p[3][2] = z_sign;
    p[2][2] = z_sign * zfar / (zfar - znear);
    p[2][3] = -(zfar * znear) / (zfar - znear);
    return p.transpose();
}

pub fn focal2fov(focal: f32, pixels: f32) -> Rad<f32> {
    return Rad(2. * (pixels / (2. * focal)).atan());
}

pub fn fov2focal(fov: Rad<f32>, pixels: f32) -> f32 {
    pixels / (2. * (fov * 0.5).tan())
}
