// based on https://sotrh.github.io/learn-wgpu/intermediate/tutorial12-camera/#cleaning-up-lib-rs
// (camera controller code mostly)

use cgmath::*;

#[derive(Debug, Clone, Copy)]
pub struct PerspectiveCamera {
    pub position: Point3<f32>,
    pub yaw: Rad<f32>,
    pub pitch: Rad<f32>,
    pub projection: PerspectiveProjection,
}

impl PerspectiveCamera {
    pub fn new(
        pos: Point3<f32>,
        yaw: Rad<f32>,
        pitch: Rad<f32>,
        projection: PerspectiveProjection,
    ) -> Self {
        PerspectiveCamera {
            position: pos,
            yaw: yaw,
            pitch: pitch,
            projection: projection,
        }
    }
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            position: Point3::new(0., 0., -1.),
            yaw: Rad::zero(),
            pitch: Rad::zero(),
            projection: PerspectiveProjection {
                aspect: 1.,
                fovy: Deg(45.).into(),
                znear: 0.1,
                zfar: 100.,
            },
        }
    }
}

impl Camera for PerspectiveCamera {
    fn view_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.position,
            Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
            Vector3::unit_y(),
        )
    }

    fn proj_matrix(&self) -> Matrix4<f32> {
        self.projection.projection_matrix()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PerspectiveProjection {
    pub aspect: f32,
    pub fovy: Rad<f32>,
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
    pub fn new<F: Into<Rad<f32>>>(aspect: f32, fovy: F, znear: f32, zfar: f32) -> Self {
        Self {
            aspect,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SimpleCamera {
    pub view: Matrix4<f32>,
    pub projection: Matrix4<f32>,
}
impl SimpleCamera {
    pub fn new(view: Matrix4<f32>, projection: Matrix4<f32>) -> Self {
        Self { view, projection }
    }
}

impl Camera for SimpleCamera {
    fn view_matrix(&self) -> Matrix4<f32> {
        // Matrix4::from(self.rot) * Matrix4::from_translation(self.pos.to_vec())
        self.view
    }

    fn proj_matrix(&self) -> Matrix4<f32> {
        self.projection
    }
}

pub trait Camera {
    fn view_matrix(&self) -> Matrix4<f32>;
    fn proj_matrix(&self) -> Matrix4<f32>;
}

pub fn world2view(r: Matrix3<f32>, t: Vector3<f32>) -> Matrix4<f32> {
    let mut rt = Matrix4::from(r.transpose());
    rt[3] = Vector4::new(t.x, t.y, t.z, 1.);
    return rt;
}

pub fn build_proj(znear: f32, zfar: f32, fov_x: f32, fov_y: f32) -> Matrix4<f32> {
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
