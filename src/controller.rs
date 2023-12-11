use cgmath::*;
#[cfg(target_arch = "wasm32")]
use instant::Duration;
use num_traits::Float;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use winit::event::VirtualKeyCode;

use crate::camera::{Camera, PerspectiveCamera};

#[derive(Debug)]
pub struct CameraController {
    pub center: Point3<f32>,
    amount: Vector3<f32>,
    shift: Vector2<f32>,
    rotation: Vector3<f32>,
    scroll: f32,
    pub speed: f32,
    pub sensitivity: f32,

    pub left_mouse_pressed: bool,
    pub right_mouse_pressed: bool,
    pub alt_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            center: Point3::origin(),
            amount: Vector3::zero(),
            shift: Vector2::zero(),
            rotation: Vector3::zero(),
            scroll: 0.0,
            speed,
            sensitivity,
            left_mouse_pressed: false,
            right_mouse_pressed: false,
            alt_pressed: false,
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, pressed: bool) -> bool {
        let amount = if pressed { 1.0 } else { 0.0 };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount.z += amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount.z += -amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount.x += -amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.amount.x += amount;
                true
            }
            VirtualKeyCode::Q => {
                self.rotation.z += amount / self.sensitivity;
                true
            }
            VirtualKeyCode::E => {
                self.rotation.z += -amount / self.sensitivity;
                true
            }
            VirtualKeyCode::Space => {
                self.amount.y += amount;
                true
            }
            VirtualKeyCode::LShift => {
                self.amount.y += -amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f32, mouse_dy: f32) {
        if self.left_mouse_pressed {
            self.rotation.x += mouse_dx as f32;
            self.rotation.y += mouse_dy as f32;
        }
        if self.right_mouse_pressed {
            self.shift.y += -mouse_dx as f32;
            self.shift.x += mouse_dy as f32;
        } else {
            self.shift += Vector2::zero();
        }
    }

    pub fn process_scroll(&mut self, dy: f32) {
        self.scroll += -dy;
    }

    pub fn reset_to_camera(&mut self, camera: PerspectiveCamera) {
        let inv_view = camera.view_matrix().inverse_transform().unwrap();
        let forward = inv_view.z.truncate();
        self.center = closest_point(camera.position, forward, self.center);
    }

    pub fn update_camera(&mut self, camera: &mut PerspectiveCamera, dt: Duration) {
        let dt: f32 = dt.as_secs_f32();

        let mut dir = camera.position - self.center;
        let distance = dir.magnitude();

        dir = dir.normalize_to((distance.ln() + self.scroll * dt * 10. * self.speed).exp());

        let inv_view = camera.view_matrix().inverse_transform().unwrap();

        let x_axis = inv_view.x.truncate();
        let y_axis = inv_view.y.truncate();
        let z_axis = inv_view.z.truncate();

        let offset =
            (self.shift.y * x_axis - self.shift.x * y_axis) * dt * self.speed * 0.1 * distance;
        self.center += offset;
        camera.position += offset;

        let mut theta = Rad((-self.rotation.x) * dt * self.sensitivity);
        let mut phi = Rad((-self.rotation.y) * dt * self.sensitivity);
        let mut eta = Rad::zero();

        if self.alt_pressed {
            eta = Rad(-self.rotation.y * dt * self.sensitivity);
            theta = Rad::zero();
            phi = Rad::zero();
        }

        let rot_theta = Quaternion::from_axis_angle(y_axis, -theta);
        let rot_phi = Quaternion::from_axis_angle(x_axis, phi);
        let rot_eta = Quaternion::from_axis_angle(z_axis, eta);
        let rot = rot_theta * rot_phi * rot_eta;

        let up = rot.rotate_vector(y_axis);
        let new_dir: Vector3<f32> = rot.rotate_vector(dir);
        camera.position = self.center + new_dir;

        camera.rotation = Quaternion::look_at(-new_dir, up);

        // decay based on fps
        let decay = (0.8).powf(dt * 60.);
        self.rotation *= decay;
        self.shift *= decay;
        self.scroll *= decay;
    }
}

fn closest_point(orig: Point3<f32>, dir: Vector3<f32>, point: Point3<f32>) -> Point3<f32> {
    let dir = dir.normalize();
    let lhs = point - orig;

    let dot_p = lhs.dot(dir);
    // Return result
    return orig + dir * dot_p;
}
