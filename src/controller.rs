use cgmath::*;
use instant::Duration;
use winit::event::VirtualKeyCode;

use crate::camera::{Camera, PerspectiveCamera};

#[derive(Debug)]
pub struct CameraController {
    amount: Vector3<f32>,
    shift: Vector2<f32>,
    rotation: Vector3<f32>,
    scroll: f32,
    pub speed: f32,
    pub sensitivity: f32,

    pub left_mouse_pressed: bool,
    pub right_mouse_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount: Vector3::zero(),
            shift: Vector2::zero(),
            rotation: Vector3::zero(),
            scroll: 0.0,
            speed,
            sensitivity,
            left_mouse_pressed: false,
            right_mouse_pressed: false,
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, pressed: bool) -> bool {
        let amount = if pressed { 1.0 } else { 0.0 };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount.z = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount.z = -amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount.x = -amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.amount.x = amount;
                true
            }
            VirtualKeyCode::Q => {
                self.rotation.z = amount / self.sensitivity;
                true
            }
            VirtualKeyCode::E => {
                self.rotation.z = -amount / self.sensitivity;
                true
            }
            VirtualKeyCode::Space => {
                self.amount.y = amount;
                true
            }
            VirtualKeyCode::LShift => {
                self.amount.y = -amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f32, mouse_dy: f32) {
        if self.left_mouse_pressed {
            self.rotation.y = mouse_dx as f32;
            self.rotation.x = mouse_dy as f32;
        }
        if self.right_mouse_pressed {
            self.shift.y = -mouse_dx as f32;
            self.shift.x = mouse_dy as f32;
        } else {
            self.shift = Vector2::zero();
        }
    }

    pub fn process_scroll(&mut self, dy: f32) {
        self.scroll = -dy;
    }

    pub fn update_camera(&mut self, camera: &mut PerspectiveCamera, dt: Duration) {
        let dt: f32 = dt.as_secs_f32();

        let inv_view = camera.view_matrix().inverse_transform().unwrap();

        let x_axis = inv_view.transform_vector(Vector3::new(1., 0., 0.));
        let y_axis = inv_view.transform_vector(Vector3::new(0., 1., 0.));
        let z_axis = inv_view.transform_vector(Vector3::new(0., 0., 1.));
        camera.position += z_axis * (self.amount.z) * self.speed * dt;
        camera.position += x_axis * (self.amount.x) * self.speed * dt;
        camera.position -= y_axis * (self.amount.y) * self.speed * dt;

        // zoom / scroll
        camera.position -= z_axis * self.scroll * self.speed * self.sensitivity * dt;

        // Rotate camera according to camera main axes
        let rot_y =
            Quaternion::from_axis_angle(y_axis, Rad(self.rotation.y * self.sensitivity * dt));
        let rot_x =
            Quaternion::from_axis_angle(x_axis, -Rad(self.rotation.x * self.sensitivity * dt));
        let rot_z =
            Quaternion::from_axis_angle(z_axis, -Rad(self.rotation.z * self.sensitivity * dt));
        camera.rotation = (camera.rotation * rot_x * rot_y * rot_z).normalize();

        // reset
        self.rotation = <_>::zero();
        self.scroll = 0.0;
    }
}
