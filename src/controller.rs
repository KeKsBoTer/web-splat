use cgmath::*;
use std::time::Duration;
use winit::event::VirtualKeyCode;

use crate::camera::PerspectiveCamera;

#[derive(Debug)]
pub struct CameraController {
    amount: Vector3<f32>,
    shift: Vector2<f32>,
    rotate_horizontal: f32,
    rotate_vertical: f32,
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
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
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
            self.rotate_horizontal = mouse_dx as f32;
            self.rotate_vertical = mouse_dy as f32;
        }
        if self.right_mouse_pressed {
            self.shift.x = -mouse_dx as f32;
            self.shift.y = mouse_dy as f32;
        } else {
            self.shift = Vector2::zero();
        }
    }

    pub fn process_scroll(&mut self, dy: f32) {
        self.scroll = -dy;
    }

    pub fn update_camera(&mut self, camera: &mut PerspectiveCamera, dt: Duration) {
        let dt: f32 = dt.as_secs_f32();
        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        camera.position += forward * (self.amount.z) * self.speed * dt;
        camera.position += right * (self.amount.x) * self.speed * dt;

        // Move in/out (aka. "zoom")
        // Note: this isn't an actual zoom. The camera's position
        // changes when zooming. I've added this to make it easier
        // to get closer to an object you want to focus on.
        let (pitch_sin, pitch_cos) = camera.pitch.0.sin_cos();
        let scrollward =
            Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position -= scrollward * self.scroll * self.speed * self.sensitivity * dt;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        camera.position.y += (self.amount.y) * self.speed * dt;
        println!("{:?}", camera.position);

        // Rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;
        camera.yaw = camera.yaw.normalize();

        // reset
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;
        self.scroll = 0.0;
    }
}
