use cgmath::*;
#[cfg(target_arch = "wasm32")]
use instant::Duration;
use num_traits::Float;
use std::f32::consts::PI;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

use winit::keyboard::KeyCode;

use crate::camera::PerspectiveCamera;

#[derive(Debug)]
pub struct CameraController {
    pub center: Point3<f32>,
    pub up: Option<Vector3<f32>>,
    amount: Vector3<f32>,
    shift: Vector2<f32>,
    rotation: Vector3<f32>,
    scroll: f32,
    pub speed: f32,
    pub sensitivity: f32,

    pub left_mouse_pressed: bool,
    pub right_mouse_pressed: bool,
    pub alt_pressed: bool,
    pub user_inptut: bool,

    pub touched_count: i32,
    pub last_touch: Vector3<f32>,//x,y,id
    pub last_touch_2: Vector3<f32>,//x,y,id
    pub touch_rotate_sensitivity_threshold: f32,
    pub touch_rotate_sensitivity: f32,
    pub touch_zoom_sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            center: Point3::origin(),
            amount: Vector3::zero(),
            shift: Vector2::zero(),
            rotation: Vector3::zero(),
            up: None,
            scroll: 0.0,
            speed,
            sensitivity,
            left_mouse_pressed: false,
            right_mouse_pressed: false,
            alt_pressed: false,
            user_inptut: false,
            touched_count: 0,
            last_touch: Vector3::zero(),
            last_touch_2: Vector3::zero(),
            touch_rotate_sensitivity_threshold: 3.0,
            touch_rotate_sensitivity: 2.0,
            touch_zoom_sensitivity: 0.01,
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, pressed: bool) -> bool {
        let amount = if pressed { 1.0 } else { 0.0 };
        let processed = match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount.z += amount;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount.z += -amount;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount.x += -amount;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount.x += amount;
                true
            }
            KeyCode::KeyQ => {
                self.rotation.z += amount / self.sensitivity;
                true
            }
            KeyCode::KeyE => {
                self.rotation.z += -amount / self.sensitivity;
                true
            }
            KeyCode::Space => {
                self.amount.y += amount;
                true
            }
            KeyCode::ShiftLeft => {
                self.amount.y += -amount;
                true
            }
            _ => false,
        };
        self.user_inptut = processed;
        return processed;
    }

    pub fn process_mouse(&mut self, mouse_dx: f32, mouse_dy: f32) {
        if self.left_mouse_pressed {
            self.rotation.x += mouse_dx as f32;
            self.rotation.y += mouse_dy as f32;
            self.user_inptut = true;
        }
        if self.right_mouse_pressed {
            self.shift.y += -mouse_dx as f32;
            self.shift.x += mouse_dy as f32;
            self.user_inptut = true;
        }
    }

    pub fn process_scroll(&mut self, dy: f32) {
        self.scroll += -dy;
        self.user_inptut = true;
    }

    pub fn process_touch(&mut self, touch_id: f32, touch_x: f32, touch_y: f32) {
        //println!("touched_count {}", self.touched_count);
        if self.touched_count == 1 {//rotate            
            let mut touch_dx = 0.0;
            if touch_x - self.last_touch.x < -self.touch_rotate_sensitivity_threshold {
                touch_dx = -1.0;
            }
            else if touch_x - self.last_touch.x > self.touch_rotate_sensitivity_threshold {
                touch_dx = 1.0;
            }
            let mut touch_dy = 0.0;
            if touch_y - self.last_touch.y < -self.touch_rotate_sensitivity_threshold {
                touch_dy = -1.0;
            }
            else if touch_y - self.last_touch.y > self.touch_rotate_sensitivity_threshold {
                touch_dy = 1.0;
            }
            self.rotation.x += touch_dx * self.touch_rotate_sensitivity;
            self.rotation.y += touch_dy * self.touch_rotate_sensitivity;
            self.user_inptut = true;
            //save last
            self.last_touch.x = touch_x;
            self.last_touch.y = touch_y;
            self.last_touch.z = touch_id;
        }
        else if self.touched_count == 2 {//zoom
            let last_distance = ((self.last_touch.x - self.last_touch_2.x) * (self.last_touch.x - self.last_touch_2.x) + (self.last_touch.y - self.last_touch_2.y) * (self.last_touch.y - self.last_touch_2.y)).sqrt();
            let mut cur_distance = 0.0;
            if self.last_touch.z == touch_id {//1st point
                cur_distance = ((self.last_touch_2.x - touch_x) * (self.last_touch_2.x - touch_x) + (self.last_touch_2.y - touch_y) * (self.last_touch_2.y - touch_y)).sqrt();
                //save last
                self.last_touch.x = touch_x;
                self.last_touch.y = touch_y;
            }
            else if self.last_touch_2.z == touch_id{//2nd point
                cur_distance = ((self.last_touch.x - touch_x) * (self.last_touch.x - touch_x) + (self.last_touch.y - touch_y) * (self.last_touch.y - touch_y)).sqrt();
                //save last
                self.last_touch_2.x = touch_x;
                self.last_touch_2.y = touch_y;
                }           
            let delta_distance = cur_distance - last_distance;
            if delta_distance > 0.0 {
                self.scroll -= self.touch_zoom_sensitivity;
                self.user_inptut = true;
            }
            else if delta_distance < 0.0 {
                self.scroll += self.touch_zoom_sensitivity;
                self.user_inptut = true;
            }
        }
    }

    /// moves the controller center to the closest point on a line defined by the camera position and rotation
    /// ajusts the controller up vector by projecting the current up vector onto the plane defined by the camera right vector
    pub fn reset_to_camera(&mut self, camera: PerspectiveCamera) {
        let inv_view = camera.rotation.invert();
        let forward = inv_view * Vector3::unit_z();
        let right = inv_view * Vector3::unit_x();

        // move center point
        self.center = closest_point(camera.position, forward, self.center);
        // adjust up vector by projecting it onto the plane defined by the right vector of the camera
        if let Some(up) = &self.up {
            let new_up = up - up.project_on(right);
            self.up = Some(new_up.normalize());
        }
    }

    pub fn update_camera(&mut self, camera: &mut PerspectiveCamera, dt: Duration) {
        let dt: f32 = dt.as_secs_f32();
        let mut dir = camera.position - self.center;
        let distance = dir.magnitude();

        dir = dir.normalize_to((distance.ln() + self.scroll * dt * 10. * self.speed).exp());

        let view_t: Matrix3<f32> = camera.rotation.invert().into();

        let x_axis = view_t.x;
        let y_axis = self.up.unwrap_or(view_t.y);
        let z_axis = view_t.z;

        let offset =
            (self.shift.y * x_axis - self.shift.x * y_axis) * dt * self.speed * 0.1 * distance;
        self.center += offset;
        camera.position += offset;
        let mut theta = Rad((self.rotation.x) * dt * self.sensitivity);
        let mut phi = Rad((-self.rotation.y) * dt * self.sensitivity);
        let mut eta = Rad::zero();

        if self.alt_pressed {
            eta = Rad(-self.rotation.y * dt * self.sensitivity);
            theta = Rad::zero();
            phi = Rad::zero();
        }

        let rot_theta = Quaternion::from_axis_angle(y_axis, theta);
        let rot_phi = Quaternion::from_axis_angle(x_axis, phi);
        let rot_eta = Quaternion::from_axis_angle(z_axis, eta);
        let rot = rot_theta * rot_phi * rot_eta;

        let mut new_dir = rot.rotate_vector(dir);

        if angle_short(y_axis, new_dir) < Rad(0.1) {
            new_dir = dir;
        }
        camera.position = self.center + new_dir;

        // update rotation
        // camera.rotation = (rot * camera.rotation.invert()).invert();
        camera.rotation = Quaternion::look_at(-new_dir, y_axis);

        // decay based on fps
        let mut decay = (0.8).powf(dt * 60.);
        if decay < 1e-4 {
            decay = 0.;
        }
        self.rotation *= decay;
        if self.rotation.magnitude() < 1e-4 {
            self.rotation = Vector3::zero();
        }
        self.shift *= decay;
        if self.shift.magnitude() < 1e-4 {
            self.shift = Vector2::zero();
        }
        self.scroll *= decay;
        if self.scroll.abs() < 1e-4 {
            self.scroll = 0.;
        }
        self.user_inptut = false;
    }
}

fn closest_point(orig: Point3<f32>, dir: Vector3<f32>, point: Point3<f32>) -> Point3<f32> {
    let dir = dir.normalize();
    let lhs = point - orig;

    let dot_p = lhs.dot(dir);
    // Return result
    return orig + dir * dot_p;
}

fn angle_short(a: Vector3<f32>, b: Vector3<f32>) -> Rad<f32> {
    let angle = a.angle(b);
    if angle > Rad(PI / 2.) {
        return Rad(PI) - angle;
    } else {
        return angle;
    }
}
