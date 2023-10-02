use std::time::Duration;

pub trait Lerp {
    fn lerp(&self, other: &Self, amount: f32) -> Self;
}

pub struct Animation<T: Lerp> {
    from: T,
    to: T,
    time_left: Duration,
    duration: Duration,
}
impl<T: Lerp + Clone> Animation<T> {
    pub fn new(from: T, to: T, duration: Duration) -> Self {
        Self {
            from,
            to,
            time_left: duration,
            duration: duration,
        }
    }
    pub fn update(&mut self, dt: Duration) -> T {
        match self.time_left.checked_sub(dt) {
            Some(new_left) => {
                // set time left
                self.time_left = new_left;
                let elapsed = 1. - new_left.as_secs_f32() / self.duration.as_secs_f32();
                let amount = smoothstep(elapsed);
                let new_value = self.from.lerp(&self.to, amount);
                return new_value;
            }
            None => {
                self.time_left = Duration::ZERO;
                return self.to.clone();
            }
        }
    }

    pub fn done(&self) -> bool {
        self.time_left.is_zero()
    }
}

pub fn smoothstep(x: f32) -> f32 {
    return x * x * (3.0 - 2.0 * x);
}
