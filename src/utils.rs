use winit::event::VirtualKeyCode;

pub fn key_to_num(key: VirtualKeyCode) -> Option<u32> {
    match key {
        VirtualKeyCode::Key0 => Some(0),
        VirtualKeyCode::Key1 => Some(1),
        VirtualKeyCode::Key2 => Some(2),
        VirtualKeyCode::Key3 => Some(3),
        VirtualKeyCode::Key4 => Some(4),
        VirtualKeyCode::Key5 => Some(5),
        VirtualKeyCode::Key6 => Some(6),
        VirtualKeyCode::Key7 => Some(7),
        VirtualKeyCode::Key8 => Some(8),
        VirtualKeyCode::Key9 => Some(9),
        _ => None,
    }
}
