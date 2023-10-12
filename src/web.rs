#[wasm_bindgen]
pub fn run_wasm(pc: Vec<u8>, scene: Option<Vec<u8>>) {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    let pc_reader = Cursor::new(pc);
    let scene_reader = scene.map(|d: Vec<u8>| Cursor::new(d));

    wasm_bindgen_futures::spawn_local(open_window(pc_reader, scene_reader));
}
