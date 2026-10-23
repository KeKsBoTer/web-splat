//! Runs the [`DepthSorter`] in the background: on a thread, or in a web worker on the web
//! (`public/sort_worker.js`, which runs the same sort compiled to wasm).
//!
//! Only one sort is in flight at a time. The caller asks for a new one when the camera moved,
//! while the sort of an older view is running the views are not queued.

#[cfg(not(target_arch = "wasm32"))]
use crate::sort::DepthSorter;

pub struct CpuSorter {
    backend: Backend,
    in_flight: bool,
}

impl CpuSorter {
    pub fn new() -> Self {
        Self {
            backend: Backend::new(),
            in_flight: false,
        }
    }

    /// Adds points (three values per point) to the points that are sorted.
    pub fn add_positions(&mut self, xyz: Vec<f32>) {
        self.backend.add_positions(xyz);
    }

    /// Starts sorting for the view (see [`crate::sort::DepthSorter::sort`]). Does nothing if a
    /// sort is still running, returns whether it started.
    pub fn start(&mut self, view: [f32; 4]) -> bool {
        if self.in_flight {
            return false;
        }
        self.backend.start(view);
        self.in_flight = true;
        true
    }

    /// the indices of the latest finished sort
    pub fn poll(&mut self) -> Option<Vec<u32>> {
        let result = self.backend.poll();
        if result.is_some() {
            self.in_flight = false;
        }
        result
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum Message {
    Positions(Vec<f32>),
    Sort([f32; 4]),
}

#[cfg(not(target_arch = "wasm32"))]
struct Backend {
    messages: std::sync::mpsc::Sender<Message>,
    results: std::sync::mpsc::Receiver<Vec<u32>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Backend {
    fn new() -> Self {
        let (messages, message_rx) = std::sync::mpsc::channel::<Message>();
        let (result_tx, results) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("splat sorter".to_string())
            .spawn(move || {
                let mut sorter = DepthSorter::new();
                // ends when the sender is dropped
                while let Ok(message) = message_rx.recv() {
                    let mut view = None;
                    let mut message = Some(message);
                    // positions are added in order, of several views only the latest is sorted
                    while let Some(m) = message.take().or_else(|| message_rx.try_recv().ok()) {
                        match m {
                            Message::Positions(p) => sorter.add_positions(&p),
                            Message::Sort(v) => view = Some(v),
                        }
                    }
                    if let Some(view) = view {
                        if result_tx.send(sorter.sort(view)).is_err() {
                            return;
                        }
                    }
                }
            })
            .expect("could not start the sort thread");
        Self { messages, results }
    }

    fn add_positions(&mut self, xyz: Vec<f32>) {
        self.messages.send(Message::Positions(xyz)).ok();
    }

    fn start(&mut self, view: [f32; 4]) {
        self.messages.send(Message::Sort(view)).ok();
    }

    fn poll(&mut self) -> Option<Vec<u32>> {
        // only the latest result is of interest
        self.results.try_iter().last()
    }
}

#[cfg(target_arch = "wasm32")]
struct Backend {
    worker: web_sys::Worker,
    result: std::rc::Rc<std::cell::RefCell<Option<Vec<u32>>>>,
    _on_message: wasm_bindgen::closure::Closure<dyn FnMut(web_sys::MessageEvent)>,
}

#[cfg(target_arch = "wasm32")]
impl Backend {
    fn new() -> Self {
        use wasm_bindgen::JsCast;

        let options = web_sys::WorkerOptions::new();
        options.set_type(web_sys::WorkerType::Module);
        let worker = web_sys::Worker::new_with_options("sort_worker.js", &options)
            .expect("could not start the sort worker");

        let result = std::rc::Rc::new(std::cell::RefCell::new(None));
        let on_message = {
            let result = result.clone();
            wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::MessageEvent)>::new(
                move |event: web_sys::MessageEvent| {
                    let indices = js_sys::Reflect::get(&event.data(), &"indices".into())
                        .ok()
                        .and_then(|v| v.dyn_into::<js_sys::Uint32Array>().ok());
                    if let Some(indices) = indices {
                        *result.borrow_mut() = Some(indices.to_vec());
                    }
                },
            )
        };
        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        Self {
            worker,
            result,
            _on_message: on_message,
        }
    }

    fn post(&self, kind: &str, data: &js_sys::Float32Array) {
        let message = js_sys::Object::new();
        js_sys::Reflect::set(&message, &"type".into(), &kind.into()).unwrap();
        js_sys::Reflect::set(&message, &"data".into(), data).unwrap();
        // the buffer is moved to the worker instead of copied
        let transfer = js_sys::Array::of1(&data.buffer());
        self.worker
            .post_message_with_transfer(&message, &transfer)
            .unwrap();
    }

    fn add_positions(&mut self, xyz: Vec<f32>) {
        self.post("positions", &js_sys::Float32Array::from(xyz.as_slice()));
    }

    fn start(&mut self, view: [f32; 4]) {
        self.post("sort", &js_sys::Float32Array::from(view.as_slice()));
    }

    fn poll(&mut self) -> Option<Vec<u32>> {
        self.result.borrow_mut().take()
    }
}
