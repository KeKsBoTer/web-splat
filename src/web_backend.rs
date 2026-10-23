//! Choosing the graphics backend on the web with the `backend` parameter of the page url:
//! `?backend=webgpu` or `?backend=webgl`. Without it WebGPU is used if the browser has it and
//! WebGL2 otherwise.

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum WebBackend {
    WebGpu,
    WebGl,
}

impl WebBackend {
    fn name(&self) -> &'static str {
        match self {
            WebBackend::WebGpu => "webgpu",
            WebBackend::WebGl => "webgl",
        }
    }
}

const PARAMETER: &str = "backend";

/// the backend that is forced by the url, if any
pub fn requested_backend() -> Option<WebBackend> {
    let search = web_sys::window()?.location().search().ok()?;
    search
        .trim_start_matches('?')
        .split('&')
        .find_map(|p| p.strip_prefix("backend="))
        .and_then(|value| match value.to_lowercase().as_str() {
            "webgpu" => Some(WebBackend::WebGpu),
            "webgl" | "webgl2" => Some(WebBackend::WebGl),
            _ => None,
        })
}

/// Reloads the page with the backend forced to `backend`, all other parameters are kept.
pub fn switch_backend(backend: WebBackend) {
    let Some(location) = web_sys::window().map(|w| w.location()) else {
        return;
    };
    let search = location.search().unwrap_or_default();
    let mut params: Vec<&str> = search
        .trim_start_matches('?')
        .split('&')
        .filter(|p| !p.is_empty() && !p.starts_with(&format!("{PARAMETER}=")))
        .collect();
    let new_parameter = format!("{PARAMETER}={}", backend.name());
    params.push(&new_parameter);
    // changing the search navigates to the new url
    location.set_search(&params.join("&")).ok();
}
