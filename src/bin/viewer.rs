use clap::Parser;
use std::str::FromStr;
#[allow(unused_imports)]
use std::{fmt::Debug, fs::File, path::PathBuf};
use url::Url;
#[allow(unused_imports)]
use web_splats::{open_window, RenderConfig};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: Url,

    /// Scene json file
    scene: Option<Url>,

    #[arg(long, default_value_t = false)]
    no_vsync: bool,

    /// Support HDR rendering
    #[arg(long, default_value_t = false)]
    hdr: bool,
}

/// check if there is a scene file in the same directory or parent directory as the input file
#[allow(unused)]
fn try_find_scene_file(input: &Url, depth: u32) -> Option<Url> {
    let path = PathBuf::from_str(input.path()).unwrap();
    if let Some(parent) = path.parent() {
        let scene = parent.join("cameras.json");
        let mut new_url = input.clone();
        if scene.exists() {
            new_url.set_path(scene.to_str().unwrap());
            return Some(new_url);
        }
        if depth == 0 {
            return None;
        }
        new_url.set_path(parent.to_str().unwrap());
        return try_find_scene_file(&new_url, depth - 1);
    }
    return None;
}

#[cfg(not(target_arch = "wasm32"))]
#[pollster::main]
async fn main() {
    use web_splats::io;

    let mut opt = Opt::parse();

    if opt.scene.is_none() {
        opt.scene = try_find_scene_file(&opt.input,2);
        log::warn!("No scene file specified, using {:?}", opt.scene);
    }

    let data_file = io::read_from_url(&opt.input).await.unwrap();

    let scene_file = if let Some(a) = opt.scene.as_ref().map(|s| io::read_from_url(s)) {
        Some(a.await.unwrap())
    } else {
        None
    };

    if opt.no_vsync {
        log::info!("V-sync disabled");
    }

    open_window(
        data_file,
        scene_file,
        RenderConfig {
            no_vsync: opt.no_vsync,
            hdr: opt.hdr,
        },
        None, //Some(opt.input),
        None, //opt.scene,
    )
    .await;
}
#[cfg(target_arch = "wasm32")]
fn main() {
    todo!("not implemented")
}
