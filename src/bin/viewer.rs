use clap::Parser;
use std::{fs::File, path::PathBuf};
use web_splats::{open_window, RenderConfig};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    /// Scene json file
    scene: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    no_vsync: bool,

    /// Sky box image
    #[arg(long)]
    skybox: Option<PathBuf>,
}

#[pollster::main]
async fn main() {
    let opt = Opt::parse();

    let data_file = File::open(opt.input).unwrap();
    let scene_file = opt.scene.map(|p| File::open(p).unwrap());

    if opt.no_vsync {
        log::info!("V-sync disabled");
    }

    open_window(
        data_file,
        scene_file,
        RenderConfig {
            no_vsync: opt.no_vsync,
            skybox: opt.skybox,
        },
    )
    .await;
}
