use clap::Parser;
use std::{fs::File, path::PathBuf};
use web_splats::{open_window, PCDataType, RenderConfig};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    /// Scene json file
    scene: Option<PathBuf>,

    #[arg(long, default_value_t = true)]
    no_vsync: bool,
}

#[pollster::main]
async fn main() {
    let opt = Opt::parse();

    // we dont need to close these
    // rust is smart enough to close/drop them once they are no longer needed
    // thank you rust <3
    let data_type = match opt.input.as_path().extension().unwrap().to_str().unwrap() {
        "ply" => PCDataType::PLY,
        #[cfg(feature = "npz")]
        "npz" => PCDataType::NPZ,
        _ => panic!("Unknown data type for input file"),
    };
    let data_file = File::open(opt.input).unwrap();
    let scene_file = opt.scene.map(|p| File::open(p).unwrap());

    if opt.no_vsync {
        println!("V-sync disabled");
    }

    open_window(
        data_file,
        data_type,
        scene_file,
        RenderConfig {
            no_vsync: opt.no_vsync,
        },
    )
    .await;
}
