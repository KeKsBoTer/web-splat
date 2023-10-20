use clap::Parser;
use std::{fs::File, path::PathBuf};
use web_splats::{open_window, RenderConfig, SHDType};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Opt {
    /// Input file
    input: PathBuf,

    /// Scene json file
    scene: Option<PathBuf>,

    /// maximum allowed Spherical Harmonics (SH) degree
    #[arg(long, default_value_t = 3)]
    max_sh_deg: u32,

    /// datatype used for SH coefficients
    #[arg(long,value_enum, default_value_t = SHDType::Byte)]
    sh_dtype: SHDType,

    #[arg(long)]
    no_vsync: bool,
}

#[pollster::main]
async fn main() {
    let opt = Opt::parse();

    // we dont need to close these
    // rust is smart enough to close/drop them once they are no longer needed
    // thank you rust <3
    let ply_file = File::open(opt.input).unwrap();
    let scene_file = opt.scene.map(|p| File::open(p).unwrap());

    open_window(
        ply_file,
        scene_file,
        RenderConfig {
            max_sh_deg: opt.max_sh_deg,
            sh_dtype: opt.sh_dtype,
            no_vsync: opt.no_vsync,
        },
    )
    .await;
}
