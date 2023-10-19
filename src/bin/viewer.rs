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
    
    /// select renderer, "rast" for rasterizer, "comp" for software rasterization via compute shader
    #[arg(long, default_value_t = String::from("comp"))]
    renderer: String,
}

#[pollster::main]
async fn main() {
    let opt = Opt::parse();

    // TODO this is suboptimal as it is never closed
    let ply_file = File::open(opt.input).unwrap();
    let scene_file = opt.scene.map(|p| File::open(p).unwrap());
    
    println!("Using renderer {}", opt.renderer);

    open_window(
        ply_file,
        scene_file,
        RenderConfig {
            max_sh_deg: opt.max_sh_deg,
            sh_dtype: opt.sh_dtype,
            no_vsync: opt.no_vsync,
            renderer: opt.renderer,
        },
    )
    .await;
}
