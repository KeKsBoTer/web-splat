use clap::Parser;
use std::path::PathBuf;
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

    open_window(
        opt.input,
        opt.scene,
        RenderConfig {
            max_sh_deg: opt.max_sh_deg,
            sh_dtype: opt.sh_dtype,
            no_vsync: opt.no_vsync,
        },
    )
    .await;
}
