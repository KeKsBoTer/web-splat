use clap::Parser;
use std::{fs::File, path::PathBuf};
use web_splats::{open_window, RenderConfig, SHDType, PCDataType};

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

    #[arg(long, default_value_t = true)]
    no_vsync: bool,
    
    /// select renderer, "rast" for rasterizer, "comp" for software rasterization via compute shader
    #[arg(long, default_value_t = String::from("rast"))]
    renderer: String,
    
    /// decide if the data should be used in compressed format for rendering
    #[arg(long, default_value_t = false)]
    use_compressed_data: bool,
    
    /// activate ui (ui slows rendering down for 50%)
    #[arg(long, default_value_t = true)]
    render_ui: bool,
}

#[pollster::main]
async fn main() {
    let opt = Opt::parse();

    // we dont need to close these
    // rust is smart enough to close/drop them once they are no longer needed
    // thank you rust <3
    let data_type = match opt.input.as_path().extension().unwrap().to_str().unwrap() {
        "ply" => PCDataType::PLY,
        #[cfg(feature="npz")]
        "npz" => PCDataType::NPZ,
        _ => panic!("Unknown data type for input file"),
    };
    let data_file = File::open(opt.input).unwrap();
    let scene_file = opt.scene.map(|p| File::open(p).unwrap());
    
    println!("Using renderer {}", opt.renderer);
    if opt.no_vsync {
        println!("V-sync disabled");
    }

    open_window(
        data_file,
        data_type,
        scene_file,
        RenderConfig {
            max_sh_deg: opt.max_sh_deg,
            sh_dtype: opt.sh_dtype,
            no_vsync: opt.no_vsync,
            renderer: opt.renderer,
            use_compressed_data: opt.use_compressed_data,
            render_ui: opt.render_ui,
        },
    )
    .await;
}
