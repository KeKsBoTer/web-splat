use std::path::PathBuf;
use structopt::StructOpt;
use web_splats::open_window;

#[derive(Debug, StructOpt)]
#[structopt(name = "viewer", about = "3D gaussian splats viewer")]
struct Opt {
    /// Input file
    #[structopt(parse(from_os_str))]
    input: PathBuf,

    /// Scene json file
    #[structopt(parse(from_os_str))]
    scene: Option<PathBuf>,
}

#[pollster::main]
async fn main() {
    let opt = Opt::from_args();

    open_window(opt.input, opt.scene).await;
}
