use std::{fs::File, path::PathBuf};
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

fn main() {
    let opt = Opt::from_args();

    let pc = File::open(opt.input).unwrap();
    let scene = opt.scene.map(|f| File::open(f).unwrap());

    pollster::block_on(open_window(pc, scene));
}
