use std::{
    fs,
    io::{self, BufRead},
    path::Path,
};

use cgmath::Vector3;

pub fn load_color_schema<P: AsRef<Path>>(path: P) -> Result<Vec<Vector3<f32>>, anyhow::Error> {
    let f = fs::File::open(path.as_ref())?;
    let reader = io::BufReader::new(f);
    let mut colors = Vec::new();
    for line in reader.lines() {
        let parts: Vec<f32> = line?
            .split_whitespace()
            .map(|s| s.parse::<f32>())
            .collect::<Result<_, _>>()?;
        colors.push(Vector3::new(parts[0], parts[1], parts[2]));
    }
    Ok(colors)
}
