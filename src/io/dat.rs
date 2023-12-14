use std::{
    io::{self},
    marker::PhantomData,
};

use cgmath::{InnerSpace, Point3, Vector3, Quaternion};
use log::info;
use num_traits::One;

use crate::{
    pointcloud::GaussianSplatFloat,
    utils::{build_cov, sh_deg_from_num_coefs, sigmoid},
};

pub struct DatReader<R> {
    num_points: usize,
    reader: R,
}

impl<R> DatReader<R> {
    pub fn magic_bytes() -> &'static [u8] {
        "".as_bytes()
    }
}

impl<R: io::BufRead+ io::Seek> DatReader<R> {
    pub fn new(reader: R) -> Result<Self, anyhow::Error> {
        let mut reader = reader;
        reader.rewind()?;
        let mut first_line = String::new();
        reader.read_line(&mut first_line)?;
        let num_string = first_line.split(" ").next().unwrap();
        let num_points = num_string.parse::<usize>()?;
        Ok(Self {
            reader,
            num_points,
        })
    }

    fn read_line(&mut self) -> GaussianSplatFloat {
        let mut line = String::new();
        self.reader.read_line(&mut line).unwrap();

        let parts:Vec<f32> = line.trim().split(" ").map(|v|v.parse::<f32>().unwrap()).collect();
        let pos = Point3::new(parts[0], parts[1], parts[2]);
        let tension = parts[4];



        let mut sh = [[0.; 3]; 16];
        sh[0][0] = tension;
        

        let opacity = 0.5;

        let diameter = parts[3]/2.;
        let scale = Vector3::new(1.,1.,1.).normalize()*diameter;

        let rot = Quaternion::one();

        let cov = build_cov(rot, scale);


        return GaussianSplatFloat {
            xyz: pos,
            opacity,
            cov,
            sh,
            _pad: [0; 2],
        };
    }

    pub fn read(&mut self) -> Result<Vec<GaussianSplatFloat>, anyhow::Error> {
        let num_points = self.num_points()?;
        let vertices: Vec<GaussianSplatFloat> = (0..num_points)
            .map(|_| self.read_line())
            .collect();
        let max_v = vertices.iter().map(|v|((v.sh[0][0]*0.282+0.5)*10.) as u32).max().unwrap();
        println!("max_v: {}", max_v);
        return Ok(vertices);
    }

    pub fn file_sh_deg(&self) -> Result<u32, anyhow::Error> {
        Ok(0)
    }

    pub fn num_points(&self) -> Result<usize, anyhow::Error> {
        Ok(self.num_points)
    }
}
