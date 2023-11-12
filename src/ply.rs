#[cfg(target_arch = "wasm32")]
use instant::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use std::{
    io::{self},
    marker::PhantomData,
};

use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use cgmath::{InnerSpace, Point3, Quaternion, Vector3};
use log::info;

use crate::{
    pointcloud::GaussianSplatFloat,
    utils::{build_cov, sh_deg_from_num_coefs, sigmoid},
};

pub struct PlyReader<R> {
    header: ply_rs::ply::Header,
    reader: R,
    _data: PhantomData<R>,
}

impl<R: io::BufRead + io::Seek> PlyReader<R> {
    pub fn new(reader: R) -> Result<Self, anyhow::Error> {
        let mut reader = reader;
        let parser = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let header = parser.read_header(&mut reader).unwrap();
        Ok(Self {
            header,
            reader,
            _data: PhantomData::default(),
        })
    }

    fn read_line<B: ByteOrder>(&mut self, idx: usize, sh_deg: u32) -> GaussianSplatFloat {
        let mut pos = [0.; 3];
        self.reader.read_f32_into::<B>(&mut pos).unwrap();

        // skip normals
        // for what ever reason it is faster to call read than seek ...
        // so we just read them and never use them again
        let mut _normals = [0.; 3];
        self.reader.read_f32_into::<B>(&mut _normals).unwrap();

        let mut sh = [[0.; 3]; 16];
        self.reader.read_f32_into::<B>(&mut sh[0]).unwrap();
        let mut sh_rest = [0.; 15 * 3];
        self.reader.read_f32_into::<B>(&mut sh_rest).unwrap();

        // higher order coefficients are stored with channel first (shape:[N,3,C])
        for i in 0..15 {
            for j in 0..3 {
                sh[i + 1][j] = sh_rest[j * 15 + i];
            }
        }

        let opacity = sigmoid(self.reader.read_f32::<B>().unwrap());

        let scale_1 = self.reader.read_f32::<B>().unwrap().exp();
        let scale_2 = self.reader.read_f32::<B>().unwrap().exp();
        let scale_3 = self.reader.read_f32::<B>().unwrap().exp();
        let scale = Vector3::new(scale_1, scale_2, scale_3);

        let rot_0 = self.reader.read_f32::<B>().unwrap();
        let rot_1 = self.reader.read_f32::<B>().unwrap();
        let rot_2 = self.reader.read_f32::<B>().unwrap();
        let rot_3 = self.reader.read_f32::<B>().unwrap();
        let rot = Quaternion::new(rot_0, rot_1, rot_2, rot_3).normalize();

        let cov = build_cov(rot, scale);

        return GaussianSplatFloat {
            xyz: Point3::from(pos).cast().unwrap(),
            opacity,
            cov,
            sh,
            _pad: [0; 2],
        };
    }

    pub fn read(&mut self, sh_deg: u32) -> Result<Vec<GaussianSplatFloat>, anyhow::Error> {
        let start = Instant::now();
        let num_points = self.num_points()?;
        let vertices: Vec<GaussianSplatFloat> = match self.header.encoding {
            ply_rs::ply::Encoding::Ascii => todo!("acsii ply format not supported"),
            ply_rs::ply::Encoding::BinaryBigEndian => (0..num_points)
                .map(|i| self.read_line::<BigEndian>(i, sh_deg))
                .collect(),
            ply_rs::ply::Encoding::BinaryLittleEndian => (0..num_points)
                .map(|i| self.read_line::<LittleEndian>(i, sh_deg))
                .collect(),
        };
        info!(
            "reading ply file took {:}ms",
            (Instant::now() - start).as_millis()
        );
        return Ok(vertices);
    }

    pub fn file_sh_deg(&self) -> Result<u32, anyhow::Error> {
        let num_sh_coefs = self.header.elements["vertex"]
            .properties
            .keys()
            .filter(|k| k.starts_with("f_"))
            .count();

        let file_sh_deg = sh_deg_from_num_coefs(num_sh_coefs as u32 / 3).ok_or(anyhow::anyhow!(
            "number of sh coefficients {num_sh_coefs} cannot be mapped to sh degree"
        ))?;
        Ok(file_sh_deg)
    }

    pub fn num_points(&self) -> Result<usize, anyhow::Error> {
        Ok(self
            .header
            .elements
            .get("vertex")
            .ok_or(anyhow::anyhow!("missing element vertex"))?
            .count as usize)
    }
}
