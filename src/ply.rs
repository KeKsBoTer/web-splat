#[cfg(target_arch = "wasm32")]
use instant::{Duration, Instant};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use std::{
    io::{self},
    marker::PhantomData,
};

use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use cgmath::{InnerSpace, Point3, Quaternion, Vector3};
use log::info;

use crate::{
    pointcloud::{GaussianSplat, PointCloudReader},
    utils::{build_cov, sh_deg_from_num_coefs, sh_num_coefficients, sigmoid},
    SHDType,
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

    fn read_line<B: ByteOrder, W: io::Write>(
        &mut self,
        idx: u32,
        sh_deg: u32,
        sh_dtype: SHDType,
        sh_coefs_buffer: &mut W,
    ) -> GaussianSplat {
        let mut pos = [0.; 3];
        self.reader.read_f32_into::<B>(&mut pos).unwrap();

        // skip normals
        self.reader
            .seek(io::SeekFrom::Current(std::mem::size_of::<f32>() as i64 * 3))
            .unwrap();

        let mut sh_coefs_raw = [0.; 16 * 3];
        self.reader.read_f32_into::<B>(&mut sh_coefs_raw).unwrap();

        sh_dtype
            .write_to(sh_coefs_buffer, sh_coefs_raw[0], 0)
            .unwrap();
        sh_dtype
            .write_to(sh_coefs_buffer, sh_coefs_raw[1], 0)
            .unwrap();
        sh_dtype
            .write_to(sh_coefs_buffer, sh_coefs_raw[2], 0)
            .unwrap();

        // higher order coefficients are stored with channel first (shape:[N,3,C])
        for i in 1..sh_num_coefficients(sh_deg) {
            for j in 0..3 {
                sh_dtype
                    .write_to(
                        sh_coefs_buffer,
                        sh_coefs_raw[(2 + j * 15 + i) as usize],
                        i as u32,
                    )
                    .unwrap();
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
        let rot_q = Quaternion::new(rot_0, rot_1, rot_2, rot_3).normalize();

        return GaussianSplat {
            xyz: Point3::from(pos),
            opacity,
            covariance: build_cov(rot_q, scale),
            sh_idx: idx,
            ..Default::default()
        };
    }
}

impl<R: io::BufRead + io::Seek> PointCloudReader for PlyReader<R> {
    fn read(
        &mut self,
        sh_dtype: SHDType,
        sh_deg: u32,
    ) -> Result<(Vec<GaussianSplat>, Vec<u8>), anyhow::Error> {
        let start = Instant::now();
        let mut sh_coef_buffer = Vec::new();
        let num_points = self.num_points()?;
        let vertices: Vec<GaussianSplat> = match self.header.encoding {
            ply_rs::ply::Encoding::Ascii => todo!("acsii ply format not supported"),
            ply_rs::ply::Encoding::BinaryBigEndian => (0..num_points)
                .map(|i| {
                    self.read_line::<BigEndian, _>(i as u32, sh_deg, sh_dtype, &mut sh_coef_buffer)
                })
                .collect(),
            ply_rs::ply::Encoding::BinaryLittleEndian => (0..num_points)
                .map(|i| {
                    self.read_line::<LittleEndian, _>(
                        i as u32,
                        sh_deg,
                        sh_dtype,
                        &mut sh_coef_buffer,
                    )
                })
                .collect(),
        };
        info!(
            "reading ply file took {:}ms",
            (Instant::now() - start).as_millis()
        );
        return Ok((vertices, sh_coef_buffer));
    }

    fn file_sh_deg(&self) -> Result<u32, anyhow::Error> {
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

    fn num_points(&self) -> Result<usize, anyhow::Error> {
        Ok(self
            .header
            .elements
            .get("vertex")
            .ok_or(anyhow::anyhow!("missing element vertex"))?
            .count as usize)
    }
}
