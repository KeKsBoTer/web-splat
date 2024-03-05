use anyhow::Ok;
use half::f16;
#[cfg(target_arch = "wasm32")]
use instant::Instant;
use ply_rs::ply;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use std::io::{self, BufReader, Read, Seek};

use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use cgmath::{InnerSpace, Point3, Quaternion, Vector3};
use log::info;

use crate::{
    pointcloud::Gaussian,
    utils::{build_cov, sh_deg_from_num_coefs, sigmoid},
};

use super::{GenericGaussianPointCloud, PointCloudReader};

pub struct PlyReader<R: Read + Seek> {
    header: ply_rs::ply::Header,
    reader: BufReader<R>,
    sh_deg: u32,
    num_points: usize,
    mip_splatting: Option<bool>,
    kernel_size: Option<f32>,
    background_color: Option<[f32; 3]>,
}

impl<R: io::Read + io::Seek> PlyReader<R> {
    pub fn new(reader: R) -> Result<Self, anyhow::Error> {
        let mut reader = BufReader::new(reader);
        let parser = ply_rs::parser::Parser::<ply_rs::ply::DefaultElement>::new();
        let header = parser.read_header(&mut reader).unwrap();
        let sh_deg = Self::file_sh_deg(&header)?;
        let num_points = Self::num_points(&header)?;
        let mip_splatting = Self::mip_splatting(&header)?;
        let kernel_size = Self::kernel_size(&header)?;
        let background_color = Self::background_color(&header)
            .map_err(|e| log::warn!("could not parse background_color: {}", e))
            .unwrap_or_default();
        Ok(Self {
            header,
            reader,
            sh_deg,
            num_points,
            mip_splatting,
            kernel_size,
            background_color,
        })
    }

    fn read_line<B: ByteOrder>(
        &mut self,
        sh_deg: usize,
    ) -> anyhow::Result<(Gaussian, [[f16; 3]; 16])> {
        let mut pos = [0.; 3];
        self.reader.read_f32_into::<B>(&mut pos)?;

        // skip normals
        // for what ever reason it is faster to call read than seek ...
        // so we just read them and never use them again
        let mut _normals = [0.; 3];
        self.reader.read_f32_into::<B>(&mut _normals)?;

        let mut sh: [[f32; 3]; 16] = [[0.; 3]; 16];
        self.reader.read_f32_into::<B>(&mut sh[0])?;
        let mut sh_rest = [0.; 15 * 3];
        let num_coefs = (sh_deg + 1) * (sh_deg + 1);
        self.reader
            .read_f32_into::<B>(&mut sh_rest[..(num_coefs - 1) * 3])?;

        // higher order coefficients are stored with channel first (shape:[N,3,C])
        for i in 0..(num_coefs - 1) {
            for j in 0..3 {
                sh[i + 1][j] = sh_rest[j * (num_coefs - 1) + i];
            }
        }

        let opacity = sigmoid(self.reader.read_f32::<B>()?);

        let scale_1 = self.reader.read_f32::<B>()?.exp();
        let scale_2 = self.reader.read_f32::<B>()?.exp();
        let scale_3 = self.reader.read_f32::<B>()?.exp();
        let scale = Vector3::new(scale_1, scale_2, scale_3);

        let rot_0 = self.reader.read_f32::<B>()?;
        let rot_1 = self.reader.read_f32::<B>()?;
        let rot_2 = self.reader.read_f32::<B>()?;
        let rot_3 = self.reader.read_f32::<B>()?;
        let rot = Quaternion::new(rot_0, rot_1, rot_2, rot_3).normalize();

        let cov = build_cov(rot, scale);

        return Ok((
            Gaussian {
                xyz: Point3::from(pos).cast().unwrap(),
                opacity: f16::from_f32(opacity),
                cov: cov.map(|x| f16::from_f32(x)),
            },
            sh.map(|x| x.map(|y| f16::from_f32(y))),
        ));
    }

    fn file_sh_deg(header: &ply::Header) -> Result<u32, anyhow::Error> {
        let num_sh_coefs = header.elements["vertex"]
            .properties
            .keys()
            .filter(|k| k.starts_with("f_"))
            .count();

        let file_sh_deg = sh_deg_from_num_coefs(num_sh_coefs as u32 / 3).ok_or(anyhow::anyhow!(
            "number of sh coefficients {num_sh_coefs} cannot be mapped to sh degree"
        ))?;
        Ok(file_sh_deg)
    }

    fn num_points(header: &ply::Header) -> Result<usize, anyhow::Error> {
        Ok(header
            .elements
            .get("vertex")
            .ok_or(anyhow::anyhow!("missing element vertex"))?
            .count as usize)
    }

    fn mip_splatting(header: &ply::Header) -> Result<Option<bool>, anyhow::Error> {
        Ok(header
            .comments
            .iter()
            .find(|c| c.contains("mip"))
            .map(|c| c.split('=').last().unwrap().parse::<bool>())
            .transpose()?)
    }
    fn kernel_size(header: &ply::Header) -> Result<Option<f32>, anyhow::Error> {
        Ok(header
            .comments
            .iter()
            .find(|c| c.contains("kernel_size"))
            .map(|c| c.split('=').last().unwrap().parse::<f32>())
            .transpose()?)
    }

    fn background_color(header: &ply::Header) -> anyhow::Result<Option<[f32; 3]>> {
        header
            .comments
            .iter()
            .find(|c| c.contains("background_color"))
            .map(|c| {
                let value = c.split('=').last();
                let parts = value.map(|c| {
                    c.split(",")
                        .map(|v| v.parse::<f32>())
                        .collect::<Result<Vec<f32>, _>>()
                });
                parts.map_or_else(
                    || Err(anyhow::anyhow!("could not parse:")),
                    |x| {
                        x.map_err(|e| anyhow::anyhow!("could not parse: {}", e))
                            .map(|x| [x[0], x[1], x[2]])
                    },
                )
            })
            .transpose()
    }
}

impl<R: io::Read + io::Seek> PointCloudReader for PlyReader<R> {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error> {
        let start = Instant::now();
        let mut gaussians = Vec::with_capacity(self.num_points);
        let mut sh_coefs = Vec::with_capacity(self.num_points);
        match self.header.encoding {
            ply_rs::ply::Encoding::Ascii => todo!("acsii ply format not supported"),
            ply_rs::ply::Encoding::BinaryBigEndian => {
                for _ in 0..self.num_points {
                    let (g, s) = self.read_line::<BigEndian>(self.sh_deg as usize)?;
                    gaussians.push(g);
                    sh_coefs.push(s);
                }
            }
            ply_rs::ply::Encoding::BinaryLittleEndian => {
                for _ in 0..self.num_points {
                    let (g, s) = self.read_line::<LittleEndian>(self.sh_deg as usize)?;
                    gaussians.push(g);
                    sh_coefs.push(s);
                }
            }
        };
        return Ok(GenericGaussianPointCloud::new(
            gaussians,
            sh_coefs,
            self.sh_deg,
            self.num_points,
            self.kernel_size,
            self.mip_splatting,
            self.background_color,
            None,
            None,
        ));
    }

    fn magic_bytes() -> &'static [u8] {
        "ply".as_bytes()
    }

    fn file_ending() -> &'static str {
        "ply"
    }
}
