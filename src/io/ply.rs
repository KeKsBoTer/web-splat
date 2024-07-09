use anyhow::Ok;
use half::f16;
use ply_rs::ply;

use std::io::{self, BufReader, Read, Seek};

use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use cgmath::{InnerSpace, Point3, Quaternion, Vector3};

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
        // let sh_deg = Self::file_sh_deg(&header)?;
        let sh_deg = 0;
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


        let trbf_center = self.reader.read_f32::<B>()?;
        let trbf_scale = self.reader.read_f32::<B>()?.exp();
        // skip normals
        // for what ever reason it is faster to call read than seek ...
        // so we just read them and never use them again
        let mut _normals = [0.; 3];
        self.reader.read_f32_into::<B>(&mut _normals)?;

        let mut motion = [0.;10];
        self.reader.read_f32_into::<B>(&mut motion[..9])?;

        let mut color = [0.; 6];
        self.reader.read_f32_into::<B>(&mut color)?;

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

        let mut omega = [0.;4];
        self.reader.read_f32_into::<B>(&mut omega)?;

        let mut f_t = [0.; 3];
        self.reader.read_f32_into::<B>(&mut f_t)?;

        let cov = build_cov(rot, scale);


        let mut color_sh = [[0.;3];16];
        color_sh[0] = [color[0], color[1], color[2]];
        color_sh[1] = [color[3], color[4], color[5]];
        color_sh[2] = f_t;
        return Ok((
            Gaussian {
                xyz: Point3::from(pos).cast().unwrap(),
                opacity: f16::from_f32(opacity),
                cov: cov.map(|x| f16::from_f32(x)),
                trbf: [f16::from_f32(trbf_center), f16::from_f32(trbf_scale)],
                motion: motion.map(|x| f16::from_f32(x)),
                omega: omega.map(|x| f16::from_f32(x)),
            },
            color_sh.map(|x| x.map(|y| f16::from_f32(y))),
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
