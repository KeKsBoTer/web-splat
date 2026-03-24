use half::f16;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Quaternion, Vector3};

use crate::{
    pointcloud::{Aabb, Gaussian, PointCloudMetadata},
    utils::{build_cov, sh_deg_from_num_coefs, sigmoid},
};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use serde_ply::RowVisitor;

pub struct PlyReader{
    read_points: usize,
    parser: serde_ply::PlyChunkedReader,
}


impl PlyReader {


    pub fn new() -> Self {
        let parser = serde_ply::PlyChunkedReader::new();
        Self {
            parser,
            read_points: 0,
        }
    }

    fn metadata(header: &serde_ply::PlyHeader) -> PointCloudMetadata {
        PointCloudMetadata {
            num_points: Self::num_points(header).unwrap_or(0),
            sh_deg: Self::file_sh_deg(header).unwrap_or(0),
            center: Point3::origin(),
            up: None,
            mip_splatting: Self::mip_splatting(header).unwrap_or(None),
            kernel_size: Self::kernel_size(header).unwrap_or(None),
            background_color: Self::background_color(header).unwrap_or(None),
            compressed: false,
        }
    }

    fn file_sh_deg(header: &serde_ply::PlyHeader) -> Result<u32, anyhow::Error> {
        let num_sh_coefs = header
            .get_element("vertex")
            .unwrap()
            .properties
            .iter()
            .filter(|k| k.name.starts_with("f_"))
            .count();

        let file_sh_deg = sh_deg_from_num_coefs(num_sh_coefs as u32 / 3).ok_or(anyhow::anyhow!(
            "number of sh coefficients {num_sh_coefs} cannot be mapped to sh degree"
        ))?;
        Ok(file_sh_deg)
    }

    fn num_points(header: &serde_ply::PlyHeader) -> Result<usize, anyhow::Error> {
        Ok(header
            .get_element("vertex")
            .ok_or(anyhow::anyhow!("missing element vertex"))?
            .count as usize)
    }

    fn mip_splatting(header: &serde_ply::PlyHeader) -> Result<Option<bool>, anyhow::Error> {
        Ok(header
            .comments
            .iter()
            .find(|c| c.contains("mip"))
            .map(|c| c.split('=').last().unwrap().to_lowercase().parse::<bool>())
            .transpose()?)
    }
    fn kernel_size(header: &serde_ply::PlyHeader) -> Result<Option<f32>, anyhow::Error> {
        Ok(header
            .comments
            .iter()
            .find(|c| c.contains("kernel_size"))
            .map(|c| c.split('=').last().unwrap().parse::<f32>())
            .transpose()?)
    }

    fn background_color(header: &serde_ply::PlyHeader) -> anyhow::Result<Option<[f32; 3]>> {
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

    pub fn read_metadata(&mut self,data: &[u8]) -> Option<PointCloudMetadata> {
        self.parser.buffer_mut().extend_from_slice(data);
        let header = self.parser.header();
        return header.map(|h| Self::metadata(h));
    }

    pub fn load_next_chunk(
        &mut self,
        data: &[u8]
    ) -> Result<(Vec<Gaussian>, Vec<[[f16; 3]; 16]>, Aabb<f32>), anyhow::Error> {
        let mut gaussians = Vec::new();
        let mut sh_coefs = Vec::new();
        let mut bbox = Aabb::new(
            Point3::new(f32::MAX, f32::MAX, f32::MAX),
            Point3::new(f32::MIN, f32::MIN, f32::MIN),
        );

        self.parser.buffer_mut().extend_from_slice(data);


        if let Some(current_element) = self.parser.current_element() {
            if current_element.name == "vertex" {
                RowVisitor::new(|g: GaussianPly| {
                    let gaussian = g.gaussian();
                    bbox.grow(&gaussian.xyz);
                    gaussians.push(gaussian);
                    sh_coefs.push(g.sh_coefs());
                })
                .deserialize(&mut self.parser)?;
            } else {
            }
        } else {

        }
        self.read_points += gaussians.len();
        return Ok((gaussians, sh_coefs, bbox));
    }

    fn magic_bytes() -> &'static [u8] {
        "ply".as_bytes()
    }

    fn file_ending() -> &'static str {
        "ply"
    }
}

#[derive(Deserialize, Serialize, Debug)]
struct GaussianPly {
    x: f32,
    y: f32,
    z: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    f_dc_0: f32,
    f_dc_1: f32,
    f_dc_2: f32,
    f_rest_0: f32,
    f_rest_1: f32,
    f_rest_2: f32,
    f_rest_3: f32,
    f_rest_4: f32,
    f_rest_5: f32,
    f_rest_6: f32,
    f_rest_7: f32,
    f_rest_8: f32,
    f_rest_9: f32,
    f_rest_10: f32,
    f_rest_11: f32,
    f_rest_12: f32,
    f_rest_13: f32,
    f_rest_14: f32,
    f_rest_15: f32,
    f_rest_16: f32,
    f_rest_17: f32,
    f_rest_18: f32,
    f_rest_19: f32,
    f_rest_20: f32,
    f_rest_21: f32,
    f_rest_22: f32,
    f_rest_23: f32,
    f_rest_24: f32,
    f_rest_25: f32,
    f_rest_26: f32,
    f_rest_27: f32,
    f_rest_28: f32,
    f_rest_29: f32,
    f_rest_30: f32,
    f_rest_31: f32,
    f_rest_32: f32,
    f_rest_33: f32,
    f_rest_34: f32,
    f_rest_35: f32,
    f_rest_36: f32,
    f_rest_37: f32,
    f_rest_38: f32,
    f_rest_39: f32,
    f_rest_40: f32,
    f_rest_41: f32,
    f_rest_42: f32,
    f_rest_43: f32,
    f_rest_44: f32,
    opacity: f32,
    scale_0: f32,
    scale_1: f32,
    scale_2: f32,
    rot_0: f32,
    rot_1: f32,
    rot_2: f32,
    rot_3: f32,
}

impl GaussianPly {
    fn gaussian(&self) -> Gaussian {
        let rot = Quaternion::new(self.rot_0, self.rot_1, self.rot_2, self.rot_3).normalize();
        let scale = Vector3::new(self.scale_0, self.scale_1, self.scale_2).map(|v| v.exp());
        let cov = build_cov(rot, scale);
        let opacity = sigmoid(self.opacity);
        Gaussian::new(
            Point3::new(self.x, self.y, self.z),
            f16::from_f32(opacity),
            cov.map(|f| f16::from_f32(f)),
        )
    }

    fn sh_coefs(&self) -> [[f16; 3]; 16] {
        let mut sh_coefs = [[f16::from_f32(0.0); 3]; 16];
        for i in 0..16 {
            sh_coefs[i][0] = f16::from_f32(*self.get_sh_coef(i, 0));
            sh_coefs[i][1] = f16::from_f32(*self.get_sh_coef(i, 1));
            sh_coefs[i][2] = f16::from_f32(*self.get_sh_coef(i, 2));
        }
        sh_coefs
    }

    fn get_sh_coef(&self, i: usize, c: usize) -> &f32 {
        match (i, c) {
            (0, 0) => &self.f_dc_0,
            (0, 1) => &self.f_dc_1,
            (0, 2) => &self.f_dc_2,
            (1, 0) => &self.f_rest_0,
            (2, 0) => &self.f_rest_1,
            (3, 0) => &self.f_rest_2,
            (4, 0) => &self.f_rest_3,
            (5, 0) => &self.f_rest_4,
            (6, 0) => &self.f_rest_5,
            (7, 0) => &self.f_rest_6,
            (8, 0) => &self.f_rest_7,
            (9, 0) => &self.f_rest_8,
            (10, 0) => &self.f_rest_9,
            (11, 0) => &self.f_rest_10,
            (12, 0) => &self.f_rest_11,
            (13, 0) => &self.f_rest_12,
            (14, 0) => &self.f_rest_13,
            (15, 0) => &self.f_rest_14,
            (1, 1) => &self.f_rest_15,
            (2, 1) => &self.f_rest_16,
            (3, 1) => &self.f_rest_17,
            (4, 1) => &self.f_rest_18,
            (5, 1) => &self.f_rest_19,
            (6, 1) => &self.f_rest_20,
            (7, 1) => &self.f_rest_21,
            (8, 1) => &self.f_rest_22,
            (9, 1) => &self.f_rest_23,
            (10, 1) => &self.f_rest_24,
            (11, 1) => &self.f_rest_25,
            (12, 1) => &self.f_rest_26,
            (13, 1) => &self.f_rest_27,
            (14, 1) => &self.f_rest_28,
            (15, 1) => &self.f_rest_29,
            (1, 2) => &self.f_rest_30,
            (2, 2) => &self.f_rest_31,
            (3, 2) => &self.f_rest_32,
            (4, 2) => &self.f_rest_33,
            (5, 2) => &self.f_rest_34,
            (6, 2) => &self.f_rest_35,
            (7, 2) => &self.f_rest_36,
            (8, 2) => &self.f_rest_37,
            (9, 2) => &self.f_rest_38,
            (10, 2) => &self.f_rest_39,
            (11, 2) => &self.f_rest_40,
            (12, 2) => &self.f_rest_41,
            (13, 2) => &self.f_rest_42,
            (14, 2) => &self.f_rest_43,
            (15, 2) => &self.f_rest_44,
            _ => panic!("invalid sh coef index"),
        }
    }
}
