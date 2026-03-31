use half::f16;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Quaternion, Vector3};

use crate::{
    pointcloud::{Aabb, Gaussian, PointCloudMetadata},
    utils::{build_cov, sh_deg_from_num_coefs, sigmoid},
};
use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use serde_ply::RowVisitor;

pub struct PlyReader {
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
            quantization: None,
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

    pub fn read_metadata(&mut self, data: &[u8]) -> Option<PointCloudMetadata> {
        self.parser.buffer_mut().extend_from_slice(data);
        let header = self.parser.header();
        return header.map(|h| Self::metadata(h));
    }

    pub fn load_next_chunk(
        &mut self,
        data: &[u8],
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
                    sh_coefs.push(g.features.as_array().map(|v| {
                        [
                            f16::from_f32(v[0]),
                            f16::from_f32(v[1]),
                            f16::from_f32(v[2]),
                        ]
                    }));
                })
                .deserialize(&mut self.parser)?;
            } else {
                log::warn!(
                    "current element is {}, expected vertex",
                    current_element.name
                );
            }
        } else {
        }
        self.read_points += gaussians.len();
        return Ok((gaussians, sh_coefs, bbox));
    }
}

#[derive(Deserialize, Serialize)]
struct GaussianPly {
    x: f32,
    y: f32,
    z: f32,

    #[serde(default)]
    nx: f32,
    #[serde(default)]
    ny: f32,
    #[serde(default)]
    nz: f32,

    #[serde(flatten)]
    features: FeaturePly<f32>,

    opacity: f32,

    #[serde(flatten)]
    cov: CovariancePly<f32>,
}

impl GaussianPly {
    fn gaussian(&self) -> Gaussian {
        let cov = self.cov.covariance();
        let opacity = sigmoid(self.opacity);
        Gaussian::new(
            Point3::new(self.x, self.y, self.z),
            f16::from_f32(opacity),
            cov,
        )
    }
}

// #[derive(Deserialize, Serialize, Debug)]
// struct VertexPly {
//     x: f32,
//     y: f32,
//     z: f32,
//     opacity: u8,
//     scaling_factor: u8,
//     gaussian_indices: u32,
//     feature_indices: u32,
// }

#[derive(Deserialize, Serialize, Debug)]
struct CovariancePly<T> {
    scale_0: T,
    scale_1: T,
    scale_2: T,
    rot_0: T,
    rot_1: T,
    rot_2: T,
    rot_3: T,
}

impl CovariancePly<f32> {
    fn covariance(&self) -> [f16; 6] {
        let rot = Quaternion::new(self.rot_0, self.rot_1, self.rot_2, self.rot_3).normalize();
        let scale = Vector3::new(self.scale_0, self.scale_1, self.scale_2).map(|v| v.exp());
        let cov = build_cov(rot, scale);
        cov.map(|f| f16::from_f32(f))
    }
}

#[derive(Deserialize, Serialize)]
struct FeaturePly<T> {
    f_dc_0: T,
    f_dc_1: T,
    f_dc_2: T,
    #[serde(default)]
    f_rest_0: T,
    #[serde(default)]
    f_rest_1: T,
    #[serde(default)]
    f_rest_2: T,
    #[serde(default)]
    f_rest_3: T,
    #[serde(default)]
    f_rest_4: T,
    #[serde(default)]
    f_rest_5: T,
    #[serde(default)]
    f_rest_6: T,
    #[serde(default)]
    f_rest_7: T,
    #[serde(default)]
    f_rest_8: T,
    #[serde(default)]
    f_rest_9: T,
    #[serde(default)]
    f_rest_10: T,
    #[serde(default)]
    f_rest_11: T,
    #[serde(default)]
    f_rest_12: T,
    #[serde(default)]
    f_rest_13: T,
    #[serde(default)]
    f_rest_14: T,
    #[serde(default)]
    f_rest_15: T,
    #[serde(default)]
    f_rest_16: T,
    #[serde(default)]
    f_rest_17: T,
    #[serde(default)]
    f_rest_18: T,
    #[serde(default)]
    f_rest_19: T,
    #[serde(default)]
    f_rest_20: T,
    #[serde(default)]
    f_rest_21: T,
    #[serde(default)]
    f_rest_22: T,
    #[serde(default)]
    f_rest_23: T,
    #[serde(default)]
    f_rest_24: T,
    #[serde(default)]
    f_rest_25: T,
    #[serde(default)]
    f_rest_26: T,
    #[serde(default)]
    f_rest_27: T,
    #[serde(default)]
    f_rest_28: T,
    #[serde(default)]
    f_rest_29: T,
    #[serde(default)]
    f_rest_30: T,
    #[serde(default)]
    f_rest_31: T,
    #[serde(default)]
    f_rest_32: T,
    #[serde(default)]
    f_rest_33: T,
    #[serde(default)]
    f_rest_34: T,
    #[serde(default)]
    f_rest_35: T,
    #[serde(default)]
    f_rest_36: T,
    #[serde(default)]
    f_rest_37: T,
    #[serde(default)]
    f_rest_38: T,
    #[serde(default)]
    f_rest_39: T,
    #[serde(default)]
    f_rest_40: T,
    #[serde(default)]
    f_rest_41: T,
    #[serde(default)]
    f_rest_42: T,
    #[serde(default)]
    f_rest_43: T,
    #[serde(default)]
    f_rest_44: T,
}

impl<T: Copy> FeaturePly<T> {
    fn as_array(self) -> [[T; 3]; 16] {
        std::array::from_fn(|i| {
            [
                self.get_sh_coef(i, 0),
                self.get_sh_coef(i, 1),
                self.get_sh_coef(i, 2),
            ]
        })
    }
    fn get_sh_coef(&self, i: usize, c: usize) -> T {
        match (i, c) {
            (0, 0) => self.f_dc_0,
            (0, 1) => self.f_dc_1,
            (0, 2) => self.f_dc_2,
            (1, 0) => self.f_rest_0,
            (2, 0) => self.f_rest_1,
            (3, 0) => self.f_rest_2,
            (4, 0) => self.f_rest_3,
            (5, 0) => self.f_rest_4,
            (6, 0) => self.f_rest_5,
            (7, 0) => self.f_rest_6,
            (8, 0) => self.f_rest_7,
            (9, 0) => self.f_rest_8,
            (10, 0) => self.f_rest_9,
            (11, 0) => self.f_rest_10,
            (12, 0) => self.f_rest_11,
            (13, 0) => self.f_rest_12,
            (14, 0) => self.f_rest_13,
            (15, 0) => self.f_rest_14,
            (1, 1) => self.f_rest_15,
            (2, 1) => self.f_rest_16,
            (3, 1) => self.f_rest_17,
            (4, 1) => self.f_rest_18,
            (5, 1) => self.f_rest_19,
            (6, 1) => self.f_rest_20,
            (7, 1) => self.f_rest_21,
            (8, 1) => self.f_rest_22,
            (9, 1) => self.f_rest_23,
            (10, 1) => self.f_rest_24,
            (11, 1) => self.f_rest_25,
            (12, 1) => self.f_rest_26,
            (13, 1) => self.f_rest_27,
            (14, 1) => self.f_rest_28,
            (15, 1) => self.f_rest_29,
            (1, 2) => self.f_rest_30,
            (2, 2) => self.f_rest_31,
            (3, 2) => self.f_rest_32,
            (4, 2) => self.f_rest_33,
            (5, 2) => self.f_rest_34,
            (6, 2) => self.f_rest_35,
            (7, 2) => self.f_rest_36,
            (8, 2) => self.f_rest_37,
            (9, 2) => self.f_rest_38,
            (10, 2) => self.f_rest_39,
            (11, 2) => self.f_rest_40,
            (12, 2) => self.f_rest_41,
            (13, 2) => self.f_rest_42,
            (14, 2) => self.f_rest_43,
            (15, 2) => self.f_rest_44,
            _ => panic!("invalid sh coef index"),
        }
    }
}
