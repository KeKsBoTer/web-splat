use std::io::{Read, Seek};

use cgmath::{InnerSpace, Point3, Quaternion, Vector3};
use npyz::npz::{self, NpzArchive};
use half::f16;

use crate::{
    pointcloud::{GaussianSplat, PointCloudReader, GeometricInfo},
    utils::{build_cov, sh_deg_from_num_coefs, sh_num_coefficients},
    SHDType,
};

pub struct NpzReader<'a, R: Read + Seek> {
    npz_file: NpzArchive<&'a mut R>,
    sh_deg: u32,
    num_points: usize,
}

impl<'a, R: Read + Seek> NpzReader<'a, R> {
    pub fn new(reader: &'a mut R) -> Result<Self, anyhow::Error> {
        let mut npz_file = npz::NpzArchive::new(reader)?;

        let mut sh_deg = npz_file
            .by_name("features_dc")?
            .map(|_| 1u32)
            .ok_or(anyhow::anyhow!("missing array 'features_dc'"))?;
        if let Some(rest) = npz_file.by_name("features_rest")? {
            sh_deg = sh_deg_from_num_coefs(rest.shape()[1] as u32 + 1)
                .ok_or(anyhow::anyhow!("num sh coefs not valid"))?;
        }
        let num_points = npz_file
            .by_name("xyz")?
            .ok_or(anyhow::anyhow!("array xyz missing"))?
            .shape()[0] as usize;

        Ok(NpzReader {
            npz_file,
            sh_deg,
            num_points,
        })
    }
}
fn get_npz_const<'a, T: npyz::Deserialize, R: Read + Seek>(npz_file: &mut NpzArchive<&'a mut R>, field_name: &str) -> Result<T, anyhow::Error> {
    Ok(npz_file.by_name(field_name)
        .unwrap()
        .unwrap()
        .into_vec()?
        .as_slice()[0])
}
impl<'a, R: Read + Seek> PointCloudReader for NpzReader<'a, R> {
    fn read(
        &mut self,
        sh_dtype: SHDType,
        sh_deg: u32,
    ) -> Result<(Vec<GaussianSplat>, Vec<u8>, Vec<GeometricInfo>), anyhow::Error> {
        if sh_dtype != SHDType::Float {
            return Err(anyhow::anyhow!("only Float sh coefs supported"));
        }
        
        let opacity_scale: f32 = get_npz_const(&mut self.npz_file, "opacity_scale").unwrap_or(1.0);
        let opacity_zero_point: f32 = get_npz_const(&mut self.npz_file, "opacity_zero_point").unwrap_or(0.0);
        let scaling_scale: f32 = get_npz_const(&mut self.npz_file, "scaling_scale").unwrap_or(1.0);
        let scaling_zero_point: f32 = get_npz_const(&mut self.npz_file, "scaling_zero_point").unwrap_or(0.0);
        let features_scale: f32 = get_npz_const(&mut self.npz_file, "features_scale").unwrap_or(1.0);
        let features_zero_point: f32 = get_npz_const(&mut self.npz_file, "features_zero_point").unwrap_or(0.0);
        let gaussian_scale: f32 = get_npz_const(&mut self.npz_file, "gaussian_scale").unwrap_or(1.0);
        let gaussian_zero_point: f32 = get_npz_const(&mut self.npz_file, "gaussian_zero_point").unwrap_or(0.0);

        let xyz: Vec<Point3<f16>> = self
            .npz_file
            .by_name("xyz")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice()
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]).cast().unwrap())
            .collect();

        let scaling: Vec<Vector3<f32>> = self
            .npz_file
            .by_name("scaling")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice().iter()
            .map(|c: &f32| ((c - scaling_zero_point) * scaling_scale).exp()).into_iter()
            .chunks_exact(3)
            .map(|c: &[f32]| Vector3::new(c[0], c[1], c[2]))
            .collect();
        
        let rotation: Vec<Quaternion<f32>> = self
            .npz_file
            .by_name("rotation")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice()
            .chunks_exact(4)
            .map(|c| Quaternion::new(c[0], c[1], c[2], c[3]).normalize())
            .collect();

        let opacity: Vec<f32> = self
            .npz_file
            .by_name("opacity")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slie()
            .map(|c| (c - opacity_zero_point) * opacity_scale)
            .collect();

        let feature_indices: Option<Vec<i32>> =
            if let Some(idx_array) = self.npz_file.by_name("feature_indices")? {
                Some(idx_array.into_vec()?)
            } else {
                None
            };

        let gaussian_indices: Option<Vec<i32>> =
            if let Some(idx_array) = self.npz_file.by_name("gaussian_indices")? {
                log::warn!("indexed gaussian splats (scaling & rotation) are not supported");
                Some(idx_array.into_vec()?)
            } else {
                None
            };

        let features_dc: Vec<Vector3<f32>> = self
            .npz_file
            .by_name("features_dc")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice()
            .chunks_exact(3)
            .map(|c| Vector3::new(c[0], c[1], c[2]))
            .collect();

        let features_rest: Vec<Vector3<f32>> = self
            .npz_file
            .by_name("features_rest")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice()
            .chunks_exact(3)
            .map(|c| Vector3::new(c[0], c[1], c[2]))
            .collect();

        let num_points = xyz.len();

        let vertices: Vec<GaussianSplat> = (0..num_points)
            .map(|i| GaussianSplat {
                xyz: xyz[i],
                sh_idx: feature_indices
                    .as_ref()
                    .map(|f| f[i] as u32)
                    .unwrap_or(i as u32),
                covariance: if let Some(indices) = &gaussian_indices {
                    build_cov(rotation[indices[i] as usize], scaling[indices[i] as usize])
                } else {
                    build_cov(rotation[i], scaling[i])
                },
                opacity: opacity[i],
            })
            .collect();

        let mut sh_coef_buffer = Vec::new();
        let num_coefs_h = sh_num_coefficients(sh_deg) as usize - 1;
        for i in 0..features_dc.len() {
            sh_dtype
                .write_to(&mut sh_coef_buffer, features_dc[i].x, 0)
                .unwrap();
            sh_dtype
                .write_to(&mut sh_coef_buffer, features_dc[i].y, 0)
                .unwrap();
            sh_dtype
                .write_to(&mut sh_coef_buffer, features_dc[i].z, 0)
                .unwrap();
            for o in 0..num_coefs_h {
                // iterate higher order coefs
                for c in 0..3 {
                    // iterate RGB channels
                    sh_dtype
                        .write_to(
                            &mut sh_coef_buffer,
                            features_rest[i * num_coefs_h + o][c],
                            i as u32 + 1,
                        )
                        .unwrap();
                }
            }
        }

        return Ok((vertices, sh_coef_buffer));
    }

    fn file_sh_deg(&self) -> Result<u32, anyhow::Error> {
        Ok(self.sh_deg)
    }

    fn num_points(&self) -> Result<usize, anyhow::Error> {
        Ok(self.num_points)
    }
}
