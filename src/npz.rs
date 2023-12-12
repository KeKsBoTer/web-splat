use std::io::{Read, Seek};

use byteorder::WriteBytesExt;
use cgmath::{InnerSpace, Point3, Quaternion, Vector3};
use half::f16;
use image::EncodableLayout;
use npyz::npz::{self, NpzArchive};

use crate::{
    pointcloud::{Gaussian, GeometricInfo, PointCloudReader, Quantization, QuantizationUniform},
    utils::{build_cov, sh_deg_from_num_coefs, sh_num_coefficients},
};

pub struct NpzReader<'a, R: Read + Seek> {
    npz_file: NpzArchive<&'a mut R>,
    sh_deg: u32,
    num_points: usize,
}

impl<'a, R: Read + Seek> NpzReader<'a, R> {
    pub fn new(reader: &'a mut R) -> Result<Self, anyhow::Error> {
        let mut npz_file = npz::NpzArchive::new(reader)?;

        let mut sh_deg = 0;
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

    pub fn magic_bytes() -> &'static [u8] {
        (b"\x50\x4B\x03\x04").as_bytes()
    }
}
fn get_npz_const<'a, T: npyz::Deserialize + Copy, R: Read + Seek>(
    npz_file: &mut NpzArchive<&'a mut R>,
    field_name: &str,
) -> Result<T, anyhow::Error> {
    Ok(npz_file
        .by_name(field_name)?
        .ok_or(anyhow::anyhow!("field not present in npz file"))?
        .into_vec::<T>()?[0]
        .clone())
}
impl<'a, R: Read + Seek> PointCloudReader for NpzReader<'a, R> {
    fn read(
        &mut self,
        sh_deg: u32,
    ) -> Result<
        (
            Vec<Gaussian>,
            Vec<u8>,
            Vec<GeometricInfo>,
            QuantizationUniform,
        ),
        anyhow::Error,
    > {
        let opacity_scale: f32 = get_npz_const(&mut self.npz_file, "opacity_scale").unwrap_or(1.0);
        let opacity_zero_point: i32 =
            get_npz_const(&mut self.npz_file, "opacity_zero_point").unwrap_or(0);

        let scaling_scale: f32 = get_npz_const(&mut self.npz_file, "scaling_scale").unwrap_or(1.0);
        let scaling_zero_point: f32 =
            get_npz_const::<i32, _>(&mut self.npz_file, "scaling_zero_point").unwrap_or(0) as f32;

        let rotation_scale: f32 =
            get_npz_const(&mut self.npz_file, "rotation_scale").unwrap_or(1.0);
        let rotation_zero_point: f32 =
            get_npz_const::<i32, _>(&mut self.npz_file, "rotation_zero_point").unwrap_or(0) as f32;

        let features_dc_scale: f32 =
            get_npz_const(&mut self.npz_file, "features_dc_scale").unwrap_or(1.0);
        let features_dc_zero_point: i32 =
            get_npz_const(&mut self.npz_file, "features_dc_zero_point").unwrap_or(0);

        let features_rest_scale: f32 =
            get_npz_const(&mut self.npz_file, "features_rest_scale").unwrap_or(1.0);
        let features_rest_zero_point: i32 =
            get_npz_const(&mut self.npz_file, "features_rest_zero_point").unwrap_or(0);

        let scaling_factor_scale: f32 =
            get_npz_const(&mut self.npz_file, "scaling_factor_scale").unwrap_or(1.);
        let scaling_factor_zero_point: i32 =
            get_npz_const(&mut self.npz_file, "scaling_factor_zero_point").unwrap_or(0);

        let xyz: Vec<Point3<f16>> = self
            .npz_file
            .by_name("xyz")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice()
            .chunks_exact(3)
            .map(|c: &[f16]| Point3::new(c[0], c[1], c[2]).cast().unwrap())
            .collect();

        let scaling: Vec<Vector3<f32>> = self
            .npz_file
            .by_name("scaling")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice()
            .iter()
            .map(|c: &i8| ((*c as f32 - scaling_zero_point) * scaling_scale).max(0.))
            .collect::<Vec<f32>>()
            .chunks_exact(3)
            .map(|c: &[f32]| Vector3::new(c[0], c[1], c[2]).normalize())
            .collect();

        let scaling_factor: Vec<i8> = self
            .npz_file
            .by_name("scaling_factor")?
            .unwrap()
            .into_vec()?;

        let rotation: Vec<Quaternion<f32>> = self
            .npz_file
            .by_name("rotation")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice()
            .iter()
            .map(|c: &i8| ((*c as f32 - rotation_zero_point) * rotation_scale))
            .collect::<Vec<f32>>()
            .chunks_exact(4)
            .map(|c| Quaternion::new(c[0], c[1], c[2], c[3]).normalize())
            .collect();

        let opacity = self
            .npz_file
            .by_name("opacity")?
            .unwrap()
            .into_vec::<i8>()?;

        let features_indices: Vec<u32> = self
            .npz_file
            .by_name("feature_indices")?
            .unwrap()
            .into_vec()?
            .as_slice()
            .iter()
            .map(|c: &i32| *c as u32)
            .collect::<Vec<u32>>();

        let gaussian_indices = self
            .npz_file
            .by_name("gaussian_indices")?
            .unwrap()
            .into_vec()?
            .as_slice()
            .iter()
            .map(|c: &i32| *c as u32)
            .collect::<Vec<u32>>();

        let features_dc: Vec<i8> = self
            .npz_file
            .by_name("features_dc")
            .unwrap()
            .unwrap()
            .into_vec()?;

        let features_rest: Vec<i8> = self
            .npz_file
            .by_name("features_rest")
            .unwrap()
            .unwrap()
            .into_vec()?;

        let num_points: usize = xyz.len();
        let num_sh_coeffs = sh_num_coefficients(sh_deg);

        let vertices: Vec<Gaussian> = (0..num_points)
            .map(|i| Gaussian {
                xyz: xyz[i],
                opacity: opacity[i],
                scale_factor: scaling_factor[i],
                geometry_idx: gaussian_indices[i],
                sh_idx: features_indices[i],
            })
            .collect();

        let mut sh_buffer = Vec::new();

        let sh_coeffs_length = num_sh_coeffs as usize * 3;
        let rest_num_coefs = sh_coeffs_length - 3;
        for i in 0..(features_dc.len() / 3) {
            sh_buffer.write_i8(features_dc[i * 3 + 0]).unwrap();
            sh_buffer.write_i8(features_dc[i * 3 + 1]).unwrap();
            sh_buffer.write_i8(features_dc[i * 3 + 2]).unwrap();
            for j in 0..rest_num_coefs {
                sh_buffer
                    .write_i8(features_rest[i * rest_num_coefs + j])
                    .unwrap();
            }
        }
        let covar_buffer = (0..rotation.len())
            .map(|i| {
                let cov = build_cov(rotation[i], scaling[i]);
                GeometricInfo {
                    covariance: cov.map(|v| f16::from_f32(v)),
                    ..Default::default()
                }
            })
            .collect();

        let quantization = QuantizationUniform {
            color_dc: Quantization::new(features_dc_zero_point, features_dc_scale),
            color_rest: Quantization::new(features_rest_zero_point, features_rest_scale),
            opacity: Quantization::new(opacity_zero_point, opacity_scale),
            scaling_factor: Quantization::new(scaling_factor_zero_point, scaling_factor_scale),
        };

        return Ok((vertices, sh_buffer, covar_buffer, quantization));
    }

    fn file_sh_deg(&self) -> Result<u32, anyhow::Error> {
        Ok(self.sh_deg)
    }

    fn num_points(&self) -> Result<usize, anyhow::Error> {
        Ok(self.num_points)
    }
}
