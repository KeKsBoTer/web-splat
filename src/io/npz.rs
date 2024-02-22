use std::io::{Read, Seek};

use byteorder::WriteBytesExt;
use cgmath::{InnerSpace, Point3, Quaternion, Vector3};
use half::f16;
use image::EncodableLayout;
use npyz::npz::{self, NpzArchive};

#[cfg(target_arch = "wasm32")]
use instant::Instant;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use crate::{
    pointcloud::{Covariance3D, GaussianCompressed, GaussianQuantization, Quantization},
    utils::{build_cov, sh_deg_from_num_coefs, sh_num_coefficients},
};

use super::{GenericGaussianPointCloud, PointCloudReader};

pub struct NpzReader<'a, R: Read + Seek> {
    npz_file: NpzArchive<&'a mut R>,
    sh_deg: u32,
    kernel_size: Option<f32>,
    mip_splatting: Option<bool>,
    background_color: Option<[f32; 3]>,
}

impl<'a, R: Read + Seek> NpzReader<'a, R> {
    pub fn new(reader: &'a mut R) -> Result<Self, anyhow::Error> {
        let mut npz_file = npz::NpzArchive::new(reader)?;

        let mut sh_deg = 0;
        if let Some(rest) = npz_file.by_name("features_rest")? {
            sh_deg = sh_deg_from_num_coefs(rest.shape()[1] as u32 + 1)
                .ok_or(anyhow::anyhow!("num sh coefs not valid"))?;
        }

        let kernel_size: Option<f32> = get_npz_value(&mut npz_file, "kernel_size")?;
        let mip_splatting: Option<bool> = get_npz_value(&mut npz_file, "mip_splatting")?;

        let background_color: Option<[f32; 3]> =
            if let Some(bg) = get_npz_array_optional::<f32>(&mut npz_file, "background_color")? {
                Some(bg.as_slice().try_into()?)
            } else {
                None
            };
        Ok(NpzReader {
            npz_file,
            sh_deg,
            kernel_size,
            mip_splatting,
            background_color,
        })
    }
}

impl<'a, R: Read + Seek> PointCloudReader for NpzReader<'a, R> {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error> {
        let opacity_scale: f32 = get_npz_value(&mut self.npz_file, "opacity_scale")?.unwrap_or(1.0);
        let opacity_zero_point: i32 =
            get_npz_value(&mut self.npz_file, "opacity_zero_point")?.unwrap_or(0);

        let scaling_scale: f32 = get_npz_value(&mut self.npz_file, "scaling_scale")?.unwrap_or(1.0);
        let scaling_zero_point: f32 =
            get_npz_value::<i32>(&mut self.npz_file, "scaling_zero_point")?.unwrap_or(0) as f32;

        let rotation_scale: f32 =
            get_npz_value(&mut self.npz_file, "rotation_scale")?.unwrap_or(1.0);
        let rotation_zero_point: f32 =
            get_npz_value::<i32>(&mut self.npz_file, "rotation_zero_point")?.unwrap_or(0) as f32;

        let features_dc_scale: f32 =
            get_npz_value(&mut self.npz_file, "features_dc_scale")?.unwrap_or(1.0);
        let features_dc_zero_point: i32 =
            get_npz_value(&mut self.npz_file, "features_dc_zero_point")?.unwrap_or(0);

        let features_rest_scale: f32 =
            get_npz_value(&mut self.npz_file, "features_rest_scale")?.unwrap_or(1.0);
        let features_rest_zero_point: i32 =
            get_npz_value(&mut self.npz_file, "features_rest_zero_point")?.unwrap_or(0);

        let scaling_factor_scale: f32 =
            get_npz_value(&mut self.npz_file, "scaling_factor_scale")?.unwrap_or(1.);
        let scaling_factor_zero_point: i32 =
            get_npz_value(&mut self.npz_file, "scaling_factor_zero_point")?.unwrap_or(0);

        let now = Instant::now();
        let xyz: Vec<Point3<f16>> = try_get_npz_array::<f16>(&mut self.npz_file, "xyz")?
            .as_slice()
            .chunks_exact(3)
            .map(|c: &[f16]| Point3::new(c[0], c[1], c[2]).cast().unwrap())
            .collect();

        let scaling: Vec<Vector3<f32>> = try_get_npz_array::<i8>(&mut self.npz_file, "scaling")?
            .as_slice()
            .iter()
            .map(|c: &i8| ((*c as f32 - scaling_zero_point) * scaling_scale).max(0.))
            .collect::<Vec<f32>>()
            .chunks_exact(3)
            .map(|c: &[f32]| Vector3::new(c[0], c[1], c[2]).normalize())
            .collect();

        let scaling_factor: Vec<i8> = try_get_npz_array(&mut self.npz_file, "scaling_factor")?;

        let rotation: Vec<Quaternion<f32>> = try_get_npz_array(&mut self.npz_file, "rotation")?
            .as_slice()
            .iter()
            .map(|c: &i8| ((*c as f32 - rotation_zero_point) * rotation_scale))
            .collect::<Vec<f32>>()
            .chunks_exact(4)
            .map(|c| Quaternion::new(c[0], c[1], c[2], c[3]).normalize())
            .collect();

        let opacity: Vec<i8> = try_get_npz_array(&mut self.npz_file, "opacity")?;

        let features_indices: Vec<u32> =
            try_get_npz_array::<i32>(&mut self.npz_file, "feature_indices")?
                .as_slice()
                .iter()
                .map(|c: &i32| *c as u32)
                .collect::<Vec<u32>>();

        let gaussian_indices = try_get_npz_array::<i32>(&mut self.npz_file, "gaussian_indices")?
            .as_slice()
            .iter()
            .map(|c: &i32| *c as u32)
            .collect::<Vec<u32>>();

        let features_dc: Vec<i8> = try_get_npz_array(&mut self.npz_file, "features_dc")?;

        let features_rest: Vec<i8> = try_get_npz_array(&mut self.npz_file, "features_rest")?;

        let num_points: usize = xyz.len();
        let sh_deg = self.sh_deg;
        let num_sh_coeffs = sh_num_coefficients(sh_deg);

        let gaussians: Vec<GaussianCompressed> = (0..num_points)
            .map(|i| GaussianCompressed {
                xyz: xyz[i],
                opacity: opacity[i],
                scale_factor: scaling_factor[i],
                geometry_idx: gaussian_indices[i],
                sh_idx: features_indices[i],
            })
            .collect();

        let mut sh_coefs = Vec::new();

        let sh_coeffs_length = num_sh_coeffs as usize * 3;
        let rest_num_coefs = sh_coeffs_length - 3;
        for i in 0..(features_dc.len() / 3) {
            sh_coefs.write_i8(features_dc[i * 3 + 0]).unwrap();
            sh_coefs.write_i8(features_dc[i * 3 + 1]).unwrap();
            sh_coefs.write_i8(features_dc[i * 3 + 2]).unwrap();
            for j in 0..rest_num_coefs {
                sh_coefs
                    .write_i8(features_rest[i * rest_num_coefs + j])
                    .unwrap();
            }
        }
        let covars = (0..rotation.len())
            .map(|i| {
                let cov = build_cov(rotation[i], scaling[i]);
                Covariance3D(cov.map(|v| f16::from_f32(v)))
            })
            .collect();

        let duration = now.elapsed();
        log::info!("reading took {:?}", duration);

        let quantization = GaussianQuantization {
            color_dc: Quantization::new(features_dc_zero_point, features_dc_scale),
            color_rest: Quantization::new(features_rest_zero_point, features_rest_scale),
            opacity: Quantization::new(opacity_zero_point, opacity_scale),
            scaling_factor: Quantization::new(scaling_factor_zero_point, scaling_factor_scale),
        };

        return Ok(GenericGaussianPointCloud::new_compressed(
            gaussians,
            sh_coefs,
            sh_deg,
            num_points,
            self.kernel_size,
            self.mip_splatting,
            self.background_color,
            Some(covars),
            Some(quantization),
        ));
    }

    fn magic_bytes() -> &'static [u8] {
        (b"\x50\x4B\x03\x04").as_bytes()
    }

    fn file_ending() -> &'static str {
        "npz"
    }
}

// tries to read an array
// if the array is not present None is returned
fn get_npz_array_optional<T: npyz::Deserialize + Copy>(
    reader: &mut NpzArchive<impl Read + Seek>,
    field_name: &str,
) -> Result<Option<Vec<T>>, anyhow::Error> {
    let a = reader.by_name(field_name)?;
    if let Some(a) = a {
        let val = a.into_vec::<T>()?;
        Ok(Some(val))
    } else {
        Ok(None)
    }
}

// tries to read an array
// if the array is not present it is treated as an error
fn try_get_npz_array<T: npyz::Deserialize + Copy>(
    reader: &mut NpzArchive<impl Read + Seek>,
    field_name: &str,
) -> Result<Vec<T>, anyhow::Error> {
    return Ok(reader
        .by_name(field_name)?
        .ok_or(anyhow::format_err!("array missing"))?
        .into_vec::<T>()?);
}

// reads a single optional value
fn get_npz_value<T: npyz::Deserialize + Copy>(
    reader: &mut NpzArchive<impl Read + Seek>,
    field_name: &str,
) -> Result<Option<T>, anyhow::Error> {
    if let Some(arr) = get_npz_array_optional(reader, field_name)? {
        arr.get(0)
            .ok_or(anyhow::format_err!("array empty"))
            .map(|v| Some(*v))
    } else {
        Ok(None)
    }
}
