use std::io::{Read, Seek};

use byteorder::{ReadBytesExt, LittleEndian};
use cgmath::{InnerSpace, Point3, Quaternion, Vector3};
use npyz::npz::{self, NpzArchive};
use half::f16;

use crate::{
    pointcloud::{GaussianSplat, PointCloudReader, GeometricInfo},
    utils::{build_cov, sh_deg_from_num_coefs, sh_num_coefficients},
    SHDType,
};
#[derive(Debug, PartialEq, Clone)]
struct f16d(f16);
struct HalfReader;
impl npyz::TypeRead for HalfReader {
    type Value = f16d;

    #[inline]
    fn read_one<R: Read>(&self, mut reader: R) -> std::io::Result<Self::Value> {
        Ok(f16d{0: f16::from_bits(reader.read_u16::<LittleEndian>()?)})
    }
}
impl npyz::Deserialize for f16d {
    type TypeReader = HalfReader;

    fn reader(dtype: &npyz::DType) -> Result<Self::TypeReader, npyz::DTypeError> {
        if true {
            Ok(HalfReader)
        } else {
            Err(npyz::DTypeError::custom("Vector5 only supports '<i4' format!"))
        }
    }
}


pub struct NpzReader<'a, R: Read + Seek> {
    npz_file: NpzArchive<&'a mut R>,
    sh_deg: u32,
    num_points: usize,
}

impl<'a, R: Read + Seek> NpzReader<'a, R> {
    pub fn new(reader: &'a mut R) -> Result<Self, anyhow::Error> {
        let mut npz_file = npz::NpzArchive::new(reader)?;

        let mut sh_deg = 0;
        if let Some(rest) = npz_file.by_name("features")? {
            sh_deg = sh_deg_from_num_coefs(rest.shape()[1] as u32)// + 1)
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
fn get_npz_const<'a, T: npyz::Deserialize + Copy, R: Read + Seek>(npz_file: &mut NpzArchive<&'a mut R>, field_name: &str) -> Result<T, anyhow::Error> {
    Ok(npz_file.by_name(field_name)?
        .ok_or(anyhow::anyhow!("field not present in npz file"))?
        .into_vec::<T>()?
        [0].clone())
}
impl<'a, R: Read + Seek> PointCloudReader for NpzReader<'a, R> {
    fn read(
        &mut self,
        sh_dtype: SHDType,
        sh_deg: u32,
    ) -> Result<(Vec<GaussianSplat>, Vec<u8>, Vec<GeometricInfo>), anyhow::Error> {
        let opacity_scale: f32 = get_npz_const(&mut self.npz_file, "opacity_scale").unwrap_or(1.0);
        let opacity_zero_point: f32 = get_npz_const(&mut self.npz_file, "opacity_zero_point").unwrap_or(0.0);
        let scaling_scale: f32 = get_npz_const(&mut self.npz_file, "scaling_scale").unwrap_or(1.0);
        let scaling_zero_point: f32 = get_npz_const(&mut self.npz_file, "scaling_zero_point").unwrap_or(0.0);
        let features_scale: f32 = get_npz_const(&mut self.npz_file, "features_scale").unwrap_or(1.0);
        let features_zero_point: f32 = get_npz_const(&mut self.npz_file, "features_zero_point").unwrap_or(0.0);
        let gaussian_scale: f32 = get_npz_const(&mut self.npz_file, "gaussian_scale").unwrap_or(1.0);
        let gaussian_zero_point: f32 = get_npz_const(&mut self.npz_file, "gaussian_zero_point").unwrap_or(0.0);
        
        println!("npz importer: o_s {opacity_scale}, o_zp {opacity_zero_point}, s_s {scaling_scale}, s_zp {scaling_zero_point},
                f_s: {features_scale}, f_zp {features_zero_point}, g_s {gaussian_scale}, g_zp {gaussian_zero_point}");

        println!("{:?}", self.npz_file.by_name("xyz").unwrap().unwrap().dtype());
        let xyz: Vec<Point3<f16>> = self
            .npz_file
            .by_name("xyz")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice()
            .chunks_exact(3)
            .map(|c: &[f16d]| Point3::new(c[0].0, c[1].0, c[2].0).cast().unwrap())
            .collect();
        println!("parsing position done");

        let scaling: Vec<Vector3<f32>> = self
            .npz_file
            .by_name("scaling")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice().iter()
            .map(|c: &i8| ((*c as f32 - scaling_zero_point) * scaling_scale).exp()).collect::<Vec::<f32>>()
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

        let opacity: Vec<f16> = self
            .npz_file
            .by_name("opacity")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice().iter()
            .map(|c: &i8| f16::from_f32((*c as f32 - opacity_zero_point) * opacity_scale))
            .collect();
        
        let features_indices: Vec<u32> = if let Some(idx_array) = self.npz_file.by_name("feature_indices")? {
                                            idx_array.into_vec()?
                                        } else {
                                            (0..xyz.len() as u32).collect()
                                        };

        let gaussian_indices: Vec<u32> = if let Some(idx_array) = self.npz_file.by_name("gaussian_indices")? {
                                            idx_array.into_vec()?
                                        } else {
                                            (0..xyz.len() as u32).collect()
                                        };

        let features: Vec<f32> = self
            .npz_file
            .by_name("features")
            .unwrap()
            .unwrap()
            .into_vec()?
            .as_slice().iter()
            .map(|c: &i8| ((*c as f32 - features_zero_point) * features_scale).exp())
            .collect();

        let num_points = xyz.len();
        let num_sh_coeffs = sh_num_coefficients(sh_deg);
        if true {
            // safety checks for the feature indices and gaussian indices
            assert_eq!(num_points, features_indices.len());
            assert_eq!(num_points, gaussian_indices.len());
            assert_eq!(scaling.len(), rotation.len());
            let features_len = features.len() / 48;
            let gaussian_len = scaling.len();
            assert_eq!(*features_indices.iter().max().unwrap(), features_len as u32 - 1);
            assert_eq!(*gaussian_indices.iter().max().unwrap(), gaussian_len as u32 - 1);
        }

        let vertices: Vec<GaussianSplat> = (0..num_points)
            .map(|i| GaussianSplat{
                xyz: xyz[i],
                opacity: opacity[i],
                geometry_idx: gaussian_indices[i],
                sh_idx: features_indices[i],
            }).collect();

        let mut sh_buffer = Vec::new();
        for i in 0..features.len() {
            let f = i / num_sh_coeffs as usize;
            let idx = f as u32 / 3;
            sh_dtype.write_to(&mut sh_buffer, features[i], idx);
        }
        let covar_buffer = (0..scaling.len())
            .map(|i| GeometricInfo{covariance: build_cov(rotation[i], scaling[i]), ..Default::default()}).collect();

        return Ok((vertices, sh_buffer, covar_buffer));
    }

    fn file_sh_deg(&self) -> Result<u32, anyhow::Error> {
        Ok(self.sh_deg)
    }

    fn num_points(&self) -> Result<usize, anyhow::Error> {
        Ok(self.num_points)
    }
}
