#[cfg(feature = "npz")]
use std::io::BufReader;
use std::io::{Read, Seek};

use bytemuck::Zeroable;
use cgmath::{Point3, Vector3};
use half::f16;

use crate::pointcloud::{
    Aabb, Covariance3D, Gaussian, GaussianCompressed, GaussianQuantization, PlaneFit, Quantization,
};

#[cfg(feature = "npz")]
use self::npz::NpzReader;

use self::ply::PlyReader;
pub use self::ply::{GaussianChunk, PlyInfo};
pub use self::stream::{PlyFeeder, PointCloudStream};

#[cfg(feature = "npz")]
pub mod npz;
pub mod ply;
pub mod stream;

pub trait PointCloudReader {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error>;

    fn magic_bytes() -> &'static [u8];
    fn file_ending() -> &'static str;
}

pub struct GenericGaussianPointCloud {
    gaussians: Vec<u8>,
    sh_coefs: Vec<u8>,
    compressed: bool,
    pub covars: Option<Vec<Covariance3D>>,
    pub quantization: Option<GaussianQuantization>,
    pub sh_deg: u32,
    pub num_points: usize,
    pub kernel_size: Option<f32>,
    pub mip_splatting: Option<bool>,
    pub background_color: Option<[f32; 3]>,

    pub up: Option<Vector3<f32>>,
    pub center: Point3<f32>,
    pub aabb: Aabb<f32>,
}

impl GenericGaussianPointCloud {
    pub fn load<'a, R: Read + Seek>(f: R) -> Result<Self, anyhow::Error> {
        let mut signature: [u8; 4] = [0; 4];
        let mut f = f;
        f.read_exact(&mut signature)?;
        f.rewind()?;
        if signature.starts_with(PlyReader::<R>::magic_bytes()) {
            let mut ply_reader = PlyReader::new(f)?;
            return ply_reader.read();
        }
        #[cfg(feature = "npz")]
        if signature.starts_with(NpzReader::<R>::magic_bytes()) {
            let mut reader = BufReader::new(f);
            let mut npz_reader = NpzReader::new(&mut reader)?;
            return npz_reader.read();
        }
        #[cfg(not(feature = "npz"))]
        if signature.starts_with(b"PK\x03\x04") {
            return Err(anyhow::anyhow!(
                "npz files need the `npz` feature (cargo run --release --features npz ...)"
            ));
        }
        return Err(anyhow::anyhow!("Unknown file format"));
    }

    fn new(
        gaussians: Vec<Gaussian>,
        sh_coefs: Vec<[[f16; 3]; 16]>,
        sh_deg: u32,
        num_points: usize,
        kernel_size: Option<f32>,
        mip_splatting: Option<bool>,
        background_color: Option<[f32; 3]>,
        covars: Option<Vec<Covariance3D>>,
        quantization: Option<GaussianQuantization>,
    ) -> Self {
        let mut bbox: Aabb<f32> = Aabb::zeroed();
        for v in &gaussians {
            bbox.grow(&v.xyz);
        }

        let mut plane_fit = PlaneFit::default();
        for g in &gaussians {
            plane_fit.add(g.xyz.cast().unwrap());
        }
        let (center, mut up) = plane_fit.finish();

        if bbox.radius() < 10. {
            up = None;
        }
        Self {
            gaussians: bytemuck::cast_slice(&gaussians).to_vec(),
            sh_coefs: bytemuck::cast_slice(&sh_coefs).to_vec(),
            sh_deg,
            num_points,
            kernel_size,
            mip_splatting,
            background_color,
            covars,
            quantization,
            up: up,
            center,
            aabb: bbox,
            compressed: false,
        }
    }

    #[cfg(feature = "npz")]
    fn new_compressed(
        gaussians: Vec<GaussianCompressed>,
        sh_coefs: Vec<u8>,
        sh_deg: u32,
        num_points: usize,
        kernel_size: Option<f32>,
        mip_splatting: Option<bool>,
        background_color: Option<[f32; 3]>,
        covars: Option<Vec<Covariance3D>>,
        quantization: Option<GaussianQuantization>,
    ) -> Self {
        let mut bbox: Aabb<f32> = Aabb::unit();
        for v in &gaussians {
            bbox.grow(&v.xyz);
        }

        let mut plane_fit = PlaneFit::default();
        for g in &gaussians {
            plane_fit.add(g.xyz.cast().unwrap());
        }
        let (center, mut up) = plane_fit.finish();

        if bbox.radius() < 10. {
            up = None;
        }
        Self {
            gaussians: bytemuck::cast_slice(&gaussians).to_vec(),
            sh_coefs: bytemuck::cast_slice(&sh_coefs).to_vec(),
            sh_deg,
            num_points,
            kernel_size,
            mip_splatting,
            background_color,
            covars,
            quantization,
            up: up,
            center,
            aabb: bbox,
            compressed: true,
        }
    }

    pub fn gaussians(&self) -> anyhow::Result<&[Gaussian]> {
        if self.compressed {
            Err(anyhow::anyhow!("Gaussians are compressed"))
        } else {
            Ok(bytemuck::cast_slice(&self.gaussians))
        }
    }

    pub fn gaussians_compressed(&self) -> anyhow::Result<&[GaussianCompressed]> {
        if self.compressed {
            Err(anyhow::anyhow!("Gaussians are compressed"))
        } else {
            Ok(bytemuck::cast_slice(&self.gaussians))
        }
    }

    /// The gaussians and their sh coefficients as they are used for rendering without
    /// compression: compressed point clouds are decoded the way `preprocess_compressed.wgsl`
    /// does it.
    /// Only the points in `range` are converted, so that not all points have to be in memory
    /// twice.
    pub(crate) fn to_chunk(&self, range: std::ops::Range<usize>) -> GaussianChunk {
        if !self.compressed {
            let gaussians: &[Gaussian] = bytemuck::cast_slice(&self.gaussians);
            let sh_coefs: &[[[f16; 3]; 16]] = bytemuck::cast_slice(&self.sh_coefs);
            return GaussianChunk {
                gaussians: gaussians[range.clone()].to_vec(),
                sh_coefs: sh_coefs[range].to_vec(),
            };
        }
        let compressed: &[GaussianCompressed] =
            &bytemuck::cast_slice::<_, GaussianCompressed>(&self.gaussians)[range];
        let covars = self
            .covars
            .as_ref()
            .expect("compressed point cloud without covariances");
        let quantization = self
            .quantization
            .as_ref()
            .expect("compressed point cloud without quantization");
        let sh_values: &[i8] = bytemuck::cast_slice(&self.sh_coefs);
        let num_coefs = ((self.sh_deg + 1) * (self.sh_deg + 1)) as usize;
        let dequantize =
            |value: i8, q: &Quantization| (value as f32 - q.zero_point as f32) * q.scale;

        let mut gaussians = Vec::with_capacity(compressed.len());
        let mut sh_coefs = Vec::with_capacity(compressed.len());
        for g in compressed {
            let opacity = dequantize(g.opacity, &quantization.opacity);
            let scaling_factor = dequantize(g.scale_factor, &quantization.scaling_factor).exp();
            let cov = covars[g.geometry_idx as usize]
                .0
                .map(|c| f16::from_f32(c.to_f32() * scaling_factor * scaling_factor));
            gaussians.push(Gaussian::new(g.xyz, f16::from_f32(opacity), cov));

            let mut sh = [[f16::ZERO; 3]; 16];
            for (c, coef) in sh.iter_mut().enumerate().take(num_coefs) {
                let q = if c == 0 {
                    &quantization.color_dc
                } else {
                    &quantization.color_rest
                };
                for (k, value) in coef.iter_mut().enumerate() {
                    let i = 3 * (g.sh_idx as usize * num_coefs + c) + k;
                    *value = f16::from_f32(dequantize(sh_values[i], q));
                }
            }
            sh_coefs.push(sh);
        }
        GaussianChunk {
            gaussians,
            sh_coefs,
        }
    }

    pub fn sh_coefs_buffer(&self) -> &[u8] {
        &self.sh_coefs
    }

    pub fn gaussian_buffer(&self) -> &[u8] {
        &self.gaussians
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }
}
