use anyhow::Context;
use half::f16;
use serde::de::DeserializeSeed;
use serde::{Deserialize, Serialize};
use serde_ply::{PlyChunkedReader, PlyHeader, RowVisitor};

use std::io::{self, Read, Seek};

use cgmath::{InnerSpace, Point3, Quaternion, Vector3};

use crate::{
    pointcloud::Gaussian,
    utils::{build_cov, sh_deg_from_num_coefs, sigmoid},
};

use super::{GenericGaussianPointCloud, PointCloudReader};

/// the header is expected to be smaller than this
const MAX_HEADER_SIZE: usize = 1 << 20;

/// number of spherical harmonics coefficients per color channel for degree 3
const MAX_SH_REST: usize = 15;

/// One row of the vertex element, in the order of the original 3DGS file layout
/// (which lets serde-ply take its fast path).
/// SH coefficients of degree < 3 are missing in the file and default to zero.
#[derive(Deserialize, Serialize)]
struct Vertex {
    x: f32,
    y: f32,
    z: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    f_dc_0: f32,
    f_dc_1: f32,
    f_dc_2: f32,
    #[serde(default)]
    f_rest_0: f32,
    #[serde(default)]
    f_rest_1: f32,
    #[serde(default)]
    f_rest_2: f32,
    #[serde(default)]
    f_rest_3: f32,
    #[serde(default)]
    f_rest_4: f32,
    #[serde(default)]
    f_rest_5: f32,
    #[serde(default)]
    f_rest_6: f32,
    #[serde(default)]
    f_rest_7: f32,
    #[serde(default)]
    f_rest_8: f32,
    #[serde(default)]
    f_rest_9: f32,
    #[serde(default)]
    f_rest_10: f32,
    #[serde(default)]
    f_rest_11: f32,
    #[serde(default)]
    f_rest_12: f32,
    #[serde(default)]
    f_rest_13: f32,
    #[serde(default)]
    f_rest_14: f32,
    #[serde(default)]
    f_rest_15: f32,
    #[serde(default)]
    f_rest_16: f32,
    #[serde(default)]
    f_rest_17: f32,
    #[serde(default)]
    f_rest_18: f32,
    #[serde(default)]
    f_rest_19: f32,
    #[serde(default)]
    f_rest_20: f32,
    #[serde(default)]
    f_rest_21: f32,
    #[serde(default)]
    f_rest_22: f32,
    #[serde(default)]
    f_rest_23: f32,
    #[serde(default)]
    f_rest_24: f32,
    #[serde(default)]
    f_rest_25: f32,
    #[serde(default)]
    f_rest_26: f32,
    #[serde(default)]
    f_rest_27: f32,
    #[serde(default)]
    f_rest_28: f32,
    #[serde(default)]
    f_rest_29: f32,
    #[serde(default)]
    f_rest_30: f32,
    #[serde(default)]
    f_rest_31: f32,
    #[serde(default)]
    f_rest_32: f32,
    #[serde(default)]
    f_rest_33: f32,
    #[serde(default)]
    f_rest_34: f32,
    #[serde(default)]
    f_rest_35: f32,
    #[serde(default)]
    f_rest_36: f32,
    #[serde(default)]
    f_rest_37: f32,
    #[serde(default)]
    f_rest_38: f32,
    #[serde(default)]
    f_rest_39: f32,
    #[serde(default)]
    f_rest_40: f32,
    #[serde(default)]
    f_rest_41: f32,
    #[serde(default)]
    f_rest_42: f32,
    #[serde(default)]
    f_rest_43: f32,
    #[serde(default)]
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

impl Vertex {
    fn sh_rest(&self) -> [f32; MAX_SH_REST * 3] {
        [
            self.f_rest_0,
            self.f_rest_1,
            self.f_rest_2,
            self.f_rest_3,
            self.f_rest_4,
            self.f_rest_5,
            self.f_rest_6,
            self.f_rest_7,
            self.f_rest_8,
            self.f_rest_9,
            self.f_rest_10,
            self.f_rest_11,
            self.f_rest_12,
            self.f_rest_13,
            self.f_rest_14,
            self.f_rest_15,
            self.f_rest_16,
            self.f_rest_17,
            self.f_rest_18,
            self.f_rest_19,
            self.f_rest_20,
            self.f_rest_21,
            self.f_rest_22,
            self.f_rest_23,
            self.f_rest_24,
            self.f_rest_25,
            self.f_rest_26,
            self.f_rest_27,
            self.f_rest_28,
            self.f_rest_29,
            self.f_rest_30,
            self.f_rest_31,
            self.f_rest_32,
            self.f_rest_33,
            self.f_rest_34,
            self.f_rest_35,
            self.f_rest_36,
            self.f_rest_37,
            self.f_rest_38,
            self.f_rest_39,
            self.f_rest_40,
            self.f_rest_41,
            self.f_rest_42,
            self.f_rest_43,
            self.f_rest_44,
        ]
    }
}

fn convert_vertex(v: &Vertex, sh_deg: usize) -> (Gaussian, [[f16; 3]; 16]) {
    let num_coefs = (sh_deg + 1) * (sh_deg + 1);
    let sh_rest = v.sh_rest();
    let mut sh: [[f32; 3]; 16] = [[0.; 3]; 16];
    sh[0] = [v.f_dc_0, v.f_dc_1, v.f_dc_2];

    // higher order coefficients are stored with channel first (shape:[N,3,C])
    for i in 0..(num_coefs - 1) {
        for j in 0..3 {
            sh[i + 1][j] = sh_rest[j * (num_coefs - 1) + i];
        }
    }

    let opacity = sigmoid(v.opacity);
    let scale = Vector3::new(v.scale_0.exp(), v.scale_1.exp(), v.scale_2.exp());
    let rot = Quaternion::new(v.rot_0, v.rot_1, v.rot_2, v.rot_3).normalize();
    let cov = build_cov(rot, scale);

    (
        Gaussian::new(
            Point3::new(v.x, v.y, v.z).cast().unwrap(),
            f16::from_f32(opacity),
            cov.map(|x| f16::from_f32(x)),
        ),
        sh.map(|x| x.map(|y| f16::from_f32(y))),
    )
}

/// Properties of a point cloud that are known from the ply header, before any point is read.
#[derive(Clone, Debug)]
pub struct PlyInfo {
    pub num_points: usize,
    pub sh_deg: u32,
    pub mip_splatting: Option<bool>,
    pub kernel_size: Option<f32>,
    pub background_color: Option<[f32; 3]>,
}

impl PlyInfo {
    fn from_header(header: &PlyHeader) -> Result<Self, anyhow::Error> {
        if header.elem_defs.first().map(|e| e.name.as_str()) != Some("vertex") {
            anyhow::bail!("vertex must be the first element in the ply file");
        }
        Ok(Self {
            num_points: num_points(header)?,
            sh_deg: file_sh_deg(header)?,
            mip_splatting: mip_splatting(header)?,
            kernel_size: kernel_size(header)?,
            background_color: background_color(header)
                .map_err(|e| log::warn!("could not parse background_color: {}", e))
                .unwrap_or_default(),
        })
    }
}

/// A batch of points that was read from a ply file and converted for rendering.
#[derive(Default)]
pub struct GaussianChunk {
    pub gaussians: Vec<Gaussian>,
    pub sh_coefs: Vec<[[f16; 3]; 16]>,
}

impl GaussianChunk {
    pub fn len(&self) -> usize {
        self.gaussians.len()
    }

    fn append(&mut self, mut other: GaussianChunk) {
        self.gaussians.append(&mut other.gaussians);
        self.sh_coefs.append(&mut other.sh_coefs);
    }
}

/// Incremental ply parser: feed it bytes of a ply file with [`Self::push`] (in pieces of any
/// size) and get the points that are complete so far.
pub struct PlyStream {
    reader: PlyChunkedReader,
    info: Option<PlyInfo>,
    loaded: usize,
}

impl PlyStream {
    pub fn new() -> Self {
        Self {
            reader: PlyChunkedReader::new(),
            info: None,
            loaded: 0,
        }
    }

    /// available as soon as the header was received
    pub fn info(&self) -> Option<&PlyInfo> {
        self.info.as_ref()
    }

    pub fn loaded_points(&self) -> usize {
        self.loaded
    }

    pub fn is_complete(&self) -> bool {
        self.info
            .as_ref()
            .is_some_and(|i| self.loaded >= i.num_points)
    }

    /// Adds data and returns the points that could be read from it.
    /// [`Self::info`] is set once the header was received, check it before using the chunk.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Option<GaussianChunk>, anyhow::Error> {
        self.reader.buffer_mut().extend_from_slice(bytes);

        if self.info.is_none() {
            let buffered = self.reader.buffer_mut().len();
            match self.reader.header() {
                Some(header) => self.info = Some(PlyInfo::from_header(header)?),
                None if buffered > MAX_HEADER_SIZE => {
                    anyhow::bail!("could not parse ply header")
                }
                // header is not complete yet
                None => return Ok(None),
            }
        }
        if self.is_complete() {
            // ignore everything after the vertices
            self.reader.buffer_mut().clear();
            return Ok(None);
        }
        let info = self.info.as_ref().unwrap();

        let mut rows: Vec<Vertex> = Vec::new();
        RowVisitor::new(|v: Vertex| rows.push(v))
            .deserialize(&mut self.reader)
            .context("could not read vertices")?;
        self.loaded += rows.len();
        Ok(Some(convert_vertices(&rows, info.sh_deg as usize)))
    }
}

/// converts rows to gaussians, in parallel where threads are available
fn convert_vertices(rows: &[Vertex], sh_deg: usize) -> GaussianChunk {
    #[cfg(not(target_arch = "wasm32"))]
    let (gaussians, sh_coefs) = {
        use rayon::prelude::*;
        rows.par_iter()
            .with_min_len(1024)
            .map(|v| convert_vertex(v, sh_deg))
            .unzip()
    };
    #[cfg(target_arch = "wasm32")]
    let (gaussians, sh_coefs) = rows.iter().map(|v| convert_vertex(v, sh_deg)).unzip();
    GaussianChunk {
        gaussians,
        sh_coefs,
    }
}

fn file_sh_deg(header: &PlyHeader) -> Result<u32, anyhow::Error> {
    let num_sh_coefs = header
        .elem_defs
        .iter()
        .find(|e| e.name == "vertex")
        .ok_or(anyhow::anyhow!("missing element vertex"))?
        .properties
        .iter()
        .filter(|p| p.name.starts_with("f_dc_") || p.name.starts_with("f_rest_"))
        .count();

    let file_sh_deg = sh_deg_from_num_coefs(num_sh_coefs as u32 / 3).ok_or(anyhow::anyhow!(
        "number of sh coefficients {num_sh_coefs} cannot be mapped to sh degree"
    ))?;
    Ok(file_sh_deg)
}

fn num_points(header: &PlyHeader) -> Result<usize, anyhow::Error> {
    Ok(header
        .elem_defs
        .iter()
        .find(|e| e.name == "vertex")
        .ok_or(anyhow::anyhow!("missing element vertex"))?
        .count)
}

fn mip_splatting(header: &PlyHeader) -> Result<Option<bool>, anyhow::Error> {
    Ok(header
        .comments
        .iter()
        .find(|c| c.contains("mip"))
        .map(|c| c.split('=').last().unwrap().to_lowercase().parse::<bool>())
        .transpose()?)
}
fn kernel_size(header: &PlyHeader) -> Result<Option<f32>, anyhow::Error> {
    Ok(header
        .comments
        .iter()
        .find(|c| c.contains("kernel_size"))
        .map(|c| c.split('=').last().unwrap().parse::<f32>())
        .transpose()?)
}

fn background_color(header: &PlyHeader) -> anyhow::Result<Option<[f32; 3]>> {
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

/// Reads a whole ply file at once.
pub struct PlyReader<R: Read + Seek> {
    reader: R,
}

impl<R: io::Read + io::Seek> PlyReader<R> {
    pub fn new(reader: R) -> Result<Self, anyhow::Error> {
        Ok(Self { reader })
    }
}

impl<R: io::Read + io::Seek> PointCloudReader for PlyReader<R> {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error> {
        let mut stream = PlyStream::new();
        let mut points = GaussianChunk::default();
        let mut buf = vec![0u8; 16 << 20];
        while !stream.is_complete() {
            let n = self.reader.read(&mut buf)?;
            if n == 0 {
                anyhow::bail!(
                    "unexpected end of file: read {} of {} points",
                    stream.loaded_points(),
                    stream.info().map_or(0, |i| i.num_points)
                );
            }
            let chunk = stream.push(&buf[..n])?;
            if let Some(info) = stream.info() {
                points
                    .gaussians
                    .reserve_exact(info.num_points.saturating_sub(points.len()));
                points
                    .sh_coefs
                    .reserve_exact(info.num_points.saturating_sub(points.len()));
            }
            if let Some(chunk) = chunk {
                points.append(chunk);
            }
        }
        let info = stream.info().unwrap().clone();
        Ok(GenericGaussianPointCloud::new(
            points.gaussians,
            points.sh_coefs,
            info.sh_deg,
            info.num_points,
            info.kernel_size,
            info.mip_splatting,
            info.background_color,
            None,
            None,
        ))
    }

    fn magic_bytes() -> &'static [u8] {
        "ply".as_bytes()
    }

    fn file_ending() -> &'static str {
        "ply"
    }
}
