use bytemuck::Zeroable;
use cgmath::{
    Array, BaseNum, ElementWise, EuclideanSpace, InnerSpace, MetricSpace, Point3, Vector2, Vector3,
    Vector4,
};
use half::f16;
use num_traits::Float;
use std::fmt::Debug;
use std::io::{BufReader, Read, Seek};
use std::mem;
use wgpu::util::DeviceExt;

#[cfg(feature = "npz")]
use crate::npz::NpzReader;
use crate::ply::PlyReader;
#[cfg(feature = "npz")]
use crate::uniform::UniformBuffer;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Gaussian {
    pub xyz: Point3<f16>,
    pub opacity: i8,
    pub scale_factor: i8,
    pub geometry_idx: u32,
    pub sh_idx: u32,
}

impl Default for Gaussian {
    fn default() -> Self {
        Gaussian::zeroed()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GeometricInfo {
    pub covariance: [f16; 6],
}
impl Default for GeometricInfo {
    fn default() -> Self {
        GeometricInfo::zeroed()
    }
}

#[allow(dead_code)]
pub struct PointCloud {
    splat_2d_buffer: wgpu::Buffer,

    bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
    num_points: u32,
    sh_deg: u32,
    bbox: Aabb<f32>,
    compressed: bool,

    pub center: Point3<f32>,
    pub up: Option<Vector3<f32>>,

    mip_splatting: Option<bool>,
    kernel_size: Option<f32>,
    background_color: Option<wgpu::Color>,
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.num_points)
            .finish()
    }
}

impl PointCloud {
    pub fn load<R: Read + Seek>(device: &wgpu::Device, f: R) -> Result<Self, anyhow::Error> {
        let mut signature: [u8; 4] = [0; 4];
        let mut f = f;
        f.read_exact(&mut signature)?;
        f.rewind()?;
        if signature.starts_with(PlyReader::<R>::magic_bytes()) {
            return Self::load_ply(&device, f);
        }
        #[cfg(feature = "npz")]
        if signature.starts_with(NpzReader::<R>::magic_bytes()) {
            return Self::load_npz(&device, f);
        }
        return Err(anyhow::anyhow!("Unknown file format"));
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }

    #[cfg(feature = "npz")]
    pub fn load_npz<R: Read + Seek>(device: &wgpu::Device, f: R) -> Result<Self, anyhow::Error> {
        let mut reader = BufReader::new(f);
        let mut npz_reader = NpzReader::new(&mut reader)?;
        let (file_sh_deg, num_points) = (npz_reader.file_sh_deg()?, npz_reader.num_points()?);
        let sh_deg = file_sh_deg;
        log::info!("num_points: {num_points}, sh_deg: {sh_deg}");

        let bind_group_layout = Self::bind_group_layout(device);

        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d gaussians buffer"),
            size: (num_points * mem::size_of::<Splat>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud rendering bind group"),
            layout: &Self::bind_group_layout_render(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 2,
                resource: splat_2d_buffer.as_entire_binding(),
            }],
        });

        let (vertices, sh_coefs, covars, quantization) = npz_reader.read(sh_deg)?;
        let points: Vec<Point3<f32>> = vertices.iter().map(|g| g.xyz.map(|v| v.to_f32())).collect();

        let mut bbox: Aabb<f16> = Aabb::unit();
        for v in &vertices {
            bbox.grow(v.xyz);
        }
        let (center, mut up) = plane_from_points(points.as_slice());

        if bbox.sphere() < f16::from_f32_const(10.) {
            up = None;
        }
        log::info!("up vector: {:?}", up);
        let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh coefs buffer"),
            contents: bytemuck::cast_slice(sh_coefs.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let covars_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Covariances buffer"),
            contents: bytemuck::cast_slice(covars.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3d gaussians buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let quantization_uniform =
            UniformBuffer::new(device, quantization, Some("quantization uniform buffer"));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: covars_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: splat_2d_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: quantization_uniform.buffer().as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            splat_2d_buffer,

            bind_group,
            render_bind_group,
            num_points: num_points as u32,
            sh_deg,
            bbox: bbox.into(),
            compressed: true,
            center,
            up,
            mip_splatting: None,
            kernel_size: None,
            background_color: None,
        })
    }

    pub fn load_ply<R: Read + Seek>(device: &wgpu::Device, f: R) -> Result<Self, anyhow::Error> {
        let reader = BufReader::new(f);

        let mut ply_reader = PlyReader::new(reader).unwrap();

        let sh_deg = ply_reader.file_sh_deg()?;
        let num_points = ply_reader.num_points()?;
        log::info!("num_points: {num_points}, sh_deg: {sh_deg}");

        let bind_group_layout = Self::bind_group_layout_float(device);

        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d gaussians buffer"),
            size: (num_points * mem::size_of::<Splat>()) as u64,
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud rendering bind group"),
            layout: &Self::bind_group_layout_render(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 2,
                resource: splat_2d_buffer.as_entire_binding(),
            }],
        });

        let (vertices, sh_coefs) = ply_reader.read()?;
        let mut bbox = Aabb::unit();
        for v in &vertices {
            bbox.grow(v.xyz);
        }
        let points: Vec<Point3<f32>> = vertices.iter().map(|g| g.xyz.map(|v| v.to_f32())).collect();

        let (center, up) = plane_from_points(points.as_slice());
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3d gaussians buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh coefs buffer"),
            contents: bytemuck::cast_slice(
                sh_coefs
                    .into_iter()
                    .flatten()
                    .collect::<Vec<[f16; 3]>>()
                    .as_slice(),
            ),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: splat_2d_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            splat_2d_buffer,

            bind_group,
            render_bind_group,
            num_points: num_points as u32,
            sh_deg,
            compressed: false,
            bbox: bbox.into(),
            center,
            up,
            mip_splatting: ply_reader
                .mip_splatting()
                .inspect_err(|e| log::warn!("cannot read parameter mip_splatting: {e:?}"))
                .ok(),
            kernel_size: ply_reader
                .kernel_size()
                .inspect_err(|e| log::warn!("cannot read kernel size: {e:?}"))
                .ok(),
            background_color: ply_reader
                .background_color()
                .inspect_err(|e| log::warn!("cannot read background color: {e:?}"))
                .ok(),
        })
    }

    pub fn num_points(&self) -> u32 {
        self.num_points
    }

    pub fn sh_deg(&self) -> u32 {
        self.sh_deg
    }

    pub fn bbox(&self) -> &Aabb<f32> {
        &self.bbox
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn bind_group_layout_float(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud float bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn bind_group_layout_render(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud rendering bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub fn mip_splatting(&self) -> Option<bool> {
        self.mip_splatting
    }
    pub fn dilation_kernel_size(&self) -> Option<f32> {
        self.kernel_size
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat {
    pub v: Vector4<f16>,
    pub pos: Vector2<f16>,
    pub color: Vector4<f16>,
}

pub trait PointCloudReader {
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
    >;

    fn file_sh_deg(&self) -> Result<u32, anyhow::Error>;

    fn num_points(&self) -> Result<usize, anyhow::Error>;
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Quantization {
    pub zero_point: i32,
    pub scale: f32,
    _pad: [u32; 2],
}

impl Quantization {
    #[cfg(feature = "npz")]
    pub fn new(zero_point: i32, scale: f32) -> Self {
        Quantization {
            zero_point,
            scale,
            ..Default::default()
        }
    }
}

impl Default for Quantization {
    fn default() -> Self {
        Self {
            zero_point: 0,
            scale: 1.,
            _pad: [0, 0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct QuantizationUniform {
    pub color_dc: Quantization,
    pub color_rest: Quantization,
    pub opacity: Quantization,
    pub scaling_factor: Quantization,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianFloat {
    pub xyz: Point3<f16>,
    pub opacity: f16,
    pub cov: [f16; 6],
}

pub struct Aabb<F: Float + BaseNum> {
    min: Point3<F>,
    max: Point3<F>,
}

impl<F: Float + BaseNum> Aabb<F> {
    pub fn new(min: Point3<F>, max: Point3<F>) -> Self {
        Self { min, max }
    }

    pub fn grow(&mut self, pos: Point3<F>) {
        self.min.x = self.min.x.min(pos.x);
        self.min.y = self.min.y.min(pos.y);
        self.min.z = self.min.z.min(pos.z);
        self.max.x = self.max.x.max(pos.x);
        self.max.y = self.max.y.max(pos.y);
        self.max.z = self.max.z.max(pos.z);
    }

    pub fn corners(&self) -> [Point3<F>; 8] {
        [
            Vector3::new(F::zero(), F::zero(), F::zero()),
            Vector3::new(F::one(), F::zero(), F::zero()),
            Vector3::new(F::zero(), F::one(), F::zero()),
            Vector3::new(F::one(), F::one(), F::zero()),
            Vector3::new(F::zero(), F::zero(), F::one()),
            Vector3::new(F::one(), F::zero(), F::one()),
            Vector3::new(F::zero(), F::one(), F::one()),
            Vector3::new(F::one(), F::one(), F::one()),
        ]
        .map(|d| self.min + self.max.to_vec().mul_element_wise(d))
    }

    pub fn unit() -> Self {
        Self {
            min: Point3::new(-F::one(), -F::one(), -F::one()),
            max: Point3::new(F::one(), F::one(), F::one()),
        }
    }

    pub fn center(&self) -> Point3<F> {
        self.min.midpoint(self.max)
    }

    pub fn sphere(&self) -> F {
        self.min.distance(self.max) / (F::one() + F::one())
    }
}

impl Into<Aabb<f32>> for Aabb<f16> {
    fn into(self) -> Aabb<f32> {
        Aabb {
            min: self.min.map(|v| v.into()),
            max: self.max.map(|v| v.into()),
        }
    }
}

// Fit a plane to a collection of points.
// Fast, and accurate to within a few degrees.
// Returns None if the points do not span a plane.
// see http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
fn plane_from_points(points: &[Point3<f32>]) -> (Point3<f32>, Option<Vector3<f32>>) {
    let n = points.len();

    let mut sum = Point3 {
        x: 0.0f32,
        y: 0.0f32,
        z: 0.0f32,
    };
    for p in points {
        sum = &sum + p.to_vec();
    }
    let centroid = &sum * (1.0 / (n as f32));
    if n < 3 {
        return (centroid, None);
    }

    // Calculate full 3x3 covariance matrix, excluding symmetries:
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut xz = 0.0;
    let mut yy = 0.0;
    let mut yz = 0.0;
    let mut zz = 0.0;

    for p in points {
        let r = p - centroid;
        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    xx /= n as f32;
    xy /= n as f32;
    xz /= n as f32;
    yy /= n as f32;
    yz /= n as f32;
    zz /= n as f32;

    let mut weighted_dir = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    {
        let det_x = yy * zz - yz * yz;
        let axis_dir = Vector3 {
            x: det_x,
            y: xz * yz - xy * zz,
            z: xy * yz - xz * yy,
        };
        let mut weight = det_x * det_x;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += &axis_dir * weight;
    }

    {
        let det_y = xx * zz - xz * xz;
        let axis_dir = Vector3 {
            x: xz * yz - xy * zz,
            y: det_y,
            z: xy * xz - yz * xx,
        };
        let mut weight = det_y * det_y;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += &axis_dir * weight;
    }

    {
        let det_z = xx * yy - xy * xy;
        let axis_dir = Vector3 {
            x: xy * yz - xz * yy,
            y: xy * xz - yz * xx,
            z: det_z,
        };
        let mut weight = det_z * det_z;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += &axis_dir * weight;
    }

    let mut normal = weighted_dir.normalize();

    if normal.dot(Vector3::unit_y()) < 0. {
        normal = -normal;
    }
    if normal.is_finite() {
        (centroid, Some(normal))
    } else {
        (centroid, None)
    }
}
