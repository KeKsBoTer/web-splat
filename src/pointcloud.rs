use bytemuck::Zeroable;
use byteorder::{LittleEndian, WriteBytesExt};
use cgmath::{Point3, Vector4};
use clap::ValueEnum;
use half::f16;
use wgpu::Features;
use std::fmt::{Debug, Display};
use std::io::{self, BufReader, Read, Seek};
use std::mem;
use wgpu::util::DeviceExt;

use crate::gpu_rs::{GPURSSorter};
use crate::ply::PlyReader;
#[cfg(feature="npz")]
use crate::npz::NpzReader;
use crate::utils::max_supported_sh_deg;
use crate::PCDataType;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GaussianSplat {
    pub xyz: Point3<f16>,
    pub opacity: f16,
    pub geometry_idx: u32,
    pub sh_idx: u32,
}

impl Default for GaussianSplat {
    fn default() -> Self {
        GaussianSplat::zeroed()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GeometricInfo {
    pub covariance: [f16; 6],
    pub padding: f32,
}
impl Default for GeometricInfo {
    fn default() -> Self {
        GeometricInfo::zeroed()
    }
}

#[allow(dead_code)]
pub struct PointCloud {
    vertex_buffer: wgpu::Buffer,        // contains only 2 indices: index to the sh and covar entries
    sh_coef_buffer: wgpu::Buffer,       // contains the spherical harmonics
    covars_buffer: Option<wgpu::Buffer>,        // contains the covariances (includes alpha value and splat center)
    splat_2d_buffer: wgpu::Buffer,      
    
    // extra buffers needed for compressed rendering
    scaling: Option<wgpu::Buffer>,
    scaling_factor: Option<wgpu::Buffer>,
    rotation: Option<wgpu::Buffer>,
    opacity: Option<wgpu::Buffer>,
    features_indices: Option<wgpu::Buffer>,
    gaussian_indices: Option<wgpu::Buffer>,
    pc_uniform_infos: Option<wgpu::Buffer>,

    bind_group: wgpu::BindGroup,
    pub render_bind_group: wgpu::BindGroup,
    points: Option<Vec<GaussianSplat>>,
    num_points: u32,
    sh_deg: u32,
    sh_dtype: SHDType,
    // Fields needed for data sorting
    pub sorter: GPURSSorter,
    sorter_b_a: wgpu::Buffer,              // buffer a for keyval
    sorter_b_b: wgpu::Buffer,              // buffer b for keyval
    sorter_p_a: wgpu::Buffer,              // payload buffer a
    sorter_p_b: wgpu::Buffer,              // payload buffer b
    sorter_int: wgpu::Buffer,              // internal memory storage (used for histgoram calcs)
    pub sorter_uni: wgpu::Buffer,          // uniform buffer information
    pub sorter_dis: wgpu::Buffer,          // dispatch buffer
    pub sorter_dis_bg: wgpu::BindGroup,    // sorter dispatch bind group (needed mainly for the preprocess pipeline to set the correct dispatch count in shader)
    pub sorter_bg: wgpu::BindGroup,        // sorter bind group
    pub sorter_render_bg: wgpu::BindGroup, // bind group only with the sorted indices for rendering
    pub sorter_bg_pre: wgpu::BindGroup,    // bind group for the preprocess (is the sorter_dis and sorter_bg merged as we only have a limited amount of bgs for the preprocessing)
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.num_points)
            .finish()
    }
}

fn ply_header_info<R: io::BufRead + io::Seek>(buf_reader:&mut R) -> (u32, usize){
    let mut reader = crate::ply::PlyReader::new(buf_reader).unwrap();
    (reader.file_sh_deg().unwrap(), reader.num_points().unwrap())    
}
#[cfg(feature="npz")]
fn npz_header_info<R: io::BufRead + io::Seek>(buf_reader:&mut R) -> (u32, usize){
    let mut reader = crate::npz::NpzReader::new(buf_reader).unwrap();
    (reader.file_sh_deg().unwrap(), reader.num_points().unwrap())    
}

unsafe fn to_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts(
        (p as *const T) as  *const u8,
        ::core::mem::size_of::<T>())
}

impl PointCloud {
    pub fn load<R: Read + Seek>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        f: R,
        pc_data_type: PCDataType,
        load_compressed: bool,
        sh_dtype: SHDType,
        max_sh_deg: Option<u32>,
    ) -> Result<Self, anyhow::Error> {
        let mut reader = BufReader::new(f);
        let (file_sh_deg, num_points) = match pc_data_type {
            PCDataType::PLY => ply_header_info(&mut reader),
            #[cfg(feature="npz")]
            PCDataType::NPZ => npz_header_info(&mut reader),
            _ => return Err(anyhow::anyhow!("Cannot parse data from point cloud data type")), 
        };
        let max_buffer_size = device
            .limits()
            .max_buffer_size
            .max(device.limits().max_storage_buffer_binding_size as u64);

        let target_sh_deg = max_sh_deg.unwrap_or(file_sh_deg);
        let sh_deg =
            max_supported_sh_deg(max_buffer_size, num_points as u64, sh_dtype, target_sh_deg)
                .ok_or(anyhow::anyhow!(
                    "cannot fit sh coefs into a buffer (exceeds WebGPU's max_buffer_size)"
                ))?;
        log::info!("num_points: {num_points}, sh_deg: {sh_deg}, sh_dtype: {sh_dtype}");
        if sh_deg < target_sh_deg {
            log::warn!("color sh degree (file: {file_sh_deg}) was decreased to degree {sh_deg} to fit the coefficients into memory")
        }

        let bind_group_layout = Self::bind_group_layout(device, load_compressed);

        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d gaussians buffer"),
            size: (num_points * mem::size_of::<Splat2D>()) as u64,
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

        let mut vertex_buffer;
        let mut sh_buffer;
        let mut covars_buffer = None;        
        let mut bind_group;
        let mut vertices = None;
        let mut sh_coefs;
        let mut covars;
        
        let mut scaling_buffer = None;
        let mut scaling_factor_buffer = None;
        let mut rotation_buffer = None;
        let mut opacity_buffer = None;
        let mut feature_indices_buffer = None;
        let mut gaussian_indices_buffer = None;
        let mut pc_uniform_infos_buffer = None;
        if !load_compressed {
            reader.seek(std::io::SeekFrom::Start(0)); // have to reset reader to start of file
            let (vertices_t, sh_coefs_t, covars_t) = match pc_data_type {
                PCDataType::PLY => PlyReader::new(reader).unwrap().read(sh_dtype, sh_deg)?,
                #[cfg(feature="npz")]
                PCDataType::NPZ => NpzReader::new(&mut reader).unwrap().read(sh_dtype, sh_deg)?,
                _ => return Err(anyhow::anyhow!("Unknown point cloud data type")),
            };
            vertices = Some(vertices_t);
            sh_coefs = sh_coefs_t;
            covars = covars_t;

            sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sh coefs buffer"),
                contents: bytemuck::cast_slice(sh_coefs.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            covars_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Covariances buffer"),
                contents: bytemuck::cast_slice(covars.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("3d gaussians buffer"),
                contents: bytemuck::cast_slice(vertices.as_ref().expect("missing vertices").as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        resource: covars_buffer.as_ref().expect("missing covars buffer").as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: splat_2d_buffer.as_entire_binding(),
                    },
                ],
            });
        }
        else {
            reader.seek(std::io::SeekFrom::Start(0)); // have to reset reader to start of file
            let data = match pc_data_type {
                PCDataType::PLY => PlyReader::new(reader).unwrap().read_compressed(sh_deg)?,
                #[cfg(feature="npz")]
                PCDataType::NPZ => NpzReader::new(&mut reader).unwrap().read_compressed(sh_deg)?,
                _ => return Err(anyhow::anyhow!("Unknown point cloud data type")),
            };

            sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sh coefs buffer"),
                contents: bytemuck::cast_slice(data.features.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let scaling_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Covariances buffer"),
                contents: bytemuck::cast_slice(data.scaling.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let scaling_factor_buffer = match data.scaling_factor{
                Some(scale) => Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Scale factor"),
                    contents: bytemuck::cast_slice(scale.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })),
                None => None
            };
            rotation_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Rotation buffer"),
                contents: bytemuck::cast_slice(data.rotation.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("3d positions buffer"),
                contents: bytemuck::cast_slice(data.xyz.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            opacity_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
                label: Some("opacity buffer"),
                contents: bytemuck::cast_slice(data.opacity.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            feature_indices_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("feature indices"),
                contents: bytemuck::cast_slice(data.feature_indices.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            gaussian_indices_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gaussian indices"),
                contents: bytemuck::cast_slice(data.gaussian_indices.as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            pc_uniform_infos_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("point cloud uniforms"),
                contents: unsafe {to_u8_slice(&data.compressed_s_zp)},
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }));

            bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("point cloud cpmoressed bind group"),
                layout: &bind_group_layout ,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: vertex_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: scaling_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: scaling_factor_buffer.as_ref().unwrap_or(&scaling_buffer).as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: rotation_buffer.as_ref().expect("missing rotation buffer").as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: opacity_buffer.as_ref().expect("missing opacity buffer").as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: sh_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: feature_indices_buffer.as_ref().expect("missing feature indices bufer").as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: gaussian_indices_buffer.as_ref().expect("missing gaussian indices buffer").as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: splat_2d_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: pc_uniform_infos_buffer.as_ref().expect("missing pc uniform infos buffer").as_entire_binding(),
                    },
                ],
            });
        }

        let sorter = GPURSSorter::new(device, queue);
        let (sorter_b_a, sorter_b_b, sorter_p_a, sorter_p_b) =
            GPURSSorter::create_keyval_buffers(device, num_points, 4);
        let sorter_int = sorter.create_internal_mem_buffer(device, num_points);
        let (sorter_uni, sorter_dis, sorter_bg, sorter_dis_bg) = sorter.create_bind_group(
            device,
            num_points,
            &sorter_int,
            &sorter_b_a,
            &sorter_b_b,
            &sorter_p_a,
            &sorter_p_b,
        );
        let sorter_render_bg = sorter.create_bind_group_render(device, &sorter_uni, &sorter_p_a);
        let sorter_bg_pre = sorter.create_bind_group_preprocess(
            device,
            &sorter_uni,
            &sorter_dis,
            &sorter_int,
            &sorter_b_a,
            &sorter_b_b,
            &sorter_p_a,
            &sorter_p_b,
        );

        Ok(Self {
            vertex_buffer,
            sh_coef_buffer: sh_buffer,
            covars_buffer,
            splat_2d_buffer,
            
            scaling: scaling_buffer,
            scaling_factor: scaling_factor_buffer,
            rotation: rotation_buffer,
            opacity: opacity_buffer,
            features_indices: feature_indices_buffer,
            gaussian_indices: gaussian_indices_buffer,
            pc_uniform_infos: pc_uniform_infos_buffer,
            
            bind_group,
            render_bind_group,
            num_points: num_points as u32,
            points: vertices,
            sh_deg,
            sh_dtype,
            sorter,
            sorter_b_a,
            sorter_b_b,
            sorter_p_a,
            sorter_p_b,
            sorter_int,
            sorter_uni,
            sorter_dis,
            sorter_dis_bg,
            sorter_bg,
            sorter_render_bg,
            sorter_bg_pre,
        })
    }

    pub fn num_points(&self) -> u32 {
        self.num_points
    }

    pub fn points(&self) -> &Vec<GaussianSplat> {
        &self.points.as_ref().expect("Points are not available")
    }
    pub fn sh_deg(&self) -> u32 {
        self.sh_deg
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    pub fn sh_dtype(&self) -> SHDType {
        self.sh_dtype
    }

    pub fn bind_group_layout(device: &wgpu::Device, data_compressed: bool) -> wgpu::BindGroupLayout {
        if !data_compressed {
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
                ],
            })
        }
        else {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("point cloud compressed bind group layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
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
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat2D {
    v: Vector4<f16>,
    pos: Vector4<f16>,
    color: Vector4<u8>,
    _pad: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, ValueEnum)]
pub enum SHDType {
    Float = 0,
    Half = 1,
    Byte = 2,
}

impl Display for SHDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SHDType::Float => f.write_str("f32"),
            SHDType::Half => f.write_str("f16"),
            SHDType::Byte => f.write_str("i8"),
        }
    }
}

impl SHDType {
    pub fn write_to<W: io::Write>(&self, writer: &mut W, v: f32, idx: u32) -> io::Result<()> {
        let scale = if idx == 0 { 4. } else { 0.5 };
        match self {
            SHDType::Float => writer.write_f32::<LittleEndian>(v),
            SHDType::Half => writer.write_u16::<LittleEndian>(f16::from_f32(v).to_bits()),
            SHDType::Byte => writer.write_i8((v * 127. / scale) as i8),
        }
    }

    pub fn packed_size(&self) -> usize {
        match self {
            SHDType::Float => mem::size_of::<f32>(),
            SHDType::Half => mem::size_of::<f16>(),
            SHDType::Byte => mem::size_of::<i8>(),
        }
    }
}

pub struct CompressedScaleZeroPoint {
    pub opacity_s: f32,
    pub opacity_zp: i32,
    pub scaling_s: f32,
    pub scaling_zp: i32,
    pub rotation_s: f32,
    pub rotation_zp: i32,
    pub features_s: f32,
    pub features_zp: i32,
    pub features_rest_s: f32,
    pub features_rest_zp: i32,
    pub scaling_factor_s: f32,
    pub scaling_factor_zp: i32,
}

pub struct PCCompressed{
    pub compressed_s_zp: CompressedScaleZeroPoint,

    pub xyz: Vec<f16>,
    pub scaling: Vec<i8>,
    pub scaling_factor: Option<Vec<i8>>,
    pub rotation: Vec<i8>,
    pub opacity: Vec<i8>,
    pub features: Vec<i8>,
    pub feature_indices: Vec<u32>,
    pub gaussian_indices: Vec<u32>,
}
pub trait PointCloudReader {
    fn read(
        &mut self,
        sh_dtype: SHDType,
        sh_deg: u32,
    ) -> Result<(Vec<GaussianSplat>, Vec<u8>, Vec<GeometricInfo>), anyhow::Error>;
    
    fn read_compressed(
        &mut self,
        sh_deg: u32,
    ) -> Result<PCCompressed, anyhow::Error>;

    fn file_sh_deg(&self) -> Result<u32, anyhow::Error>;

    fn num_points(&self) -> Result<usize, anyhow::Error>;
}
