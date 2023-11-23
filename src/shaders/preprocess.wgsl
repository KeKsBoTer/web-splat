
//const MAX_SH_DEG:u32 = <injected>u;

const SH_C0:f32 = 0.28209479177387814;

const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);

const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);


struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Quantization {
    zero_point: i32,
    scaling: f32
}

struct QuantizationUniforms {
    @align(16) color_dc: Quantization,
    @align(16) color_rest: Quantization,
    @align(16) opacity: Quantization,
    @align(16) scaling_factor: Quantization,
}

struct GaussianSplat {
    // 2x f16 xy
    pos_xy: u32,
    // 1x f16 z, int8 opacity, int8 scale_factor, 
    pos_zw: u32,
    geometry_idx: u32,
    sh_idx: u32,
};

struct GeometricInfo {
    cov: array<u32, 3>,
};

struct Splats2D {
     // 4x f16 packed as u32
    v_0: u32, v_1: u32,
    // 2x f16 packed as u32
    pos: u32,
    // rgba packed as u8
    color: u32
};

struct DrawIndirect {
    /// The number of vertices to draw.
    vertex_count: u32,
    /// The number of instances to draw.
    instance_count: atomic<u32>,
    /// The Index of the first vertex to draw.
    base_vertex: u32,
    /// The instance ID of the first instance to draw.
    /// Has to be 0, unless [`Features::INDIRECT_FIRST_INSTANCE`](crate::Features::INDIRECT_FIRST_INSTANCE) is enabled.
    base_instance: u32,
}

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,     // essentially contains the same info as instance_count in DrawIndirect
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0) 
var<storage,read> vertices : array<GaussianSplat>;

// sh coefs packed as 4x u8 = 1x u32
@group(1) @binding(1) 
var<storage,read> sh_coefs : array<u32>;
@group(1) @binding(2)
var<storage,read> geometries : array<GeometricInfo>;
@group(1) @binding(3) 
var<storage,read_write> points_2d : array<Splats2D>;
@group(1) @binding(4) 
var<uniform> quantization : QuantizationUniforms;

@group(2) @binding(0) 
var<storage,read_write> indirect_draw_call : DrawIndirect;
@group(3) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(3) @binding(2)
var<storage, read_write> sort_depths : array<u32>;
@group(3) @binding(4)
var<storage, read_write> sort_indices : array<u32>;
@group(3) @binding(6)
var<storage, read_write> sort_dispatch: DispatchIndirect;


fn dequantize(value: i32, quantization: Quantization) -> f32 {
    return (f32(value) - f32(quantization.zero_point)) * quantization.scaling;
}

fn dequantizef4(value: vec4<f32>, quantization: Quantization) -> vec4<f32> {
    return (value - f32(quantization.zero_point)) * quantization.scaling;
}


/// reads the ith sh coef from the vertex buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let n = (MAX_SH_DEG + 1u) * (MAX_SH_DEG + 1u);
    let coef_idx = 3u * (splat_idx * n + c_idx);
    // coefs are packed as  bytes (4x per u32)
    let buff_idx = coef_idx / 4u;
    var v1 = unpack4x8snorm(sh_coefs[buff_idx]);
    var v2 = unpack4x8snorm(sh_coefs[buff_idx + 1u]);
    if c_idx == 0u {
        v1 = dequantizef4(v1 * 127., quantization.color_dc);
        v2 = dequantizef4(v2 * 127., quantization.color_dc);
    } else {
        v1 = dequantizef4(v1 * 127., quantization.color_rest);
        v2 = dequantizef4(v2 * 127., quantization.color_rest);
    }
    let r = coef_idx % 4u;
    if r == 0u {
        return vec3<f32>(v1.xyz);
    } else if r == 1u {
        return vec3<f32>(v1.yzw);
    } else if r == 2u {
        return vec3<f32>(v1.zw, v2.x);
    } else { // r == 3u
        return vec3<f32>(v1.w, v2.xy);
    }
}

// spherical harmonics evaluation with Condon–Shortley phase
fn evaluate_sh(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return result;
}

@compute @workgroup_size(256,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if idx > arrayLength(&vertices) {
        return;
    }

    let focal = camera.focal;
    let viewport = camera.viewport;
    let vertex = vertices[idx];
    let geometric_info = geometries[vertex.geometry_idx];
    let xyz = vec3<f32>(unpack2x16float(vertex.pos_xy), unpack2x16float(vertex.pos_zw).x);

    var camspace = camera.view * vec4<f32>(xyz, 1.);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;
    let z = pos2d.z / pos2d.w;

    // frustum culling hack
    if z < 0. || z > 1. || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
        return;
    }

    // let opacity = unpack2x16float(vertex.pos_zw).y;
    let opacity = dequantize(extractBits(i32(vertex.pos_zw), 2u * 8u, 8u), quantization.opacity);
    let scaling_factor = exp(dequantize(extractBits(i32(vertex.pos_zw), 3u * 8u, 8u), quantization.scaling_factor));

    let s2 = scaling_factor * scaling_factor;
    let cov1: vec2<f32> = unpack2x16float(geometric_info.cov[0]) * s2;
    let cov2: vec2<f32> = unpack2x16float(geometric_info.cov[1]) * s2;
    let cov3: vec2<f32> = unpack2x16float(geometric_info.cov[2]) * s2;
    let Vrk = mat3x3<f32>(
        cov1[0], cov1[1], cov2[0],
        cov1[1], cov2[1], cov3[0],
        cov2[0], cov3[0], cov3[1]
    );
    let J = mat3x3<f32>(
        focal.x / camspace.z,
        0.,
        -(focal.x * camspace.x) / (camspace.z * camspace.z),
        0.,
        -focal.y / camspace.z,
        (focal.y * camspace.y) / (camspace.z * camspace.z),
        0.,
        0.,
        0.
    );

    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let T = W * J;
    let cov = transpose(T) * Vrk * T;

    let diagonal1 = cov[0][0] + 0.3;
    let offDiagonal = cov[0][1];
    let diagonal2 = cov[1][1] + 0.3;

    let mid = 0.5 * (diagonal1 + diagonal2);
    let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
    // eigenvalues of the 2D screen space splat
    let lambda1 = mid + max(radius, 0.1);
    let lambda2 = mid - max(radius, 0.1);

    let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
    // scaled eigenvectors in screen space 
    let v1 = sqrt(2.0 * lambda1) * diagonalVector;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    let v_center = pos2d.xyzw / pos2d.w;

    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(xyz - camera_pos);
    let color = vec4<f32>(
        max(vec3<f32>(0.), evaluate_sh(dir, vertex.sh_idx, MAX_SH_DEG)),
        opacity
    );

    let store_idx = atomicAdd(&indirect_draw_call.instance_count, 1u);
    let v = vec4<f32>(v1 / viewport, v2 / viewport);
    points_2d[store_idx] = Splats2D(
        pack2x16float(v.xy), pack2x16float(v.zw),
        pack2x16float(v_center.xy),
        pack4x8unorm(color)
    );
    
    // filling the sorting buffers and the indirect sort dispatch buffer
    let znear = -camera.proj[3][2] / camera.proj[2][2];
    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - (1.));
    // filling the sorting buffers and the indirect sort dispatch buffer
    sort_depths[store_idx] = u32(f32(0xffffffu) - pos2d.z / zfar * f32(0xffffffu));
    sort_indices[store_idx] = store_idx;
    if idx == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);   // safety addition to always have an unfull block at the end of the buffer
    }
    let cur_key_size = atomicAdd(&sort_infos.keys_size, 1u);
    let keys_per_wg = 256u * 15u;         // Caution: if workgroup size (256) or keys per thread (15) changes the dispatch is wrong!!
    if (cur_key_size % keys_per_wg) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}