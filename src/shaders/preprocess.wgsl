
//const MAX_SH_DEG:u32 = <injected>u;
//const SH_DTYPE:u32 = <injected>u;

// which precision/datatype the sh coefficients use
const SH_DTYPE_FLOAT:u32 = 0u;
const SH_DTYPE_HALF:u32 = 1u;
const SH_DTYPE_BYTE:u32 = 2u;

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


struct GaussianSplat {
    // 3x f16 xyz, 1x f16 opacity
    xyz_opacity: array<u32,2>,
    sh_idx: u32,
    // 6 f16 values
    cov: array<u32,3>,
};

struct Splats2D {
    // 4x f16 packed as u32
    v: vec2<u32>,
    // 4x f16 packed as u32
    pos: vec2<u32>,
    // rgba packed as u8
    color: u32,
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
var<storage,read_write> points_2d : array<Splats2D>;
@group(2) @binding(0) 
var<storage,read_write> indirect_draw_call : DrawIndirect;
@group(3) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(3) @binding(2)
var<storage, read_write> sort_depths : array<f32>;
@group(3) @binding(4)
var<storage, read_write> sort_indices : array<u32>;
@group(3) @binding(6)
var<storage, read_write> sort_dispatch: DispatchIndirect;


/// reads the ith sh coef from the vertex buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let n = (MAX_SH_DEG + 1u) * (MAX_SH_DEG + 1u);
    let coef_idx = 3u * (splat_idx * n + c_idx);
    if SH_DTYPE == SH_DTYPE_BYTE {
        // coefs are packed as  bytes (4x per u32)
        let buff_idx = coef_idx / 4u;
        var v1 = unpack4x8snorm(sh_coefs[buff_idx]);
        var v2 = unpack4x8snorm(sh_coefs[buff_idx + 1u]);
        if c_idx == 0u {
            v1 *= 4.;
            v2 *= 4.;
        } else {
            v1 *= 0.5;
            v2 *= 0.5;
        }
        let r = coef_idx % 4u;
        if r == 0u {
            return vec3<f32>(v1.xyz);
        } else if r == 1u {
            return vec3<f32>(v1.yzw);
        } else if r == 2u {
            return vec3<f32>(v1.zw, v2.x);
        } else if r == 3u {
            return vec3<f32>(v1.w, v2.xy);
        }
    } else if SH_DTYPE == SH_DTYPE_HALF {
        // coefs are packed as half (2x per u32)
        let buff_idx = coef_idx / 2u;
        let v = vec4<f32>(unpack2x16float(sh_coefs[buff_idx]), unpack2x16float(sh_coefs[buff_idx + 1u]));
        let r = coef_idx % 2u;
        if r == 0u {
            return v.rgb;
        } else if r == 1u {
            return v.gba;
        }
    } else if SH_DTYPE == SH_DTYPE_FLOAT {
        // coefs are packed as float (1x per u32)
        return bitcast<vec3<f32>>(vec3<u32>(sh_coefs[coef_idx], sh_coefs[coef_idx + 1u], sh_coefs[coef_idx + 2u]));
    }
    
    // unreachable
    return vec3<f32>(0.);
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

@compute @workgroup_size(16,16,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x * wgs.y * 16u + gid.y;
    if idx > arrayLength(&vertices) {
        return;
    }

    let focal = camera.focal;
    let viewport = camera.viewport;
    var vertex = vertices[idx];

    let p1 = unpack2x16float(vertex.xyz_opacity[0]);
    let p2 = unpack2x16float(vertex.xyz_opacity[1]);
    let xyz = vec3<f32>(p1.xy,p2.x);
    let opacity = p2.y;

    var camspace = camera.view * vec4<f32>(xyz, 1.);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;

    // frustum culling hack
    if pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
        return;
    }

    let cov1: vec2<f32> = unpack2x16float(vertex.cov[0]);
    let cov2: vec2<f32> = unpack2x16float(vertex.cov[1]);
    let cov3: vec2<f32> = unpack2x16float(vertex.cov[2]);
    let covPacked = array<f32,6>(cov1[0], cov1[1], cov2[0], cov2[1], cov3[0], cov3[1]);
    let Vrk = mat3x3<f32>(
        covPacked[0],
        covPacked[1],
        covPacked[2],
        covPacked[1],
        covPacked[3],
        covPacked[4],
        covPacked[2],
        covPacked[4],
        covPacked[5]
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
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);

    let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
    // scaled eigenvectors in screen space 
    let v1 = sqrt(2.0 * lambda1) * diagonalVector;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    let v_center = pos2d.xyzw / pos2d.w;

    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(xyz - camera_pos);
    let color = vec4<f32>(
        saturate(evaluate_sh(dir, vertex.sh_idx, MAX_SH_DEG)),
        opacity
    );

    let store_idx = atomicAdd(&indirect_draw_call.instance_count, 1u);
    let v = vec4<f32>(v1 / viewport, v2 / viewport);
    points_2d[store_idx] = Splats2D(
        vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw)),
        vec2<u32>(pack2x16float(v_center.xy), pack2x16float(v_center.zw)),
        pack4x8unorm(color),
    );
    
    // filling the sorting buffers and the indirect sort dispatch buffer
    sort_depths[store_idx] = 1. - v_center.z;    // z is already larger than 1, as OpenGL projection is used
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