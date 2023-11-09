// injected variables block -------------------------------

// const max_sh_deg:u32;
// const sh_dtype:u32;
// const opacity_s:f32;
// const opacity_zp:i32;
// const scaling_s:f32;
// const scaling_zp:i32;
// const rotation_s:f32;
// const rotation_zp:i32;
// const features_s:f32;
// const features_zp:i32;
// const scaling_factor_s:f32;
// const scaling_factor_zp:i32;

// injected variables block end ---------------------------

// which precision/datatype the sh coefficients use
const sh_dtype_float:u32 = 0u;
const sh_dtype_half:u32 = 1u;
const sh_dtype_byte:u32 = 2u;

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


struct camerauniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct PointcloudUniforms {
    opacity_s: f32,
    opacity_zp: i32,
    scaling_s: f32,
    scaling_zp: i32,
    rotation_s: f32,
    rotation_zp: i32,
    features_s: f32,
    features_zp: i32,
    scaling_factor_s: f32,
    scaling_factor_zp: i32,
};

struct gaussiansplat {
    pos_xy: u32,
    pos_zw: u32,
    geometry_idx: u32,
    sh_idx: u32,
};

struct geometricinfo {
    cov: array<u32, 3>,
    padding: u32,
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
    /// the number of vertices to draw.
    vertex_count: u32,
    /// the number of instances to draw.
    instance_count: atomic<u32>,
    /// the index of the first vertex to draw.
    base_vertex: u32,
    /// the instance id of the first instance to draw.
    /// has to be 0, unless [`features::indirect_first_instance`](crate::features::indirect_first_instance) is enabled.
    base_instance: u32,
}

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,     // essentially contains the same info as instance_count in drawindirect
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0)
var<uniform> camera: camerauniforms;

@group(1) @binding(0)
var<storage,read> xyz: array<u32>;      // xyz are vec3<half>s, for readout one has to calculate the two u32s by hand and extract half pos
@group(1) @binding(1)
var<storage,read> scaling: array<u32>;  // scaling are vec3<i8>s, upon extraction normalize with ((s - scaling_zp) * scaling_s).exp, pos has to be calculated by hand
@group(1) @binding(2)
var<storage,read> scaling_factor: array<u32>; // scaling factors are i8s, if scaling_factor_s != 0 the scaling has to adopted as: scale = scale.normalize() * ((scaling_factor - scaling_factor_zp)*scaling_factor_scale).exp()
@group(1) @binding(3)
var<storage,read> rotation: array<u32>; // rotation are vec4<i8>s, have to be extracted and the normalized, extraction by simple normalization with zp and s
@group(1) @binding(4)
var<storage,read> opacity: array<u32>;  // scale are i8s, standard normalization via zp and s
@group(1) @binding(5)
var<storage,read> features: array<u32>; // sh features (combined average and high frequency) as vec<i8>
@group(1) @binding(6)
var<storage,read> feature_indices: array<u32>; // indices which map from gaussian splat index to the feature vec(sh vec)
@group(1) @binding(7)
var<storage,read> gaussian_indices: array<u32>;// indices which map from gaussian splat index to the scaling and rotation value
@group(1) @binding(8) 
var<storage,read_write> points_2d : array<Splats2D>;
@group(1) @binding(9)
var<uniform> pc_uniforms : PointcloudUniforms;

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

fn i8tof32(v: u32) -> f32{
    var o = f32(v);
    if o > 127.0 {
        o -= 256.0;
    }
    return o;    
}
fn unpack4xi8(v: u32) -> vec4<f32> {
    return vec4<f32>(i8tof32(v & 0xffu), i8tof32((v >> 8u) & 0xffu), i8tof32((v>>16u) & 0xffu), i8tof32(v >> 24u)).wzyx;
}

fn get_pos(splat_idx: u32) -> vec4<f32> {
    let base_pos = splat_idx * 3u / 2u;
    let a = xyz[base_pos];
    let b = xyz[base_pos + 1u];
    
    var v: vec4<f32>;
    if (splat_idx & 1u) == 0u {
        v = vec4<f32>(unpack2x16float(a),
                        unpack2x16float(b));
    }
    else {
        v = vec4<f32>(unpack2x16float(a).y,
                        unpack2x16float(b), 1.0);
    }
    v.w = 1.0;
    return v;
}

fn get_covar(geometry_idx: u32) -> mat3x3<f32> {
    let base_pos = geometry_idx * 3u / 4u;
    var scale: vec3<f32>;
    switch (geometry_idx & 3u) {
        case 0u: {scale = unpack4xi8(scaling[base_pos]).xyz;}
        case 1u: {scale = unpack4xi8(scaling[base_pos]).yzw;}
        case 2u: {scale = vec3<f32>(unpack4xi8(scaling[base_pos]).zw, unpack4xi8(scaling[base_pos]).x);}
        case 3u: {scale = vec3<f32>(unpack4xi8(scaling[base_pos]).w, unpack4xi8(scaling[base_pos]).xy);}
        default: {}
    }
    scale = (-scale - f32(pc_uniforms.scaling_zp)) * pc_uniforms.scaling_s;
    scale = exp(scale);
    
    if false && pc_uniforms.scaling_factor_s > 0.0 {
        let s = unpack4x8snorm(scaling_factor[geometry_idx / 4u])[geometry_idx & 3u] * 127.0;
        scale *= (s - f32(pc_uniforms.scaling_factor_zp)) * pc_uniforms.scaling_factor_s;
    }
    
    // note that the x component is the scalar in this case...
    var rotation = unpack4xi8(rotation[base_pos]);
    rotation = (rotation - f32(pc_uniforms.rotation_zp)) * pc_uniforms.rotation_s;
    rotation = normalize(rotation);
    
    let x2 = 2.0 * rotation.y;
    let y2 = 2.0 * rotation.z;
    let z2 = 2.0 * rotation.w;

    let xx2 = x2 * rotation.y;
    let xy2 = x2 * rotation.z;
    let xz2 = x2 * rotation.w;
    
    let yy2 = y2 * rotation.z;
    let yz2 = y2 * rotation.w;
    let zz2 = z2 * rotation.w;

    let sx2 = x2 * rotation.x;
    let sy2 = y2 * rotation.x;
    let sz2 = z2 * rotation.x;
    let r = mat3x3<f32>(
        1.0 - yy2 - zz2, xy2 + sz2, xz2 - sy2,
        xy2 - sz2, 1.0 - xx2 - zz2, yz2 + sx2,
        xz2 + sy2, yz2 - sx2, 1.0 - xx2 - yy2,
    );
    let s = mat3x3<f32>(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
    let l = r * s;
    return l * transpose(l);
}

fn get_opacity(geometry_idx: u32) -> f32 {
    let v = unpack4xi8(opacity[geometry_idx / 4u])[geometry_idx & 3u];
    return (v - f32(pc_uniforms.opacity_zp)) * pc_uniforms.opacity_s;
}

/// reads the ith sh coef from the vertex buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let n = (MAX_SH_DEG + 1u) * (MAX_SH_DEG + 1u);
    let coef_idx = 3u * (splat_idx * n + c_idx);
    // coefs are packed as  bytes (4x per u32)
    let buff_idx = coef_idx / 4u;
    var v1 = unpack4x8snorm(features[buff_idx]) * 127.0;
    var v2 = unpack4x8snorm(features[buff_idx + 1u]) * 127.0;
    v1 = (v1 - f32(pc_uniforms.features_zp)) * pc_uniforms.features_s;
    v2 = (v2 - f32(pc_uniforms.features_zp)) * pc_uniforms.features_s;
    //if c_idx == 0u {
    //    v1 *= 4.;
    //    v2 *= 4.;
    //} else {
    //    v1 *= 0.5;
    //    v2 *= 0.5;
    //}
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
    
    if false {
        return vec3<f32>(1.0);
    }

    return result;
}

@compute @workgroup_size(256,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if idx * 3u > arrayLength(&xyz) {
        return;
    }

    let focal = camera.focal;
    let viewport = camera.viewport;
    let xyz = get_pos(idx);

    var camspace = camera.view * xyz;
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;

    // frustum culling hack
    if pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
        return;
    }

    let geometry_idx = gaussian_indices[idx];
    let Vrk = get_covar(geometry_idx);
    let opacity = get_opacity(idx);
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
    let dir = normalize(xyz.xyz - camera_pos);
    let sh_idx = feature_indices[idx];
    let color = vec4<f32>(
        saturate(evaluate_sh(dir, sh_idx, MAX_SH_DEG)),
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