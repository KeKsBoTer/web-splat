const KERNEL_SIZE:f32 = 0.3;
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

struct Gaussian {
    pos_opacity: array<u32,2>,
    cov: array<u32,3>,
    trbf:u32,
    motion:array<u32,5>,
    omega:array<u32,2>
}

struct Splat {
     // 4x f16 packed as u32
    v_0: u32, v_1: u32,
    // 2x f16 packed as u32
    pos: u32,
    // 9 color features and opacity as f16 packed as u32
    color_opacity:array<u32,5>
};

struct DrawIndirect {
    /// The number of gaussians to draw.
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

struct RenderSettings {
    clipping_box_min: vec4<f32>,
    clipping_box_max: vec4<f32>,
    gaussian_scaling: f32,
    max_sh_deg: u32,
    show_env_map: u32,
    mip_spatting: u32,
    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    time: f32,
    center: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0) 
var<storage,read> gaussians : array<Gaussian>;
@group(1) @binding(1) 
var<storage,read> sh_coefs : array<array<u32,24>>;

@group(1) @binding(2) 
var<storage,read_write> points_2d : array<Splat>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(3) @binding(0)
var<uniform> render_settings: RenderSettings;


/// reads the ith sh coef from the vertex buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let a = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 0u) / 2u])[(c_idx * 3u + 0u) % 2u];
    let b = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 1u) / 2u])[(c_idx * 3u + 1u) % 2u];
    let c = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 2u) / 2u])[(c_idx * 3u + 2u) % 2u];
    return vec3<f32>(
        a, b, c
    );
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

fn cov_coefs(v_idx: u32) -> array<f32,6> {
    let a = unpack2x16float(gaussians[v_idx].cov[0]);
    let b = unpack2x16float(gaussians[v_idx].cov[1]);
    let c = unpack2x16float(gaussians[v_idx].cov[2]);
    return array<f32,6>(a.x, a.y, b.x, b.y, c.x, c.y);
}

fn trbf_function(x: f32) -> f32 {
    return exp(-pow(x,2.));
}

fn unpack_motion(motion: array<u32,5>) -> array<vec3<f32>,3> {
    let a = unpack2x16float(motion[0]);
    let b = unpack2x16float(motion[1]);
    let c = unpack2x16float(motion[2]);
    let d = unpack2x16float(motion[3]);
    let e = unpack2x16float(motion[4]);
    return array<vec3<f32>,3>(vec3<f32>(a.x, a.y, b.x), vec3<f32>(b.y, c.x, c.y), vec3<f32>(d.x, d.y, e.x));
}

@compute @workgroup_size(256,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let focal = camera.focal;
    let viewport = camera.viewport;
    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    var xyz = vec3<f32>(a.x, a.y, b.x);
    var opacity = b.y;



    let trbf = unpack2x16float(vertex.trbf);
    let trbf_center = trbf.x;
    let trbf_scale = trbf.y;

    let trbf_distance = render_settings.time-trbf_center;
    let trbf_distance2 = trbf_distance*trbf_distance;
    let trbf_distance3 = trbf_distance2*trbf_distance;

    let motion = unpack_motion(vertex.motion);

    xyz += trbf_distance*motion[0] + trbf_distance2*motion[1] + trbf_distance3*motion[2];

    // scale opacity based on time
    let trbf_scale2 = trbf_scale*trbf_scale;
    opacity *= exp(-trbf_distance2/trbf_scale2);


    if any(xyz < render_settings.clipping_box_min.xyz) || any(xyz > render_settings.clipping_box_max.xyz) {
        return;
    }

    var camspace = camera.view * vec4<f32>(xyz, 1.);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;
    let z = pos2d.z / pos2d.w;

    if idx == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);   // safety addition to always have an unfull block at the end of the buffer
    }
    // frustum culling hack
    if z <= 0. || z >= 1. || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
        return;
    }

    let cov_sparse = cov_coefs(idx);

    let walltime = render_settings.walltime;
    var scale_mod = 0.;
    let dd = 5. * distance(render_settings.center, xyz) / render_settings.scene_extend;
    if walltime > dd {
        scale_mod = smoothstep(0., 1., (walltime - dd));
    }

    let scaling = render_settings.gaussian_scaling * scale_mod;
    // TODO: we need to store scaling and rotation seperately
    // rotation needs to be adjusted over time
    let Vrk = mat3x3<f32>(
        cov_sparse[0], cov_sparse[1], cov_sparse[2],
        cov_sparse[1], cov_sparse[3], cov_sparse[4],
        cov_sparse[2], cov_sparse[4], cov_sparse[5]
    ) * scaling * scaling;
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

    let kernel_size = render_settings.kernel_size;
    if bool(render_settings.mip_spatting) {
        // according to Mip-Splatting by Yu et al. 2023
        let det_0 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
        let det_1 = max(1e-6, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
        var coef = sqrt(det_0 / (det_1 + 1e-6) + 1e-6);

        if det_0 <= 1e-6 || det_1 <= 1e-6 {
            coef = 0.0;
        }
        opacity *= coef;
    }

    let diagonal1 = cov[0][0] + kernel_size;
    let offDiagonal = cov[0][1];
    let diagonal2 = cov[1][1] + kernel_size;

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
    // let color = vec4<f32>(
    //     max(vec3<f32>(0.), evaluate_sh(dir, idx, render_settings.max_sh_deg)),
    //     opacity
    // );
    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);
    let v = vec4<f32>(v1 / viewport, v2 / viewport);


    let feat_1 = sh_coef(idx, 0u);
    let feat_2 = sh_coef(idx, 1u);
    let feat_3 = sh_coef(idx, 2u);

    points_2d[store_idx] = Splat(
        pack2x16float(v.xy), pack2x16float(v.zw),
        pack2x16float(v_center.xy),
        array<u32,5>(
            pack2x16float(feat_1.rg),
            pack2x16float(vec2<f32>(feat_1.b,feat_2.r)),
            pack2x16float(feat_2.gb),
            pack2x16float(feat_3.rg),
            pack2x16float(vec2<f32>(feat_3.b,opacity)),
        )
    );
    // filling the sorting buffers and the indirect sort dispatch buffer
    let znear = -camera.proj[3][2] / camera.proj[2][2];
    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - (1.));
    // filling the sorting buffers and the indirect sort dispatch buffer
    sort_depths[store_idx] = bitcast<u32>(zfar - pos2d.z) ;//u32(f32(0xffffffu) - pos2d.z / zfar * f32(0xffffffu));
    sort_indices[store_idx] = store_idx;

    let keys_per_wg = 256u * 15u;         // Caution: if workgroup size (256) or keys per thread (15) changes the dispatch is wrong!!
    if (store_idx % keys_per_wg) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}