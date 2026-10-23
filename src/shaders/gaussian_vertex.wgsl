// Rendering without compute shaders (WebGL2): every gaussian is an instance of a quad and is
// projected in the vertex shader. The instances are the indices of the gaussians in draw order,
// they are sorted on the CPU (see cpu_sort.rs). The gaussians are read from a texture, storage
// buffers are not available. Splats that are not visible are moved out of the view.
// The math is the one of preprocess.wgsl, see gaussian.wgsl for the meaning of the output.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    // squared distance from the splat's center where its contribution is below the cutoff
    @location(2) @interpolate(flat) max_a: f32,
};

// A gaussian is 8 texels (128 bytes) of the data texture, the texture is as wide as
// `points per row * 8` texels (see PointCloud::new_streaming). The 32 words of a gaussian are:
//   0..3    x, y, z (f32)
//   3       opacity and padding (2x f16)
//   4..7    covariance (6x f16)
//   7..31   16 * 3 sh coefficients (f16), rgb of one coefficient after the other
//   31      padding
@group(2) @binding(0)
var gaussian_data: texture_2d<u32>;

struct Gaussian {
    xyz: vec3<f32>,
    opacity: f32,
    cov: array<f32, 6>,
    words: array<u32, 32>,
};

fn load_gaussian(index: u32) -> Gaussian {
    let points_per_row = textureDimensions(gaussian_data).x / 8u;
    let row = i32(index / points_per_row);
    let column = i32((index % points_per_row) * 8u);

    var g: Gaussian;
    for (var i = 0; i < 8; i++) {
        let texel = textureLoad(gaussian_data, vec2<i32>(column + i, row), 0);
        g.words[4 * i] = texel.x;
        g.words[4 * i + 1] = texel.y;
        g.words[4 * i + 2] = texel.z;
        g.words[4 * i + 3] = texel.w;
    }
    g.xyz = vec3<f32>(bitcast<f32>(g.words[0]), bitcast<f32>(g.words[1]), bitcast<f32>(g.words[2]));
    g.opacity = unpack2x16float(g.words[3]).x;
    let c0 = unpack2x16float(g.words[4]);
    let c1 = unpack2x16float(g.words[5]);
    let c2 = unpack2x16float(g.words[6]);
    g.cov = array<f32, 6>(c0.x, c0.y, c1.x, c1.y, c2.x, c2.y);
    return g;
}

/// the rgb of the ith sh coefficient
fn sh_coef(g: Gaussian, i: u32) -> vec3<f32> {
    let j = i * 3u;
    return vec3<f32>(
        unpack2x16float(g.words[7u + (j + 0u) / 2u])[(j + 0u) % 2u],
        unpack2x16float(g.words[7u + (j + 1u) / 2u])[(j + 1u) % 2u],
        unpack2x16float(g.words[7u + (j + 2u) / 2u])[(j + 2u) % 2u],
    );
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,

    viewport: vec2<f32>,
    focal: vec2<f32>
};

// the same uniform buffer as the preprocess (see SplattingArgsUniform in renderer.rs)
struct RenderSettings {
    clipping_box_min: vec4<f32>,
    clipping_box_max: vec4<f32>,
    gaussian_scaling: f32,
    max_sh_deg: u32,
    mip_spatting: u32,
    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    alpha_cutoff: f32,
    num_points: u32,
    center: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(0)
var<uniform> render_settings: RenderSettings;

const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32, 5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32, 7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

// spherical harmonics evaluation with Condon–Shortley phase
fn evaluate_sh(dir: vec3<f32>, g: Gaussian, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(g, 0u);

    if sh_deg > 0u {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += -SH_C1 * y * sh_coef(g, 1u) + SH_C1 * z * sh_coef(g, 2u) - SH_C1 * x * sh_coef(g, 3u);

        if sh_deg > 1u {
            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(g, 4u) + SH_C2[1] * yz * sh_coef(g, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(g, 6u) + SH_C2[3] * xz * sh_coef(g, 7u) + SH_C2[4] * (xx - yy) * sh_coef(g, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(g, 9u) + SH_C3[1] * xy * z * sh_coef(g, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(g, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(g, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(g, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(g, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(g, 15u);
            }
        }
    }
    result += 0.5;

    return result;
}

fn hidden() -> VertexOutput {
    var out: VertexOutput;
    // outside of the view
    out.position = vec4<f32>(2., 2., 2., 1.);
    out.screen_pos = vec2<f32>(0.);
    out.color = vec4<f32>(0.);
    out.max_a = 0.;
    return out;
}

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    // index of the gaussian
    @location(0) index: u32,
) -> VertexOutput {
    let g = load_gaussian(index);
    let xyz = g.xyz;
    var opacity = g.opacity;

    if any(xyz < render_settings.clipping_box_min.xyz) || any(xyz > render_settings.clipping_box_max.xyz) {
        return hidden();
    }

    let camspace = camera.view * vec4<f32>(xyz, 1.);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;
    let z = pos2d.z / pos2d.w;

    // frustum culling hack
    if z <= 0. || z >= 1. || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
        return hidden();
    }
    if opacity < render_settings.alpha_cutoff {
        return hidden();
    }

    let cov_sparse = g.cov;

    let walltime = render_settings.walltime;
    var scale_mod = 0.;
    let dd = 5. * distance(render_settings.center, xyz) / render_settings.scene_extend;
    if walltime > dd {
        scale_mod = smoothstep(0., 1., (walltime - dd));
    }

    let focal = camera.focal;
    let scaling = render_settings.gaussian_scaling * scale_mod;
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
    // the squared distance where opacity * exp(-a) falls below the cutoff
    let max_a = log(opacity / render_settings.alpha_cutoff);
    if !(max_a > 0.) {
        return hidden();
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
    let v1 = sqrt(2.0 * lambda1) * diagonalVector / camera.viewport;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x) / camera.viewport;

    let v_center = pos2d.xy / pos2d.w;

    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(xyz - camera_pos);
    let rgb = max(vec3<f32>(0.), evaluate_sh(dir, g, render_settings.max_sh_deg));

    // splat rectangle with left lower corner at (-1,-1)
    // and upper right corner at (1,1)
    let x = f32(in_vertex_index % 2u == 0u) * 2. - (1.);
    let y = f32(in_vertex_index < 2u) * 2. - (1.);
    let position = vec2<f32>(x, y) * sqrt(max_a);

    let offset = 2. * mat2x2<f32>(v1, v2) * position;
    var out: VertexOutput;
    out.position = vec4<f32>(v_center + offset, 0., 1.);
    out.screen_pos = position;
    out.color = vec4<f32>(rgb, opacity);
    out.max_a = max_a;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = dot(in.screen_pos, in.screen_pos);
    let b = select(0., min(0.99, exp(-a) * in.color.a), a <= in.max_a);
    return vec4<f32>(in.color.rgb, 1.) * b;
}
