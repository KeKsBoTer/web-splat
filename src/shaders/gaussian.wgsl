// A splat contributes alpha = opacity * exp(-a) at the squared distance a from its center
// (in the units of its scaled eigenvectors). Contributions below alpha_cutoff (a uniform,
// 1/255 by default) are not drawn: the quad of a splat only covers the pixels with
// a <= ln(opacity / alpha_cutoff), which is smaller the lower the opacity and the higher
// the cutoff.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    // squared distance from the splat's center where its contribution is below the cutoff
    @location(2) @interpolate(flat) max_a: f32,
};

struct VertexInput {
    @location(0) v: vec4<f32>,
    @location(1) pos: vec4<f32>,
    @location(2) color: vec4<f32>,
};

struct Splat {
     // 4x f16 packed as u32
    v_0: u32, v_1: u32,
    // 2x f16 packed as u32
    pos: u32,
    // rgba packed as f16
    color_0: u32,color_1: u32,
};

@group(0) @binding(2)
var<storage, read> points_2d : array<Splat>;
@group(1) @binding(4)
var<storage, read> indices : array<u32>;

// the same uniform buffer as the preprocess (see SplattingArgsUniform in renderer.rs), only
// the fields up to alpha_cutoff are declared
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
}
@group(2) @binding(0)
var<uniform> render_settings: RenderSettings;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;

    let vertex = points_2d[indices[in_instance_index] + 0u];

    // scaled eigenvectors in screen space 
    let v1 = unpack2x16float(vertex.v_0);
    let v2 = unpack2x16float(vertex.v_1);

    let v_center = unpack2x16float(vertex.pos);

    let color = vec4<f32>(unpack2x16float(vertex.color_0), unpack2x16float(vertex.color_1));
    // the squared distance where opacity * exp(-a) falls below the cutoff
    let max_a = log(color.a / render_settings.alpha_cutoff);
    if !(max_a > 0.) {
        // the splat is below the cutoff everywhere: a degenerate quad outside of the view
        out.position = vec4<f32>(2., 2., 2., 1.);
        out.screen_pos = vec2<f32>(0.);
        out.color = color;
        out.max_a = 0.;
        return out;
    }

    // splat rectangle with left lower corner at (-1,-1)
    // and upper right corner at (1,1)
    let x = f32(in_vertex_index % 2u == 0u) * 2. - (1.);
    let y = f32(in_vertex_index < 2u) * 2. - (1.);

    let position = vec2<f32>(x, y) * sqrt(max_a);

    let offset = 2. * mat2x2<f32>(v1, v2) * position;
    out.position = vec4<f32>(v_center + offset, 0., 1.);
    out.screen_pos = position;
    out.color = color;
    out.max_a = max_a;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = dot(in.screen_pos, in.screen_pos);
    if a > in.max_a {
        discard;
    }
    let b = min(0.99, exp(-a) * in.color.a);
    return vec4<f32>(in.color.rgb, 1.) * b;
}