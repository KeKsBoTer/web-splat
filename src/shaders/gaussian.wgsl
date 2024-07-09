// we cutoff at 1/255 alpha value 
const CUTOFF:f32 = 2.3539888583335364; // = sqrt(log(255))

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color_opacity: vec4<f32>,
};

struct Splat {
     // 4x f16 packed as u32
    v_0: u32, v_1: u32,
    // 2x f16 packed as u32
    pos: u32,
    // rgb color features and opacity as f16 packed as u32
    color_opacity:array<u32,2>
};

@group(0) @binding(2)
var<storage, read> points_2d : array<Splat>;
@group(1) @binding(4)
var<storage, read> indices : array<u32>;

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

    // splat rectangle with left lower corner at (-1,-1)
    // and upper right corner at (1,1)
    let x = f32(in_vertex_index % 2u == 0u) * 2. - (1.);
    let y = f32(in_vertex_index < 2u) * 2. - (1.);

    let position = vec2<f32>(x, y) * CUTOFF;

    let offset = 2. * mat2x2<f32>(v1, v2) * position;
    out.position = vec4<f32>(v_center + offset, 0., 1.);
    out.screen_pos = position;

    out.color_opacity = vec4<f32>(unpack2x16float(vertex.color_opacity[0]),unpack2x16float(vertex.color_opacity[1]));
    return out;
}


@fragment
fn fs_main(in: VertexOutput) ->@location(0) vec4<f32>{
    let a = dot(in.screen_pos, in.screen_pos);
    if a > 2. * CUTOFF {
        discard;
    }
    let b = min(0.99, exp(-a) * in.color_opacity.a);
    return vec4<f32>(in.color_opacity.rgb, 1.) * b;
}