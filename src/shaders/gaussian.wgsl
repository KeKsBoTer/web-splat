// we cutoff at 1/255 alpha value 
const CUTOFF:f32 = 2.3539888583335364; // = sqrt(log(255))

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) depth: f32
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
    pos: u32,depth: f32,
    // rgba packed as f16
    color_0: u32,color_1: u32,
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
    let depth = vertex.depth;

    // splat rectangle with left lower corner at (-1,-1)
    // and upper right corner at (1,1)
    let x = f32(in_vertex_index % 2u == 0u) * 2. - (1.);
    let y = f32(in_vertex_index < 2u) * 2. - (1.);

    let position = vec2<f32>(x, y) * CUTOFF;

    let offset = 2. * mat2x2<f32>(v1, v2) * position;
    out.position = vec4<f32>(v_center + offset, 0., 1.);
    out.screen_pos = position;
    out.color = vec4<f32>(unpack2x16float(vertex.color_0), unpack2x16float(vertex.color_1));
    out.depth = vertex.depth;
    return out;
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @location(1) depth: vec4<f32>
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOut {
    let a = dot(in.screen_pos, in.screen_pos);
    if a > 2. * CUTOFF {
        discard;
    }
    let b = min(0.99, exp(-a) * in.color.a);
    return FragmentOut(vec4<f32>(in.color.rgb * b, b), vec4<f32>(in.depth / 100. * b, 0., 0., b));
}