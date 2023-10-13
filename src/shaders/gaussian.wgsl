
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexInput {
    @location(0) v: vec4<f32>,
    @location(1) pos: vec4<f32>,
    @location(2) color: vec4<f32>,
};

struct Splats2D {
    // 4x f16 packed as u32
    v: vec2<u32>,
    // 4x f16 packed as u32
    pos: vec2<u32>,
    // rgba packed as u8
    color: u32,
};

@group(0) @binding(2)
var<storage, read> points_2d : array<Splats2D>;
@group(1) @binding(4)
var<storage, read> indices : array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    
    let vertex = points_2d[indices[in_instance_index]];

    // scaled eigenvectors in screen space 
    let v1 = unpack2x16float(vertex.v.x);
    let v2 = unpack2x16float(vertex.v.y);

    let v_center = unpack2x16float(vertex.pos.x);

    // splat rectangle with left lower corner at (-2,-2)
    // and upper right corner at (2,2)
    let x = f32(in_vertex_index % 2u == 0u) * 4. - (2.);
    let y = f32(in_vertex_index < 2u) * 4. - (2.);

    let position = vec2<f32>(x, y);

    // let offset = position * 0.01;
    let offset = position.x * v1 * 2.0 + position.y * v2 * 2.0;
    out.position = vec4<f32>(v_center + offset, unpack2x16float(vertex.pos.y));
    out.screen_pos = position;
    out.color = unpack4x8unorm(vertex.color);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = -dot(in.screen_pos, in.screen_pos);
    if a < -4.0 {discard;}
    let b = exp(a) * in.color.a;
    return vec4<f32>(in.color.rgb * b, b);
}