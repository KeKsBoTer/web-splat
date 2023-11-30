@group(0) @binding(0)
var source_img : texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

@group(0) @binding(2)
var depth_img : texture_2d<f32>;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOut {

    // creates two vertices that cover the whole screen
    let xy = vec2<f32>(
        f32(in_vertex_index % 2u == 0u),
        f32(in_vertex_index < 2u)
    );
    return VertexOut(vec4<f32>(xy * 2. - (1.), 0., 1.), vec2<f32>(xy.x, 1. - xy.y));
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let mask = vec3<f32>(color <= vec3<f32>(0.04045));
    let a = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
    let b = color / 12.92;
    return mask * b + a;
}

@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> {
    let color = textureSample(source_img, texture_sampler, vertex_in.tex_coord);
    return vec4<f32>(linear_to_srgb(color.rgb), color.a);
}