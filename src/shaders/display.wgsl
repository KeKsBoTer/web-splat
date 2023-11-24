@group(0) @binding(0)
var source_img : texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

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

@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(source_img, texture_sampler, vertex_in.tex_coord);
}