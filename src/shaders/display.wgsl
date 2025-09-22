const PI:f32 = 3.1415926535897932384626433832795;
const TWO_PI:f32 = 6.283185307179586476925286766559;

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport: vec2<f32>,
    focal: vec2<f32>
};


struct RenderSettings {
    gaussian_scaling: f32,
    max_sh_deg: u32,
}

@group(0) @binding(0)
var source_img : texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;

@group(1) @binding(0)
var<uniform> camera: CameraUniforms;

@group(2) @binding(0)
var<uniform> render_settings: RenderSettings;


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
    let color = textureSample(source_img, texture_sampler, vertex_in.tex_coord);
    return color;
}