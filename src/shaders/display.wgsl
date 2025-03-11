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

@group(3) @binding(0)
var colorBuffer: texture_storage_2d<rgba8unorm, read_write>;
@group(3) @binding(1)
var gradientBuffer_x: texture_storage_2d<rgba16float, read_write>;
@group(3) @binding(2)
var gradientBuffer_y: texture_storage_2d<rgba16float, read_write>;
@group(3) @binding(3)
var gradientBuffer_xy: texture_storage_2d<rgba16float, read_write>;



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
    // return textureLoad(colorBuffer, vec2<i32>(i32(vertex_in.pos.x),i32(vertex_in.pos.y)));
    // return textureSample(source_img, texture_sampler, vertex_in.tex_coord);

    let grad = textureLoad(gradientBuffer_x, vec2<i32>(i32(vertex_in.pos.x),i32(vertex_in.pos.y)));
    // let color = mix(vec4<f32>(1.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 0.0, 1.0, 1.0), (grad.r+1.)/2.);
    var color: vec4<f32>;
    if grad.r < 0.0 {
        color = mix(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(1.0, 0.0, 0.0, 1.0), -grad.r*2.);
    } else {
        color = mix(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 0.0, 1.0, 1.0), grad.r*2.);
    }

    return color;
}