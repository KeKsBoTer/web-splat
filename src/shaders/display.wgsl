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
    show_env_map: u32,
}

@group(0) @binding(0)
var source_img : texture_2d<f32>;
@group(0) @binding(1)
var texture_sampler: sampler;
@group(0) @binding(2)
var depth_img : texture_2d<f32>;
@group(0) @binding(3)
var depth_sampler: sampler;

@group(1) @binding(0)
var env_map : texture_2d<f32>;
@group(1) @binding(1)
var env_map_sampler: sampler;

@group(2) @binding(0)
var<uniform> camera: CameraUniforms;

@group(3) @binding(0)
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

fn sample_env_map(dir: vec3<f32>) -> vec4<f32> {
    let texcoord = vec2<f32>(atan2(dir.z, dir.x) / TWO_PI + 0.5, -asin(dir.y) / PI + 0.5);
    return textureSample(env_map, env_map_sampler, texcoord);
}

@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> {
    let color = textureSample(source_img, texture_sampler, vertex_in.tex_coord);
    let depth = textureSample(depth_img, depth_sampler, vertex_in.tex_coord).r;
    if render_settings.show_env_map == 1u {
        let local_pos = camera.proj_inv * vec4<f32>((vertex_in.tex_coord.xy * 2. - (1.)), 1., 1.);
        let dir = camera.view_inv * vec4<f32>(local_pos.xyz, 0.);
        let env_color = sample_env_map(normalize(dir.xyz));
        return vec4<f32>(env_color.rgb * (1. - color.a) + color.rgb, 1.);
    } else {
        // return color;
        return vec4<f32>(depth,depth,depth,1.);
    }
}