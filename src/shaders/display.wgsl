const PI:f32 = 3.1415926535897932384626433832795;
const TWO_PI:f32 = 6.283185307179586476925286766559;

const CHANNEL_COLOR:u32 = 0;
const CHANNEL_GRAD_X:u32 = 1;
const CHANNEL_GRAD_Y:u32 = 2;
const CHANNEL_GRAD_XY:u32 = 3;


const INTERPOLATION_NEAREST:u32 = 0u;
const INTERPOLATION_BILINEAR:u32 = 1u;
const INTERPOLATION_BICUBIC:u32 = 2u;
const INTERPOLATION_SPLINE:u32 = 3u;

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport: vec2<f32>,
    focal: vec2<f32>
};


struct RenderSettings {
    clipping_box_min: vec4<f32>,
    clipping_box_max: vec4<f32>,
    gaussian_scaling: f32,
    max_sh_deg: u32,
    mip_spatting: u32,
    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    selected_channel:u32,
    upscaling_method:u32,
    center: vec3<f32>,
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
var colorBuffer: texture_storage_2d<rgba16float, read_write>;
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

fn sample_nearest(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = vec2<f32>(textureDimensions(colorBuffer));
    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    return textureLoad(source_img, pixel_pos, 0);
}

fn sample_bilinear(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = vec2<f32>(textureDimensions(source_img));

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);

    let z00 = textureLoad(source_img, pixel_pos,0);
    let z10 = textureLoad(source_img, clamp(pixel_pos + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
    let z01 = textureLoad(source_img, clamp(pixel_pos + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
    let z11 = textureLoad(source_img, clamp(pixel_pos + vec2<i32>(1, 1), vec2<i32>(0), vec2<i32>(tex_size-1)),0);

    return mix(
        mix(z00, z10, p_frac.x),
        mix(z01, z11, p_frac.x),
        p_frac.y
    );
}


fn sample_bicubic(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = vec2<f32>(textureDimensions(source_img));

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);
    var z:  array<mat2x2<f32>,4>;
    var dx: array<mat2x2<f32>,4>;
    var dy: array<mat2x2<f32>,4>;
    var dxy:array<mat2x2<f32>,4>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = clamp(pixel_pos + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(tex_size-1));

            let z_v = textureLoad(source_img, sample_pos,0);
            let z_up = textureLoad(source_img, clamp(sample_pos + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
            let z_left = textureLoad(source_img, clamp(sample_pos + vec2<i32>(-1, 0), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
            let z_right = textureLoad(source_img, clamp(sample_pos + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(tex_size-1)),0);
            let z_down = textureLoad(source_img, clamp(sample_pos + vec2<i32>(0, -1), vec2<i32>(0), vec2<i32>(tex_size-1)),0);

            for (var c = 0u; c < 4u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = (z_right[c] - z_left[c]) * 0.5;
                dy[c][i][j] = (z_up[c] - z_down[c]) * 0.5;
                dxy[c][i][j] = (z_right[c] + z_left[c] - 2.0 * z_v[c]) * 0.5;
            }

        }
    }

    return vec4<f32>(
        spline_interp(z[0], dx[0], dy[0], dxy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[1], dx[1], dy[1], dxy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[2], dx[2], dy[2], dxy[2], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[3], dx[3], dy[3], dxy[3], vec2<f32>(p_frac.y, p_frac.x))
    ); 
}

fn sample_spline(pos_in:vec2<f32>)->vec4<f32>{
    let tex_size = vec2<f32>(textureDimensions(colorBuffer));

    let pos = pos_in*(tex_size);
    let pixel_pos = vec2<i32>(pos);
    let p_frac = fract(pos);
    var z:  array<mat2x2<f32>,4>;
    var dx: array<mat2x2<f32>,4>;
    var dy: array<mat2x2<f32>,4>;
    var dxy:array<mat2x2<f32>,4>;

    for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
            let sample_pos = clamp(pixel_pos + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(tex_size-1));

            let z_v = textureLoad(source_img, sample_pos,0);
            let dx_v = textureLoad(gradientBuffer_x, sample_pos);
            let dy_v = -textureLoad(gradientBuffer_y, sample_pos);
            let dxy_v = -textureLoad(gradientBuffer_xy, sample_pos);

            for (var c = 0u; c < 4u; c++) {
                z[c][i][j] = z_v[c];
                dx[c][i][j] = dx_v[c];
                dy[c][i][j] = dy_v[c];
                dxy[c][i][j] = dxy_v[c];
            }

        }
    }

    return vec4<f32>(
        spline_interp(z[0], dx[0], dy[0], dxy[0], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[1], dx[1], dy[1], dxy[1], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[2], dx[2], dy[2], dxy[2], vec2<f32>(p_frac.y, p_frac.x)),
        spline_interp(z[3], dx[3], dy[3], dxy[3], vec2<f32>(p_frac.y, p_frac.x))
    ); 
}


@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> 
{
    let tex_size = vec2<f32>(textureDimensions(colorBuffer));
    let pixel_pos = vec2<i32>(tex_size * vertex_in.tex_coord);
    var grad: vec4<f32>;
    switch (render_settings.selected_channel) {
        case CHANNEL_COLOR: {
            switch render_settings.upscaling_method{
                case INTERPOLATION_BILINEAR:{
                    return sample_bilinear(vertex_in.tex_coord);
                }
                case INTERPOLATION_BICUBIC:{
                    return sample_bicubic(vertex_in.tex_coord);
                }
                case INTERPOLATION_SPLINE:{
                    return sample_spline(vertex_in.tex_coord);
                }
                default:{
                    return sample_nearest(vertex_in.tex_coord);
                }
            }
        }
        case CHANNEL_GRAD_X: {
            grad = textureLoad(gradientBuffer_x, pixel_pos);
        }
        case CHANNEL_GRAD_Y: {
            grad = textureLoad(gradientBuffer_y, pixel_pos);
        }
        case CHANNEL_GRAD_XY: {
            grad = textureLoad(gradientBuffer_xy, pixel_pos);
        }
        default: {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }

    grad *= 10.;

    var color: vec4<f32>;
    if grad.r < 0.0 {
        color = mix(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(1.0, 0.0, 0.0, 1.0), -grad.r);
    } else {
        color = mix(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 0.0, 1.0, 1.0), grad.r);
    }

    return color;
}

fn spline_interp( z:mat2x2<f32>, dx:mat2x2<f32>, dy: mat2x2<f32>, dxy: mat2x2<f32>, p: vec2<f32>) -> f32
{
    let f = mat4x4<f32>(
        z[0][0], z[0][1], dy[0][0], dy[0][1],
        z[1][0], z[1][1], dy[1][0], dy[1][1],
        dx[0][0], dx[0][1], dxy[0][0], dxy[0][1],
        dx[1][0], dx[1][1], dxy[1][0], dxy[1][1]
    );
    let m = mat4x4<f32>(
        1., 0., 0., 0.,
        0., 0., 1., 0.,
        -3., 3., -2., -1.,
        2., -2., 1., 1.
    );
    let a = transpose(m) * f * (m);

    let tx = vec4<f32>(1., p.x, p.x * p.x, p.x * p.x * p.x);
    let ty = vec4<f32>(1., p.y, p.y * p.y, p.y * p.y * p.y);
    return dot(tx, a * ty);
}

