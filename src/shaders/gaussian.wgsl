// we cutoff at 1/255 alpha value 
const CUTOFF:f32 = 2.3539888583335364; // = sqrt(log(255))

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) center: vec2<f32>,
    @location(3) pos: vec2<f32>,
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


@group(2) @binding(0)
var colorBuffer: texture_storage_2d<rgba8unorm, read_write>;
@group(2) @binding(1)
var gradientBuffer_x: texture_storage_2d<rgba16float, read_write>;
@group(2) @binding(2)
var gradientBuffer_y: texture_storage_2d<rgba16float, read_write>;
@group(2) @binding(3)
var gradientBuffer_xy: texture_storage_2d<rgba16float, read_write>;




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
    out.pos = v_center + offset;
    out.screen_pos = position;
    out.center = v_center;
    out.color = vec4<f32>(unpack2x16float(vertex.color_0), unpack2x16float(vertex.color_1));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    let a = dot(in.screen_pos, in.screen_pos);
    if a > 2. * CUTOFF {
        discard;
    }
    let b = min(0.99, exp(-a) * in.color.a);

    let d = (in.pos - in.center)*100.;
    let b_dx = -2. * d.x * b;
    let b_dy = -2. * d.y * b;
    let b_dxy = 4. * d.x * d.y * b;

    fragmentBarrierBegin();

    let old_color = textureLoad(colorBuffer, vec2<i32>(i32(in.position.x), i32(in.position.y)));
    
    // if old_color.a > 1.-1./255. {
    //     // early out...not sure if this is a good idea or really does anything
    //     discard;
    // }

    let old_grad_x = textureLoad(gradientBuffer_x, vec2<i32>(i32(in.position.x), i32(in.position.y)));
    let old_grad_y = textureLoad(gradientBuffer_y, vec2<i32>(i32(in.position.x), i32(in.position.y)));
    let old_grad_xy = textureLoad(gradientBuffer_xy, vec2<i32>(i32(in.position.x), i32(in.position.y)));
    
    let new_color = old_color.rgb + (1.-old_color.a) * in.color.rgb * b;
    let new_alpha = old_color.a + (1.-old_color.a) * b;

    // gradient in x direction
    let c_dx = old_grad_x.rgb + in.color.rgb * (b_dx*(1.-old_color.a) - old_grad_x.a * b);
    let a_dx = old_grad_x.a * (1.-b) + b_dx * (1.-old_color.a);

    // gradient in y direction
    let c_dy = old_grad_x.rgb + in.color.rgb * (b_dy*(1.-old_color.a) - old_grad_x.a * b);
    let a_dy = old_grad_y.a * (1.-b) + b_dy * (1.-old_color.a);

    // gradient in xy direction
    let c_dxy = vec3<f32>(0.);
    let a_dxy = 0.;

    let new_grad_x = vec4<f32>(c_dx, a_dx);
    let new_grad_y = vec4<f32>(c_dy, a_dy);
    let new_grad_xy = vec4<f32>(c_dxy, a_dxy);

    textureStore(colorBuffer, vec2<i32>(i32(in.position.x), i32(in.position.y)), vec4<f32>(new_color, new_alpha));
    textureStore(gradientBuffer_x, vec2<i32>(i32(in.position.x), i32(in.position.y)), new_grad_x);
    textureStore(gradientBuffer_y, vec2<i32>(i32(in.position.x), i32(in.position.y)), new_grad_y);
    textureStore(gradientBuffer_xy, vec2<i32>(i32(in.position.x), i32(in.position.y)), new_grad_xy);

    fragmentBarrierEnd();
    // return vec4<f32>(in.color.rgb, 1.) * b;
    // return vec4<f32>(new_color, new_alpha);
    return vec4<f32>(0.);
}