// we cutoff at 1/255 alpha value 
const CUTOFF:f32 = 2.3539888583335364; // = sqrt(log(255))

const INTERPOLATION_NEAREST:u32 = 0u;
const INTERPOLATION_BILINEAR:u32 = 1u;
const INTERPOLATION_BICUBIC:u32 = 2u;
const INTERPOLATION_SPLINE:u32 = 3u;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) center: vec2<f32>,
    @location(3) cov: vec4<f32>,
    @location(4) offset: vec2<f32>,
};

struct VertexInput {
    @location(0) v: vec4<f32>,
    @location(1) pos: vec4<f32>,
    @location(2) color: vec4<f32>,
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

struct Splat {
     // 4x f16 packed as u32
    v_0: u32, v_1: u32,
    // 2x f16 packed as u32
    pos: u32,
    // rgba packed as f16
    color_0: u32,color_1: u32,
    // cov packed as 4xf16
    cov_1: u32, cov_2: u32
};

@group(0) @binding(2)
var<storage, read> points_2d : array<Splat>;
@group(1) @binding(4)
var<storage, read> indices : array<u32>;


@group(2) @binding(0)
var colorBuffer: texture_storage_2d<rgba16float, read_write>;
@group(2) @binding(1)
var gradientBuffer_x: texture_storage_2d<rgba16float, read_write>;
@group(2) @binding(2)
var gradientBuffer_y: texture_storage_2d<rgba16float, read_write>;
@group(2) @binding(3)
var gradientBuffer_xy: texture_storage_2d<rgba16float, read_write>;


@group(3) @binding(0)
var<uniform> render_settings: RenderSettings;


@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;

    let vertex = points_2d[indices[in_instance_index] + 0u];

    // scaled eigenvectors in screen space 
    var v1 = unpack2x16float(vertex.v_0);
    var v2 = unpack2x16float(vertex.v_1);

    let cov = vec4<f32>(unpack2x16float(vertex.cov_1), unpack2x16float(vertex.cov_2));

    let v_center = unpack2x16float(vertex.pos);

    // splat rectangle with left lower corner at (-1,-1)
    // and upper right corner at (1,1)
    let x = f32(in_vertex_index % 2u == 0u) * 2. - (1.);
    let y = f32(in_vertex_index < 2u) * 2. - (1.);

    let position = vec2<f32>(x, y) * CUTOFF;

    let offset = 2. * mat2x2<f32>(v1, v2) * position;
    out.position = vec4<f32>(v_center + offset, 0., 1.);
    out.cov = cov;
    out.screen_pos = position;
    out.center = v_center;
    out.color = vec4<f32>(unpack2x16float(vertex.color_0), unpack2x16float(vertex.color_1));
    out.offset = offset;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    let screen_size = vec2<f32>(textureDimensions(colorBuffer));
    var cov =mat2x2<f32>(in.cov.xy, in.cov.yz);

    // cov = cov * mat2x2<f32>(1./screen_size.x, 0., 0., 1./screen_size.y);

    let offset = in.offset*screen_size*0.5;

    let c_a = cov[0][0];
    let c_b = cov[1][0];
    let c_c = cov[1][1];
    let c_d = cov[0][1];

    let x = offset.x;
    let y = offset.y;


    // let d = 0.5 * cov * offset;
    // // let a = -(d.x*d.x+d.y*d.y);
    // let a = - dot(d, d);

    // let power = -0.5 * (c_a * x * x+ c_c* y * y) - c_b * x * y;
    let power = -dot(in.screen_pos, in.screen_pos);

    let alpha_pre = exp(power) * in.color.a;
    // let alpha_pre = max(0.,1-sqrt(-a));
    let alpha = min(0.99, alpha_pre);

    if alpha < exp(-(CUTOFF*CUTOFF)) {
        discard;
    }

        // let d = 2. * cov * in.screen_pos;
        // let d = in.offset;

        // let dg_dx = (-c_a*(c_a*x+c_b*y)-c_d*(c_c*y+c_d*x));
        // let dg_dy = (-c_b*(c_a*x+c_b*y)-c_c*(c_c*y+c_d*x));
        // // let dg_dxy = (-c_a*c_b-c_c*c_d+(c_a*(c_a*x+c_b*y)+c_d*(c_c*y+c_d*x))*(c_b*(c_a*x+c_b*y)+c_c*(c_c*y+c_d*x)));
        // let dg_dxy = -2.*(c_a*c_b-c_d*c_c);

        // let alpha_dx = dpdxFine(alpha_pre);
        // let alpha_dy = alpha_dx_a;//dpdyFine(alpha_pre);
        // let alpha_dxy = dpdyFine(alpha_dx);

    fragmentBarrierBegin();

    let old_color = textureLoad(colorBuffer, vec2<i32>(i32(in.position.x), i32(in.position.y)));
    let T = (1.-old_color.a);
    let new_color = old_color.rgb + T * in.color.rgb * alpha;
    let new_alpha = old_color.a + T * alpha;

    if render_settings.upscaling_method == INTERPOLATION_SPLINE{

        let dg_dx = (- c_a * x - c_b * y);
        let dg_dy = (- c_b * x - c_c * y);
        let dg_dxy = -c_b;

        var alpha_dx = dg_dx * alpha;
        var alpha_dy = dg_dy * alpha;
        var alpha_dxy = (dg_dx * dg_dy+ dg_dxy ) * alpha;

        if alpha !=alpha_pre{
            alpha_dx = 0.;
            alpha_dy = 0.;
            alpha_dxy = 0.;
        }

        
        // if old_color.a > 1.-1./255. {
        //     // early out...not sure if this is a good idea or really does anything
        //     discard;
        // }

        let old_grad_x = textureLoad(gradientBuffer_x, vec2<i32>(i32(in.position.x), i32(in.position.y)));
        let old_grad_y = textureLoad(gradientBuffer_y, vec2<i32>(i32(in.position.x), i32(in.position.y)));
        let old_grad_xy = textureLoad(gradientBuffer_xy, vec2<i32>(i32(in.position.x), i32(in.position.y)));
        

        // gradient in x direction
        let c_dx  = old_grad_x.rgb  + in.color.rgb * (T*alpha_dx  - old_grad_x.a  * alpha);
        let c_dy  = old_grad_y.rgb  + in.color.rgb * (T*alpha_dy  - old_grad_y.a  * alpha);
        let c_dxy = old_grad_xy.rgb + in.color.rgb * (T*alpha_dxy - old_grad_xy.a * alpha - old_grad_y.a * alpha_dx - old_grad_x.a * alpha_dy);

        let a_dx  = old_grad_x.a  * (1.-alpha) + alpha_dx  * T;
        let a_dy  = old_grad_y.a  * (1.-alpha) + alpha_dy  * T;
        let a_dxy = old_grad_xy.a * (1.-alpha) + alpha_dxy * T - old_grad_x.a * alpha_dy - old_grad_y.a * alpha_dx;

        let new_grad_x = vec4<f32>(c_dx, a_dx);
        let new_grad_y = vec4<f32>(c_dy, a_dy);
        let new_grad_xy = vec4<f32>(c_dxy, a_dxy);

        textureStore(gradientBuffer_x, vec2<i32>(i32(in.position.x), i32(in.position.y)), new_grad_x);
        textureStore(gradientBuffer_y, vec2<i32>(i32(in.position.x), i32(in.position.y)), new_grad_y);
        textureStore(gradientBuffer_xy, vec2<i32>(i32(in.position.x), i32(in.position.y)), new_grad_xy);
    }
    textureStore(colorBuffer, vec2<i32>(i32(in.position.x), i32(in.position.y)), vec4<f32>(new_color, new_alpha));

    fragmentBarrierEnd();
    return vec4<f32>(in.color.rgb, alpha);
}
