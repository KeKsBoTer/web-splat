
struct Splats2D {
    // 4x f16 packed as u32
    v: vec2<u32>,
    // 4x f16 packed as u32
    pos: vec2<u32>,
    // rgba packed as u8
    color: u32,
};

struct GeneralInfo{
    keys_size: u32,
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport: vec2<f32>,
    focal: vec2<f32>
};

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(1) @binding(2)
var<storage, read> points_2d : array<Splats2D>;
@group(2) @binding(0)
var<storage, read> infos: GeneralInfo;          // only needed for the keys_size to check if the thread has to do work at all
@group(2) @binding(4)
var<storage, read> indices : array<u32>;
@group(3) @binding(0)
var<storage, read_write> out_alpha : array<atomic<u32>>;
@group(3) @binding(1)
var<storage, read_write> out_color : array<atomic<u32>>;

const MAX_ALPHA: u32 = 0xFFFFFFu;
const MAX_COL: u32 = 0xFFFFu;

@compute @workgroup_size(256, 1, 1)
fn draw_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    // each worker processes exactly one splat and draws it to the out images.
    // To do this with atomic images one simply converts the alpha value of the current splat
    // to inv alpha value and takes the logarithm of this to get into linear space
    // then get the alpha visibility at the current pixel with an atomic add into the out_alpha image
    // to get the true alpha form this linearized alpha value, simply do 1 - exp(lin_alpha);
    
    // as an optimization one can do a previous lookup of the alpha value, and if (alpha & 1 << 31), then already stop, as
    // the pixel is already fully opaque

    if gid.x >= infos.keys_size {return;}
    
    let vertex = points_2d[indices[gid.x]];
    let w_h = vec2<u32>(camera.viewport);
    let color_channel_diff = w_h.x * w_h.y;

    // scaled eigenvectors in screen space
    let v1 = unpack2x16float(vertex.v.x);
    let v2 = unpack2x16float(vertex.v.y);

    let v_center = unpack2x16float(vertex.pos.x);
    
    // splat rectangle with left lower corner at (-2, -2);
    // and upper right corner at (2, 2)
    let c1 = vec2<f32>(-2., -2.);
    let c2 = vec2<f32>(2., -2.);
    let c3 = vec2<f32>(-2., 2.);
    let c4 = vec2<f32>(2., 2.);
    
    let o1 = c1.x * v1 * 2. + c1.y * v2 * 2.;
    let o2 = c2.x * v1 * 2. + c2.y * v2 * 2.;
    let o3 = c3.x * v1 * 2. + c3.y * v2 * 2.;
    let o4 = c4.x * v1 * 2. + c4.y * v2 * 2.;
    
    let p1 = v_center + o1;
    let p2 = v_center + o2;
    let p3 = v_center + o3;
    let p4 = v_center + o4;
    
    // software rasterization
    // p1...p4 are the coordinates in normalized device coordinates
    // as a first approach get bounding box and got trough the bounding box for rasterization
    let bb_min = (min(p1, min(p2, min(p3, p4))) * .5 + .5) * (vec2<f32>(w_h) - 1.) + .5;
    let bb_max = (max(p1, max(p2, max(p3, p4))) * .5 + .5) * (vec2<f32>(w_h) - 1.) + .5;
    let v_center_p = (v_center * .5 + .5) * (vec2<f32>(w_h) - 1.) + .5;
    let diff = vec2<i32>(bb_max - bb_min);
    let color = unpack4x8unorm(vertex.color);
    for(var y = 0; y < diff.y; y++) {
        for(var x = 0; x < diff.x; x++) {
            let cur_p = bb_min  + vec2<f32>(f32(x), f32(y));
            // check if outside image
            if any(clamp(cur_p, vec2<f32>(0.), vec2<f32>(w_h) - 1.) != cur_p) {
                continue;
            }
            
            let linear_idx = u32(cur_p.y) * w_h.x + u32(cur_p.x);
            // getting the normalized (with sigma normalized) distance squared
            let dir = (cur_p - v_center_p) / vec2<f32>(diff) * 2.;
            let a = -dot(dir, dir);
            if a < -4. {continue;}                      // if too far from the center, continue
            if out_alpha[linear_idx] > MAX_ALPHA {continue;}  // if pixel is already opaque, do nothing TODO: correct value here
            let alpha = exp(a) * color.a;
            let u_alpha = u32(-log((1. - alpha) * f32(MAX_ALPHA))); // - as the result of log will always be negative (1 - alpha in [0,1 -> log in (-inf,0])
            let last_alpha = atomicAdd(&out_alpha[linear_idx], u_alpha);
            if last_alpha > MAX_ALPHA {continue;}
            let cur_alpha = 1. - exp(-f32(last_alpha + u_alpha));
            let multiplied_color = vec4<u32>(vec4<f32>(color) * alpha);
            atomicAdd(&out_color[linear_idx], multiplied_color.x);
            atomicAdd(&out_color[linear_idx + color_channel_diff], multiplied_color.y);
            atomicAdd(&out_color[linear_idx + 2u * color_channel_diff], multiplied_color.z);
        }
    }
}

struct VertexOutput{
    @builtin(position) position: vec4<f32>,
}
@vertex
fn vs_resolve(@builtin(vertex_index) vertex_id: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(f32(vertex_id & 1u) * 2. - 1., f32(vertex_id >> 1u) * 2. - 1., 0., 1.);
    return out;
}

@fragment
fn fs_resolve(in: VertexOutput) -> @location(0) vec4<f32> {
    let w_h = vec2<u32>(camera.viewport);
    let gid = vec2<u32>(in.position.xy);//vec2<u32>((in.position.xy * .5 + .5) * vec2<f32>(w_h - 1u) + .5);
    if any(min(gid.xy, w_h - 1u) != gid.xy) {
        return vec4<f32>(0.);
    }
    let color_channel_diff = w_h.x * w_h.y;
    let linear_idx = gid.y * w_h.x + gid.x;
    let alpha = out_alpha[linear_idx];
    out_alpha[linear_idx] = 0u;
    out_color[linear_idx] = 0u;
    out_color[linear_idx + color_channel_diff] = 0u;
    out_color[linear_idx + 2u * color_channel_diff] = 0u;

    let col = vec4<u32>(out_color[linear_idx], out_color[linear_idx + color_channel_diff], out_color[linear_idx + 2u * color_channel_diff], MAX_COL);
    return vec4<f32>(f32(alpha));//vec4<f32>(col) / f32(MAX_COL);
}
