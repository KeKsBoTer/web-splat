
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

@group(0) @binding(2)
var<storage, read> points_2d : array<Splats2D>;
@group(1) @binding(0)
var<storage, read> infos: GeneralInfo;          // only needed for the keys_size to check if the thread has to do work at all
@group(1) @binding(4)
var<storage, read> indices : array<u32>;
@group(2) @binding(0)
var<storage, read_write> out_alpha : array<u32>;
@group(2) @binding(1)
var<storage, read_write> out_color : array<vec4<u32>>;
@group(2) @binding(3)
var<unifrom, read> final_image : texture_2d<Rgba8Unorm>;

const MAX_ALPHA: u32 = 0xFFFFFF;
const MAX_COL: u32 = 0xFFFF;

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
    let w_h = textureDimensions(final_image);

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
    let bb_min = (min(p1, p2, p3, p4) * .5 + .5) * (w_h - 1) + .5;
    let bb_max = (max(p1, p2, p3, p4) * .5 + .5) * (w_h - 1) + .5;
    let v_center_p = (v_center * .5 + .5) * (w_h - 1) + .5;
    let diff = bb_max - bb_min;
    let color = unpack4x8unorm(vertex.color);
    for(int y = 0; y < diff.y; ++y) {
        for(int x = 0; x < diff.x; ++x) {
            let cur_p = bb_min  + vec2<int>(x, y);
            // check if outside image
            if clamp(cur_p, vec2<int>(0), w_h - 1) != cur_p {
                continue;
            }
            
            // getting the normalized (with sigma normalized) distance squared
            let dir = (cur_p - v_center_p) / diff * 2;
            let a = -dot(dir, dir);
            if a < -4. {continue;}                      // if too far from the center, continue
            if out_alpha[cur_p] > MAX_ALPHA {continue;}  // if pixel is already opaque, do nothing TODO: correct value here
            let alpha = exp(a) * color.a;
            let u_alpha = u32(-log((1 - alpha) * MAX_ALPHA)); // - as the result of log will always be negative (1 - alpha in [0,1 -> log in (-inf,0])
            let last_alpha = atomicAdd(&out_alpha[cur_p], u_alpha);
            if last_alpha > MAX_ALPHA {continue;}
            let cur_alpha = 1. - exp(f32(-last_alpha - u_alpha));
            let multiplied_color = vec4<u32>(vec4<f32>(color) * alpha);
            let linear_idx = y * w_h.x + x;
            atomicAdd(&out_color[linear_idx], multiplied_color.xyz);
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn resolve_color(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w_h = textureDimensions(final_image);
    if(min(gid.xy, w_h - 1) != w_h)
        return;
    let linear_idx = gid.y * w_h.x + gid.x;
    let col = out_color[linear_idx];
    let float_col = vec4<f32>(col / MAX_COL, 1.);
    imageStore(final_image, gid.xy, float_col);
    out_alpha[linear_idx] = 0u;
    out_color[linear_idx] = vec4<u32>(0u);
}
