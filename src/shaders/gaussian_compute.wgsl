
struct Splats2D {
    // 4x f16 packed as u32
    v: vec2<u32>,
    // 4x f16 packed as u32
    pos: vec2<u32>,
    // rgba packed as u8
    color: u32,
};

@group(0) @binding(2)
var<storage, read> points_2d : array<Splats2D>;
@group(1) @binding(4)
var<storage, read> indices : array<u32>;

@compute @workgroup_size(256, 1, 1)
fn draw_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    
}