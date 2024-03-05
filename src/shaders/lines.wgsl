struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct LineParams {
    @location(0) start: vec3<f32>,
    @location(1) end: vec3<f32>,
    @location(2) color: vec4<f32>
};


@group(0) @binding(0)
var<uniform> camera: CameraUniforms;


struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    params: LineParams,
) -> VertexOut {
    var start = camera.proj * camera.view * vec4<f32>(params.start, 1.0);
    var end = camera.proj * camera.view * vec4<f32>(params.end, 1.0);

    let a = normalize(end.xy / end.w - start.xy / start.w);
    let b = vec2<f32>(a.y, -a.x) * 0.001;

    let x = f32(vertex_index % 2u == 0u);
    let y = f32(vertex_index < 2u) * 2.0-1.0;

    var pos2d = start.xyz / start.w * x + end.xyz / end.w * (1. - x) + vec3<f32>(b, 0.) * y;
    let w = start.w * x + end.w * (1. - x);

    return VertexOut(vec4<f32>(pos2d * w, w), params.color);
}

@fragment
fn fs_main(vertex_in: VertexOut) -> @location(0) vec4<f32> {
    return vertex_in.color;
}

