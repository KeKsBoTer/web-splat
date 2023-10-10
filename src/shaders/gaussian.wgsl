
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};


struct VertexInput {
    @location(0) v: vec4<f32>,
    @location(1) pos: vec4<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    vertex: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    // scaled eigenvectors in screen space 
    let v1 = vertex.v.xy;
    let v2 = vertex.v.zw;

    let v_center = vertex.pos.xy;

    // splat rectangle with left lower corner at (-2,-2)
    // and upper right corner at (2,2)
    let x = f32(in_vertex_index % 2u == 0u) * 4. - (2.);
    let y = f32(in_vertex_index < 2u) * 4. - (2.);

    let position = vec2<f32>(x, y);

    // let offset = position * 0.01;
    let offset = position.x * v1 * 2.0 + position.y * v2 * 2.0;
    out.position = vec4<f32>(v_center + offset, vertex.pos.zw);
    out.screen_pos = position;
    out.color = vertex.color;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = -dot(in.screen_pos, in.screen_pos);
    if a < -4.0 {discard;}
    let b = exp(a) * in.color.a;
    return vec4<f32>(in.color.rgb * b, b);
}