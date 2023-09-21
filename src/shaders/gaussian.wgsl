const SH_C0:f32 = 0.28209479177387814;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};


struct CameraUniforms{
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
};


struct VertexInput {
    @location(0) xyz: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) cov1: vec3<f32>,
    @location(4) cov2: vec3<f32>,
};


@group(0) @binding(0)
var<uniform> camera: CameraUniforms;


@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    vertex: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    let focal = vec2<f32>(1200.,1164.);
    let viewport = vec2<f32>(1920.,995.);

    let camspace = camera.view *  vec4<f32>(vertex.xyz,1.);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;

     if pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds
		 || pos2d.y < -bounds || pos2d.y > bounds {
        out.position = vec4(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let Vrk = mat3x3<f32>(
        vertex.cov1.x, vertex.cov1.y, vertex.cov1.z, 
        vertex.cov1.y, vertex.cov2.x, vertex.cov2.y,
        vertex.cov1.z, vertex.cov2.y, vertex.cov2.z
    );


    let J = mat3x3<f32>(
        focal.x / camspace.z, 0., -(focal.x * camspace.x) / (camspace.z * camspace.z), 
        0., -focal.y / camspace.z, (focal.y * camspace.y) / (camspace.z * camspace.z), 
        0., 0., 0.
    );


    let W = transpose(mat3x3<f32>(camera.view[0].xyz,camera.view[1].xyz,camera.view[2].xyz));
    let T = W * J;
    let cov = transpose(T) * Vrk * T;
    

    let diagonal1 = cov[0][0] + 0.3; // x 
    let off_diagonal = cov[0][1]; // y
    let diagonal2 = cov[1][1] + 0.3; // z


    let v_center = pos2d.xy / pos2d.w;

	let mid = 0.5 * (diagonal1 + diagonal2);
	let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, off_diagonal));
	let lambda1 = mid + radius;
	let lambda2 = max(mid - radius, 0.1);
	let diagonal_vector = normalize(vec2<f32>(off_diagonal, lambda1 - diagonal1));
	let v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonal_vector;
	let v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonal_vector.y, -diagonal_vector.x);

    let x = f32(in_vertex_index%2u == 0u)*4.-(2.);
    let y = f32(in_vertex_index<2u)*4.-(2.);

    let position = vec2<f32>(x,y);

    out.position = vec4<f32>(v_center+position*0.01,0.,1.);
    // let offset = position.x * v1 / viewport * 2.0  + position.y * v2 / viewport * 2.0;
    // out.position = vec4<f32>(v_center  + offset,0.,1.);
    out.screen_pos = position;

    let color = saturate(vertex.color*SH_C0 +0.5);
    out.color = vec4<f32>(color,vertex.opacity);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = -dot(in.screen_pos, in.screen_pos);
    if a < -4.0 {discard;}
    let b = exp(a) * in.color.a;
    return vec4<f32>(in.color.rgb*b,b);
    // return vec4<f32>(in.color.rgb,1.);
}