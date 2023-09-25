const SH_C0:f32 = 0.28209479177387814;

const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);

const SH_C3 = array<f32,7>(
    -0.5900435899266435,
	2.890611442640554,
	-0.4570457994644658,
	0.3731763325901154,
	-0.4570457994644658,
	1.445305721320277,
	-0.5900435899266435
);


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
    
    viewport:vec2<f32>,
    focal:vec2<f32>
};


struct VertexInput {
    @location(0) xyz: vec3<f32>,
    @location(1) sh_0: vec3<f32>,
    @location(2) sh_1: vec3<f32>,
    @location(3) sh_2: vec3<f32>,
    @location(4) sh_3: vec3<f32>,
    @location(5) sh_4: vec3<f32>,
    @location(6) sh_5: vec3<f32>,
    @location(7) sh_6: vec3<f32>,
    @location(8) sh_7: vec3<f32>,
    @location(9) sh_8: vec3<f32>,
    @location(10) sh_9: vec3<f32>,
    @location(11) sh_10: vec3<f32>,
    @location(12) sh_11: vec3<f32>,
    @location(13) sh_12: vec3<f32>,
    @location(14) sh_13: vec3<f32>,
    @location(15) sh_14: vec3<f32>,
    @location(16) sh_15: vec3<f32>,
    @location(17) cov1: vec3<f32>,
    @location(18) cov2: vec3<f32>,
    @location(19) opacity: f32,
};


@group(0) @binding(0)
var<uniform> camera: CameraUniforms;



// spherical harmonics evaluation with Condon–Shortley phase
fn evaluate_sh(dir:vec3<f32>,vertex:VertexInput,sh_deg:u32)->vec3<f32>{


    var result = SH_C0 * vertex.sh_0;

    if sh_deg > 0u{

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result +=  
            - SH_C1 * y * vertex.sh_1
            + SH_C1 * z * vertex.sh_2
            - SH_C1 * x * vertex.sh_3;

        if sh_deg > 1u{

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result +=  SH_C2[0] * xy * vertex.sh_4
            + SH_C2[1] * yz * vertex.sh_5
            + SH_C2[2] * (2.0 * zz - xx - yy) * vertex.sh_6
            + SH_C2[3] * xz * vertex.sh_7
            + SH_C2[4] * (xx - yy) * vertex.sh_8;

            if sh_deg > 2u{
                result +=  SH_C3[0] * y * (3.0 * xx - yy) * vertex.sh_9 
                    + SH_C3[1] * xy * z * vertex.sh_10 
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * vertex.sh_11 
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * vertex.sh_12 
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * vertex.sh_13 
                    + SH_C3[5] * z * (xx - yy) * vertex.sh_14 
                    + SH_C3[6] * x * (xx - 3.0 * yy) * vertex.sh_15;
            }
        }
    }
    result += 0.5;

    return result;
}

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    vertex: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    let focal = camera.focal;
    let viewport = camera.viewport;


    var camspace = camera.view *  vec4<f32>(vertex.xyz,1.);
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
    
    let vCenter = pos2d.xy / pos2d.w;

    let diagonal1 = cov[0][0] + 0.3;
    let offDiagonal = cov[0][1];
    let diagonal2 = cov[1][1] + 0.3;

	let mid = 0.5 * (diagonal1 + diagonal2);
	let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
	let lambda1 = mid + radius;
	let lambda2 = max(mid - radius, 0.1);
	let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
	let v1 = sqrt(2.0 * lambda1) * diagonalVector;
	let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    let v_center = pos2d.xy / pos2d.w;

    let x = f32(in_vertex_index%2u == 0u)*4.-(2.);
    let y = f32(in_vertex_index<2u)*4.-(2.);

    let position = vec2<f32>(x,y);

    let offset = position.x * v1/viewport * 2.0  + position.y * v2/viewport * 2.0;
    out.position = vec4<f32>(v_center  + offset,0.,1.);
    out.screen_pos = position;

    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(vertex.xyz-camera_pos);

    let color = saturate(evaluate_sh(dir,vertex,3u));
    out.color = vec4<f32>(color,vertex.opacity);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = -dot(in.screen_pos, in.screen_pos);
    if a < -4.0 {discard;}
    let b = exp(a) * in.color.a;
    return vec4<f32>(in.color.rgb*b,b);
}