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


struct CameraUniforms{
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport:vec2<f32>,
    focal:vec2<f32>
};


struct GaussianSplat {
    xyz: vec3<f32>,
    color:array<vec4<f32>,16>,
    cov1: vec3<f32>,
    cov2: vec3<f32>,
    opacity: f32,
};

struct Splats2D {
    color: vec4<f32>,
    v: vec4<f32>,
    pos: vec3<f32>,
};
struct DrawIndirect {
    /// The number of vertices to draw.
    vertex_count: u32,
    /// The number of instances to draw.
    instance_count: atomic<u32>,
    /// The Index of the first vertex to draw.
    base_vertex: u32,
    /// The instance ID of the first instance to draw.
    /// Has to be 0, unless [`Features::INDIRECT_FIRST_INSTANCE`](crate::Features::INDIRECT_FIRST_INSTANCE) is enabled.
    base_instance: u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0) 
var<storage,read> vertices : array<GaussianSplat>;
@group(1) @binding(1) 
var<storage,write> points_2d : array<Splats2D>;
@group(2) @binding(0) 
var<storage,write> indirect_draw_call : DrawIndirect;


// spherical harmonics evaluation with Condon–Shortley phase
fn evaluate_sh(dir:vec3<f32>,vertex:GaussianSplat,sh_deg:u32)->vec3<f32>{
    var result = SH_C0 * vertex.color[0].rgb;

    if sh_deg > 0u{

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result +=  
            - SH_C1 * y * vertex.color[1].rgb
            + SH_C1 * z * vertex.color[2].rgb
            - SH_C1 * x * vertex.color[3].rgb;

        if sh_deg > 1u{

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result +=  SH_C2[0] * xy * vertex.color[4].rgb
            + SH_C2[1] * yz * vertex.color[5].rgb
            + SH_C2[2] * (2.0 * zz - xx - yy) * vertex.color[6].rgb
            + SH_C2[3] * xz * vertex.color[7].rgb
            + SH_C2[4] * (xx - yy) * vertex.color[8].rgb;

            if sh_deg > 2u {
                result +=  SH_C3[0] * y * (3.0 * xx - yy) * vertex.color[9].rgb
                    + SH_C3[1] * xy * z * vertex.color[10].rgb
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * vertex.color[11].rgb
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * vertex.color[12].rgb
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * vertex.color[13].rgb
                    + SH_C3[5] * z * (xx - yy) * vertex.color[14].rgb
                    + SH_C3[6] * x * (xx - 3.0 * yy) * vertex.color[15].rgb;
            }
        }
    }
    result += 0.5;

    return result;
}

@compute @workgroup_size(16,16,1)
fn preprocess(@builtin(global_invocation_id) gid : vec3<u32>,@builtin(num_workgroups) wgs : vec3<u32>){
    let idx = gid.x*wgs.y*16u + gid.y;
    if idx > arrayLength(&vertices){
        return;
    }

    let store_idx = atomicAdd(&indirect_draw_call.instance_count,1u);
    
    let focal = camera.focal;
    let viewport = camera.viewport;
    let vertex = vertices[idx];

    var camspace = camera.view *  vec4<f32>(vertex.xyz,1.);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;

    // frustum culling hack
    if pos2d.z < -pos2d.w || pos2d.x < -bounds || pos2d.x > bounds
		 || pos2d.y < -bounds || pos2d.y > bounds {
        points_2d[idx].pos = vec3<f32>(-10.,-10.,-10.);
        points_2d[idx].color = vec4<f32>(1.,0.,0.,1.);
        return;
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
    
    let diagonal1 = cov[0][0] + 0.3;
    let offDiagonal = cov[0][1];
    let diagonal2 = cov[1][1] + 0.3;

	let mid = 0.5 * (diagonal1 + diagonal2);
	let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
    // eigenvalues of the 2D screen space splat
	let lambda1 = mid + radius;
	let lambda2 = max(mid - radius, 0.1);

	let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
    // scaled eigenvectors in screen space 
	let v1 = sqrt(2.0 * lambda1) * diagonalVector;
	let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    let v_center = pos2d.xyz / pos2d.w;


    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(vertex.xyz-camera_pos);
    let color = saturate(evaluate_sh(dir,vertex,3u));
    let color_a = vec4<f32>(color,vertex.opacity);
    let v = vec4<f32>(v1/viewport,v2/viewport);
    points_2d[idx] = Splats2D(color_a,v,v_center);
}