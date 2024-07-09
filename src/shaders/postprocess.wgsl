const mlp1 : array<array<f32,12>,6> = array<array<f32,12>,6>(array<f32,12>(-0.0830461, 0.01828, -0.380667, -0.686259, -0.650126, 1.10293, -0.0382432, 0.345367, -0.274539, 0.101835, -0.102817, -0.0485223), array<f32,12>(-0.44028, -0.237361, -0.246021, 0.233389, 0.563848, -0.102364, -0.125592, -0.101625, 0.0662293, 0.142045, -0.015178, 0.146841), array<f32,12>(-0.0228797, -0.0488746, 0.130377, -1.02685, -0.0936284, -0.943207, -0.127569, 0.180326, -0.502049, -0.0944073, -0.0403834, -0.311223), array<f32,12>(-0.333418, 0.487763, 0.309167, 0.223141, -0.788702, -0.574983, 0.0704187, -0.224897, -0.0595356, -0.157118, 0.109221, 0.0722155), array<f32,12>(-0.202443, 0.42973, 0.459457, 0.871593, 0.273756, 0.053463, 0.138366, -0.391168, -0.0189382, -0.154796, -0.0157429, -0.170406), array<f32,12>(0.221166, 0.358617, -0.370399, 0.0938435, 0.555776, -0.259338, 0.0136085, 0.00988081, 0.188859, 0.298862, -0.152734, -0.0671567), );
const mlp2 : array<array<f32,6>,3> = array<array<f32,6>,3>(array<f32,6>(0.286231, 0.501636, -0.377501, 0.597989, 0.296465, -0.41476), array<f32,6>(0.304235, 0.36491, -0.501831, 0.615221, 0.377721, -0.428405), array<f32,6>(0.319008, 0.243025, -0.645524, 0.587118, 0.489349, -0.463296), );

@group(0) @binding(0)
var image_in1: texture_storage_2d<rgba16float, read>;
@group(0) @binding(1)
var image_in2: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2)
var image_in3: texture_storage_2d<rgba16float, read>;

@group(1) @binding(0)
var image_out: texture_storage_2d<rgba16float, write>;

fn mlp(spec:array<f32,6>,time_feature:vec3<f32>,ray_dir:vec3<f32>)->vec3<f32>{
    // let in_vec = array<f32,12>(time_feature.x, time_feature.y, time_feature.z, ray_dir.x, ray_dir.y, ray_dir.z, time_feature.x*ray_dir.x, time_feature.y*ray_dir.y, time_feature.z*ray_dir.z, time_feature.x*ray_dir.y, time_feature.y*ray_dir.z, time_feature.z*ray_dir.x);
    // for()
    return vec3<f32>(0.,0.,0.);
}

fn sigmoid(color:vec3<f32>)->vec3<f32>{
    return exp(color)/(1. + exp(-color));
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let size = textureDimensions(image_in1);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let albedo = textureLoad(image_in1, vec2<u32>(gid.x, gid.y));
    let spec = textureLoad(image_in2, vec2<u32>(gid.x, gid.y)).rgb;
    let time_feature = textureLoad(image_in3, vec2<u32>(gid.x, gid.y)).rgb;

    let ray_dir = vec3<f32>(0.,0.,1.);
    let color_dir = vec3<f32>(0.);//mlp(spec,time_feature,ray_dir);
    let out_color = sigmoid(albedo.rgb + color_dir);
    textureStore(image_out, vec2<u32>(gid.x, gid.y), vec4<f32>(out_color,albedo.a));
}