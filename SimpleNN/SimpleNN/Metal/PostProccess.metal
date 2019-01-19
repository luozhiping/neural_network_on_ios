//
//  PostProccess.metal
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/29.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct BatchNormParams {
    float epsilon;
    ushort neuronType;
    float neuronA;
    float neuronB;
};


//kernel void batch_norm_3(texture2d<half, access::read> inTexture [[texture(0)]],
//                         texture2d<half, access::write> outTexture [[texture(1)]],
//                         constant PaddingParam& params [[buffer(0)]],
//                         const device half4* weights [[buffer(1)]],
//                         device float4 &printBuffer [[buffer(2)]],
//                         ushort2 gid [[thread_position_in_grid]]) {
//    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
//        return;
//    }
//    half4 i = inTexture.read(gid);
//    half epsilon = params.epsilon;
//    half4 out = weights[2] * (i - weights[0]) / sqrt(weights[1] + epsilon) + weights[3];
//    outTexture.write(out, gid);
//}
enum NeuronType: ushort {
    NeuronTypeNone = 0,
    NeuronTypeReLU = 1,
    NeuronTypeLinear = 2,
    NeuronTypeSigmoid = 3,
    NeuronTypeTanH = 4,
    NeuronTypeAbsolute = 5,
    };

inline half4 applyNeuron(ushort neuronType, half4 x, half a, half b) {
    if (neuronType == NeuronTypeReLU)
        return fmax(x, 0.0h) + a*fmin(x, 0.0h);
    if (neuronType == NeuronTypeLinear)
        return a*x + b;
    if (neuronType == NeuronTypeSigmoid)
        return 1.0h / (1.0h + exp(-x));
    if (neuronType == NeuronTypeTanH)
        return a * tanh(b * x);
    if (neuronType == NeuronTypeAbsolute)
        return fabs(x);
    return x;
}
    
kernel void softmax_1x1(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                        texture2d_array<half, access::write> outTexture [[texture(1)]],
                        device float4 &printBuffer [[buffer(0)]],
                        ushort3 gid [[thread_position_in_grid]]) {
    ushort outCount = outTexture.get_array_size();
    
    if (gid.x >= outCount) return;
    
    //
    threadgroup float4 shared_mem[512];
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    //
    //
    
    half4 in = inTexture.sample(s, float2(0, 0), gid.x);
    shared_mem[gid.x] = float4(exp(in));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(gid.x == 0) {
        for (ushort i = 1; i < outCount; i ++) {
            //                sum[0]+= shared_mem[i].x;
            //                sum[0]+= shared_mem[i].y;
            //                sum[0]+= shared_mem[i].z;
            //                sum[0]+= shared_mem[i].w;
            shared_mem[0] += shared_mem[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 sum = float4(shared_mem[0].x + shared_mem[0].y + shared_mem[0].z + shared_mem[0].w);
    float4 out = float4(exp(in)) / sum;
    //        out.x /= sum;
    //        out.y /= sum;
    //        out.z /= sum;
    //        out.w /= sum;
    //        exp(in)/sum;
    
    if (gid.x == 0 && gid.y==0) {
        
        float4 bbb = float4(out);
        //                                    bbb.y = sum;
        //                            bbb.y = th.y;
        //            sharedbb.z = index;
        //            bbb.y = float(sum[0]);
        //            bbb.x = outTexture.get_array_size();
        printBuffer = float4(bbb);
    }
    
    outTexture.write(half4(out), uint2(gid.yz), gid.x);
}

kernel void batch_norm(texture2d_array<half, access::read> inTexture [[texture(0)]],
                   texture2d_array<half, access::write> outTexture [[texture(1)]],
                   constant BatchNormParams& params [[buffer(0)]],
                   const device half4* weights [[buffer(1)]],
                   device float4 &printBuffer [[buffer(2)]],
                   ushort3 gid [[thread_position_in_grid]]
                   ) {
if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() || gid.z >= inTexture.get_array_size()) {
    return;
}
half4 i = inTexture.read(uint2(gid.x, gid.y), gid.z);
ushort depth = inTexture.get_array_size();
ushort variance = 3 * depth;
ushort mean = 2 * depth;
ushort gamma = 0 * depth;
ushort beta = 1 * depth;
//half4 tempVariance = half4(1);
half epsilon = params.epsilon;
half4 out = weights[gamma + gid.z] * (i - weights[mean + gid.z]) / sqrt(weights[variance + gid.z] + epsilon) + weights[beta + gid.z];

//    out = applyNeuron(params.neuronType, out, params.neuronA, params.neuronB);
outTexture.write(out, uint2(gid.x, gid.y), gid.z);

if (gid.x == 0 && gid.y ==0 && gid.z == 0) {
    half4 bbb = half4(0);
    bbb.x = weights[0].x;
    bbb.y = weights[beta + gid.z].x;
    bbb.z = weights[mean + gid.z].x;
    bbb.w = weights[variance + gid.z].x;
    //            t.xy = half2(pos);
    printBuffer = float4(bbb);
    
}
    
    
}
    
struct ReluParams {
    float max_value;
    float negative_slope;
    float threshold;
};

kernel void relu(texture2d_array<half, access::read> inTexture [[texture(0)]],
                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                       constant ReluParams& params [[buffer(0)]],
                       device float4 &printBuffer [[buffer(1)]],
                       ushort3 gid [[thread_position_in_grid]]
                       ) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() || gid.z >= inTexture.get_array_size()) {
        return;
    }
    half4 i = inTexture.read(uint2(gid.x, gid.y), gid.z);
    
//    `f(x) = max_value` for `x >= max_value`,
//        `f(x) = x` for `threshold <= x < max_value`,
//            `f(x) = negative_slope * (x - threshold)` otherwise.
    half4 out = fmin(i, params.max_value);
    if(out.x < 0) {
        out.x = params.negative_slope * (out.x - params.threshold);
    }
    if(out.y < 0) {
        out.y = params.negative_slope * (out.y - params.threshold);
    }
    if(out.z < 0) {
        out.z = params.negative_slope * (out.z - params.threshold);
    }
    if(out.w < 0) {
        out.w = params.negative_slope * (out.w - params.threshold);
    }
    
    outTexture.write(out, uint2(gid.x, gid.y), gid.z);
    
    if (gid.x == 0 && gid.y ==0 && gid.z == 1) {
//        half4 bbb = half4(0);
        //            t.xy = half2(pos);
        printBuffer = float4(out);
        
    }
}
    
kernel void leaky_relu(texture2d_array<half, access::read> inTexture [[texture(0)]],
                 texture2d_array<half, access::write> outTexture [[texture(1)]],
                 constant ReluParams& params [[buffer(0)]],
                 device float4 &printBuffer [[buffer(1)]],
                 ushort3 gid [[thread_position_in_grid]]
                 ) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() || gid.z >= inTexture.get_array_size()) {
        return;
    }
    half4 i = inTexture.read(uint2(gid.x, gid.y), gid.z);
    
    //    `f(x) = max_value` for `x >= max_value`,
    //        `f(x) = x` for `threshold <= x < max_value`,
    //            `f(x) = negative_slope * (x - threshold)` otherwise.
    half4 out = half4(i);
    if(out.x < 0) {
        out.x *= params.negative_slope;
    }
    if(out.y < 0) {
        out.y *= params.negative_slope;
    }
    if(out.z < 0) {
        out.z *= params.negative_slope;
    }
    if(out.w < 0) {
        out.w *= params.negative_slope;
    }
    
    outTexture.write(out, uint2(gid.x, gid.y), gid.z);
    
    if (gid.x == 0 && gid.y ==0 && gid.z == 0) {
//        half4 bbb = half4(0);
        //            t.xy = half2(pos);
        printBuffer = float4(out);
        
    }
}

kernel void add(texture2d_array<half, access::read> inTexture [[texture(0)]],
                 texture2d_array<half, access::read> inTexture2 [[texture(1)]],
                 texture2d_array<half, access::write> outTexture [[texture(2)]],
                 device float4 &printBuffer [[buffer(0)]],
                 ushort3 gid [[thread_position_in_grid]]
                 ) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() || gid.z >= inTexture.get_array_size()) {
        return;
    }
    half4 i = inTexture.read(uint2(gid.x, gid.y), gid.z);
    half4 i2 = inTexture2.read(uint2(gid.x, gid.y), gid.z);

    
    half4 out = i + i2;
    outTexture.write(out, uint2(gid.x, gid.y), gid.z);
    
    if (gid.x == 0 && gid.y ==0 && gid.z == 0) {
        //        half4 bbb = half4(0);
        //            t.xy = half2(pos);
        printBuffer = float4(out);
        
    }
}


struct UpSamplingParam {
    ushort factors_w;
    ushort factors_h;
};

kernel void upsampling(texture2d_array<half, access::read> inTexture [[texture(0)]],
                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                       constant UpSamplingParam& params [[buffer(0)]],
                       device float4 &printBuffer [[buffer(1)]],
                       ushort3 gid [[thread_position_in_grid]]
                       ) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height() || gid.z >= inTexture.get_array_size()) {
        return;
    }
    half4 i = inTexture.read(uint2(gid.x, gid.y), gid.z);
    ushort w = params.factors_w;
    ushort h = params.factors_h;
    for (ushort x = 0; x < w; x++) {
        for (ushort y = 0; y < h; y++) {
            outTexture.write(i, uint2(gid.x*w+x, gid.y*h+y),gid.z);
        }
    }
    if (gid.x == 1 && gid.y ==1 && gid.z == 0) {
        half4 bbb = half4(0);
        bbb.xy = half2(gid.x*w+1, h);
        printBuffer = float4(bbb);
        
    }
}

kernel void concatenate_family2(texture2d_array<half, access::read> inTexture0 [[texture(0)]],
                                texture2d_array<half, access::read> inTexture1 [[texture(1)]],
                                texture2d_array<half, access::read> inTexture2 [[texture(2)]],
                                texture2d_array<half, access::read> inTexture3 [[texture(3)]],
                        texture2d_array<half, access::write> outTexture [[texture(4)]],
                        device float4 &printBuffer [[buffer(0)]],
                        ushort3 gid [[thread_position_in_grid]]
                        ) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() || gid.z >= outTexture.get_array_size()) {
        return;
    }
    half4 i;
    if (gid.z < inTexture0.get_array_size()) {
        i = inTexture0.read(uint2(gid.x, gid.y), gid.z);

    } else if (gid.z < inTexture0.get_array_size() + inTexture1.get_array_size()) {
        i = inTexture1.read(uint2(gid.x, gid.y), gid.z-inTexture0.get_array_size());


    } else if (gid.z < inTexture0.get_array_size() + inTexture1.get_array_size()+inTexture2.get_array_size()) {
        i = inTexture2.read(uint2(gid.x, gid.y), gid.z-inTexture0.get_array_size()-inTexture1.get_array_size());
    } else if (gid.z < inTexture0.get_array_size() + inTexture1.get_array_size()+inTexture2.get_array_size()+inTexture3.get_array_size()) {
        i = inTexture3.read(uint2(gid.x, gid.y), gid.z-inTexture0.get_array_size()-inTexture1.get_array_size()-inTexture2.get_array_size());
    }
    outTexture.write(i, uint2(gid.x, gid.y), gid.z);
    
    if (gid.x == 0 && gid.y ==0 && gid.z == 16) {
        float4 bbb = float4(0);
        bbb.x = inTexture0.get_array_size();
        printBuffer = float4(bbb);
        
    }
    
}

kernel void concatenate(array<texture2d_array<half, access::read>, 4> inTexture [[texture(0)]],
                        texture2d_array<half, access::write> outTexture [[texture(4)]],
                        device float4 &printBuffer [[buffer(0)]],
                        ushort3 gid [[thread_position_in_grid]]
                        ) {
    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() || gid.z >= outTexture.get_array_size()) {
        return;
    }
    half4 i;
    if (gid.z < inTexture[0].get_array_size()) {
        i = inTexture[0].read(uint2(gid.x, gid.y), gid.z);

    } else if (gid.z < inTexture[0].get_array_size() + inTexture[1].get_array_size()) {
        i = inTexture[1].read(uint2(gid.x, gid.y), gid.z-inTexture[0].get_array_size());


    } else if (gid.z < inTexture[0].get_array_size() + inTexture[1].get_array_size()+inTexture[2].get_array_size()) {
        i = inTexture[2].read(uint2(gid.x, gid.y), gid.z-inTexture[0].get_array_size()-inTexture[1].get_array_size());
    } else if (gid.z < inTexture[0].get_array_size() + inTexture[1].get_array_size()+inTexture[2].get_array_size()+inTexture[3].get_array_size()) {
        i = inTexture[3].read(uint2(gid.x, gid.y), gid.z-inTexture[0].get_array_size()-inTexture[1].get_array_size()-inTexture[2].get_array_size());
    }
    outTexture.write(i, uint2(gid.x, gid.y), gid.z);

    if (gid.x == 0 && gid.y ==0 && gid.z == 16) {
        printBuffer = float4(i);

    }

}
    
    
//kernel void output(<texture2d_array<half, access::read> inTexture [[texture(0)]],
//                        texture2d_array<half, access::write> outTexture [[texture(1)]],
//                        ushort3 gid [[thread_position_in_grid]]
//                        ) {
//    if (gid.x >= outTexture.get_width() || gid.y >= outTexture.get_height() || gid.z >= outTexture.get_array_size()) {
//        return;
//    }
////    ushort width = inTexture
//}
