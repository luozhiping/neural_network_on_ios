#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

enum NeuronType: ushort {
    NeuronTypeNone = 0,
    NeuronTypeReLU = 1,
    NeuronTypeLinear = 2,
    NeuronTypeSigmoid = 3,
    NeuronTypeTanH = 4,
    NeuronTypeAbsolute = 5,
    };
    
    // Applying the activation function in the shader is quicker than creating
    // a new layer for it.
    inline float4 applyNeuron(ushort neuronType, float4 x, float a, float b) {
        if (neuronType == NeuronTypeReLU)
            return fmax(x, 0.0f) + a*fmin(x, 0.0f);
        if (neuronType == NeuronTypeLinear)
            return a*x + b;
        if (neuronType == NeuronTypeSigmoid)
            return 1.0f / (1.0f + exp(-x));
        if (neuronType == NeuronTypeTanH)
            return a * tanh(b * x);
        if (neuronType == NeuronTypeAbsolute)
            return fabs(x);
        return x;
    }
    
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
    
    struct ConvParams {
        ushort offsetX;
        ushort offsetY;
        ushort offsetZ;
        ushort strideX;
        ushort strideY;
        ushort neuronType;
        float neuronA;
        float neuronB;
        bool useBias;
    };
    
    struct ConvParams2 {
        ushort offsetX;
        ushort offsetY;
        ushort offsetZ;
        ushort strideX;
        ushort strideY;
        ushort neuronType;
//        float neuronA;
//        float neuronB;
//        ushort useBias;
    };
    
    kernel void conv3x3 (
                         texture2d<half, access::sample> inTexture [[texture(0)]],
                         texture2d<half, access::write> outTexture [[texture(1)]],
                         constant ConvParams& params [[buffer(0)]],
                         const device half4* weights [[buffer(1)]],
                         const device half4* biasTerms [[buffer(2)]],
                         device float4 &printBuffer [[buffer(3)]],
                         ushort3 gid [[thread_position_in_grid]],
                         ushort3 th [[thread_position_in_threadgroup]],
                         ushort index [[thread_index_in_threadgroup]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort kW = 3;
        const ushort kH = 3;
        ushort3 currentGid = ushort3(0, 0, 0);
        
        currentGid.x = gid.x * params.strideX;
        currentGid.y = gid.y * params.strideY;
        const ushort2 pos = currentGid.xy + ushort2(params.offsetX, params.offsetY);
        
        // Note: If we use half4, then we lose too much precision.
        float4 out = float4(0.0f);
        
        half4 in[9];
        in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1));
        in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1));
        in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1));
        in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ));
        in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ));
        in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ));
        in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1));
        in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1));
        in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1));
        
        for (ushort t = 0; t < kH*kW; ++t) {
            half4 wx = weights[0*kH*kW + t];
            out.x += dot(float4(in[t]), float4(wx));
            
            half4 wy = weights[1*kH*kW + t];
            out.y += dot(float4(in[t]), float4(wy));
            
            half4 wz = weights[2*kH*kW + t];
            out.z += dot(float4(in[t]), float4(wz));
            
            half4 ww = weights[3*kH*kW + t];
            out.w += dot(float4(in[t]), float4(ww));
        }
        threadgroup float ggg=1;
        float a = 1;
        //        a = a + 1;
        ggg += a + 1;
        if (gid.x == 31 && gid.y == 13) {
            //            x = float(gid.x);
            //            x = gid.x;
            
        }
        //        if (gid.x == 223 && gid.y == 223) {
        //            x = 6;
        //        }
        if (gid.x == 20 && gid.y==13) {
            float4 bbb = float4(0);
            bbb.x = th.x;
            bbb.y = th.y;
            bbb.z = index;
            bbb.w = ggg;
            printBuffer = float4(bbb);
            
        }
        if (params.useBias) {
            out += float4(biasTerms[0]);
        }
        
        out = applyNeuron(params.neuronType, out, params.neuronA, params.neuronB);
        
        
        
        outTexture.write(half4(out), uint2(gid.xy));
    }
    
    kernel void conv3x3_out_array (
                                   texture2d<half, access::sample> inTexture [[texture(0)]],
                                   texture2d_array<half, access::write> outTexture [[texture(1)]],
                                   constant ConvParams& params [[buffer(0)]],
                                   const device half4* weights [[buffer(1)]],
                                   const device half4* biasTerms [[buffer(2)]],
                                   device float4 &printBuffer [[buffer(3)]],
                                   ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        //
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        //
        const ushort kW = 3;
        const ushort kH = 3;
        //

        ushort3 currentGid = ushort3(0, 0, 0);

        currentGid.x = gid.x * params.strideX;
        currentGid.y = gid.y * params.strideY;

        const ushort2 pos = currentGid.xy + ushort2(params.offsetX, params.offsetY);
        const ushort outSlice = gid.z;
        //
        float4 out = float4(0.0f);
        //
        half4 in[9];
        in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1));
        in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1));
        in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1));
        in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ));
        in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ));
        in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ));
        in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1));
        in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1));
        in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1));

        for (ushort t = 0; t < kH*kW; ++t) {
            half4 wx = weights[(outSlice*4 + 0)*kH*kW + t];
            out.x += dot(float4(in[t]), float4(wx));

            half4 wy = weights[(outSlice*4 + 1)*kH*kW + t];
            out.y += dot(float4(in[t]), float4(wy));

            half4 wz = weights[(outSlice*4 + 2)*kH*kW + t];
            out.z += dot(float4(in[t]), float4(wz));

            half4 ww = weights[(outSlice*4 + 3)*kH*kW + t];
            out.w += dot(float4(in[t]), float4(ww));
        }
        
        if (params.useBias) {
            out += float4(biasTerms[outSlice]);
        }
        out = applyNeuron(params.neuronType, out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), uint2(gid.xy), outSlice);
        
        if (gid.x == 0 && gid.y==0 && gid.z == 0) {
            half4 t = inTexture.sample(s, float2(0, 0));
            float4 bbb = float4(0);
            bbb.x = weights[100].x;
            bbb.y = in[4].w;
            bbb.zw = float2(pos.xy);
//            t.xy = half2(pos);
            half4 wx = weights[(outSlice*4 + 0)*kH*kW + 0];
            printBuffer = float4(t);
//            printBuffer[1] = float4(bbb);

        }
    }
    
    
    
    
    kernel void conv3x3_array(
                              texture2d_array<half, access::sample> inTexture [[texture(0)]],
                              texture2d_array<half, access::write> outTexture [[texture(1)]],
                              constant ConvParams& params [[buffer(0)]],
                              const device half4* weights [[buffer(1)]],
                              const device half4* biasTerms [[buffer(2)]],
                              device float4 &printBuffer [[buffer(3)]],
                              ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort kW = 3;
        const ushort kH = 3;
        
        ushort3 currentGid = ushort3(0, 0, 0);
        
        currentGid.x = gid.x * params.strideX;
        currentGid.y = gid.y * params.strideY;
        
        const ushort2 pos = currentGid.xy + ushort2(params.offsetX, params.offsetY);
        const ushort inSlices = inTexture.get_array_size();
        const ushort outSlice = gid.z;
        
        float4 out = float4(0.0f);
        
        for (ushort inSlice = 0; inSlice < inSlices; ++inSlice) {
            half4 in[9];
            in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1), inSlice);
            in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1), inSlice);
            in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1), inSlice);
            in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ), inSlice);
            in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ), inSlice);
            in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ), inSlice);
            in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1), inSlice);
            in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1), inSlice);
            in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1), inSlice);
            
            for (ushort t = 0; t < kH*kW; ++t) {
                half4 wx = weights[(outSlice*4 + 0)*kH*kW*inSlices + t*inSlices + inSlice];
                out.x += dot(float4(in[t]), float4(wx));
                
                half4 wy = weights[(outSlice*4 + 1)*kH*kW*inSlices + t*inSlices + inSlice];
                out.y += dot(float4(in[t]), float4(wy));
                
                half4 wz = weights[(outSlice*4 + 2)*kH*kW*inSlices + t*inSlices + inSlice];
                out.z += dot(float4(in[t]), float4(wz));
                
                half4 ww = weights[(outSlice*4 + 3)*kH*kW*inSlices + t*inSlices + inSlice];
                out.w += dot(float4(in[t]), float4(ww));
            }
        }
        if (params.useBias) {
            out += float4(biasTerms[outSlice]);
        }
        out = applyNeuron(params.neuronType, out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), uint2(gid.xy), outSlice);
    }
    
    
    kernel void depthwiseConv3x3_array(
                                       texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                                       constant ConvParams& params [[buffer(0)]],
                                       const device half* weights [[buffer(1)]],
                                       const device half4* biasTerms [[buffer(2)]],
                                       device float4 &printBuffer [[buffer(3)]],
                                       ushort3 gid [[thread_position_in_grid]])
    {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        ushort3 currentGid = ushort3(0, 0, 0);
        
        currentGid.x = gid.x * params.strideX;
        currentGid.y = gid.y * params.strideY;
        const ushort2 pos = currentGid.xy + ushort2(params.offsetX, params.offsetY);
        const ushort slice = gid.z;
        
        half4 in[9];
        in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1), slice);
        in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1), slice);
        in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1), slice);
        in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ), slice);
        in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ), slice);
        in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ), slice);
        in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1), slice);
        in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1), slice);
        in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1), slice);
        
        const ushort kW = 3;
        const ushort kH = 3;
        
        float4 out = float4(0.0f);
        for (ushort t = 0; t < kH*kW; ++t) {
            const auto pixel = float4(in[t]);
            out.x += pixel.x * float(weights[(slice*4 + 0)*kH*kW + t]);
            out.y += pixel.y * float(weights[(slice*4 + 1)*kH*kW + t]);
            out.z += pixel.z * float(weights[(slice*4 + 2)*kH*kW + t]);
            out.w += pixel.w * float(weights[(slice*4 + 3)*kH*kW + t]);
        }
        
        
        if (params.useBias) {
            out += float4(biasTerms[slice]);
        }
        
        out = applyNeuron(params.neuronType, out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), uint2(gid.xy), gid.z);
        
        if (gid.x == 0 && gid.y==0 && gid.z==0) {
            float4 bbb = float4(0);
            bbb.x = outTexture.get_array_size();
            printBuffer = float4(out);
            
        }
    }
    
    
    
    
    kernel void conv_array(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                           texture2d_array<half, access::write> outTexture [[texture(1)]],
                           constant ConvParams& params [[buffer(0)]],
                           const device half4* weights [[buffer(1)]],
                           const device half4* biasTerms [[buffer(2)]],
                           device float4 &printBuffer [[buffer(3)]],
                           ushort3 gid [[thread_position_in_grid]]) {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort kW = 1;
        const ushort kH = 1;
        ushort3 currentGid = ushort3(0, 0, 0);
        
        currentGid.x = gid.x * params.strideX;
        currentGid.y = gid.y * params.strideY;
        const ushort2 pos = currentGid.xy + ushort2(params.offsetX, params.offsetY);
        const ushort inSlices = inTexture.get_array_size();
        const ushort outSlice = gid.z;
        
        
        
        float4 out = float4(0.0f);
        
        for (ushort inSlice = 0; inSlice < inSlices; ++inSlice) {
            half4 in;
            in = inTexture.sample(s, float2(pos.x, pos.y), inSlice);
            
            half4 wx = weights[(outSlice*4 + 0)*kH*kW*inSlices + inSlice];
            out.x += dot(float4(in), float4(wx));
            
            half4 wy = weights[(outSlice*4 + 1)*kH*kW*inSlices + inSlice];
            out.y += dot(float4(in), float4(wy));
            
            half4 wz = weights[(outSlice*4 + 2)*kH*kW*inSlices + inSlice];
            out.z += dot(float4(in), float4(wz));
            
            half4 ww = weights[(outSlice*4 + 3)*kH*kW*inSlices + inSlice];
            out.w += dot(float4(in), float4(ww));
            
            
        }
        
        
        if (params.useBias) {
            out += float4(biasTerms[outSlice]);
        }
        out = applyNeuron(params.neuronType, out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), uint2(gid.xy), outSlice);
        
        if (gid.x == 0 && gid.y==0 && gid.z == 0 && outSlice == 0) {
            half4 in = inTexture.sample(s, float2(0, 0), 0);
//            in.w = pos.x;
//            in.z = pos.y;
//            float4 bbb = float4(out.x);
            //            bbb.x = th.x;
            //                            bbb.y = th.y;
            //            bbb.z = index;
            //            bbb.w = ggg;
            printBuffer = float4(in);
            
        }
    }
    
    
    kernel void global_average_pooling(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                                       device float4 &printBuffer [[buffer(0)]],
                                       ushort3 tid [[thread_position_in_threadgroup]],
                                       uint tin[[thread_index_in_threadgroup]],
                                       ushort3 gid [[thread_position_in_grid]]) {
        if (gid.x >= inTexture.get_width() ||
            gid.y >= inTexture.get_height() ||
            gid.z >= inTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        const ushort count = inTexture.get_width() * inTexture.get_height();
        threadgroup half4 shared_mem[100];
        const ushort outSlice = gid.z;
        const ushort index = tid.x * inTexture.get_width() + tid.y;
        shared_mem[index] = inTexture.sample(s, float2(tid.x, tid.y), outSlice);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tin == 0) {
            half4 sum = 0;
            for(ushort i = 0; i < count; i++) {
                sum += shared_mem[i];
            }
            sum/=count;
            outTexture.write(sum, uint2(gid.xy), outSlice);
        
        if (gid.z == 1 && gid.x == 0 && gid.y == 0) {
            float4 bbb = float4(shared_mem[2]);
            //            half4 i = inTexture.sample(s, float2(0, 0), 0);
            //            bbb.x = inTexture.get_width();
            //            bbb.y = inTexture.get_height();
                        bbb.z = index;
            //            bbb.w = ggg;
            printBuffer = float4(sum);
            
        }}

//        outTexture.write(sum, gid.xy, outSlice);
        
    }
    
    
    kernel void global_average_pooling1(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                                       device float4 &printBuffer [[buffer(0)]],
                                       ushort3 gid [[thread_position_in_grid]]) {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort outSlice = gid.z;
        half4 sum = half4(0);
        half4 in;
        for (ushort x = 0; x < inTexture.get_width(); x++) {
            for (ushort y = 0; y < inTexture.get_height(); y++) {
                in = inTexture.sample(s, float2(x, y), outSlice);
                sum += in;
            }
        }
        
        //        }
        if (gid.x == 0 && gid.y==0 && gid.z == 0) {
            float4 bbb = float4(sum);
//            half4 i = inTexture.sample(s, float2(0, 0), 0);
//            bbb.x = inTexture.get_width();
//            bbb.y = inTexture.get_height();
//            bbb.z = i.x;
            //            bbb.w = ggg;
            printBuffer = float4(bbb);
            
        }
        sum /= inTexture.get_width()*inTexture.get_height();

        outTexture.write(sum, uint2(gid.xy), outSlice);
        
    }

    kernel void fully_connected_conv(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                                       texture2d_array<half, access::write> outTexture [[texture(1)]],
                                     constant ConvParams& params [[buffer(0)]],
                                     const device half4* weights [[buffer(1)]],
                                     const device half4* biasTerms [[buffer(2)]],
                                       device float4 &printBuffer [[buffer(3)]],
                                       ushort3 gid [[thread_position_in_grid]]) {
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
//        const ushort kW = 1;
//        const ushort kH = 1;
        
        const ushort inSlices = inTexture.get_array_size();
        const ushort outSlice = gid.z;
        
        
        
        float4 out = float4(0.0f);
        
        for (ushort inSlice = 0; inSlice < inSlices; ++inSlice) {
            half4 in;
            in = inTexture.sample(s, float2(0, 0), inSlice);
            
            half4 wx = weights[(outSlice*4 + 0)*inSlices + inSlice];
            out.x += dot(float4(in), float4(wx));
            
            half4 wy = weights[(outSlice*4 + 1)*inSlices + inSlice];
            out.y += dot(float4(in), float4(wy));
            
            half4 wz = weights[(outSlice*4 + 2)*inSlices + inSlice];
            out.z += dot(float4(in), float4(wz));
            
            half4 ww = weights[(outSlice*4 + 3)*inSlices + inSlice];
            out.w += dot(float4(in), float4(ww));
            
            
        }
        
        
        if (params.useBias) {
            out += float4(biasTerms[outSlice]);
        }
        out = applyNeuron(params.neuronType, out, params.neuronA, params.neuronB);
        
        outTexture.write(half4(out), uint2(gid.xy), outSlice);
        
        if (gid.x == 0 && gid.y==0 && gid.z == 0) {
//            float4 bbb = float4(biasTerms[0]);
            //            bbb.x = th.x;
            //                            bbb.y = th.y;
            //            bbb.z = index;
            //            bbb.w = ggg;
            printBuffer = float4(biasTerms[0]);
        }
    }
    
    
    
    kernel void max_pooling(
                              texture2d_array<half, access::sample> inTexture [[texture(0)]],
                              texture2d_array<half, access::write> outTexture [[texture(1)]],
                              constant ConvParams& params [[buffer(0)]],
                              device float4 &printBuffer [[buffer(1)]],
                              ushort3 gid [[thread_position_in_grid]]) {
        
        if (gid.x >= outTexture.get_width() ||
            gid.y >= outTexture.get_height() ||
            gid.z >= outTexture.get_array_size()) return;
        
        constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
        
        const ushort kW = 3;
        const ushort kH = 3;
        
        ushort3 currentGid = ushort3(0, 0, 0);
        
        currentGid.x = gid.x * params.strideX;
        currentGid.y = gid.y * params.strideY;
        
        const ushort2 pos = currentGid.xy + ushort2(params.offsetX, params.offsetY);
        const ushort outSlice = gid.z;
        
        half4 in[9];
        in[0] = inTexture.sample(s, float2(pos.x - 1, pos.y - 1), outSlice);
        in[1] = inTexture.sample(s, float2(pos.x    , pos.y - 1), outSlice);
        in[2] = inTexture.sample(s, float2(pos.x + 1, pos.y - 1), outSlice);
        in[3] = inTexture.sample(s, float2(pos.x - 1, pos.y    ), outSlice);
        in[4] = inTexture.sample(s, float2(pos.x    , pos.y    ), outSlice);
        in[5] = inTexture.sample(s, float2(pos.x + 1, pos.y    ), outSlice);
        in[6] = inTexture.sample(s, float2(pos.x - 1, pos.y + 1), outSlice);
        in[7] = inTexture.sample(s, float2(pos.x    , pos.y + 1), outSlice);
        in[8] = inTexture.sample(s, float2(pos.x + 1, pos.y + 1), outSlice);
        
        half4 out = in[0];
        for (ushort t = 1; t < kH*kW; ++t) {
            out.x = fmax(out.x, in[t].x);
            out.y = fmax(out.y, in[t].y);

            out.z = fmax(out.z, in[t].z);
            out.w = fmax(out.w, in[t].w);

        }
        
        outTexture.write(half4(out), uint2(gid.xy), outSlice);
        if (gid.x == 0 && gid.y==0 && gid.z == 0) {
//            float4 bbb = float4(5);
            //            bbb.x = th.x;
            //                            bbb.y = th.y;
            //            bbb.z = index;
            //            bbb.w = ggg;
            printBuffer = float4(in[4]);
        }
    }
