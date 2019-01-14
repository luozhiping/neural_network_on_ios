//
//  Preproccess.metal
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/27.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;
struct PaddingParam {
    
    ushort paddingTop;
    ushort paddingBottom;
    ushort paddingLeft;
    ushort paddingRight;
};

kernel void zeropadding(texture2d<half, access::sample> inTexture [[texture(0)]],
                        texture2d<half, access::write> outTexture [[texture(1)]],
                        constant PaddingParam& params [[buffer(0)]],
                        device float4 &printBuffer [[buffer(1)]],
                        ushort3 gid [[thread_position_in_grid]]) {
   
    if (gid.x + params.paddingLeft >= outTexture.get_width()||
        gid.y + params.paddingTop >= outTexture.get_height()) {
        return;
    }
    
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);

    half4 in = inTexture.sample(s, float2(gid.x, gid.y));
    outTexture.write(in, uint2(gid.x + params.paddingLeft, gid.y + params.paddingTop));
    
    if (gid.x == 0 && gid.y==0) {
//        float4 bbb = float4(in);
        
        printBuffer = float4(in);
        
    }
}

kernel void zeropadding_array(texture2d_array<half, access::sample> inTexture [[texture(0)]],
                        texture2d_array<half, access::write> outTexture [[texture(1)]],
                        constant PaddingParam& params [[buffer(0)]],
                        device float4 &printBuffer [[buffer(1)]],
                        ushort3 gid [[thread_position_in_grid]]) {
    
    if (gid.x + params.paddingLeft >= outTexture.get_width()||
        gid.y + params.paddingTop >= outTexture.get_height()||
        gid.z >= outTexture.get_array_size()) {
        return;
    }
    
    constexpr sampler s(coord::pixel, filter::nearest, address::clamp_to_zero);
    
    half4 in = inTexture.sample(s, float2(gid.x, gid.y), gid.z);
    outTexture.write(in, uint2(gid.x + params.paddingLeft, gid.y + params.paddingTop), gid.z);
    
    if (gid.x == 0 && gid.y==0) {
//        float4 bbb = float4(in);
        
        printBuffer = float4(in);
        
    }
}

