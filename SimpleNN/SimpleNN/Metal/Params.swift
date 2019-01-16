//
//  Params.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/18.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation

public struct ConvParams {
    var offsetX: Int16 = 0
    var offsetY: Int16 = 0
    
    var offsetZ: Int16 = 0
    
    var strideY: Int16 = 0
    var strideX: Int16 = 0
    var neuronType: Int16 = 0
    var neuronA: Float = 0;
    var neuronB: Float = 0;
    var useBias: Bool = true;
}

public struct ConvParams2 {
    var offsetX: Int16 = 0
    var offsetY: Int16 = 0
    
    var offsetZ: Int16 = 0
    
    var strideY: Int16 = 0
    var strideX: Int16 = 0
    var neuronType: Int16 = 0
    var neuronA: Float = 0;
    var neuronB: Float = 0;
    var useBias: Bool = true;
}


public struct PaddingParams {
    var paddingTop: Int16 = 0;
    var paddingBottom: Int16 = 0;
    var paddingLeft: Int16 = 0;
    var paddingRight: Int16 = 0;

}

public struct BatchNormParams {
    var epsilon: Float = 0.001;
    var neuronType: Int16 = 0
    var neuronA: Float = 0;
    var neuronB: Float = 0;
//    var
}

public struct ReluParams {
    var max_value: Float = 10000;
    var negative_slope: Float = 0
    var threshold: Float = 0;
    //    var
}

public struct UpSamplingParams {
    var factors_w: Int16 = 1;
    var factors_h: Int16 = 1;
    //    var
}
