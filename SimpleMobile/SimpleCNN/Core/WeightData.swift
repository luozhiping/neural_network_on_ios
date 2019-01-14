//
//  WeightData.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/13.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation


open class Weights {
    var weightsPointer: UnsafePointer<Float>
    var biasPointer: UnsafePointer<Float>
    
    public init(weightsPointer: UnsafePointer<Float>, biasPointer: UnsafePointer<Float>) {
        self.weightsPointer = weightsPointer
        self.biasPointer = biasPointer
    }
}
