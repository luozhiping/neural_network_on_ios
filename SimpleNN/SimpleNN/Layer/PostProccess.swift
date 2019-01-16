//
//  PostProccess.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/29.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders
@available(iOS 10, *)
public class Softmax: SimpleCNNLayer {
    
    public init(name: String) {
        super.init(name: name)
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        self.outputShape = inputShape
        super.setShape(inputShape: self.outputShape)
    }
    
    override public func createNetWork(device: MTLDevice) {
        cnn = MPSCNNSoftMax(device: device)
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
    }
}

@available(iOS 11.3, *)
public class BatchNormalization: SimpleCNNLayer {
//    var dataSource: MPSCNNBatchNormalizationDataSource
    public init(name: String) {
        super.init(name: name)
    }
    
//    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
//        self.inputShape = inputShape
//        self.outputShape = inputShape
//        super.setShape(inputShape: self.outputShape)
//    }
//
//    public override func insertWeight(pointer: UnsafePointer<Float>) -> UnsafePointer<Float> {
//        dataSource = MPSCNNBatchNormalizationDataSource(coder: <#T##NSCoder#>)
//    }
//
//    override public func createNetWork(device: MTLDevice) {
//        cnn = MPSCNNBatchNormalization(device: device, dataSource: dataSource)
//    }
//
//    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
//        super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
//    }
}


