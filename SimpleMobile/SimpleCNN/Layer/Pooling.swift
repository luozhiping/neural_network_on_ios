//
//  Pooling.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/15.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders
import MetalKit


@available(iOS 10, *)
public class GlobalAveragePooling: SimpleCNNLayer {
    public init(name: String) {
        super.init(name: name)
    }
    
    override public func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        if let inputShape = inputShape {
            self.inputShape = inputShape
            self.outputShape = DataShape(width: 1, height: 1, channels: inputShape.channels)
            super.setShape(inputShape: self.outputShape)
        }

    }
    
    override public func getOutputData(commandBuffer: MTLCommandBuffer, device: MTLDevice) -> DataWrapper {
        guard self.imageDesc != nil else {
            fatalError("should init imageDesc first")
        }
        //        if self.outputData == nil {
        self.outputData = ImageData(imageDesc: self.imageDesc!, commandBuffer: commandBuffer, device: device, isTemporary: isTemporary)
        //        }
        return self.outputData!
    }
    
    override public func createNetWork(device: MTLDevice) {
        
        let pool = MPSCNNPoolingAverage(device: device,
                                        kernelWidth: self.inputShape!.width,
                                        kernelHeight: self.inputShape!.height,
                                        strideInPixelsX: self.inputShape!.width,
                                        strideInPixelsY: self.inputShape!.height)
        
        pool.offset = MPSOffset(x: inputShape!.width/2, y: inputShape!.height/2, z: 0)
        pool.edgeMode = .clamp
        self.cnn = pool
    }
}

@available(iOS 10, *)
public class MaxPooling: SimpleCNNLayer {
    var params = ConvParams()
    let poolSize: (Int, Int)
    let stride: (Int, Int)
    let padding: PaddingType
    
    public init(poolSize: (Int, Int),
                stride: (Int, Int) = (1, 1),
                padding: PaddingType = .same,name: String) {
        self.poolSize = poolSize
        self.stride = stride
        self.padding = padding
        super.init(name: name)
//        forceNotTemp = true
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        if let inputShape = inputShape {
            self.inputShape = inputShape
            if padding == .same {
                self.outputShape = DataShape(width: Int(ceil(Float(inputShape.width)/Float(stride.0))), height: Int(ceil(Float(inputShape.height)/Float(stride.1))), channels: inputShape.channels)
                
            } else {
                self.outputShape = DataShape(width: (inputShape.width-poolSize.0)/stride.0+1, height: (inputShape.height-poolSize.0)/stride.0+1, channels: inputShape.channels)
                
            }
            super.setShape(inputShape: self.outputShape)
        }
    }
    
    override public func createNetWork(device: MTLDevice) {
        cnn = MPSCNNPoolingMax(device: device, kernelWidth: self.poolSize.0, kernelHeight: self.poolSize.1, strideInPixelsX: self.stride.0, strideInPixelsY: self.stride.1)
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if poolSize.0 == 3 {
            if self.inputShape!.width % 2 == 0 {
                cnn.offset = MPSOffset(x: 1, y: 1, z: 0)
            } else {
                cnn.offset = MPSOffset(x: 0, y: 0, z: 0)
            }
        } else if poolSize.0 == 2 {
            cnn.offset = MPSOffset(x: 1, y: 1, z: 0)
        }
//        cnn.offset = offsetForConvolution(padding: .valid,
//                                           sourceWidth: inputShape!.width,
//                                           sourceHeight: inputShape!.height,
//                                           destinationWidth: outputShape!.width,
//                                           destinationHeight: outputShape!.height,
//                                           kernelWidth: poolSize.0,
//                                           kernelHeight: poolSize.1,
//                                           strideInPixelsX: stride.0,
//                                           strideInPixelsY: stride.1)
        
        cnn.edgeMode = .clamp
//        print("\(self.name)offset,", cnn.offset)
        super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
    }
}
