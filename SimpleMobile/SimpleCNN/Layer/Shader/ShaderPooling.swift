//
//  ShaderPooling.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/22.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

public class ShaderGlobalAveragePooling: ShaderLayer {
    public init(device: MTLDevice, name: String) {
        super.init(device: device, name: name)
    }
    
    override public func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        if let inputShape = inputShape {
            self.inputShape = inputShape
            self.outputShape = DataShape(width: 1, height: 1, channels: inputShape.channels)
            super.setShape(inputShape: self.outputShape)
        }
        
    }
    
    override public func createNetWork(device: MTLDevice) {
        let functionName = "global_average_pooling"
        self.device = device
        pipeline = initFunction(device: device, name: functionName)
        
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture(), index: 0)
                encoder.setTexture(destinationData.getTexture(), index: 1)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 0)
                encoder.dispatch(pipeline: pipeline, width: destinationData.getTexture()!.width, height: destinationData.getTexture()!.height, featureChannels: outputShape!.channels)
                encoder.endEncoding()
                commandBuffer.addCompletedHandler {commandBuffer in
                    let data = NSData(bytes: outVectorBuffer!.contents(), length: 20)
                    //            var a = outVectorBuffer!.contents() as! UnsafePointer<Float>
                    var out = [Float](repeating:0, count:5)
                    data.getBytes(&out, length: 20)
//                    print("ShaderGlobalAveragePooling: \(out)")
                }
            }
            super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
        }
    }
    
}

public class ShaderMaxPooling: ShaderLayer {
    var params = ConvParams()
    let poolSize: (Int, Int)
    let stride: (Int, Int)
    let padding: PaddingType
    
    
    public init(device: MTLDevice, poolSize: (Int, Int),
                stride: (Int, Int) = (1, 1),
                padding: PaddingType = .same,
                name: String = "ShaderMaxPooling") {
        self.poolSize = poolSize
        self.stride = stride
        self.padding = padding
        super.init(device: device, name: name)
    }
    
    override public func setShape(inputShape: DataShape?, outputShape: DataShape?) {
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
        let functionName = "max_pooling"
        self.device = device
        pipeline = initFunction(device: device, name: functionName)
        params.strideX = Int16(stride.0)
        params.strideY = Int16(stride.1)
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            let offset = offsetForConvolution(padding: .same,
                                              sourceWidth: inputShape!.width,
                                              sourceHeight: inputShape!.height,
                                              destinationWidth: outputShape!.width,
                                              destinationHeight: outputShape!.height,
                                              kernelWidth: poolSize.0,
                                              kernelHeight: poolSize.1,
                                              strideInPixelsX: stride.0,
                                              strideInPixelsY: stride.1)
            params.offsetX = Int16(offset.x)
            params.offsetY = Int16(offset.y)
            params.offsetZ = Int16(offset.z)
//            print("offset maxpool:", offset, params, destinationData.texture!.arrayLength)

            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture(), index: 0)
                encoder.setTexture(destinationData.getTexture(), index: 1)
                encoder.setBytes(&params, length: MemoryLayout<ConvParams>.size+3, index: 0)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 1)
                encoder.dispatch(pipeline: pipeline, width: destinationData.texture!.width, height: destinationData.texture!.height, featureChannels: outputShape!.channels)
                encoder.endEncoding()
                commandBuffer.addCompletedHandler {commandBuffer in
                    let data = NSData(bytes: outVectorBuffer!.contents(), length: 20)
                    //            var a = outVectorBuffer!.contents() as! UnsafePointer<Float>
                    var out = [Float](repeating:0, count:5)
                    data.getBytes(&out, length: 20)
//                    print("\(self.name): \(out)")
                }
            }
            super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
        }
    }
    
}
