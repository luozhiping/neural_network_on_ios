//
//  ShaderPreproccess.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/24.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

public class ShaderImageInput: ShaderLayer {
    let imageWidth: Int
    let imageHeight: Int
    let channels: Int
    var lanczos: MPSImageLanczosScale!
    
    
    public init(device: MTLDevice, imageWidth: Int, imageHeight: Int, channels: Int, name: String) {
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.channels = channels
        //        self.outputShape = self.inputShape
        super.init(device: device, name: name)
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.outputShape = DataShape(width: imageWidth, height: imageHeight, channels: channels)
        super.setShape(inputShape: self.outputShape)
    }
    
    override public func createNetWork(device: MTLDevice) {
        lanczos = MPSImageLanczosScale(device: device)
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        let inputData = sourceData
        let outputData = destinationData
        if inputData.getTexture()!.width != outputData.getTexture()!.width || inputData.getTexture()!.width != outputData.getTexture()!.height {
            lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputData.texture!, destinationTexture: outputData.texture!)
        } else {
            destinationData.texture = inputData.texture
        }
    }
}

public class ShaderZeroPadding: ShaderLayer {
    var padding: [[Int]]
    var params: PaddingParams
    public init(device: MTLDevice, padding: [[Int]], name: String = "ShaderZeroPadding") {
        self.padding = padding
        self.params = PaddingParams()
        super.init(device: device, name: name)
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        if let inputShape = inputShape {
            self.inputShape = inputShape
            self.outputShape = DataShape(width: inputShape.width+self.padding[0][0]+self.padding[0][1], height: inputShape.height+self.padding[1][0]+self.padding[1][1], channels: inputShape.channels)
            
            params.paddingTop = Int16(self.padding[0][0])
            params.paddingBottom = Int16(self.padding[0][1])
            params.paddingLeft = Int16(self.padding[1][0])
            params.paddingRight = Int16(self.padding[1][1])
            
            super.setShape(inputShape: self.outputShape)
        }
    }
    
    public override func getOutputData(commandBuffer: MTLCommandBuffer, device: MTLDevice) -> DataWrapper {
        if #available(*, iOS 9) {
            return super.getOutputData(commandBuffer: commandBuffer, device: device)
        } else {
            self.outputData = ImageData(imageDesc: self.imageDesc!, commandBuffer: commandBuffer, device: device, isTemporary: isTemporary)
            //        }
            return self.outputData!
        }
    }
    
    override public func createNetWork(device: MTLDevice) {
        if self.outputShape!.channels > 4 {
            pipeline = initFunction(device: device, name: "zeropadding_array")
        } else {
            pipeline = initFunction(device: device, name: "zeropadding")
        }
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
//        var inputTexture: MTLTexture? = sourceData.getTexture()
//        var outputTexture: MTLTexture? = destinationData.getTexture()
//        if let input = sourceData as? ImageData {
//            inputTexture = input.image!.texture
//            outputTexture = (destinationData as! ImageData).image!.texture
//        }
//        print(destinationData.getTexture()!)
        if let pipeline = self.pipeline {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture()!, index: 0)
                encoder.setTexture(destinationData.getTexture()!, index: 1)
                encoder.setBytes(&params, length: MemoryLayout<PaddingParams>.size, index: 0)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 1)
                encoder.dispatch(pipeline: pipeline, width: outputShape!.width, height: outputShape!.height, featureChannels: outputShape!.channels)
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
