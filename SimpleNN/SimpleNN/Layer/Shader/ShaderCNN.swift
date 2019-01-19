//
//  ShaderCNN.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/22.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

public class ShaderConvolution: ShaderLayer {
    var params = ConvParams()
    let kernel: (Int, Int)
    var outputChannels: Int
    let stride: (Int, Int)
    let activation: String?
    let padding: PaddingType
    
    
    public init(device: MTLDevice, kernel: (Int, Int),
                outputChannels: Int,
                stride: (Int, Int) = (1, 1),
                padding: PaddingType = .same,
                activation: String? = nil,
                name: String = "ShaderConvolution", useBias: Bool = true) {
        self.kernel = kernel
        self.outputChannels = outputChannels
        self.stride = stride
        self.padding = padding
        self.activation = activation
        super.init(device: device, name: name, useBias: useBias)
    }
    override public func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        if let inputShape = inputShape {
            self.inputShape = inputShape
            if padding == .same {
                let width = ceil(Float(inputShape.width)/Float(stride.0))
                self.outputShape = DataShape(width: Int(width), height: Int(ceil(Float(inputShape.height)/Float(stride.1))), channels: outputChannels)
                
            } else {
                self.outputShape = DataShape(width: (inputShape.width-kernel.0)/stride.0+1, height: (inputShape.height-kernel.1)/stride.1+1, channels: outputChannels)
                
            }
            super.setShape(inputShape: self.outputShape)
        }
    }
    
    override public func createNetWork(device: MTLDevice) {
        var functionName = "conv3x3"
        if self.inputShape!.channels <= 4 && self.outputShape!.channels > 4 {
            functionName = "conv3x3_out_array"
        } else if self.inputShape!.channels > 4 && self.outputShape!.channels > 4 {
            functionName = "conv3x3_array"
        }
        if kernel.0 < 3 && kernel.1 < 3 {
            functionName = "conv_array"
        }
        pipeline = initFunction(device: device, name: functionName)
        params.useBias = self.useBias
        let act = Activation(activation: activation)
        params.neuronType = Int16(act.activationType())
        
        params.neuronA = act.actA
//        params.neuronB = act.actB
        
        params.strideX = Int16(stride.0)
        params.strideY = Int16(stride.1)
    }
    
    override public func getOutputData(commandBuffer: MTLCommandBuffer, device: MTLDevice) -> DataWrapper {
        return super.getOutputData(commandBuffer: commandBuffer, device: device)
    }
    
    override public func insertWeight(pointer: UnsafePointer<Float>) -> UnsafePointer<Float> {
        let p = super.insertWeight(pointer: pointer)
        
        weightsBuffer = makeBuffer(device: device,
                                   kernelWidth: kernel.0,
                                   kernelHeight: kernel.1,
                                   inputFeatureChannels: inputShape!.channels,
                                   outputFeatureChannels: outputShape!.channels,
                                   weights: weight!.weightsPointer)
        if useBias {
            biasBuffer = makeBuffer(device: device,
                                    outputFeatureChannels: outputShape!.channels,
                                    biasTerms: weight!.biasPointer)
        }
        
        
        return p
    }
    
    override public func getWeightSize() -> Int {
        return inputShape!.channels * kernel.1 * kernel.0 * outputShape!.channels
    }
    
    override public func getBiasSize() -> Int {
        return useBias ? outputShape!.channels : 0
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            let offset = offsetForConvolution(padding: padding,
                                              sourceWidth: inputShape!.width,
                                              sourceHeight: inputShape!.height,
                                              destinationWidth: outputShape!.width,
                                              destinationHeight: outputShape!.height,
                                              kernelWidth: kernel.0,
                                              kernelHeight: kernel.1,
                                              strideInPixelsX: stride.0,
                                              strideInPixelsY: stride.1)
            params.offsetX = Int16(offset.x)
            params.offsetY = Int16(offset.y)
            params.offsetZ = Int16(offset.z)
//            print(destinationData.getTexture()!.width, destinationData.getTexture()!.height, outputShape!.channels, MemoryLayout<ConvParams2>.size)
//                    print("\(self.name) offset1:", offset, params, destinationData.texture!.arrayLength)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture(), index: 0)
                encoder.setTexture(destinationData.getTexture(), index: 1)
                encoder.setBytes(&params, length: MemoryLayout<ConvParams>.size+3, index: 0)
                encoder.setBuffer(weightsBuffer, offset: 0, index: 1)
                if useBias {
                    encoder.setBuffer(biasBuffer, offset: 0, index: 2)

                } else {
                    encoder.setBuffer(weightsBuffer, offset: 0, index: 2)

                }
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 3)
                encoder.dispatch(pipeline: pipeline, width: destinationData.getTexture()!.width, height: destinationData.getTexture()!.height, featureChannels: outputShape!.channels)
                encoder.endEncoding()
                commandBuffer.addCompletedHandler {commandBuffer in
                    let data = NSData(bytes: outVectorBuffer!.contents(), length: 20)
                    //            var a = outVectorBuffer!.contents() as! UnsafePointer<Float>
                    var out = [Float](repeating:0, count:5)
                    data.getBytes(&out, length: 20)
//                                    print("\(self.name): \(out)")
                }
            }
            super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
        }
    }
    
}


public class ShaderDepthwiseConvolution: ShaderLayer {
    var params = ConvParams()
    let kernel: (Int, Int)
    let stride: (Int, Int)
    let activation: String?
    let padding: PaddingType
    public init(device: MTLDevice, kernel: (Int, Int),
                stride: (Int, Int) = (1, 1),
                padding: PaddingType = .same,
                activation: String? = nil,
                name: String = "ShaderDepthwiseConvolution", useBias: Bool = true) {
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
//                self.device = device
        super.init(device:device, name: name, useBias: useBias)
    }
    
    override public func getWeightSize() -> Int {
        return inputShape!.channels * kernel.1 * kernel.0
    }
    
    override public func getBiasSize() -> Int {
        return useBias ? outputShape!.channels : 0
    }
    
    override public func insertWeight(pointer: UnsafePointer<Float>) -> UnsafePointer<Float> {
        var p = pointer
        let weight = p
        p += self.getWeightSize()
//        print(name, "weightsize:", getWeightSize(), weight[0], weight[1], weight[2])
        let bias = p
        p += self.getBiasSize()
//        print(name, "biassize:", getBiasSize(), bias[0])
        self.weight = Weights(weightsPointer: weight, biasPointer: bias)
        //        return p
        
        //        let outputSlices = (inputShape!.channels + 3) / 4
        //        let paddedOutputChannels = outputSlices * 4
        //        let count = paddedOutputChannels * kernel.0 * kernel.1
        //
        //        weightsBuffer = device.makeBuffer(length: MemoryLayout<Float32>.stride * count)!
        //        let ptr = UnsafeMutablePointer(mutating: weight!.weightsPointer)
        //        let copyCount = self.getWeightSize()
        //        float32to16(input: ptr, output: weightsBuffer.contents(), count: copyCount)
        weightsBuffer = makeBuffer(device: device,
                                   outputFeatureChannels: getWeightSize(),
                                   biasTerms: self.weight!.weightsPointer)
        
        biasBuffer = makeBuffer(device: device,
                                outputFeatureChannels: outputShape!.channels,
                                biasTerms: self.weight!.biasPointer)
        return p
    }
    
    override public func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        if let inputShape = inputShape {
            self.inputShape = inputShape
            if padding == .same {
                self.outputShape = DataShape(width: Int(ceil(Float(inputShape.width)/Float(stride.0))), height: Int(ceil(Float(inputShape.height)/Float(stride.1))), channels: inputShape.channels)
                
            } else {
                self.outputShape = DataShape(width: (inputShape.width-kernel.0)/stride.0+1, height: (inputShape.height-kernel.0)/stride.0+1, channels: inputShape.channels)
                
            }
            for layer in childrenLayers {
                layer.setShape(inputShape: self.outputShape)
            }
//            setImageDesc(isTemporary: true)

        }
    }
    
    
    public override func createNetWork(device: MTLDevice) {
        let functionName = "depthwiseConv3x3_array"
        pipeline = initFunction(device: device, name: functionName)
        
        let act = Activation(activation: activation)
        params.neuronType = Int16(act.activationType())
        
        params.neuronA = act.actA
//        params.neuronB = act.actB
        params.useBias = self.useBias
    }
    
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            let offset = offsetForConvolution(padding: padding,
                                              sourceWidth: inputShape!.width,
                                              sourceHeight: inputShape!.height,
                                              destinationWidth: outputShape!.width,
                                              destinationHeight: outputShape!.height,
                                              kernelWidth: kernel.0,
                                              kernelHeight: kernel.1,
                                              strideInPixelsX: stride.0,
                                              strideInPixelsY: stride.1)
            params.offsetX = Int16(offset.x)
            params.offsetY = Int16(offset.y)
            params.offsetZ = Int16(offset.z)
            params.strideX = Int16(stride.0)
            params.strideY = Int16(stride.1)
//            print("\(self.name) offset1:", offset, params, destinationData.texture!.arrayLength)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture(), index: 0)
                encoder.setTexture(destinationData.getTexture(), index: 1)
                encoder.setBytes(&params, length: MemoryLayout<ConvParams>.size+3, index: 0)
                encoder.setBuffer(weightsBuffer, offset: 0, index: 1)
                encoder.setBuffer(biasBuffer, offset: 0, index: 2)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 3)
                encoder.dispatch(pipeline: pipeline, width: destinationData.getTexture()!.width, height: destinationData.getTexture()!.height, featureChannels: outputShape!.channels)
                encoder.endEncoding()
                commandBuffer.addCompletedHandler {commandBuffer in
                    let data = NSData(bytes: outVectorBuffer!.contents(), length: 20)
                    //            var a = outVectorBuffer!.contents() as! UnsafePointer<Float>
                    var out = [Float](repeating:0, count:5)
                    data.getBytes(&out, length: 20)
//                                    print("\(self.name): \(out)")
                }
            }
            super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
        }
    }
}

public class ShaderFullyConnectedConvolution: ShaderConvolution {
    public init(device: MTLDevice, outputChannels: Int, activation: String?, name: String = "ShaderFullyConnectedConvolution", useBias: Bool = false) {
        super.init(device: device, kernel: (1, 1), outputChannels: outputChannels, stride: (1, 1), padding: .same, activation: activation, name: name, useBias: useBias)
    }
    
    
    override public func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        self.outputShape = DataShape(width: 1, height: 1, channels: outputChannels)
        for layer in childrenLayers {
            layer.setShape(inputShape: self.outputShape)
        }
        setImageDesc(isTemporary: true)

    }
    
    public override func getWeightSize() -> Int {
        return inputShape!.width * inputShape!.height * inputShape!.channels * outputChannels
    }
    
    public override func getBiasSize() -> Int {
        return useBias ? outputChannels : 0
    }
    
    public override func createNetWork(device: MTLDevice) {
        let functionName = "fully_connected_conv"
        pipeline = initFunction(device: device, name: functionName)
        
        let act = Activation(activation: activation)
        params.neuronType = Int16(act.activationType())
        
        params.neuronA = act.actA
//        params.neuronB = act.actB
        
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            //        print("offset1:", offset, params, destinationData.texture!.arrayLength)
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.texture, index: 0)
                encoder.setTexture(destinationData.texture, index: 1)
                encoder.setBytes(&params, length: MemoryLayout<ConvParams>.size+3, index: 0)
                encoder.setBuffer(weightsBuffer, offset: 0, index: 1)
                encoder.setBuffer(biasBuffer, offset: 0, index: 2)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 3)
                encoder.dispatch(pipeline: pipeline, width: destinationData.texture!.width, height: destinationData.texture!.height, featureChannels: outputShape!.channels)
                encoder.endEncoding()
                commandBuffer.addCompletedHandler {commandBuffer in
                    let data = NSData(bytes: outVectorBuffer!.contents(), length: 20)
                    //            var a = outVectorBuffer!.contents() as! UnsafePointer<Float>
                    var out = [Float](repeating:0, count:5)
                    data.getBytes(&out, length: 20)
//                                    print("\(self.name): \(out)")
                }
            }
        }
    }
}


public class ShaderSeparableConv: ShaderLayer {
    let depthwise: ShaderDepthwiseConvolution
//    let pointwise: ShaderConvolution
    let pointwise: Convolution

    
    public init(device: MTLDevice, kernel: (Int, Int),
                         stride: (Int, Int) = (1, 1),
                         padding: PaddingType = .same, outChannels: Int, activation: String? = nil, name: String, useBias: Bool) {
        self.depthwise = ShaderDepthwiseConvolution(device:device, kernel: kernel, stride: stride, activation: activation!,  name: name,useBias: useBias)
        self.pointwise = Convolution(kernel: (1, 1), outputChannels: outChannels, stride: (1, 1), padding: .same, activation: activation, name: name, useBias: useBias)
        super.init(device: device, name: name, useBias: useBias)
        
        self.depthwise.tmpImageCount += 1
        self.pointwise.tmpImageCount += 1
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.depthwise.setShape(inputShape: inputShape, outputShape: outputShape)
        self.pointwise.setShape(inputShape: self.depthwise.outputShape, outputShape: nil)
        
        self.inputShape = inputShape
        self.outputShape = self.pointwise.outputShape
        
        super.setShape(inputShape: self.pointwise.outputShape)
    }
    
    public override func setImageDesc(isTemporary: Bool) {
        self.depthwise.setImageDesc(isTemporary: isTemporary)
        self.pointwise.setImageDesc(isTemporary: isTemporary)
        super.setImageDesc(isTemporary: isTemporary)
    }
    
    public override func insertWeight(pointer: UnsafePointer<Float>) -> UnsafePointer<Float> {
        var p = self.depthwise.insertWeight(pointer: pointer)
        p = self.pointwise.insertWeight(pointer: p)
        return p
    }
    
    public override func getWeightSize() -> Int {
        return self.depthwise.getWeightSize() + self.pointwise.getWeightSize()
    }
    
    public override func getBiasSize() -> Int {
        return self.depthwise.getBiasSize() + self.depthwise.getBiasSize()
    }
    
    public override func createNetWork(device: MTLDevice) {
        self.depthwise.createNetWork(device: device)
        self.pointwise.createNetWork(device: device)
    }
    
    public override func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        self.depthwise.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: self.depthwise.getOutputData(commandBuffer: commandBuffer, device: device))
        self.pointwise.encode(commandBuffer: commandBuffer, sourceData: self.depthwise.getOutputData()!, destinationData: destinationData)
    }
    
    public override func getOutputData(commandBuffer: MTLCommandBuffer, device: MTLDevice) -> DataWrapper {
        return self.pointwise.getOutputData(commandBuffer:commandBuffer, device:device)
    }
}
