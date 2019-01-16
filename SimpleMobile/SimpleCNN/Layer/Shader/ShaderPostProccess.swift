//
//  ShaderPostProccess.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/27.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation

import MetalPerformanceShaders

public class ShaderSoftmax: ShaderLayer {
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        self.outputShape = inputShape
        super.setShape(inputShape: self.outputShape)
    }
    
    override public func createNetWork(device: MTLDevice) {
        pipeline = initFunction(device: device, name: "softmax_1x1")
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.texture, index: 0)
                encoder.setTexture(destinationData.texture, index: 1)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 0)
                encoder.dispatch(pipeline: pipeline, count:outputShape!.channels)
                encoder.endEncoding()
                commandBuffer.addCompletedHandler {commandBuffer in
                    let data = NSData(bytes: outVectorBuffer!.contents(), length: 20)
                    //            var a = outVectorBuffer!.contents() as! UnsafePointer<Float>
                    var out = [Float](repeating:0, count:5)
                    data.getBytes(&out, length: 20)
//                    print("Softmax: \(out)")
                }
            }
            super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
        }
        
    }
}


public class ShaderBatchNormalization: ShaderLayer {
    var params: BatchNormParams
//    var activation: String?

    public init(device: MTLDevice, eplison: Float, name: String = "ShaderBatchNormalization") {
        self.params = BatchNormParams()
        self.params.epsilon = eplison
        super.init(device: device, name: name)
    }
    
    public override func setActivation(activation: String) {
        let act = Activation(activation: activation)
        params.neuronType = Int16(act.activationType())
        
        params.neuronA = act.actA
        params.neuronB = act.actB
    }
    
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        self.outputShape = inputShape
        super.setShape(inputShape: self.outputShape)
    }
    
    public override func getWeightSize() -> Int {
        // [gamma, beta, moving_mean, moving_variance]
        return inputShape!.channels * 4
    }
    
    override public func createNetWork(device: MTLDevice) {
        pipeline = initFunction(device: device, name: "batch_norm")
    }
    
    override public func insertWeight(pointer: UnsafePointer<Float>) -> UnsafePointer<Float> {
        let p = super.insertWeight(pointer: pointer)
        
        weightsBuffer = makeBuffer(device: device,
                                   outputFeatureChannels: getWeightSize(),
                                   biasTerms: weight!.weightsPointer)
        return p
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture(), index: 0)
                encoder.setTexture(destinationData.getTexture(), index: 1)
                encoder.setBytes(&params, length: MemoryLayout<BatchNormParams>.size, index: 0)
                encoder.setBuffer(weightsBuffer, offset: 0, index: 1)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 2)
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


public class ShaderRelu: ShaderLayer {
    var params: ReluParams
    //    var activation: String?
    
    public init(device: MTLDevice, maxValue: Float, negativeSlope: Float, threshold: Float, name: String = "ShaderRelu") {
        self.params = ReluParams()
        self.params.max_value = maxValue
        self.params.negative_slope = negativeSlope
        self.params.threshold = threshold
        super.init(device: device, name: name)
        
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        self.outputShape = inputShape
        super.setShape(inputShape: self.outputShape)
    }
    
    override public func createNetWork(device: MTLDevice) {
        pipeline = initFunction(device: device, name: "relu")
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture(), index: 0)
                encoder.setTexture(destinationData.getTexture(), index: 1)
                encoder.setBytes(&params, length: MemoryLayout<ReluParams>.size, index: 0)
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
//                    print("ShaderRelu: \(out)")
                }
            }
            super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
        }
        
    }
}

public class ShaderLeakyRelu: ShaderLayer {
    var params: ReluParams
    //    var activation: String?
    
    public init(device: MTLDevice, negativeSlope: Float, name: String = "ShaderLeakyRelu") {
        self.params = ReluParams()
        self.params.negative_slope = negativeSlope
        self.params.threshold = 0
        super.init(device: device, name: name)
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        self.outputShape = inputShape
        super.setShape(inputShape: self.outputShape)
    }
    
    override public func createNetWork(device: MTLDevice) {
        pipeline = initFunction(device: device, name: "leaky_relu")
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture(), index: 0)
                encoder.setTexture(destinationData.getTexture(), index: 1)
                encoder.setBytes(&params, length: MemoryLayout<ReluParams>.size, index: 0)
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

public class ShaderAdd: ShaderLayer {
    var addLayer1: Layer?
    var addLayer2: Layer?
    public init(device: MTLDevice, name: String = "ShaderAdd") {
//        self.addLayer1 = addLayer1
//        self.addLayer2 = addLayer2
        super.init(device: device, name: name)
    }
    
    public func setLayer(layer1: Layer, layer2: Layer) {
        self.addLayer1 = layer1
        self.addLayer2 = layer2
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        self.outputShape = inputShape
        super.setShape(inputShape: self.outputShape)
    }
    
    override public func createNetWork(device: MTLDevice) {
        pipeline = initFunction(device: device, name: "add")
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture(), index: 0)
                encoder.setTexture(addLayer1!.getOutputData()!.getTexture(), index: 1)
                encoder.setTexture(destinationData.getTexture(), index: 2)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 0)
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
            addLayer1!.cleanData()
            super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
        }
        
    }
}

func address(o: UnsafeRawPointer) -> String {
    return String.init(format: "%018p", Int(bitPattern: o))
}


public class ShaderUpSampling: ShaderLayer {
    var params: UpSamplingParams
    //    var activation: String?
    
    public init(device: MTLDevice, factorW: Int, factorH: Int, name: String = "ShaderUpSampling") {
        self.params = UpSamplingParams()
        self.params.factors_w = Int16(factorW)
        self.params.factors_h = Int16(factorH)
        super.init(device: device, name: name)
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        self.outputShape = DataShape(width: inputShape!.width*Int(params.factors_w), height: inputShape!.height*Int(params.factors_h), channels: inputShape!.channels)
        super.setShape(inputShape: self.outputShape)
    }
    
    override public func createNetWork(device: MTLDevice) {
        pipeline = initFunction(device: device, name: "upsampling")
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(pipeline)
                encoder.setTexture(sourceData.getTexture(), index: 0)
                encoder.setTexture(destinationData.getTexture(), index: 1)
                encoder.setBytes(&params, length: MemoryLayout<UpSamplingParams>.size, index: 0)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 1)
                encoder.dispatch(pipeline: pipeline, width: inputShape!.width, height: inputShape!.height, featureChannels: inputShape!.channels)
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


public class ShaderConcatenate: ShaderLayer {
    var layers: [Layer]?
    public init(device: MTLDevice, name: String = "ShaderConcatenate") {
        //        self.addLayer1 = addLayer1
        //        self.addLayer2 = addLayer2
        super.init(device: device, name: name)
    }
    
    
    public func setLayers(layers: [Layer]) {
        self.layers = layers
    }
    
    public override func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        var channels = 0
        for l in layers! {
            channels += l.outputShape!.channels
        }
        self.outputShape = DataShape(width: inputShape!.width, height: inputShape!.height, channels: channels)
//        self.outputShape = DataShape(width: inputShape!.width, height: inputShape!.height, channels: addLayer1!.outputShape!.channels+addLayer2!.outputShape!.channels)
        super.setShape(inputShape: self.outputShape)
    }
    
    override public func createNetWork(device: MTLDevice) {
        pipeline = initFunction(device: device, name: "concatenate")
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if let pipeline = self.pipeline {
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                var textures = [MTLTexture]()
                for l in layers! {
                    textures.append(l.getOutputData()!.getTexture()!)
                }
                
                encoder.setComputePipelineState(pipeline)
//                encoder.setTexture(addLayer1!.getOutputData()!.getTexture(), index: 0)
                encoder.setTextures(textures, range: 0..<textures.count)
                encoder.setTexture(destinationData.getTexture(), index: 4)
                var resultdata = [Float](repeating: 0, count: 1)
                let outVectorBuffer = device.makeBuffer(bytes: &resultdata, length: 16, options: MTLResourceOptions.cpuCacheModeWriteCombined)
                encoder.setBuffer(outVectorBuffer, offset: 0, index: 0)
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
            for l in layers! {
                l.cleanData()
            }
            super.encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
        }
        
    }
}
