//
//  CNN.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/13.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

@available(iOS 10, *)
func createActivation(activation: String?, device: MTLDevice) -> MPSCNNNeuron? {
    guard let activation = activation else {
        return nil
    }
    var act: MPSCNNNeuron?
    switch activation {
    case "relu":
        act = MPSCNNNeuronReLU(device: device, a: 0)
        break
    case "linear":
        act = MPSCNNNeuronLinear(device: device, a: 1, b: 0)
        break
    default:
        act = MPSCNNNeuronLinear(device: device, a: 1, b: 0)
    }
    return act
}

enum ActivationType: Int {
    case NeuronTypeNone = 0, NeuronTypeReLU, earth, mars, jupiter, saturn, uranus, neptune
}

class Activation {
    var _activationType: ActivationType
    var actA: Float
    var actB: Float
    init(activation: String?) {
        actA = 0
        actB = 0
        if activation == nil {
            _activationType = ActivationType.NeuronTypeNone
            return
        }
        let a = activation!
        var act = ActivationType.NeuronTypeNone
        switch a {
        case "relu":
            act = ActivationType.NeuronTypeReLU
            
            break
        default:
            break
        }
        _activationType = act
    }
    
    func activationType() -> Int {
    switch _activationType {
    case .NeuronTypeNone:
        return 0
    case .NeuronTypeReLU:
        return 1
    default:
        return 0
    }
    }
}

func offsetForConvolution(padding: PaddingType,
                          sourceWidth: Int,
                          sourceHeight: Int,
                          destinationWidth: Int,
                          destinationHeight: Int,
                          kernelWidth: Int,
                          kernelHeight: Int,
                          strideInPixelsX: Int,
                          strideInPixelsY: Int) -> MPSOffset {
    if padding == .same {
        if kernelWidth == 1 && strideInPixelsX == 2{
            return MPSOffset(x: 0, y: 0, z: 0)
        }
        let padH = (destinationHeight - 1) * strideInPixelsY + kernelHeight - sourceHeight
        let padW = (destinationWidth  - 1) * strideInPixelsX + kernelWidth  - sourceWidth
        return MPSOffset(x: (kernelWidth - padW)/2, y: (kernelHeight - padH)/2, z: 0)
    } else {
        return MPSOffset(x: kernelWidth/2, y: kernelHeight/2, z: 0)
    }
}

@available(iOS 10, *)
public class SimpleCNNLayer: Layer {
    var cnn: MPSCNNKernel!
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        //        mpscnn.destinationFeatureChannelOffset = destinationTensor.destinationChannelOffset
        
        let inputData = sourceData as! ImageData
        let outputData = destinationData as! ImageData
        cnn.encode(commandBuffer: commandBuffer,
                   sourceImage: inputData.image!,
                   destinationImage: outputData.image!)
    }
}

@available(iOS 10, *)
public class Convolution: SimpleCNNLayer {
    let kernel: (Int, Int)
    var outputChannels: Int
    let stride: (Int, Int)
    let activation: String?
    let padding: PaddingType
    var conv: MPSCNNConvolution!
    
    public init(kernel: (Int, Int),
                outputChannels: Int,
                stride: (Int, Int) = (1, 1),
                padding: PaddingType = .same,
                activation: String? = nil,
                name: String = "Convolution", useBias: Bool = true) {
        self.kernel = kernel
        self.outputChannels = outputChannels
        self.stride = stride
        self.padding = padding
        self.activation = activation
        //        self.inputShape = inputShape
        //        self.outputShape = DataShape(width: inputShape.width/stride.0, height: inputShape.height/stride.0, channels: outputChannels)
        super.init(name: name, useBias: useBias)
    }
    
    override public func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        if let inputShape = inputShape {
            self.inputShape = inputShape
            if padding == .same {
                self.outputShape = DataShape(width: Int(ceil(Float(inputShape.width)/Float(stride.0))), height: Int(ceil(Float(inputShape.height)/Float(stride.1))), channels: outputChannels)
                
            } else {
                
                self.outputShape = DataShape(width: (inputShape.width-kernel.0)/stride.0+1, height: (inputShape.height-kernel.0)/stride.0+1, channels: outputChannels)
                
            }
            super.setShape(inputShape: self.outputShape)
        }
        
        
    }
    
    override public func getWeightSize() -> Int {
        return inputShape!.channels * kernel.1 * kernel.0 * outputShape!.channels
    }
    
    override public func getBiasSize() -> Int {
        return useBias ? outputShape!.channels : 0
    }
    
    override public func createNetWork(device: MTLDevice) {
        let act = createActivation(activation: activation, device: device)
        let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernel.0, kernelHeight: kernel.1, inputFeatureChannels: inputShape!.channels, outputFeatureChannels: outputShape!.channels, neuronFilter: act)
        
        desc.strideInPixelsX = stride.0
        desc.strideInPixelsY = stride.1
        conv = MPSCNNConvolution(device: device, convolutionDescriptor: desc, kernelWeights: weight!.weightsPointer, biasTerms: useBias ? weight!.biasPointer : nil, flags: .none)
        conv.edgeMode = .zero
        
        cnn = conv
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        conv.offset = offsetForConvolution(padding: padding,
                                           sourceWidth: inputShape!.width,
                                           sourceHeight: inputShape!.height,
                                           destinationWidth: outputShape!.width,
                                           destinationHeight: outputShape!.height,
                                           kernelWidth: kernel.0,
                                           kernelHeight: kernel.1,
                                           strideInPixelsX: stride.0,
                                           strideInPixelsY: stride.1)
//        print("\(self.name)offset:", conv.offset, (destinationData as! ImageData).image!.texture.width, (destinationData as! ImageData).image!.texture.textureType == MTLTextureType.type2DArray, kernel.0)
        super.encode(commandBuffer: commandBuffer,
                     sourceData: sourceData,
                     destinationData: destinationData)
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
    
}

//public class ShaderLayer: Layer {
//
//}





@available(iOS 11, *)
public class DepthwiseConvolution: SimpleCNNLayer {
    let kernel: (Int, Int)
    let stride: (Int, Int)
    let activation: String?
    var conv: MPSCNNConvolution!
    let padding: PaddingType
    public init(kernel: (Int, Int),
                stride: (Int, Int) = (1, 1),
                activation: String? = nil,
                padding: PaddingType,
                useBias: Bool = true,
                name: String = "DepthwiseConvolution") {
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.activation = activation
        super.init(name: name, useBias: useBias)
    }
    
    override public func getWeightSize() -> Int {
        return inputShape!.channels * kernel.1 * kernel.0
    }
    
    override public func getBiasSize() -> Int {
        return useBias ? outputShape!.channels : 0
    }
    
    override public func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        if let inputShape = inputShape {
            self.inputShape = inputShape
            if padding == .same {
                self.outputShape = DataShape(width: Int(ceil(Float(inputShape.width)/Float(stride.0))), height: Int(ceil(Float(inputShape.height)/Float(stride.1))), channels: inputShape.channels)
                
            } else {
                self.outputShape = DataShape(width: (inputShape.width-kernel.0)/stride.0+1, height: (inputShape.height-kernel.0)/stride.0+1, channels: inputShape.channels)
                
            }
            super.setShape(inputShape: self.outputShape)
        }
    }
    
    
    override public func createNetWork(device: MTLDevice) {
            let act = createActivation(activation: activation, device: device)

            let desc = MPSCNNDepthWiseConvolutionDescriptor(kernelWidth: kernel.0, kernelHeight: kernel.1, inputFeatureChannels: inputShape!.channels, outputFeatureChannels: outputShape!.channels, neuronFilter: act)
            
            desc.strideInPixelsX = stride.0
            desc.strideInPixelsY = stride.1
            
        conv = MPSCNNConvolution(device: device, convolutionDescriptor: desc, kernelWeights: weight!.weightsPointer, biasTerms: useBias ? weight!.biasPointer : nil, flags: .none)
            conv.edgeMode = .zero
            
            cnn = conv
        
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        conv.offset = offsetForConvolution(padding: padding,
                                           sourceWidth: inputShape!.width,
                                           sourceHeight: inputShape!.height,
                                           destinationWidth: outputShape!.width,
                                           destinationHeight: outputShape!.height,
                                           kernelWidth: kernel.0,
                                           kernelHeight: kernel.1,
                                           strideInPixelsX: stride.0,
                                           strideInPixelsY: stride.1)
//        print("\(self.name):offset:\(conv.offset)")
        super.encode(commandBuffer: commandBuffer,
                     sourceData: sourceData,
                     destinationData: destinationData)
    }
}


@available(iOS 10, *)
public class PointwiseConvolution: Convolution {
    /**
     Creates a point-wise convolution layer, which is really the same as a
     convolutional layer with a 1x1 kernel.
     */
    public init(channels: Int,
                stride: (Int, Int) = (1, 1),
                activation: String?,
                useBias: Bool = true,
                name: String = "PointwiseConvolution") {
        super.init(kernel: (1, 1), outputChannels: channels, stride: stride, padding: .same, activation: activation, name: name, useBias: useBias)
    }

}

@available(iOS 10, *)
public class FullConnectedLayer: SimpleCNNLayer {
    let channels: Int
    let activation: String?
    
    public init(channels: Int, activation: String? = nil, name: String = "FullConnectedLayer") {
        self.channels = channels
        self.activation = activation
        super.init(name: name)
    }
    
    override public func setShape(inputShape: DataShape?, outputShape: DataShape?) {
        self.inputShape = inputShape
        self.outputShape = DataShape(width: 1, height: 1, channels: channels)
        super.setShape(inputShape: self.outputShape)
    }
    
    public override func getWeightSize() -> Int {
        return inputShape!.width * inputShape!.height * inputShape!.channels * channels
    }
    
    public override func getBiasSize() -> Int {
        return useBias ? channels : 0
    }
    
    override public func createNetWork(device: MTLDevice) {
        let desc = MPSCNNConvolutionDescriptor(kernelWidth: inputShape!.width,
                                               kernelHeight: inputShape!.height,
                                               inputFeatureChannels: inputShape!.channels,
                                               outputFeatureChannels: channels,
                                               neuronFilter: createActivation(activation: activation, device: device))
        
        cnn = MPSCNNFullyConnected(device: device,
                                          convolutionDescriptor: desc,
                                          kernelWeights: weight!.weightsPointer,
                                          biasTerms: weight!.biasPointer,
                                          flags: .none)
    }
}


@available(iOS 11, *)
public class SeparableConv: Layer {
    let depthwise: DepthwiseConvolution
    let pointwise: Convolution
    
    let device: MTLDevice
    public init(device: MTLDevice, kernel: (Int, Int),
                stride: (Int, Int) = (1, 1),
                padding: PaddingType = .same, outChannels: Int, activation: String? = nil, name: String, useBias: Bool) {
        self.depthwise = DepthwiseConvolution(kernel: kernel, stride: stride, activation: activation!, padding:padding, useBias: useBias, name: name)
        self.pointwise = Convolution(kernel: (1, 1), outputChannels: outChannels, stride: (1, 1), padding: padding, activation: activation, name: name, useBias: useBias)
        self.device = device
        super.init(name: name, useBias: useBias)
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
        return self.pointwise.getOutputData(commandBuffer: commandBuffer, device: device)
}
}
