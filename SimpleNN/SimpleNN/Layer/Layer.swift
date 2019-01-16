//
//  Layer.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/12.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import Metal
import MetalPerformanceShaders

public enum PaddingType {
    case same    // add zero padding
    case valid   // don't add padding
}

public class Layer {
    internal(set) public var name: String
    
    internal(set) public var useBias: Bool
    
    internal(set) public var weight: Weights?
    
    internal(set) public var inputShape: DataShape?
    internal(set) public var outputShape: DataShape?
    
    internal(set) public var outputData: DataWrapper?
    private var _imageDesc: Any?
    
    @available(iOS 10, *)
    var imageDesc: MPSImageDescriptor? {
        get {
            return _imageDesc as? MPSImageDescriptor
        }
        set {
            _imageDesc = newValue
        }
    }
    
    internal(set) public var isTemporary: Bool = false
    internal(set) public var forceNotTemp: Bool = false
    internal(set) public var tmpImageCount: Int = 0

    internal(set) public var childrenLayers: [Layer] = []
    internal(set) public var fatherLayer: Layer?
    internal(set) public var index: Int = 0

//    internal(set) public var holdOutputLayers: [Layer] = []
//    internal(set) public var holdingData: [DataWrapper] = []
    
    public init(name: String="", useBias: Bool = true) {
        self.name = name
        self.useBias = useBias
//        self.childrenLayers = childrenLayers
//        self.fatherLayers = fatherLayers
    }
    
    open func addChild(layer: Layer) {
        self.childrenLayers.append(layer)
        layer.fatherLayer = self
        if self.childrenLayers.count == 0 {
            
        } else {
            for i in 0..<self.childrenLayers.count {
                self.childrenLayers[i].index = i
            }
            
        }
        self.tmpImageCount += 1
        
    }

    open func getEncodeSquence() -> [Layer] {
        var layers: [Layer] = [self]
        for l in childrenLayers {
            layers += l.getEncodeSquence()
        }
        return layers
    }
    
    open func findMultiSonParentNode() -> Layer? {
        var father = fatherLayer
        var current = self
        while father != nil {
            if father!.childrenLayers.count > 1 {
                return father
            } else {
                current = father!
                father = father!.fatherLayer
            }
        }
        return nil
    }
    
    open func printLayer() {
        let classNames = String(describing: self).split(separator: ".")
        let className = classNames[classNames.count - 1]
        var summary = "\(self.name)(\(className))"
        if summary.count < 40 {
            for _ in 0..<(40-summary.count) {summary += " "}
        }
        let shape = "\(getShapeString(shape: self.outputShape))"
        summary += shape
        if shape.count < 16 {
            for _ in 0..<(16-shape.count) {summary += " "}
        }
        let param = "\(self.getWeightSize()+self.getBiasSize())"
        summary += param
        if param.count < 10 {
            for _ in 0..<(10-param.count) {summary += " "}
        }
        summary += "\(self.fatherLayer == nil ? "" : self.fatherLayer!.name)"
        print(summary)
//        print("\(self.name)(\(className)) \t\t\(getShapeString(shape: self.outputShape)) \t\(self.getWeightSize()+self.getBiasSize())")
        for layer in childrenLayers {
            layer.printLayer()
        }
    }
    
    open func getShapeString(shape: DataShape?) -> String{
        if shape == nil {
            return "(nil, nil, nil)"
        } else {
            return "(\(shape!.width),\(shape!.height),\(shape!.channels))"
        }
    }
    
    open func addToChild(layer: Layer, inBoundName: String) -> Bool {
        var result = false
        if self.name == inBoundName {
            addChild(layer: layer)
            return true
        } else {
            for l in childrenLayers {
                result = l.addToChild(layer: layer, inBoundName: inBoundName)
                if result {
                    return true
                }
            }
        }
        return false
    }
    
    open func childrenCount() -> Int {
        return self.childrenLayers.count
    }
    
    open func createNetWork(device: MTLDevice) {
        
    }
    
    open func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper,  destinationData: DataWrapper) {
        
    }
    
    open func predict(device: MTLDevice, commandBuffer: MTLCommandBuffer, sourceData: DataWrapper,  destinationData: DataWrapper) -> DataWrapper {
        var outputData = destinationData
        encode(commandBuffer: commandBuffer, sourceData: sourceData, destinationData: destinationData)
        for layer in childrenLayers {
            outputData = layer.predict(device: device, commandBuffer: commandBuffer, sourceData: destinationData,
                          destinationData: layer.getOutputData(commandBuffer: commandBuffer, device: device))
        }
        return outputData
    }
    
    public func setActivation(activation: String) {
    }
    
    open func cleanData() {
        if let outputData = outputData {
            if #available(iOS 10, *) {
                if let input = outputData as? ImageData {
                    if let image = input.image as? MPSTemporaryImage {
                        image.readCount -= 1
                    }
                }
            }
        }
    }
    
    
    open func setImageDesc(isTemporary: Bool = false) {
        guard let outputShape = self.outputShape else {
            fatalError("outputShape can not be nil")
        }
        
        if #available(iOS 10, *) {
            self.isTemporary = isTemporary
            imageDesc = MPSImageDescriptor(channelFormat: .float16, width: outputShape.width, height: outputShape.height, featureChannels: outputShape.channels)
            imageDesc!.storageMode = .private
            imageDesc!.usage = [.shaderRead, .shaderWrite]
        }
        if self.forceNotTemp {
            self.isTemporary = false
        }
    }
    
    open func setShape(inputShape: DataShape? = nil, outputShape: DataShape? = nil) {
        for layer in childrenLayers {
            layer.setShape(inputShape: inputShape, outputShape: outputShape)
        }
        
//        setImageDesc(isTemporary: true)
    }
    
    open func getWeightSize() -> Int {
        return 0
    }
    
    open func getBiasSize() -> Int {
        return 0
    }
    
    open func getAllWeightSize() -> Int {
        var size = getWeightSize() + getBiasSize()
        for layer in childrenLayers {
            size += layer.getAllWeightSize()
        }
        return size
    }
    
    open func getOutputData(commandBuffer: MTLCommandBuffer, device: MTLDevice) -> DataWrapper {
        if #available(iOS 10, *) {
            let output = ImageData(imageDesc: _imageDesc as! MPSImageDescriptor, commandBuffer: commandBuffer, device: device, isTemporary: isTemporary)
            if isTemporary {
                if let temp = output.image as? MPSTemporaryImage {
                    temp.readCount = tmpImageCount
//                    print("\(self.name)tmp count:\(tmpImageCount), childs:\(self.childrenLayers.count)")
                }
            }
//            print("\(self.name) ,\(isTemporary) ")
            self.outputData = output
            return self.outputData!
        } else {
            return DataWrapper()
        }
    }
    
    open func getOutputData() -> DataWrapper? {
        return outputData
    }
    
    open func insertWeight(pointer: UnsafePointer<Float>) -> UnsafePointer<Float> {
        var p = pointer
        let weight = p
        p += self.getWeightSize()
        print(name, "weightsize:", getWeightSize(), weight[0], weight[1], weight[2])
        let bias = p
        p += self.getBiasSize()
        print(name, "biassize:", getBiasSize(), bias[0])
        self.weight = Weights(weightsPointer: weight, biasPointer: bias)
        return p
    }
}


func LayerFactory(layer: NSDictionary, device: MTLDevice, layers: LayerTree) -> [Layer] {
    var l = [Layer]()
    if let config = layer["config"] as? NSMutableDictionary {
        let name: String? = config["name"] as? String

        if let batch_input_shape = config["batch_input_shape"] as? NSArray {
            var width = 256
            if let w = batch_input_shape[1] as? Int {
                width = w
            }
            var height = 256
            if let h = batch_input_shape[2] as? Int {
                height = h
            }
            var channels = 3
            if let c = batch_input_shape[3] as? Int {
                channels = c
            }
            if #available(iOS 10, *) {
//                l.append(ShaderImageInput(device: device, imageWidth: width, imageHeight: height, channels: channels, name: name == nil ? "ShaderImageInput" : name!))
                l.append(ImageInput(imageWidth: width, imageHeight: height, channels: channels, name: name == nil ? "ImageInput" : name!))
            } else {
                l.append(ShaderImageInput(device: device, imageWidth: width, imageHeight: height, channels: channels, name: name == nil ? "ShaderImageInput" : name!))
            }
        }
        let class_name = layer["class_name"] as! String
        var channels = 0
        if let filters = config["filters"] as? Int {
            channels = filters
        }
        let kernel_size = config["kernel_size"] as? NSArray
        let strides = config["strides"] as? NSArray
        var activation = config["activation"] as? String
        let use_bias = config["use_bias"] as? Bool
        var paddingType = PaddingType.same
        if let pt = config["padding"] as? String {
            if pt == "valid" {
                paddingType = PaddingType.valid
            }
        }
        switch class_name {
        case "Conv2D":
            if #available(iOS 10, *) {
//                l.append(ShaderConvolution(device: device, kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), outputChannels: channels, stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, activation: activation, name: name == nil ? "ShaderConvolution" : name!, useBias: use_bias!))
                l.append(Convolution(kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), outputChannels: channels, stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, activation: activation, name: name == nil ? "Convolution": name!, useBias: use_bias!))
            } else {
                l.append(ShaderConvolution(device: device, kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), outputChannels: channels, stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, activation: activation, name: name == nil ? "ShaderConvolution" : name!, useBias: use_bias!))
            }
            break
        case "DepthwiseConv2D":
            if #available(iOS 11, *) {
                l.append(ShaderDepthwiseConvolution(device:device, kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), padding:paddingType, activation: activation!,  name: name == nil ? "ShaderDepthwiseConvolution": name!,useBias: use_bias!))
//                l.append(DepthwiseConvolution(kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), activation: activation!, padding: paddingType, useBias: use_bias!, name: name == nil ? "DepthwiseConvolution": name!))
            } else {
                l.append(ShaderDepthwiseConvolution(device:device, kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), padding:paddingType, activation: activation!,  name: name == nil ? "ShaderDepthwiseConvolution": name!,useBias: use_bias!))
            }
            break
        case "AveragePooling2D":
            let pool_size = config["pool_size"] as? NSArray
            if #available(iOS 10, *) {
                l.append(AveragePooling(poolSize: (pool_size![0] as! Int, pool_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, name: name!))
            }
            break
        case "GlobalAveragePooling2D":
            if #available(iOS 10, *) {
                l.append(GlobalAveragePooling(name: name == nil ? "GlobalAveragePooling": name!))
            } else {
                l.append(ShaderGlobalAveragePooling(device:device, name: name == nil ? "ShaderGlobalAveragePooling": name!))
            }
            break
        case "Dense":
            channels = config["units"] as! Int
            activation = nil
            if #available(iOS 10, *) {
                l.append(FullConnectedLayer(channels: channels, activation: activation, name: name == nil ? "FullConnectedLayer": name!))
            } else {
                l.append(ShaderFullyConnectedConvolution(device:device, outputChannels: channels, activation: activation, name: name == nil ? "ShaderFullyConnectedConvolution": name!, useBias: use_bias!))
            }
            break
        case "ZeroPadding2D":
            if let pad = config["padding"] as? NSArray {
                l.append(ShaderZeroPadding(device: device, padding: [[(pad[0] as! NSArray)[0] as! Int, (pad[0] as! NSArray)[1] as! Int], [(pad[1] as! NSArray)[0] as! Int, (pad[1] as! NSArray)[1] as! Int]], name: name == nil ? "ShaderZeroPadding": name!))
            } else {
                let pad = config["padding"] as! Int
                l.append(ShaderZeroPadding(device: device, padding: [[pad, pad], [pad, pad]], name: name == nil ? "ShaderZeroPadding": name!))
            }
        case "BatchNormalization":
            if let epsilon = config["epsilon"] as? Double {
                l.append(ShaderBatchNormalization(device: device, eplison: Float(epsilon), name: name == nil ? "ShaderBatchNormalization": name!))
            }
            break
        case "UpSampling2D":
            
            let size = config["size"] as! NSArray
            let w = size[0] as! Int
            let h = size[1] as! Int
            if #available(iOS 10, *) {
                l.append(ShaderUpSampling(device: device, factorW: w, factorH: h, name: name == nil ? "ShaderUpSampling": name!))
            } else {
                l.append(ShaderUpSampling(device: device, factorW: w, factorH: h, name: name == nil ? "ShaderUpSampling": name!))
            }
            break
        case "Concatenate":
            if #available(iOS 10, *) {
                l.append(ShaderConcatenate(device: device, name: name == nil ? "ShaderConcatenate": name!))
            } else {
                l.append(ShaderConcatenate(device: device, name: name == nil ? "ShaderConcatenate": name!))
            }
            break
        case "ReLU":
            let maxValue = config["max_value"] as! Float
            let negativeSlope = config["negative_slope"] as! Float
            let threshold = config["threshold"] as! Float

            l.append(ShaderRelu(device: device, maxValue: maxValue, negativeSlope: negativeSlope, threshold: threshold, name: name == nil ? "ShaderRelu": name!))
            break
        case "LeakyReLU":
            let negativeSlope = config["alpha"] as! Float
            
            l.append(ShaderLeakyRelu(device: device, negativeSlope: negativeSlope, name: name == nil ? "LeakyRelu": name!))
            break
        case "Activation":
            if let activation = config["activation"] as? String {
                if activation == "relu" {
                    if #available(iOS 10, *) {
                        l.append(ShaderRelu(device: device, maxValue: 1000, negativeSlope: 0, threshold: 0, name: name == nil ? "ShaderRelu": name!))
                        
                    } else {
                        l.append(ShaderRelu(device: device, maxValue: 1000, negativeSlope: 0, threshold: 0, name: name == nil ? "ShaderRelu": name!))

                    }
                }
            }
            break
        case "SeparableConv2D":
            if #available(iOS 11, *) {
//                l.append(SeparableConv(device: device, kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, outChannels: channels, activation: activation!, name: name!, useBias: use_bias!))
                l.append(ShaderSeparableConv(device: device, kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, outChannels: channels, activation: activation!, name: name!, useBias: use_bias!))
            } else {
                l.append(ShaderSeparableConv(device: device, kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, outChannels: channels, activation: activation!, name: name!, useBias: use_bias!))
//                l.append(ShaderSeparableConv(device: device, kernel: (kernel_size![0] as! Int, kernel_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, outChannels: channels, activation: activation!, name: name!, useBias: use_bias!))
                
            }
            break
        case "MaxPooling2D":
            let pool_size = config["pool_size"] as? NSArray
            if #available(iOS 10, *) {
                l.append(MaxPooling(poolSize: (pool_size![0] as! Int, pool_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, name: name!))
            } else {
                l.append(ShaderMaxPooling(device: device, poolSize: (pool_size![0] as! Int, pool_size![1] as! Int), stride: (strides![0] as! Int, strides![1] as! Int), padding: paddingType, name: name!))
            }
        case "Add":
            l.append(ShaderAdd(device: device, name: name == nil ? "ShaderAdd" : name!))
            break
        default:
//            fatalError("Unrecognize network type:\(class_name)")
            break
        }
        if let activation = config["activation"] as? String {
            if activation == "softmax" {
                if #available(iOS 10, *) {
                    l.append(Softmax(name: "Softmax"))
                } else {
                    l.append(ShaderSoftmax(device:device, name: "Softmax"))
                }
            }
        }
    } else {
        fatalError("networkFile error")
    }
    return l
}
