//
//  LayerTree.swift
//  SimpleMobile
//
//  Created by luozhiping on 2019/1/4.
//  Copyright © 2019 SimpleTech. All rights reserved.
//

import Foundation
import MetalKit
import MetalPerformanceShaders
public class LayerTree {
    var headLayer: Layer?
    var tailLayer: Layer?
    var layers: [Layer] = []
    public init() {
    }
    
    public func addLayer(layer: Layer, inBounds: [String]) {
        layers.append(layer)
        if inBounds.count == 0 {
            if #available(iOS 10, *) {
                if layer is ImageInput || layer is ShaderImageInput {
                    headLayer = layer
                } else {
                    tailLayer!.addChild(layer: layer)
                }
            } else {
                if layer is ShaderImageInput {
                    headLayer = layer
                } else {
                    tailLayer!.addChild(layer: layer)
                }
            }
            tailLayer = layer
        } else if inBounds.count == 1 {
            let inBoundName = inBounds[0]
            if let tailLayer = self.tailLayer {
                if tailLayer.name == inBoundName {
                    tailLayer.addChild(layer: layer)
                    self.tailLayer = layer
                    return
                }
            }
            let result = headLayer!.addToChild(layer: layer, inBoundName: inBoundName)
            if !result {
                tailLayer!.addChild(layer: layer)
                tailLayer = layer
//                fatalError("can not find father layer:\(inBoundName)")
            }
        } else {
            //for Add layer
            let lls = headLayer!.getEncodeSquence()
            var bounds: [Layer] = []
            for (_, l) in lls.enumerated() {
                for boundName in inBounds {
                    if boundName == l.name {
                        bounds.append(l)
                    }
                }
            }
            if bounds.count > 0 {
                bounds[bounds.count - 1].addChild(layer: layer)
                tailLayer = layer
            }
            if bounds.count == 2{
                if layer is ShaderAdd {
                    (layer as! ShaderAdd).setLayer(layer1: bounds[0], layer2: bounds[1])
                    bounds[0].forceNotTemp = true
//                    bounds[1].forceNotTemp = true
                } else if layer is ShaderConcatenate {
                    let l1 = bounds[0].name == inBounds[0] ? bounds[0] : bounds[1]
                    let l2 = bounds[1].name == inBounds[1] ? bounds[1] : bounds[0]

                    (layer as! ShaderConcatenate).setLayer(layer1: l1, layer2: l2)
                }
            }
        }
    }
    
    public func layerCount() -> Int {
        return 0
    }
    
    public func printTree() {
        print("----------------------------------------------------------------------")
        print("Layer(type) \t\t\t\t\t\t\t OutputShape \t Param \t Connected to")
        print("======================================================================")
        headLayer!.printLayer()
    }
    
    public func insertWeight(pointer: UnsafePointer<Float>) {
        var p = pointer
//        var offset = 0
        for layer in layers {
            p = layer.insertWeight(pointer: p)
        }
    }
    
    public func initShape() {
        headLayer!.setShape()
        let l = headLayer!.getEncodeSquence()
        for (index, layer) in l.enumerated() {
//            layer.setImageDesc(isTemporary: layer.childrenLayers.count > 0)
            layer.setImageDesc(isTemporary: index == l.count - 1 ? false : true)
        }
    }
    
    public func createNetwork(device: MTLDevice) {
        for layer in layers {
            layer.createNetWork(device: device)
        }
    }
    
    public func getWeightSize() -> Int {
        return headLayer!.getAllWeightSize()
    }
    
    public func predict(input: DataWrapper, device: MTLDevice) -> [Float] {
//        print("start predict......")
        let commandQueue = device.makeCommandQueue()
        var output: DataWrapper?

        autoreleasepool {
            let commandBuffer = commandQueue!.makeCommandBuffer()

            if #available(iOS 10, *) {
                var imageDescList: [MPSImageDescriptor] = []
                
                for index in 0..<layers.count {
                    let layer = layers[index]
                    if layer.isTemporary {
                        imageDescList.append(layer.imageDesc!)
                    }
                }
                MPSTemporaryImage.prefetchStorage(with: commandBuffer!,
                                                  imageDescriptorList: imageDescList)
                
            }
            output = headLayer!.predict(device: device, commandBuffer: commandBuffer!, sourceData: input, destinationData: headLayer!.getOutputData(commandBuffer: commandBuffer!, device: device))
//            output = layers[layers.count-1].getOutputData()
//            print("get layer output:\(layers[layers.count-1].name)")
            commandBuffer!.commit()
            commandBuffer!.waitUntilCompleted()
        }
//
//
        let outputImage = output as! ImageData
        let results = outputImage.image!.toFloatArray()
        return results
    }
}
