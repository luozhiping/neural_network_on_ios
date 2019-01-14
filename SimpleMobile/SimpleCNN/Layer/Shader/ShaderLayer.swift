//
//  ShaderLayer.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/24.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders
import MetalKit
open class ShaderLayer: Layer {
    internal(set) public var textureDesc: MTLTextureDescriptor?
    internal(set) public var device: MTLDevice

    internal(set) public var pipeline: MTLComputePipelineState?
    internal(set) public var weightsBuffer: MTLBuffer?
    internal(set) public var biasBuffer: MTLBuffer?
    
    public init(device: MTLDevice, name: String = "", useBias: Bool = true) {
        self.device = device
        super.init(name: name, useBias: useBias)
    }
    
    open override func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        if #available(iOS 10, *) {
            if let input = sourceData as? ImageData {
                if let image = input.image as? MPSTemporaryImage {
                    image.readCount -= 1
//                    print("clean \(self.name)")
                }
            }
        }
    }
    
    
    override open func setImageDesc(isTemporary: Bool) {
        if #available(iOS 10, *) {
            super.setImageDesc(isTemporary: isTemporary)
        } else {
            if let outputShape = self.outputShape {
                textureDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: outputShape.width, height: outputShape.height, mipmapped: false)
                textureDesc!.usage = [.shaderWrite, .shaderRead]
                if outputShape.channels > 4 {
                    textureDesc!.textureType = MTLTextureType.type2DArray
                    textureDesc!.arrayLength = (outputShape.channels + 3)/4
                }
            }
        }
        
    }
    
    override open func getOutputData(commandBuffer: MTLCommandBuffer, device: MTLDevice) -> DataWrapper {
        if #available(iOS 10, *) {
//            if self.outputData == nil {
//                self.outputData = DataWrapper(textDesc: textureDesc!, device: device)
//            }
//            return self.outputData!
            return super.getOutputData(commandBuffer: commandBuffer, device: device)
        } else {
            if self.outputData == nil {
                self.outputData = DataWrapper(textDesc: textureDesc!, device: device)
            }
            return self.outputData!
        }
    }
}
