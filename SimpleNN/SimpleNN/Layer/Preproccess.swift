//
//  Other.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/14.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

public class Input: Layer {
    
}

@available(iOS 10, *)
public class ImageInput: Input {
    let imageWidth: Int
    let imageHeight: Int
    let channels: Int
    var lanczos: MPSImageLanczosScale!
    
    
    public init(imageWidth: Int, imageHeight: Int, channels: Int, name: String) {
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.channels = channels
        //        self.outputShape = self.inputShape
        super.init(name: name)
    }
    
    override public func setShape(inputShape: DataShape? = nil, outputShape: DataShape? = nil) {
        self.inputShape = DataShape(width: imageWidth, height: imageHeight, channels: channels)
        self.outputShape = DataShape(width: imageWidth, height: imageHeight, channels: channels)
        super.setShape(inputShape: self.outputShape)
    }
    
    override public func getOutputData(commandBuffer: MTLCommandBuffer, device: MTLDevice) -> DataWrapper {
        guard self.imageDesc != nil else {
            fatalError("should init imageDesc first")
        }
        
        if self.outputData == nil || isTemporary {
            self.outputData = ImageData(imageDesc: self.imageDesc!, commandBuffer: commandBuffer, device: device, isTemporary: isTemporary)
        }
        return self.outputData!
    }
    
    override public func createNetWork(device: MTLDevice) {
        lanczos = MPSImageLanczosScale(device: device)
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer, sourceData: DataWrapper, destinationData: DataWrapper) {
        let inputData = sourceData as! ImageData
        let outputData = destinationData as! ImageData
        if inputData.image!.width != outputData.image!.width || inputData.image!.height != outputData.image!.height {
//            lanczos.encode(commandBuffer: commandBuffer, sourceImage: inputData.image!, destinationImage: outputData.image!)
            lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputData.getTexture()!, destinationTexture: outputData.getTexture()!)
            
            if let image = inputData.image as? MPSTemporaryImage {
                image.readCount -= 1
            }
            
        } else {
            if let image = outputData.image as? MPSTemporaryImage {
                image.readCount -= 1
            }
            
            outputData.image = inputData.image
        }
        
        //        lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputData.image!.texture, destinationTexture: outputData.image!.texture)
    }
}
