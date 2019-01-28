//
//  Data.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/13.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders
import MetalKit

public class DataWrapper {
    public var texture: MTLTexture?

    public init(texture: MTLTexture) {
        self.texture = texture
    }
    
    public init(textDesc: MTLTextureDescriptor, device: MTLDevice) {
        texture = device.makeTexture(descriptor: textDesc)
    }
    
    public init() {
        
    }
    
    public init(imageFileName: String, device: MTLDevice, fileExtension: String = "png") {
        let textureLoader = MTKTextureLoader(device: device)
        let url = Bundle.main.url(forResource: imageFileName, withExtension: fileExtension)
        let texture = try! textureLoader.newTexture(URL: url!, options: [
            MTKTextureLoader.Option.SRGB : NSNumber(value: false)
            ])
        self.texture = texture
    }
    
    public func getTexture() -> MTLTexture?{
        return texture;
    }
}


@available(iOS 10.0, *)
public class ImageData: DataWrapper {
    public var image: MPSImage?
    public init(image: MPSImage) {
        super.init()
        self.image = image
    }
    
    override public init(imageFileName: String, device: MTLDevice, fileExtension: String = "png") {
        let textureLoader = MTKTextureLoader(device: device)
        let url = Bundle.main.url(forResource: imageFileName, withExtension: fileExtension)
        let texture = try! textureLoader.newTexture(URL: url!, options: [
            MTKTextureLoader.Option.SRGB : NSNumber(value: false)
            ])
        super.init()
        image = MPSImage(texture: texture, featureChannels: 3)
    }
    
    public override func getTexture() -> MTLTexture? {
        return image!.texture
    }
    
//    public init(textDesc: MTLTextureDescriptor, device: MTLDevice) {
//        super.init()
//        texture = device.makeTexture(descriptor: textDesc)
//    }
    
//    public init(width: Int, height: Int, channel: Int, commandBuffer: MTLCommandBuffer) {
//        image = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: imageDesc!)
//    }
    
    public init(imageDesc: MPSImageDescriptor, commandBuffer: MTLCommandBuffer, device: MTLDevice, isTemporary: Bool) {
        super.init()
        if isTemporary {
            image = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: imageDesc)
        } else {
            imageDesc.storageMode = .shared
            image = MPSImage(device: device, imageDescriptor: imageDesc)
        }
    }
}

public struct DataShape {
    public let width: Int
    public let height: Int
    public let channels: Int
    
    public init(width: Int = -1, height: Int = -1, channels: Int = -1) {
        self.width = width
        self.height = height
        self.channels = channels
    }
    
}

public class Output {
    public var result: [Float]
    public init(layer: Layer) {
        let output = layer.getOutputData() as! ImageData
        print("\(layer.name):(\(output.image!.width), \(output.image!.height), \(output.image!.featureChannels))")
        if output.image!.width == 1 && output.image!.height == 1 {
            result = output.image!.toFloatArray()
        } else {
            result = output.image!.toFloatArray()

//            result = [Float]()
        }
    }
    
    public init() {
        result = [Float]()
    }
}
