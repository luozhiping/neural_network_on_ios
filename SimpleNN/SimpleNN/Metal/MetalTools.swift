//
//  MetalTools.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/18.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import Metal
import MetalPerformanceShaders

var metalLib: MTLLibrary!

//enum MetalLibrary {
//    case ConvLib
//}

//func loadMetalLibrary(device: MTLDevice, lib: MetalLibrary) -> MTLLibrary {
//
//}

public func initFunction(device: MTLDevice, name: String) -> MTLComputePipelineState {
    if metalLib == nil {
        guard let path = Bundle(for: Layer.self).path(forResource: "default",
                                                      ofType: "metallib") else {
                                                        fatalError("Could not find Metal library")
        }
        do {
            metalLib = try device.makeLibrary(filepath: path)
        } catch {
            fatalError("Could not load Metal library")
        }
    }
    do {
        guard let kernelFunction = metalLib.makeFunction(name: name) else {
            fatalError("Could not load compute function '\(name)'")
        }
        return try metalLib.device.makeComputePipelineState(function: kernelFunction)
    } catch {
        fatalError("Could not create compute pipeline for function '\(name)'. Error: \(error)")
    }
}


/**
 Copies user-supplied weights into an MTLBuffer.
 
 The weights must be arranged in memory as:
 
 [outputChannels][kernelHeight][kernelWidth][inputChannels]
 
 Because we're working with textures, the number of input and output channels
 must always be a multiple of 4 in the MTLBuffer.
 
 For example, the first row of a 3x3 kernel with 4 or fewer channels looks
 like this in memory: 0,1,2,3 | 0,1,2,3 | 0,1,2,3. And for 4-8 channels it
 looks like this: 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7.
 
 But when inputFeatureChannels is not a multiple of 4, the weights supplied by
 the user may look like: 0,1,2 | 0,1,2 | 0,1,2 and so on. In that case we need
 to insert zero-bytes for the missing channels when copying the weights to the
 MTLBuffer. That's why you need to use this function and not just memcpy().
 */
@available(iOS 10, *)
public func copy(weights: UnsafePointer<Float>,
                 to buffer: MTLBuffer,
                 channelFormat: MPSImageFeatureChannelFormat,
                 kernelWidth: Int,
                 kernelHeight: Int,
                 inputFeatureChannels: Int,
                 outputFeatureChannels: Int) {
    
    assert(channelFormat == .float16)
    
    let inputSlices = (inputFeatureChannels + 3) / 4
    let paddedInputChannels = inputSlices * 4
    
    // Calculate how many elements we should copy. Note that the number of output
    // channels isn't necessarily a multiple of 4 in the given weights, so we may
    // have to copy over fewer bytes than fit in the MTLBuffer (the remainder of
    // the buffer will be all zeros in that case).
    let count = outputFeatureChannels * kernelHeight * kernelWidth * paddedInputChannels
    
    assert(buffer.length / MemoryLayout<Float16>.stride >= count)
    
    // If the number of input channels is a multiple of 4, we can do a straight
    // copy from the given weights into the MTLBuffer.
    if paddedInputChannels == inputFeatureChannels {
        let ptr = UnsafeMutablePointer(mutating: weights)
        float32to16(input: ptr, output: buffer.contents(), count: count)
        
        // Otherwise, copy "inputFeatureChannels" weights at a time and add 0-bytes
        // in between to pad the length to a multiple of 4.
    } else {
        var srcPtr = UnsafeMutablePointer(mutating: weights)
        var dstPtr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        
        for _ in 0..<outputFeatureChannels * kernelHeight * kernelWidth {
            float32to16(input: srcPtr, output: dstPtr, count: inputFeatureChannels)
//            print(srcPtr[0], dstPtr[0])
            srcPtr += inputFeatureChannels
            dstPtr += paddedInputChannels
            
        }
    }
}

public func copy(weights: UnsafePointer<Float>,
                 to buffer: MTLBuffer,
                 kernelWidth: Int,
                 kernelHeight: Int,
                 inputFeatureChannels: Int,
                 outputFeatureChannels: Int) {
    
    
    let inputSlices = (inputFeatureChannels + 3) / 4
    let paddedInputChannels = inputSlices * 4
    
    // Calculate how many elements we should copy. Note that the number of output
    // channels isn't necessarily a multiple of 4 in the given weights, so we may
    // have to copy over fewer bytes than fit in the MTLBuffer (the remainder of
    // the buffer will be all zeros in that case).
    let count = outputFeatureChannels * kernelHeight * kernelWidth * paddedInputChannels
    
    assert(buffer.length / MemoryLayout<Float16>.stride >= count)
    
    // If the number of input channels is a multiple of 4, we can do a straight
    // copy from the given weights into the MTLBuffer.
    if paddedInputChannels == inputFeatureChannels {
        let ptr = UnsafeMutablePointer(mutating: weights)
        float32to16(input: ptr, output: buffer.contents(), count: count)
        
        // Otherwise, copy "inputFeatureChannels" weights at a time and add 0-bytes
        // in between to pad the length to a multiple of 4.
    } else {
        var srcPtr = UnsafeMutablePointer(mutating: weights)
        var dstPtr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
        
        for _ in 0..<outputFeatureChannels * kernelHeight * kernelWidth {
            float32to16(input: srcPtr, output: dstPtr, count: inputFeatureChannels)
            //            print(srcPtr[0], dstPtr[0])
            srcPtr += inputFeatureChannels
            dstPtr += paddedInputChannels
            
        }
    }
}

/**
 Creates an MTLBuffer to hold weights.
 
 The weights are assumed to be arranged in memory like so:
 
 [outputChannels][kernelHeight][kernelWidth][inputChannels]
 
 The number of output channels and input channels will be padded to become a
 multiple of 4. For example, a 3x3 kernel with 2 input channels and 5 output
 channels actually gets a buffer containing (5+3)*3*3*(2+2) = 288 elements,
 even though you only specified 5*3*3*2 = 90 weights.
 */
@available(iOS 10, *)
public func makeBuffer(device: MTLDevice,
                       channelFormat: MPSImageFeatureChannelFormat,
                       kernelWidth: Int,
                       kernelHeight: Int,
                       inputFeatureChannels: Int,
                       outputFeatureChannels: Int,
                       weights: UnsafePointer<Float>) -> MTLBuffer {
    
    assert(channelFormat == .float16)
    
    let inputSlices = (inputFeatureChannels + 3) / 4
    let paddedInputChannels = inputSlices * 4
    let outputSlices = (outputFeatureChannels + 3) / 4
    let paddedOutputChannels = outputSlices * 4
    let count = paddedOutputChannels * kernelHeight * kernelWidth * paddedInputChannels
    
    let buffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)!
    
    copy(weights: weights, to: buffer, channelFormat: channelFormat,
         kernelWidth: kernelWidth, kernelHeight: kernelHeight,
         inputFeatureChannels: inputFeatureChannels,
         outputFeatureChannels: outputFeatureChannels)
    
    // For debugging:
    //let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
    //print(float16to32(ptr, count: count))
    
    return buffer
}

public func makeBuffer(device: MTLDevice,
                       kernelWidth: Int,
                       kernelHeight: Int,
                       inputFeatureChannels: Int,
                       outputFeatureChannels: Int,
                       weights: UnsafePointer<Float>) -> MTLBuffer {
    
    
    let inputSlices = (inputFeatureChannels + 3) / 4
    let paddedInputChannels = inputSlices * 4
    let outputSlices = (outputFeatureChannels + 3) / 4
    let paddedOutputChannels = outputSlices * 4
    let count = paddedOutputChannels * kernelHeight * kernelWidth * paddedInputChannels
    
    let buffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)!
    
    copy(weights: weights, to: buffer,
         kernelWidth: kernelWidth, kernelHeight: kernelHeight,
         inputFeatureChannels: inputFeatureChannels,
         outputFeatureChannels: outputFeatureChannels)
    
    // For debugging:
    //let ptr = buffer.contents().bindMemory(to: Float16.self, capacity: count)
    //print(float16to32(ptr, count: count))
    
    return buffer
}

/**
 Copies user-supplied bias values into an MTLBuffer.
 
 There should be one bias value for each output channel.
 */
public func copy(biasTerms: UnsafePointer<Float>,
                 to buffer: MTLBuffer,
                 channelFormat: MPSImageFeatureChannelFormat,
                 outputFeatureChannels: Int) {
    
    if #available(iOS 10.0, *) {
        assert(channelFormat == .float16)
    }
    
    let count = outputFeatureChannels
    assert(buffer.length / MemoryLayout<Float16>.stride >= count)
    
    let ptr = UnsafeMutablePointer(mutating: biasTerms)
    float32to16(input: ptr, output: buffer.contents(), count: count)
}

public func copy(biasTerms: UnsafePointer<Float>,
                 to buffer: MTLBuffer,
                 outputFeatureChannels: Int) {
    
    let count = outputFeatureChannels
    assert(buffer.length / MemoryLayout<Float16>.stride >= count)
    
    let ptr = UnsafeMutablePointer(mutating: biasTerms)
    float32to16(input: ptr, output: buffer.contents(), count: count)
}

/**
 Creates an MTLBuffer to hold bias values.
 
 The number of output channels will be padded to become a multiple of 4.
 */
public func makeBuffer(device: MTLDevice,
                       channelFormat: MPSImageFeatureChannelFormat,
                       outputFeatureChannels: Int,
                       biasTerms: UnsafePointer<Float>?) -> MTLBuffer {
    if #available(iOS 10.0, *) {
    assert(channelFormat == .float16)
    }
    let outputSlices = (outputFeatureChannels + 3) / 4
    let count = outputSlices * 4
    let buffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)!
    
    // The bias terms are often optional. If biasTerms is nil, just allocate a
    // buffer containing zeros, otherwise copy the biases into the buffer.
    if let biasTerms = biasTerms {
        copy(biasTerms: biasTerms, to: buffer, channelFormat: channelFormat,
             outputFeatureChannels: outputFeatureChannels)
    }
    
    return buffer
}

public func makeBuffer(device: MTLDevice,
                       outputFeatureChannels: Int,
                       biasTerms: UnsafePointer<Float>?) -> MTLBuffer {
    let outputSlices = (outputFeatureChannels + 3) / 4
    let count = outputSlices * 4
    let buffer = device.makeBuffer(length: MemoryLayout<Float16>.stride * count)!
    
    // The bias terms are often optional. If biasTerms is nil, just allocate a
    // buffer containing zeros, otherwise copy the biases into the buffer.
    if let biasTerms = biasTerms {
        copy(biasTerms: biasTerms, to: buffer,
             outputFeatureChannels: outputFeatureChannels)
    }
    
    return buffer
}

extension MTLComputeCommandEncoder {
    /**
     Sets the parameters of the command encoder with less boilerplate.
     
     Example:
     
     encoder.configure(parameters: [someBuffer, someTexture, UInt32(someValue)])
     
     MTLBuffer objects are passed to the shader with `setBuffer()`, MTLTexture
     objects with `setTexture()`, and anything else with `setBytes()`.
     
     - Parameters:
     - parameters: an array that may contain MTLBuffer or MTLTexture objects,
     or types such as UInt32 or structs
     */
    public func configure(parameters: [Any]) {
        for i in 0..<parameters.count {
            var obj = parameters[i]
            if let buffer = obj as? MTLBuffer {
                setBuffer(buffer, offset: 0, index: i)
            } else if let texture = obj as? MTLTexture {
                setTexture(texture, index: i)
            } else {
                setBytes(&obj, length: MemoryLayout.size(ofValue: obj), index: i)
            }
        }
    }
}

extension MTLComputeCommandEncoder {
    /**
     Dispatches a compute kernel on a 1-dimensional grid.
     
     - Parameters:
     - count: the number of elements to process
     */
    public func dispatch(pipeline: MTLComputePipelineState, count: Int) {
        // Round off count to the nearest multiple of threadExecutionWidth.
        let width = pipeline.threadExecutionWidth
        let rounded = ((count + width - 1) / width) * width
        
        let blockSize = min(rounded, pipeline.maxTotalThreadsPerThreadgroup)
        let numBlocks = (count + blockSize - 1) / blockSize
        
        let threadGroupSize = MTLSizeMake(blockSize, 1, 1)
        let threadGroups = MTLSizeMake(numBlocks, 1, 1)
//        printGrid(threadGroups: threadGroups, threadsPerThreadgroup: threadGroupSize)

        setComputePipelineState(pipeline)
        dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }
    
    /**
     Dispatches a compute kernel on a 2-dimensional grid.
     
     - Parameters:
     - width: the first dimension
     - height: the second dimension
     */
    public func dispatch(pipeline: MTLComputePipelineState, width: Int, height: Int) {
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        
        let threadGroupSize = MTLSizeMake(w, h, 1)
        
        let threadGroups = MTLSizeMake(
            (width  + threadGroupSize.width  - 1) / threadGroupSize.width,
            (height + threadGroupSize.height - 1) / threadGroupSize.height, 1)
//        printGrid(threadGroups: threadGroups, threadsPerThreadgroup: threadGroupSize)

        setComputePipelineState(pipeline)
        dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }
    
    /**
     Dispatches a compute kernel on a 3-dimensional image grid.
     
     - Parameters:
     - width: the width of the image in pixels
     - height: the height of the image in pixels
     - featureChannels: the number of channels in the image
     - numberOfImages: the number of images in the batch (default is 1)
     */
    public func dispatch(pipeline: MTLComputePipelineState,
                         width: Int,
                         height: Int,
                         featureChannels: Int,
                         numberOfImages: Int = 1) {
        let slices = ((featureChannels + 3)/4) * numberOfImages
        
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let d = 1
        let threadGroupSize = MTLSizeMake(w, h, d)
        
        let threadGroups = MTLSizeMake(
            (width  + threadGroupSize.width  - 1) / threadGroupSize.width,
            (height + threadGroupSize.height - 1) / threadGroupSize.height,
            (slices + threadGroupSize.depth  - 1) / threadGroupSize.depth)
        
//        printGrid(threadGroups: threadGroups, threadsPerThreadgroup: threadGroupSize)
        
        setComputePipelineState(pipeline)
        dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }
    
    /**
     For debugging the threadgroup sizes.
     */
    public func printGrid(threadGroups: MTLSize, threadsPerThreadgroup: MTLSize) {
        let groups = threadGroups
        let threads = threadsPerThreadgroup
        let grid = MTLSizeMake(groups.width  * threads.width,
                               groups.height * threads.height,
                               groups.depth  * threads.depth)
        
        print("threadGroups: \(groups.width)x\(groups.height)x\(groups.depth)"
            + ", threadsPerThreadgroup: \(threads.width)x\(threads.height)x\(threads.depth)"
            + ", grid: \(grid.width)x\(grid.height)x\(grid.depth)")
    }
    
    /**
     Dispatches a compute kernel on an MPSImage's texture or texture array.
     */
    @available(iOS 10.0, *)
    public func dispatch(pipeline: MTLComputePipelineState, image: MPSImage) {
        dispatch(pipeline: pipeline,
                 width: image.width,
                 height: image.height,
                 featureChannels: image.featureChannels,
                 numberOfImages: image.numberOfImages)
    }
    
    /**
     Dispatches a compute kernel on an MPSImage's texture or texture array.
     
     Use this method if you only want to overwrite a portion of the MPSImage's
     channels (i.e. when you're using `destinationFeatureChannelOffset`).
     */
    @available(iOS 10.0, *)
    public func dispatch(pipeline: MTLComputePipelineState,
                         image: MPSImage,
                         featureChannels: Int) {
        dispatch(pipeline: pipeline,
                 width: image.width,
                 height: image.height,
                 featureChannels: featureChannels,
                 numberOfImages: image.numberOfImages)
    }
}
