/*
 Copyright (c) 2016-2017 M.I. Hollemans
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to
 deal in the Software without restriction, including without limitation the
 rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 IN THE SOFTWARE.
 */

import Foundation
import Accelerate

/* Utility functions for dealing with 16-bit floating point values in Swift. */

/**
 Since Swift has no datatype for a 16-bit float we use `UInt16`s instead,
 which take up the same amount of memory. (Note: The simd framework does
 have "half" types but only for 2, 3, or 4-element vectors, not scalars.)
 */
public typealias Float16 = UInt16

/**
 Creates a new array of Swift `Float` values from a buffer of float-16s.
 */
public func float16to32(_ input: UnsafeMutablePointer<Float16>, count: Int) -> [Float] {
    var output = [Float](repeating: 0, count: count)
    float16to32(input: input, output: &output, count: count)
    return output
}

/**
 Converts a buffer of float-16s into a buffer of `Float`s, in-place.
 */
public func float16to32(input: UnsafeMutablePointer<Float16>, output: UnsafeMutableRawPointer, count: Int) {
    var bufferFloat16 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 2)
    var bufferFloat32 = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 4)
    
    if vImageConvert_Planar16FtoPlanarF(&bufferFloat16, &bufferFloat32, 0) != kvImageNoError {
        print("Error converting float16 to float32")
    }
}

/**
 Creates a new array of float-16 values from a buffer of `Float`s.
 */
public func float32to16(_ input: UnsafeMutablePointer<Float>, count: Int) -> [Float16] {
    var output = [Float16](repeating: 0, count: count)
    float32to16(input: input, output: &output, count: count)
    return output
}

/**
 Converts a buffer of `Float`s into a buffer of float-16s, in-place.
 */
public func float32to16(input: UnsafeMutablePointer<Float>, output: UnsafeMutableRawPointer, count: Int) {
    var bufferFloat32 = vImage_Buffer(data: input,  height: 1, width: UInt(count), rowBytes: count * 4)
    var bufferFloat16 = vImage_Buffer(data: output, height: 1, width: UInt(count), rowBytes: count * 2)
    
    if vImageConvert_PlanarFtoPlanar16F(&bufferFloat32, &bufferFloat16, 0) != kvImageNoError {
        print("Error converting float32 to float16")
    }
}


public func sigmoid(_ x: Float) -> Float {
    return 1 / (1 + exp(-x))
}


public func softmax(_ x: [Float]) -> [Float] {
    var x = x
    let len = vDSP_Length(x.count)
    
    // Find the maximum value in the input array.
    var max: Float = 0
    vDSP_maxv(x, 1, &max, len)
    
    // Subtract the maximum from all the elements in the array.
    // Now the highest value in the array is 0.
    max = -max
    vDSP_vsadd(x, 1, &max, &x, 1, len)
    
    // Exponentiate all the elements in the array.
    var count = Int32(x.count)
    vvexpf(&x, x, &count)
    
    // Compute the sum of all exponentiated values.
    var sum: Float = 0
    vDSP_sve(x, 1, &sum, len)
    
    // Divide each element by the sum. This normalizes the array contents
    // so that they all add up to 1.
    vDSP_vsdiv(x, 1, &sum, &x, 1, len)
    
    return x
}

extension Array where Element: Comparable {
    /**
     Returns the index and value of the largest element in the array.
     */
    public func argmax() -> (Int, Element) {
        precondition(self.count > 0)
        var maxIndex = 0
        var maxValue = self[0]
        for i in 1..<self.count {
            if self[i] > maxValue {
                maxValue = self[i]
                maxIndex = i
            }
        }
        return (maxIndex, maxValue)
    }
}

/**
 Removes bounding boxes that overlap too much with other boxes that have
 a higher score.
 
 Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc
 
 - Parameters:
 - boxes: an array of bounding boxes and their scores
 - limit: the maximum number of boxes that will be selected
 - threshold: used to decide whether boxes overlap too much
 */
func nonMaxSuppression(boxes: [Box], limit: Int, threshold: Float) -> [Box] {
    
    // Do an argsort on the confidence scores, from high to low.
    let sortedIndices = boxes.indices.sorted { boxes[$0].confidenceInClass > boxes[$1].confidenceInClass }
    
    var selected: [Box] = []
    var active = [Bool](repeating: true, count: boxes.count)
    var numActive = active.count
    
    // The algorithm is simple: Start with the box that has the highest score.
    // Remove any remaining boxes that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    outer: for i in 0..<boxes.count {
        if active[i] {
            let boxA = boxes[sortedIndices[i]]
            selected.append(boxA)
            if selected.count >= limit { break }
            
            for j in i+1..<boxes.count {
                if active[j] {
                    let boxB = boxes[sortedIndices[j]]
                    if IOU(a: boxA.rect, b: boxB.rect) > threshold {
                        active[j] = false
                        numActive -= 1
                        if numActive <= 0 { break outer }
                    }
                }
            }
        }
    }
    return selected
}

/**
 Computes intersection-over-union overlap between two bounding boxes.
 */
public func IOU(a: CGRect, b: CGRect) -> Float {
    let areaA = a.width * a.height
    if areaA <= 0 { return 0 }
    
    let areaB = b.width * b.height
    if areaB <= 0 { return 0 }
    
    let intersectionMinX = max(a.minX, b.minX)
    let intersectionMinY = max(a.minY, b.minY)
    let intersectionMaxX = min(a.maxX, b.maxX)
    let intersectionMaxY = min(a.maxY, b.maxY)
    let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
        max(intersectionMaxX - intersectionMinX, 0)
    return Float(intersectionArea / (areaA + areaB - intersectionArea))
}
