//
//  ViewController.swift
//  SimpleNNTest
//
//  Created by JiuQu on 2019/1/16.
//  Copyright © 2019 SimpleTech. All rights reserved.
//

import UIKit
import SimpleNN
import MetalPerformanceShaders
import Accelerate

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        let device: MTLDevice?
        device = MTLCreateSystemDefaultDevice()
        guard device != nil else {
            fatalError("Error: This device does not support Metal")
        }
        
        guard MPSSupportsMTLDevice(device) else {
            fatalError("Error: This device does not support Metal Performance Shaders")
        }
        let inputImage: DataWrapper
        if #available(iOS 10, *) {
            inputImage = ImageData(imageFileName: "dog416", device: device!)
            //            inputImage = ImageData(imageFileName: "Floortje224", device: device!)
            //            inputImage = ImageData(imageFileName: "Floortje224", device: device!)
            
        } else {
            inputImage = DataWrapper(imageFileName: "Floortje", device: device!)
        }
                        let model2 = Model.init(networkFileName: "official__mobilenet", weightFileName: "keras__mobilenet")
//                        let model2 = Model.init(networkFileName: "official_mobilenetv2", weightFileName: "keras_mobilenetv2")
//        let model2 = Model.init(networkFileName: "official_xception", weightFileName: "keras_xception")
//                let model2 = YoloModel.init(networkFileName: "keras_yolov3", weightFileName: "keras_yolov3")
        //        let model2 = Model.init(networkFileName: "keras_yolov3_tiny", weightFileName: "keras_yolov3_tiny")
//                let model2 = Model.init(networkFileName: "official_inceptionv3", weightFileName: "keras_inceptionv3")
        
        model2.printNetwork()
        //        return
        //        let inputImage2 = ImageData(imageFileName: "Floortje", device: device!)
        var startTime = CFAbsoluteTimeGetCurrent()
        var output = [Output]()
        var abc = 100
        for i in 0..<abc {
            output = model2.predict(input: inputImage, device: device!)
        }
        var endTime = CFAbsoluteTimeGetCurrent()
        var cost1 = (endTime - startTime)*1000
        //        startTime = CFAbsoluteTimeGetCurrent()
        //
        //        for i in 0...400 {
        //            let output2 = model.predict(input: inputImage2, device: device!)
        //        }
        //        endTime = CFAbsoluteTimeGetCurrent()
        //
        //        var cost2 = (endTime - startTime)*1000
        //
        print("代码执行时长：%f 毫秒",cost1, cost1/Double(abc))
        
        
        
        
        
        
        
        //        print(sum)
        //        assertEqual(results, results2, tolerance: 1e-3)
        
        //        let outputImage = ImageData(width: 1, height: 1, channel: 32, device: device!)
        
        var ind = 0
        if #available(iOS 10, *) {
            
            //            let outputImage2 = output as! ImageData
            //            let a2 = outputImage2.image!
            let results2 = output[0].result
//            yolo(features: results2)
            return
            //            print(a2.width, a2.height, results2.count)
            //
            var sum: Float32 = 0
            //        var sum2: Float32 = 0
            for i in 0..<results2.count {
                //            print(results[results.count - i])
                //            print(results[i])
                if i < 85 && i >= 0 {
//                    print(results2[i], i)
                }
                if i % 4 == 0 && i >= 0*4 && i <= 10*4 {
                    print(results2[i], results2[i+1], results2[i+2], results2[i+3], i/4)
                }
                //                if i % 256 == 0 {
                //                    p += String(results2[i]) + ","
                //                    p += String(results2[i+1]) + ","
                //                    p += String(results2[i+2]) + ","
                //                    p += String(results2[i+3]) + ","
                //
                //                    //                    print(results2[i], results2[i+1], results2[i+2], results2[i+3], i/4)
                //                }
                if results2[i] > 0.01 {
//                    print(results2[i], i)
                }
                sum += results2[i]
                //            sum2 += results2[i]
                
            }
            print(sum)
            
        } else {
            
        }
    }


}

let anchors: [[Float]] = [[81,82,  135,169,  344,319], [10,14,  23,27,  37,58]]

func yolo(features: [Float]) {
    let blockSize: Float = 32
    let gridHeight = 13
    let gridWidth = 13
    let boxesPerCell = 3
    let numClasses = 80
    
    func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
        let slice = channel / 4
        let indexInSlice = channel - slice*4
        let offset = slice*gridHeight*gridWidth*4 + y*gridWidth*4 + x*4 + indexInSlice
        return offset
    }
    
    for cy in 0..<gridHeight {
        for cx in 0..<gridWidth {
            for b in 0..<boxesPerCell {
                
                let channel = b*(numClasses + 5)
                let tx = features[offset(channel, cx, cy)]
                let ty = features[offset(channel + 1, cx, cy)]
                let tw = features[offset(channel + 2, cx, cy)]
                let th = features[offset(channel + 3, cx, cy)]
                let tc = features[offset(channel + 4, cx, cy)]
                
                let scale = powf(2.0,Float(0)) // scale pos by 2^i where i is the scale pyramid level
                
//                let x = (Float(cx) * blockSize + sigmoid(tx))/scale
//                let y = (Float(cy) * blockSize + sigmoid(ty))/scale
                let x = (Float(cx) * 1 + sigmoid(tx))/13*416
                let y = (Float(cy) * 1 + sigmoid(ty))/13*416
                // The size of the bounding box, tw and th, is predicted relative to
                // the size of an "anchor" box. Here we also transform the width and
                // height into the original 416x416 image space.
                let w = exp(tw) * anchors[0][2*b    ]
                let h = exp(th) * anchors[0][2*b + 1]
                if cy == 8 && cx == 4 {
                    
                    print(x, y)
                }
                // The confidence value for the bounding box is given by tc. We use
                // the logistic sigmoid to turn this into a percentage.
                let confidence = sigmoid(tc)
                
                // Gather the predicted classes for this anchor box and softmax them,
                // so we can interpret these numbers as percentages.
                var classes = [Float](repeating: 0, count: numClasses)
                for c in 0..<numClasses {
                    // The slow way:
                    //classes[c] = features[[channel + 5 + c, cy, cx] as [NSNumber]].floatValue
                    
                    // The fast way:
                    classes[c] = Float(features[offset(channel + 5 + c, cx, cy)])
                }
                if cy == 8 && cx == 4 {
                    print(tx, ty, tw, th, tc)
                }
                classes = softmax(classes)
                
                let (detectedClass, bestClassScore) = classes.argmax()
                let confidenceInClass = bestClassScore * confidence
                
                // Since we compute 13x13x3 = 507 bounding boxes, we only want to
                // keep the ones whose combined score is over a certain threshold.
                if confidenceInClass > 0.75 {
                    let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                      width: CGFloat(w), height: CGFloat(h))
                    print(">0.3,", confidenceInClass, detectedClass, x, y, w, h, cx,cy,tx,ty, tw, th,b,rect.minX, rect.minY, rect.maxX, rect.maxY)
                }
            }
        }
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
