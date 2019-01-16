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
            inputImage = ImageData(imageFileName: "cat299", device: device!)
            //            inputImage = ImageData(imageFileName: "Floortje224", device: device!)
            //            inputImage = ImageData(imageFileName: "Floortje224", device: device!)
            
        } else {
            inputImage = DataWrapper(imageFileName: "Floortje", device: device!)
        }
                        let model2 = Model.init(networkFileName: "official__mobilenet", weightFileName: "keras__mobilenet")
        //                let model2 = Model.init(networkFileName: "official_mobilenetv2", weightFileName: "keras_mobilenetv2")
//        let model2 = Model.init(networkFileName: "official_xception", weightFileName: "keras_xception")
        //        let model2 = Model.init(networkFileName: "keras_yolov3", weightFileName: "keras_yolov3")
        //        let model2 = Model.init(networkFileName: "keras_yolov3_tiny", weightFileName: "keras_yolov3_tiny")
        //        let model2 = Model.init(networkFileName: "official_inceptionv3", weightFileName: "keras_inceptionv3")
        
        model2.printNetwork()
        //        return
        //        let inputImage2 = ImageData(imageFileName: "Floortje", device: device!)
        var startTime = CFAbsoluteTimeGetCurrent()
        var output = [Float]()
        var abc = 2
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
            let results2 = output
            
            //            print(a2.width, a2.height, results2.count)
            //
            var sum: Float32 = 0
            //        var sum2: Float32 = 0
            for i in 0..<results2.count {
                //            print(results[results.count - i])
                //            print(results[i])
                if i < ind+20 && i >= ind {
                    //                                        print(results2[i], i)
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
                    print(results2[i], i)
                }
                sum += results2[i]
                //            sum2 += results2[i]
                
            }
            print(sum)
            
        } else {
            
        }
    }


}

