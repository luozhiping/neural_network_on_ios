//
//  YoloTask.swift
//  SimpleNN
//
//  Created by luozhiping on 2019/1/23.
//  Copyright © 2019 SimpleTech. All rights reserved.
//

import Foundation
import Accelerate
// depends on https://github.com/qqwweee/keras-yolo3
public class YoloModel: Model {
    let scoreThreshold: Float
    let iouThreshold: Float
    public init(networkFileName: String, weightFileName: String, scoreThreshold: Float = 0.3, iouThreshold: Float = 0.45) {
        self.scoreThreshold = scoreThreshold
        self.iouThreshold = iouThreshold
        super.init(networkFileName: networkFileName, weightFileName: weightFileName)
    }
    
    override public func predict(input: DataWrapper, device: MTLDevice) -> [Output] {
        let output = layers.predict(input: input, device: device)
        let o = YoloOuput(scoreThreshold: scoreThreshold, iouThreshold: iouThreshold)
        for layer in output {
            o.addOutput(layer: layer)
        }
        o.selectBoxes()
        let outputs = [o]
        return outputs
    }
}


public class YoloOuput: Output {
    let tiny_yolov3_anchors: [[Float]] = [[81,82,  135,169,  344,319], [23,27,  37,58,  81,82]]
    let yolov3_anchors: [[Float]] = [[116,90,  156,198,  373,326], [30,61,  62,45,  59,119], [10,13,  16,30,  33,23]]
    public var boxes = [Box]()
    let scoreThreshold: Float
    let iouThreshold: Float
    public init(scoreThreshold: Float, iouThreshold: Float) {
        self.scoreThreshold = scoreThreshold
        self.iouThreshold = iouThreshold
        super.init()
    }
    
    public func addTinyOutput(layer: Layer) {
        let output = layer.getOutputData() as! ImageData
        let result = output.image!.toFloatArray()

        calucateBoxes(features: result, gridWidth: output.image!.width, gridHeight: output.image!.height,anchors: output.image!.width == 13 ? tiny_yolov3_anchors[0] : tiny_yolov3_anchors[1])
    }
    
    public func addOutput(layer: Layer) {
        let output = layer.getOutputData() as! ImageData
        let result = output.image!.toFloatArray()
        
        var anchors = yolov3_anchors[0]
        if output.image!.width == 26 {
            anchors = yolov3_anchors[1]
        } else if output.image!.width == 52 {
            anchors = yolov3_anchors[2]
        }
        
        calucateBoxes(features: result, gridWidth: output.image!.width, gridHeight: output.image!.height, anchors: anchors)
    }
    
    func selectBoxes() {
        self.boxes = nonMaxSuppression(boxes: self.boxes, limit: 20, threshold: iouThreshold)
    }
    
    func calucateBoxes(features: [Float], gridWidth: Int, gridHeight: Int, anchors: [Float]) {
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
                    
                    let x = (Float(cx) * 1 + sigmoid(tx))/Float(gridWidth)*416
                    let y = (Float(cy) * 1 + sigmoid(ty))/Float(gridHeight)*416
                    
                    let w = exp(tw) * anchors[2*b    ]
                    let h = exp(th) * anchors[2*b + 1]
                    
                    let confidence = sigmoid(tc)
                    
                    var classes = [Float](repeating: 0, count: numClasses)
                    for c in 0..<numClasses {
                        classes[c] = Float(features[offset(channel + 5 + c, cx, cy)])
                    }
                    
                    classes = softmax(classes)
                    
                    let (detectedClass, bestClassScore) = classes.argmax()
                    let confidenceInClass = bestClassScore * confidence
                    
                    if confidenceInClass > scoreThreshold {
                        let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                          width: CGFloat(w), height: CGFloat(h))
                        self.boxes.append(Box(rect: rect, confidenceInClass: confidenceInClass, detectedClass: detectedClass))
//                        print(">0.3,", confidenceInClass, detectedClass, x, y, w, h, cx,cy,tx,ty, tw, th,b,rect.minX, rect.minY, rect.maxX, rect.maxY)
                    }
                }
            }
        }
    }
}

public class Box {
    public let rect: CGRect
    public let confidenceInClass: Float
    public let detectedClass: Int
    public init(rect: CGRect, confidenceInClass: Float, detectedClass: Int) {
        self.rect = rect
        self.confidenceInClass = confidenceInClass
        self.detectedClass = detectedClass
    }
}







