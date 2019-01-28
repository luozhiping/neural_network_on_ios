//
//  MobileNetViewController.swift
//  SimpleMobile
//
//  Created by luozhiping on 2019/1/13.
//  Copyright © 2019 SimpleTech. All rights reserved.
//

import UIKit
import MetalKit
import MetalPerformanceShaders
import Accelerate
import AVFoundation
import SimpleNN


class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    // Outlets to label and view
    @IBOutlet weak var predictLabel: UILabel!
    @IBOutlet weak var predictView: UIImageView!
    
    // some properties used to control the app and store appropriate values
    var Net: Model! = nil
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var imageNum = 0
    var total = 6
    var textureLoader : MTKTextureLoader!
    var ciContext : CIContext!
    var sourceTexture : MTLTexture? = nil
    var camRan = false
    var labels = [String](repeating: "", count: 1000)
    var URL: URL?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Load default device.
        device = MTLCreateSystemDefaultDevice()
        
        // Make sure the current device supports MetalPerformanceShaders.
        guard MPSSupportsMTLDevice(device) else {
            print("Metal Performance Shaders not Supported on current Device")
            return
        }
        
        // Load any resources required for rendering.
        
        // Create new command queue.
        commandQueue = device!.makeCommandQueue()
        
        // make a textureLoader to get our input images as MTLTextures
        textureLoader = MTKTextureLoader(device: device!)
        
        // Load the appropriate Network
        Net = YoloModel.init(networkFileName: "yolov3", weightFileName: "yolov3")
        Net.printNetwork()
        // we use this CIContext as one of the steps to get a MTLTexture
        ciContext = CIContext.init(mtlDevice: device)
        
        let name = "dog"
        URL = Bundle.main.url(forResource:name, withExtension: "jpg")
        do{
            // display the image in UIImage View
            predictView.image = try UIImage(data: NSData(contentsOf: URL!) as Data)!
        }
        catch{
            NSLog("invalid URL")
        }
        
        if let path = Bundle.main.path(forResource: "synset_words", ofType: "txt") {
            for (i, line) in lines(filename: path).enumerated() {
                if i < 1000 {
                    // Strip off the WordNet ID (the first 10 characters).
                    labels[i] = String(line[line.index(line.startIndex, offsetBy: 10)...])
                }
            }
        }
        
        fetchImage()
    }
    private func lines(filename: String) -> [String] {
        do {
            let text = try String(contentsOfFile: filename, encoding: .ascii)
            let lines = text.components(separatedBy: NSCharacterSet.newlines)
            return lines
        } catch {
            fatalError("Could not load file: \(filename)")
        }
    }
    
    public subscript(i: Int) -> String {
        return labels[i]
    }
    
    /**
     This function is to conform to UIImagePickerControllerDelegate protocol,
     contents are executed after the user selects a picture he took via camera
     */
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        
        // get taken picture as UIImage
        var uiImg = info[.originalImage] as! UIImage
        uiImg = uiImg.fixOrientation()
//        print(uiImg.imageOrientation.rawValue)
        // display the image in UIImage View
        predictView.image = uiImg
        
        // use CGImage property of UIImage
        var cgImg = uiImg.cgImage
        
        // check to see if cgImg is valid if nil, UIImg is CIImage based and we need to go through that
        // this shouldn"t be the case with our example
        if(cgImg == nil){
            // our underlying format was CIImage
            var ciImg = uiImg.ciImage
            if(ciImg == nil){
                // this should never be needed but if for some reason both formats fail, we create a CIImage
                // change UIImage to CIImage
                ciImg = CIImage(image: uiImg)
            }
            // use CIContext to get a CGImage
            cgImg = ciContext.createCGImage(ciImg!, from: ciImg!.extent)
        }
        
        // get a texture from this CGImage
        do {
            sourceTexture = try textureLoader.newTexture(cgImage: cgImg!, options: [:])
        }
        catch let error as NSError {
            fatalError("Unexpected error ocurred: \(error.localizedDescription).")
        }
        
        
        // run inference neural network to get predictions and display them
        runNetwork()
        
        // to keep track of which image is being displayed
        camRan = true
        dismiss(animated: true, completion: nil)
    }
    
    @IBAction func camera(sender: UIButton) {
        let picker = UIImagePickerController()
        
        // set the picker to camera so the user can take an image
        picker.delegate = self
        picker.sourceType = .camera
        
        // call the camera
        present(picker, animated: true, completion: nil)
    }
    
    /**
     This function is used to fetch the appropriate image and store it in a MTLTexture
     so we can run our inference network on it
     
     
     - Returns:
     Void
     */
    func fetchImage(){
        
        // get appropriate image name and path to load it into a metalTexture
        
        do {
//            sourceTexture = try textureLoader.newTexture(URL: URL!, options: [:])
            sourceTexture = try! textureLoader.newTexture(URL: URL!, options: [
                MTKTextureLoader.Option.SRGB : NSNumber(value: false)
                ])
        }
        catch let error as NSError {
            fatalError("Unexpected error ocurred: \(error.localizedDescription).")
        }
        
        // run the neural network to get outputs
        runNetwork()
        
    }
    let label = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    /**
     This function gets a commanBuffer and encodes layers in it. It follows that by commiting the commandBuffer and getting labels
     
     
     - Returns:
     Void
     */
    func runNetwork(){
//        let image = ImageData(imageFileName: "dog416", device: device)
        let image = ImageData(image: MPSImage(texture: sourceTexture!, featureChannels: 3))
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let outputs = Net.predict(input: image, device: device!)
        let endTime = CFAbsoluteTimeGetCurrent()
        let cost1 = (endTime - startTime)*1000
        print("cost time:", cost1)
        // draw rect
        if let sublayers = predictView.layer.sublayers {
            for sub in sublayers {
                sub.removeFromSuperlayer()
            }
        }
        for output in outputs {
            if let output = output as? YoloOuput {
                for box in output.boxes {
                    print("box:", box.rect, box.confidenceInClass, box.detectedClass, label[box.detectedClass])
                    let shapeLayer = CAShapeLayer()
                    shapeLayer.fillColor = UIColor.clear.cgColor
                    shapeLayer.lineWidth = 2
                    predictView.layer.addSublayer(shapeLayer)
                    CATransaction.setDisableActions(true)
                    let color: UIColor = .random
                    var sRect: CGRect
                    if predictView.image!.size.width > predictView.image!.size.height {
                        let imgSW = predictView.image!.size.width / 416
                        let scale: CGFloat = predictView.frame.size.width/416
                        var height = predictView.image!.size.height * (predictView.frame.size.width / predictView.image!.size.width)

                        let scaleh: CGFloat = height/416
                        
                        height = (predictView.frame.size.height - height)/2
                        sRect = CGRect(x: box.rect.minX*scale, y: box.rect.minY*scaleh+height, width: box.rect.width*scale, height: box.rect.height*scaleh
                        )
                    } else {
                        let imgSH = predictView.image!.size.height / 416
                        let scale: CGFloat = predictView.frame.size.height/416
                        var width = predictView.image!.size.width * (predictView.frame.size.height / predictView.image!.size.height)

                        let scalew: CGFloat = width/416
                        
                        width = (predictView.frame.size.width - width)/2
                        sRect = CGRect(x: box.rect.minX*scalew+width, y: box.rect.minY*scale, width: box.rect.width*scalew, height: box.rect.height*scale
                        )
                    }
                    
                    let path = UIBezierPath(rect: sRect)
                    shapeLayer.path = path.cgPath
                    shapeLayer.strokeColor = color.cgColor
                    
                    
                    let textLayer = CATextLayer()
                    textLayer.foregroundColor = color.cgColor
                    textLayer.contentsScale = UIScreen.main.scale
                    textLayer.fontSize = 16
                    textLayer.font = UIFont(name: "Avenir", size: textLayer.fontSize)
                    textLayer.alignmentMode = CATextLayerAlignmentMode.center
                    predictView.layer.addSublayer(textLayer)

                    let str = label[box.detectedClass] + String(format: " %.2f", box.confidenceInClass)
                    textLayer.string = str
                    textLayer.backgroundColor = UIColor.white.cgColor
                    textLayer.isHidden = false
                    
                    let attributes = [
                        NSAttributedString.Key.font: textLayer.font as Any
                    ]
                    
                    let textRect = str.boundingRect(with: CGSize(width: 400, height: 100),
                                                      options: .truncatesLastVisibleLine,
                                                      attributes: attributes, context: nil)
                    let textSize = CGSize(width: textRect.width + 12, height: textRect.height)
                    let textOrigin = CGPoint(x: sRect.origin.x - 2, y: sRect.origin.y - textSize.height)
                    textLayer.frame = CGRect(origin: textOrigin, size: textSize)
                }
            }
        }
        
        print(predictView.frame.size.width, predictView.image!.size.width)
    }
    
    
    
}

