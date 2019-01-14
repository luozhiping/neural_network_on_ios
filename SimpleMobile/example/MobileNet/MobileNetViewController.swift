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
//        Net = Model.init(networkFileName: "official_mobilenet", weightFileName: "keras_mobilenet")
//        Net = Model.init(networkFileName: "official_mobilenetv2", weightFileName: "keras_mobilenetv2")
//        Net = Model.init(networkFileName: "official_xception", weightFileName: "keras_xception")
        Net = Model.init(networkFileName: "network", weightFileName: "weights")
        Net.printNetwork()
        // we use this CIContext as one of the steps to get a MTLTexture
        ciContext = CIContext.init(mtlDevice: device)
        
        let name = "Floortje"
        let URL = Bundle.main.url(forResource:name, withExtension: "png")
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
        let uiImg = info[.originalImage] as! UIImage
        
        // display the image in UIImage View
        predictView.image = uiImg
        
        // use CGImage property of UIImage
        var cgImg = uiImg.cgImage
        
        // check to see if cgImg is valid if nil, UIImg is CIImage based and we need to go through that
        // this shouldn't be the case with our example
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
    
    @IBAction func tap(_ sender: UITapGestureRecognizer) {
        
        // if camera was used, we must display the appropriate image in predictView
        if(camRan){
            camRan = false
            do{
                let name = "Floortje"
                let URL = Bundle.main.url(forResource:name, withExtension: "png")
                // display the image in UIImage View
                predictView.image = try UIImage(data: NSData(contentsOf: URL!) as Data)!
            }
            catch{
                NSLog("invalid URL")
            }
        }
        
        // run the neural network to get predictions
        fetchImage()
    }
    
    
    @IBAction func swipeLeft(sender: UISwipeGestureRecognizer) {
        
        // image is changing, hide predictions of previous layer
        predictLabel.isHidden = true
        
        // get the next image
        imageNum = (imageNum + 1) % total
        
        // get appropriate image name and path
        let name = "Floortje"
        let URL = Bundle.main.url(forResource:name, withExtension: "png")
        do{
            // display the image in UIImage View
            predictView.image = try UIImage(data: NSData(contentsOf: URL!) as Data)!
        }
        catch{
            NSLog("invalid URL")
        }
        
        
        
    }
    
    @IBAction func swipeRight(sender: UISwipeGestureRecognizer) {
        
        // image is changing, hide predictions of previous layer
        predictLabel.isHidden = true
        
        // get the previous image
        if((imageNum - 1) >= 0){
            imageNum = (imageNum - 1) % total
        }
        else{
            imageNum = total - 1
        }
        
        // get appropriate image name and path
        let name = "Floortje"
        let URL = Bundle.main.url(forResource:name, withExtension: "png")
        do{
            // display the image in UIImage View
            predictView.image = try UIImage(data: NSData(contentsOf: URL!) as Data)!
        }
        catch{
            NSLog("invalid URL")
        }
        
        
    }
    
    
    /**
     This function is used to fetch the appropriate image and store it in a MTLTexture
     so we can run our inference network on it
     
     
     - Returns:
     Void
     */
    func fetchImage(){
        
        // get appropriate image name and path to load it into a metalTexture
        let name = "Floortje"
        let URL = Bundle.main.url(forResource:name, withExtension: "png")
        
        do {
            sourceTexture = try textureLoader.newTexture(URL: URL!, options: [:])
        }
        catch let error as NSError {
            fatalError("Unexpected error ocurred: \(error.localizedDescription).")
        }
        
        // run the neural network to get outputs
        runNetwork()
        
    }
    
    /**
     This function gets a commanBuffer and encodes layers in it. It follows that by commiting the commandBuffer and getting labels
     
     
     - Returns:
     Void
     */
    func runNetwork(){
        let image = ImageData(image: MPSImage(texture: sourceTexture!, featureChannels: 3))
        let results = Net.predict(input: image, device: device!)
//        let outputImage = output as! ImageData
//        let results = outputImage.image!.toFloatArray()
        var indexedProbabilities = [(Float, Int)]()
        for i in 0..<results.count{
            indexedProbabilities.append((results[i], i))
            if results[i] > 0.1 {
//                print(results[i], i)
            }
        }
        indexedProbabilities.sort { (a: (prob: Float, _: Int), b: (prob: Float, _: Int)) -> Bool in
            return a.prob > b.prob
        }
        var returnString = ""
        var j = 0
        var i = 0
        while( j < 5){
            let (prob, index) = indexedProbabilities[i]
            // labels at 0 and 1001 to 1008 are invalid (no labels were provided for these indices) so we ignore these
            if((index < 1001) && (index > 0)){
                returnString = returnString + String(format: "%3.2f", prob * 100) + "%\n" + labels[index] + "\n\n\n"
                j = j + 1
            }
            i = i + 1
        }
        print(returnString)
        predictLabel.text = returnString
        predictLabel.isHidden = false
        
    }
    
    
    
}

