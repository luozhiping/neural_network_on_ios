//
//  Model.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/13.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

struct Constants {
    static let IS_UNDER_9 = false//#available(iOS 11.0, *)
}

public class Model {
    var layers: LayerTree
    var device: MTLDevice?
    public init(networkFileName: String, weightFileName: String) {
        device = MTLCreateSystemDefaultDevice()
        guard device != nil else {
            fatalError("Error: This device does not support Metal")
        }
        
        guard MPSSupportsMTLDevice(device) else {
            fatalError("Error: This device does not support Metal Performance Shaders")
        }
        // 1.读取网络结构文件
        //****
        
        // 2.读取weight
        layers = LayerTree()
        readNetworkFileAndCreateLayers(networkFileName: networkFileName)
        
        // init shape
        layers.initShape()
//
//        // 3.导入weight
        initWeight(weightFileName: weightFileName)
//
//
//        // 4.生成网络
//
//
//
        createNetwork()
    }
    
    private func readNetworkFileAndCreateLayers(networkFileName: String) {
        let path = Bundle.main.path(forResource: networkFileName, ofType: "json")
        let url = URL(fileURLWithPath: path!)
        
        do {
            let data = try Data(contentsOf: url)
            let jsonData:Any = try JSONSerialization.jsonObject(with: data, options: JSONSerialization.ReadingOptions.mutableContainers)
            let jsonArr = jsonData as! NSDictionary
            
            // get layers
            let config = jsonArr["config"] as! NSDictionary
            let all_layers = config["layers"] as! NSArray
            for index in 0..<all_layers.count {
                addLayer(layer: all_layers[index] as! NSDictionary)
                
                if index == 40 {
//                    break
                }
            }
            
            let outputLayers = config["output_layers"] as! NSArray
            for output in outputLayers {
                if let output = output as? NSArray {
                    if let output = output[0] as? String {
                        // output layer name
                        var finded = false
                        for (_, layer) in layers.layers.enumerated().reversed() {
                            if layer.name == output {
                                layers.addOutput(layer: layer)
                                finded = true
                            }
                        }
                    }
                }
            }
            if layers.outputs.count == 0 {
                layers.addOutput(layer: layers.layers[layers.layers.count - 1])
            }
            for output in layers.outputs {
                print(output.name)
            }
            
//            layers.tailLayer!.forceNotTemp = true
            
            
        } catch {
            fatalError("read networkFile error")
        }
    }
    
    private func addLayer(layer: NSDictionary) {
        let ls = LayerFactory(layer: layer, device: device!, layers:layers)
        for l in ls {
            if #available(iOS 10, *) {
                if l is Softmax {
                    let inbound = [layers.tailLayer!.name]
                    layers.addLayer(layer: l, inBounds: inbound)
                } else {
                    let inbound = parseInbound(inbound_nodes: layer["inbound_nodes"] as? NSArray)
                    layers.addLayer(layer: l, inBounds: inbound)
                }
            } else {
                if l is ShaderSoftmax {
                    let inbound = [layers.tailLayer!.name]
                    layers.addLayer(layer: l, inBounds: inbound)
                } else {
                    let inbound = parseInbound(inbound_nodes: layer["inbound_nodes"] as? NSArray)
                    layers.addLayer(layer: l, inBounds: inbound)
                }
            }
            
            
        }
    }
    
    private func parseInbound(inbound_nodes: NSArray?) -> [String]{
        var result: [String] = []
        if inbound_nodes != nil && inbound_nodes!.count > 0 {
            let nodes = inbound_nodes![0] as! NSArray
            for node in nodes {
                if let n = (node as? NSArray) {
                    result.append(n[0] as! String)
                }
            }
        }
        
        return result
    }
    
    private func createNetwork() {
        layers.createNetwork(device: device!)
    }
    
    public func printNetwork() {
        layers.printTree()
        print("=========================\n")
    }
    
    private func initWeight(weightFileName: String) {
        let pointer = loadNetworkFile(weightFileName: weightFileName, weightSize: layers.getWeightSize())
        layers.insertWeight(pointer: pointer)
        
    }
    
    // 读取网络结构文件
    private func loadNetworkFile(weightFileName: String, weightSize: Int) -> UnsafePointer<Float> {
        let fileSize = weightSize * MemoryLayout<Float>.stride
        guard let path = Bundle.main.path(forResource: weightFileName, ofType: "bin") else {
            fatalError("Error: resource \"\(weightFileName)\" not found")
        }
        
        let fd = open(path, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
        if fd == -1 {
            fatalError("Error: failed to open \"\(path)\", error = \(errno)")
        }
        
        let hdr = mmap(nil, fileSize, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)
        if hdr == nil {
            fatalError("Error: mmap failed, errno = \(errno)")
        }
        
        let pointer = UnsafePointer(hdr!.bindMemory(to: Float32.self, capacity: fileSize))
        
        close(fd)
        if pointer == UnsafePointer<Float>.init(bitPattern: -1) {
            fatalError("Error: mmap failed, errno = \(errno)")
        }
//        print("read file:", weightFileName, ", fileSize:", fileSize)
        
        return pointer
    }
    
    public func predict(input: DataWrapper, device: MTLDevice) -> [Output] {
        return layers.predict(input: input, device: device)
    }
}








