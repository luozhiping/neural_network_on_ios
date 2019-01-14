//
//  ViewController.swift
//  SimpleMobile
//
//  Created by luozhiping on 2018/12/12.
//  Copyright © 2018 SimpleTech. All rights reserved.
//

import UIKit
import MetalPerformanceShaders
import MetalKit

class ViewControllerMain: UIViewController {
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
}

func assertEqual(_ a: [Float], _ b: [Float], tolerance: Float) {
    if a.count != b.count {
        fatalError("Assertion failed: array sizes not the same")
    }
    var largestDiff: Float = 0
    var totalDiff: Float = 0
    for i in 0..<a.count {
        let diff = abs(a[i] - b[i])
        if diff > tolerance {
            print(a[i], b[i])
            fatalError("Assertion failed: difference too large at index \(i): \(a[i]) vs \(b[i])")
        }
        largestDiff = max(largestDiff, diff)
        totalDiff += diff
    }
        print("    largest difference: \(largestDiff), average: \(totalDiff/Float(a.count))")
}
