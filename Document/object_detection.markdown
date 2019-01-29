# Quick Start2: Using SimpleNN to do object detection

### 1. Download code
``` shell
git clone https://github.com/luozhiping/neural_network_on_ios.git
```

### 2. Open SimpleNN.xcworkspace

### 3. Convert yolov3 model

Yolov3 model is reference from [https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)

- Using scripts to generate yolov3 model (Convert by official [Yolov3](https://pjreddie.com/darknet/yolo/) weights to test COCO dataset 80 classes)
``` shell
cd Python
bash keras_yolo3.sh yolov3 ./yolov3.json ./yolov3.bin
```

- Generate yolov3-tiny model
``` shell
bash keras_yolo3.sh yolov3-tiny ./yolov3_tiny.json ./yolov3_tiny.bin
```

1. parameters1: yolov3 or yolov3-tiny, generate yolov3 or yolov3-tiny model
2. parameters2: path to save network file
3. parameters3: path to save weights file

### 3. Add network file and weights file to project

![](./object_detection0.jpg)

modify code to load specify files

``` swift
Net = YoloModel.init(networkFileName: "yolov3", weightFileName: "yolov3")
```

- Because Yolov3 has multi output, so you must use "YoloModel" class instead of "Model" class. YoloModel will return YoloOuput which calculates boxes.

### 4. Running ObjectDetection Target

![](./object_detection1.jpg)

### 5. Result

![](./example1.png)

### 6. Inference time

| model name(keras class name)|inference time(iphoneXR 12.1)|model size|
| :----------| :-----------| :-----------|
|[Yolov3](https://github.com/qqwweee/keras-yolo3)|409ms|248M|
|[Yolov3-tiny](https://github.com/qqwweee/keras-yolo3)|70ms|35.4M|