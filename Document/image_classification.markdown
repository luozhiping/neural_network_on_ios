# Quick Start: Using SimpleNN to do image classification

### 1. Download code
``` shell
git clone https://github.com/luozhiping/neural_network_on_ios.git
```

### 2. Open SimpleNN.xcworkspace

### 3. Running ImageClassfication Target

![](./image_classification0.jpg)

### 4. Tap "Run Network" button and see prediction result (default model is mobilenet)

![](./example.png)

### 5. Inference time

| model name(keras class name)|inference time(iphoneXR 12.1)|model size|
| :----------| :-----------| :-----------| :-----------|
|[Mobilenet](https://keras.io/applications/#mobilenet)|39ms|17M|
|[MobilenetV2](https://keras.io/applications/#mobilenetv2)|30ms|14.2M|
|[Xception](https://keras.io/applications/#xception)|117ms|91.6M|
|[InceptionV3](https://keras.io/applications/#inceptionv3)|69ms|95.5M|


### 6. Try another keras image classification model

1. [Convert another keras model](./convert_keras_model.markdown)

```shell
python convert_keras.py --network-path ./network.json --weights-path ./weights.bin --model mobilenetv2
params:
--network-path network file extension must be .json
--weights-path weights file extension must be .bin
--model mobilnet\mobilenetv2\xception\inceptionv3
```

2. Add network file and weights file to project

![](./image_classification1.jpg)

3. Load new model and do inference

```swift
let Net = Model.init(networkFileName: "network", weightFileName: "weights")
Net.predict(input: image, device: device!)
```

### Tips

#### My environment
- python 3.6.3
- Keras-2.2.4
- tensorflow 1.13.0-dev20181205
- Xcode 10.1
- iPhoneXR and IOS 12.1.2

#### Model

- This application using keras [Mobilenet](https://keras.io/applications/#mobilenet) model by default
- [Convert another keras model](./convert_keras_model.markdown) and see prediction result