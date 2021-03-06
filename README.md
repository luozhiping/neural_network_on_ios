# SimpleNN: Neural Network Running On IOS using keras model
Running neural network on IOS(version 10.0 or newer) use GPU (support Keras model)

## Features
 - Use GPU running neural network, improve performance and save more phone battery.
 - Support convert and running [keras model](./Document/model_list.markdown).
 - No need import any framework just Apple's MetalPerformanceShaders and Metalkit.

## Quick Start
- [Quick Start: Using SimpleNN to do image classification](./Document/image_classification.markdown)
- [Quick Start2: Using SimpleNN to do object detection](./Document/object_detection.markdown)
- [Convert and Using keras model in SimpleNN](./Document/convert_keras_model.markdown)
- [Convert my own keras model](./Document/convert_my_model.markdown)
- [SimpleNN: Basic API](./Document/basic_api.markdown)
- [Support keras layer](./Document/layer_list.markdown)
- [Support keras model](./Document/model_list.markdown)

## Some issue
- my environment
    - python 3.6.3
    - Keras-2.2.4
    - tensorflow 1.13.0-dev20181205
    - Xcode 10.1
    - iPhoneXR with IOS 12.1.2 && iPad Air 2 with IOS 10.3.2
    - swift 4.2
- ToDo
    - Support more keras model and layers

## Documents
- [Documents](./Document)

## Example
- [Quick Start: Using SimpleNN to do image classification](./Document/image_classification.markdown)

![](./Document/example.png)

- [Quick Start2: Using SimpleNN to do object detection](./Document/object_detection.markdown)

![](./Document/example1.png)


## Recently Change ([CHANGE LOG](./Document/change_log.markdown))

### 2019-01-29 V0.2
- V0.2 support keras yolov3 model, reference from [keras-yolo3](https://github.com/qqwweee/keras-yolo3)
### 2019-01-13 V0.1
 - V0.1 support some keras imageclassfication model: [Mobilenet](https://keras.io/applications/#mobilenet)\ [MobilenetV2](https://keras.io/applications/#mobilenetv2)\ [Xception](https://keras.io/applications/#xception)\ [InceptionV3](https://keras.io/applications/#inceptionv3)
 - V0.1 support some keras [layer list](https://github.com/luozhiping/neural_network_on_ios/blob/master/Document/layer_list.markdown)

## Reference
- Github: A neural network toolkit for Metal [https://github.com/hollance/Forge](https://github.com/hollance/Forge)
- Keras [https://keras.io/models/about-keras-models/](https://keras.io/models/about-keras-models/)
- Yolov3 [https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)