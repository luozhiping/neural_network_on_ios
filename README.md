# SimpleNN: Neural Network Running On IOS using keras model
Running neural network on IOS(version 10.0 or newer) use GPU (support Keras model)

此库为IOS(版本 >= 10.0)上调用MetalPerformanceShaders和Metal API，在GPU上运行神经网络。当前支持转换keras模型
## Features
 - Use GPU running neural network, improve performance and save more phone battery.
 - Support [keras model](./Document/model_list.markdown).
 - No need import any framework just Apple's MetalPerformanceShaders.

## 2019-01-13 V0.1
 - 支持keras图片分类模型转换,支持keras官方模型： [Mobilenet](https://keras.io/applications/#mobilenet)\ [MobilenetV2](https://keras.io/applications/#mobilenetv2)\ [Xception](https://keras.io/applications/#xception)\ [InceptionV3](https://keras.io/applications/#inceptionv3)
 - 支持keras [layer列表](https://github.com/luozhiping/neural_network_on_ios/blob/master/Document/layer_list.markdown)
 - V0.1 support some keras imageclassfication model: [Mobilenet](https://keras.io/applications/#mobilenet)\ [MobilenetV2](https://keras.io/applications/#mobilenetv2)\ [Xception](https://keras.io/applications/#xception)\ [InceptionV3](https://keras.io/applications/#inceptionv3)
 - V0.1 support some keras [layer list](https://github.com/luozhiping/neural_network_on_ios/blob/master/Document/layer_list.markdown)

## Quick Start
- [Quick Start: Using SimpleNN to do image classification](./Document/image_classification.markdown)
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
    - xcode 10.1
    - iPhoneXR and IOS 12.1.2
- ToDo
    - Support more keras model and layers

### 6.Document

- [Document](./Document)

### Reference
Github: A neural network toolkit for Metal [https://github.com/hollance/Forge](https://github.com/hollance/Forge)