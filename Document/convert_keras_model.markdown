# Convert and Using keras model in SimpleNN

### 1.SimpleNN support some keras model

see [Support keras model](./model_list.markdown)

### 2.Convert keras model

- networkfile: A json file that generate by keras [model.to_json](https://keras.io/models/about-keras-models/) func, the function returns a representation of the model as a JSON string
- weightsfile: A binary file that saved model weights

running convert_keras.py to convert keras model generate SimpleNN network file and weights file

- image classification model will loading 'imagenet' weights

```shell
python convert_keras.py --network-path ./network.json --weights-path ./weights.bin --model mobilenetv2
params:
--network-path network file extension must be .json
--weights-path weights file extension must be .bin
--model mobilnet\mobilenetv2\xception\inceptionv3\
```

### 3.Add weights file and network file to project and running

```swift
let Net = Model.init(networkFileName: "network", weightFileName: "weights")
Net.predict(input: image, device: device!)
```