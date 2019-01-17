# SimpleNN: basic API

### 1. loading model

```swift
let Net = Model.init(networkFileName: "network", weightFileName: "weights")
```

### 2. print model architecture

```swift
Net.printNetwork()
```

### 3. do inference

```swift
let inputImage = ImageData(imageFileName: "cat", device: device!)
Net.predict(input: inputImage, device: device!)
```