from keras.layers import *
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications import *
import numpy as np
from numpy import array

def convert_weights(model):
    weights = []
    layers = model.layers
    for index, layer in enumerate(layers):
        # print("**************")
        # print(layer, layer.get_config())
        if isinstance(layer, DepthwiseConv2D):
            weight = layer.get_weights()
            # print("DepthwiseConv2D", weight[0].transpose(2, 3, 0, 1).reshape(-1)[0:5])
            weights.extend(weight[0].transpose(2, 3, 0, 1).reshape(-1))
            if layer.use_bias:
                weights.extend(weight[1])
        elif isinstance(layer, Conv2D):
            weight = layer.get_weights()
            print(weight[0].shape)
            # if index == 90:
            #     print(weight[0].transpose(3,0,1,2).reshape(-1)[0:4])
            weights.extend(weight[0].transpose(3,0,1,2).reshape(-1))
            if layer.use_bias:
                weights.extend(weight[1])
        elif isinstance(layer, BatchNormalization):
            weight = layer.get_weights()
            if layer.scale:
                weights.extend(weight[0])
                weights.extend(weight[1])
                weights.extend(weight[2])
                weights.extend(weight[3])
            else:
                weights.extend(np.ones(weight[0].shape))
                weights.extend(weight[0])
                weights.extend(weight[1])
                weights.extend(weight[2])
            # print("BatchNormalization:", weight[0][0:8],weight[1][0:8],weight[2][0:8],weight[3][:8])
        elif isinstance(layer, Dense):
            weight = layer.get_weights()
            # print(weight[1][:4])
            weights.extend(weight[0].transpose(1, 0).reshape(-1))
            if layer.use_bias:
                weights.extend(weight[1])
        elif isinstance(layer, SeparableConv2D):
            weight = layer.get_weights()
            # print("SeparableConv2D:", weight[1].shape)
            weights.extend(weight[0].transpose(2, 3, 0, 1).reshape(-1))
            weights.extend(weight[1].transpose(3, 0, 1, 2).reshape(-1))
        

    return weights
