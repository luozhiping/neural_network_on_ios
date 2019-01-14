from convert_weights import *
from keras.applications import *
import argparse
from numpy import array

parser = argparse.ArgumentParser()
parser.add_argument('--network-path', metavar='DIR',
                    help='path to save network json file', default='./network.json')
parser.add_argument('--weights-path', metavar='DIR',
                    help='path to save weights file', default='./weights.bin')
parser.add_argument('--model', default='mobilenet', help='Type of the Keras Model. mobilenet|mobilenetV2|xception')


if __name__ == "__main__":
    print("begin convert keras model")
    args = parser.parse_args()

    model = args.model.lower()
    assert model in ["mobilenet", "mobilenetv2", "xception"], "model support mobilenet|mobilenetV2|xception, you can try other model by manual"

    kera_model = None
    if model == "mobilenet":
        kera_model = MobileNet(weights="imagenet")
    elif model == "mobilenetv2":
        kera_model = MobileNetV2(weights="imagenet")
    elif model == "xception":
        kera_model = Xception(weights='imagenet', input_shape=(299,299,3))

    network = kera_model.to_json()
    network_file = open(args.network_path, 'w')
    network_file.write(network)
    network_file.close()

    weights = convert_weights(kera_model)
    weights = array(weights, 'float32')
    weights_file = open(args.weights_path, 'wb')
    weights.tofile(weights_file)
    weights_file.close()

    print(kera_model.summary())
    print("Convert model success: network file:", args.network_path, ", weights_file:", args.weights_path)