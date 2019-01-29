import sys
import argparse
from keras_yolov3.yolo import YOLO, detect_video
from PIL import Image
from convert_weights import *
from keras.applications import *
import argparse
from numpy import array
from keras.layers import *

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model-path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors-path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )
    parser.add_argument('--network-path', metavar='DIR',
                        help='path to save network json file', default='./network.json')
    parser.add_argument('--weights-path', metavar='DIR',
                        help='path to save weights file', default='./weights.bin')

    FLAGS = parser.parse_args()
    # print(FLAGS.weights_path)
    yolo = YOLO(**vars(FLAGS))
    model = yolo.yolo_model
    model.layers[0] = InputLayer((416, 416, 3))
    # print(model.to_json())

    print("converting, please wait")
    network = model.to_json()
    network_file = open(FLAGS.network_path, 'w')
    network_file.write(network)
    network_file.close()

    weights = convert_weights(model)
    a = array(weights, 'float32')
    # print(a.shape)
    output_file = open(FLAGS.weights_path, 'wb')
    a.tofile(output_file)
    output_file.close()

    print(model.summary())
    print("Convert model success: network file:", FLAGS.network_path, ", weights_file:", FLAGS.weights_path)