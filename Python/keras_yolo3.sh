#!/usr/bin/env bash

#python code reference from https://github.com/qqwweee/keras-yolo3.git
#cd keras_yolov3
#wget https://pjreddie.com/media/files/yolov3.weights
#wget https://pjreddie.com/media/files/yolov3-tiny.weights
#cd ..
#python3 keras_yolov3/convert.py keras_yolov3/yolov3.cfg keras_yolov3/yolov3.weights keras_yolov3/yolov3.h5
#python3 keras_yolov3/convert.py keras_yolov3/yolov3-tiny.cfg keras_yolov3/yolov3-tiny.weights keras_yolov3/yolov3-tiny.h5

if [ "$1" == 'yolov3-tiny' ]
then
    echo 'convert yolov3-tiny'
    cd keras_yolov3
    echo 'downloading yolov3-tiny weights'
    wget https://pjreddie.com/media/files/yolov3-tiny.weights
    cd ..
    python3 keras_yolov3/convert.py keras_yolov3/yolov3-tiny.cfg keras_yolov3/yolov3-tiny.weights keras_yolov3/yolov3-tiny.h5
    python3 convert_keras_yolo3.py --model-path keras_yolov3/yolov3-tiny.h5 --anchors-path keras_yolov3/tiny_yolo_anchors.txt --network-path "$2" --weights-path "$3"
else
    echo 'convert yolov3'
    cd keras_yolov3
    echo 'downloading yolov3 weights'
    wget https://pjreddie.com/media/files/yolov3.weights
    cd ..
    python3 keras_yolov3/convert.py keras_yolov3/yolov3.cfg keras_yolov3/yolov3.weights keras_yolov3/yolov3.h5
    python3 convert_keras_yolo3.py --model-path keras_yolov3/yolov3.h5 --anchors-path keras_yolov3/yolo_anchors.txt --network-path "$2" --weights-path "$3"
fi