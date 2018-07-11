#!/bin/bash

train_script=./faster_rcnn/test_net.py

cfg=./experiments/cfgs/faster_rcnn_kitti.yml
#iters=160000
imdb=kittivoc_test
weights=./output/faster_rcnn_kitti/kittivoc_train/VGGnet_fast_rcnn_iter_95000.ckpt
network=VGGnet_test
wait1=0
gpu=0
#restore=0
#exe=python
exe=~/tf18_gpu/bin/pudb


run_train="$exe $train_script --gpu $gpu --weights $weights --imdb $imdb --cfg $cfg --network $network"

$run_train
