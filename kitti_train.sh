#!/bin/bash

train_script=./faster_rcnn/train_net.py

cfg=./experiments/cfgs/faster_rcnn_kitti.yml
iters=160000
imdb=kittivoc_train
weights=./data/pretrain_model/VGG_imagenet.npy
network=VGGnet_train
gpu=0
restore=0
exe=python


run_train="$exe $train_script --gpu $gpu --weights $weights --imdb $imdb --iters $iters --cfg $cfg --network $network --restore $restore"

$run_train
