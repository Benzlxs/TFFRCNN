#!/bin/bash

train_script=./faster_rcnn/train_net.py

export CUDA_VISIBLE_DEVICES=1


cfg=./experiments/cfgs/faster_rcnn_kitti.yml
iters=150000
imdb=kittivoc_train
weights=./data/pretrain_model/VGG_imagenet.npy
network=VGGnet_train
gpu=0
restore=0

output_dir=./output/faster_rcnn_kitti/kittivoc_train/anchor_84_scale_28_aspect_ratio_3
exe=python
#exe=~/tf18_gpu/bin/pudb



run_train="$exe $train_script --gpu $gpu --weights $weights --imdb $imdb --iters $iters --cfg $cfg --network $network --restore $restore --output_dir $output_dir"

$run_train
