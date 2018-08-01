#!/bin/bash

train_script=./faster_rcnn/train_net.py

export CUDA_VISIBLE_DEVICES=1


cfg=./experiments/cfgs/faster_rcnn_kitti.yml
iters=150000
imdb=kittivoc_train
weights=./data/pretrain_model/resnet_v1_101.ckpt
network=Resnet101_train
gpu=1
restore=1

output_dir=./output/faster_rcnn_kitti/kittivoc_train/anchor_55_scale_11_aspect_ratio_5
exe=python
#exe=pudb



run_train="$exe $train_script --gpu $gpu --weights $weights --imdb $imdb --iters $iters --cfg $cfg --network $network --restore $restore --output_dir $output_dir"

$run_train
