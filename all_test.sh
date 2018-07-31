#!/bin/bash

train_script=./faster_rcnn/test_net.py

export CUDA_VISIBLE_DEVICES=0
iters_max=150000  ## this one must be consistent with number in train file
interval=5000    ## must be the same in faster_rcnn_kitti.yml file
name_anchors=anchor_75_scale_25_aspect_ratio_3


back_slash=/
postfix_file=.txt

cfg=./experiments/cfgs/faster_rcnn_kitti.yml
imdb=kittivoc_test
#weights=./output/faster_rcnn_kitti/kittivoc_train/VGGnet_fast_rcnn_iter_95000.ckpt
network=Resnet101_train # VGGnet_test
wait1=0
gpu=0
exe=python
#exe=~/tf18_gpu/bin/pudb

### the training model
string1_1=./output/faster_rcnn_kitti/kittivoc_train/
string1_2=/VGGnet_fast_rcnn_iter_
prefix_weight=$string1_1$name_anchors$string1_2
postfix_weight=.ckpt

### output of training model to generate the prediction txt files
string2_1=./output/faster_rcnn_kitti/kittivoc_test/
string2_2=/VGGnet_fast_rcnn_iter_
label_dir=/home/b/Kitti/testing/label_2/
prefix_prediction=$string2_1$name_anchors$string2_2

### the output of results runing evaluation code of kitti official one
every_one=$back_slash$name_anchors$postfix_file
output_dir=$string2_1$name_anchors
output_result_dir=$string2_1$name_anchors$every_one

for ((i=interval; i<=iters_max; i=i+interval));do
	echo $i
	weights=$prefix_weight$i$postfix_weight
	run_train="$exe $train_script --gpu $gpu --weights $weights --imdb $imdb --cfg $cfg --network $network --output_dir $output_dir"
	$run_train

	prediction_dir=$prefix_prediction$i
	
	echo $i | tee -a $output_result_dir
	./lib/kitti_native_eval/evaluate_object_3d_offline $label_dir $prediction_dir | tee -a $output_result_dir
	back_dir=$prefix_prediction$i$every_one
	cp $output_result_dir $back_dir
done
