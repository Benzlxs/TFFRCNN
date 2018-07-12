#!/bin/bash

train_script=./faster_rcnn/test_net.py

export CUDA_VISIBLE_DEVICES=0



cfg=./experiments/cfgs/faster_rcnn_kitti.yml
imdb=kittivoc_test
#weights=./output/faster_rcnn_kitti/kittivoc_train/VGGnet_fast_rcnn_iter_95000.ckpt
network=VGGnet_test
wait1=0
gpu=0
exe=python
#exe=~/tf18_gpu/bin/pudb


prefix_weight=./output/faster_rcnn_kitti/kittivoc_train/anchor_15/VGGnet_fast_rcnn_iter_
postfix_weight=.ckpt

iters_max=320000  ## this one must be consistent with number in train file
interval=5000    ## must be the same in faster_rcnn_kitti.yml file

label_dir=/home/b/Kitti/testing/label_2/
prefix_prediction=/home/b/tf_projects/img/TFFRCNN/output/faster_rcnn_kitti/kittivoc_test/anchor_15/VGGnet_fast_rcnn_iter_

output_dir=./output/faster_rcnn_kitti/kittivoc_test/anchor_15
output_result_dir=./output/faster_rcnn_kitti/kittivoc_test/anchor_15/result_15.txt
every_one=/result_15.txt

for ((i=interval; i<=iters_max; i=i+interval));do
	echo $i
	weights=$prefix_weight$i$postfix_weight
	run_train="$exe $train_script --gpu $gpu --weights $weights --imdb $imdb --cfg $cfg --network $network --output_dir $output_dir"
	$run_train

	prediction_dir=$prefix_prediction$i
	
	echo $i | tee -a $output_result_dir
	./lib/kitti_native_eval/evaluate_object_3d_offline $label_dir $prediction_dir | tee -a $output_result_dir
	#$run_eval
	back_dir=$prefix_prediction$i$every_one
	cp $output_result_dir $back_dir
done
