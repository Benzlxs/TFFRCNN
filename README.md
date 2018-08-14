# Experiments on how anchor number affect detection results

This repo is forded from https://github.com/CharlesShang/TFFRCNN on implimentation of Faster RCNN[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

Code is run to show how anchor number affect detection results,

![alt text](/all_results.png)


### Acknowledgments: 

1. [TFFRCNN](https://github.com/CharlesShang/TFFRCNN)

### Platform

1. Tensorflow 1.7 and Python 2.7
2. Ubuntu 16.04
3. GPU Titan


### To repeat the experiments

1. Follow instruction on [TFFRCNN] (https://github.com/CharlesShang/TFFRCNN) to install requirements
2. Download the KITTI dataset and creat dataset folder as instructed in [TFFRCNN] (https://github.com/CharlesShang/TFFRCNN)
4. Download pretrained [VGG16 model](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) and locate it in under data folder
5. Design the anchor scale and aspect ratio in the [config.py](/lib/fast_rcnn/config.py)
6. Change the output_dir name in [kitti_train.sh](/kitti_train.sh) and run [kitti_train.sh](/kitti_train.sh)
7. change the name_anchor in [all_test.sh](/all_test.sh) and run [all_test.sh](/all_test.sh)

### Plotting
The plotting result code is located under experiments/plotting folder and written with matlab, after you finish all the experiments, set the correct directory of your result in the [plotting_all_data.m](/experiments/plotting/plotting_all_data.m). [Our experimental results](/data/experiments_results) have been uploaded under the data folder



### Demo


### Download list

1. [VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM)


### Training on KITTI detection dataset

1. Download the KITTI detection dataset

    ```
    http://www.cvlibs.net/datasets/kitti/eval_object.php
    ```

2. Extract all of these tar into `./TFFRCNN/data/` and the directory structure looks like this:
    
    ```
    KITTI
        |-- training
                |-- image_2
                    |-- [000000-007480].png
                |-- label_2
                    |-- [000000-007480].txt
        |-- testing
                |-- image_2
                    |-- [000000-007517].png
                |-- label_2
                    |-- [000000-007517].txt
    ```

3. Convert KITTI into Pascal VOC format
    
    ```Shell
    cd $TFFRCNN
    ./experiments/scripts/kitti2pascalvoc.py \
    --kitti $TFFRCNN/data/KITTI --out $TFFRCNN/data/KITTIVOC
    ```

4. The output directory looks like this:

    ```
    KITTIVOC
        |-- Annotations
                |-- [000000-007480].xml
        |-- ImageSets
                |-- Main
                    |-- [train|val|trainval].txt
        |-- JPEGImages
                |-- [000000-007480].jpg
    ```

5. Training on `KITTIVOC` is just like on Pascal VOC 2007

    ```Shell
    python ./faster_rcnn/train_net.py \
    --gpu 0 \
    --weights ./data/pretrain_model/VGG_imagenet.npy \
    --imdb kittivoc_train \
    --iters 160000 \
    --cfg ./experiments/cfgs/faster_rcnn_kitti.yml \
    --network VGGnet_train
    ```


