# Experiments on how anchor number affect detection results

This repo is forded from https://github.com/CharlesShang/TFFRCNN on implimentation of Faster RCNN[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.

Code is run to show how anchor number affect detection results,



### Acknowledgments: 

1. [TFFRCNN](https://github.com/CharlesShang/TFFRCNN)



### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/))


### Requirements: hardware

1. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Following instruction on [TFFRCNN] (https://github.com/CharlesShang/TFFRCNN) to install requirements


![alt text](/all_results.png)
https://github.com/Benzlxs/TFFRCNN/edit/master

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


