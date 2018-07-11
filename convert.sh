#!/bin/bash

script=./experiments/scripts/kitti2pascalvoc.py
kitti_dir=./data/KITTI
out_dir=./data/KITTIVOC
exe=python

## this is for testing data

run_code="$exe $script --kitti $kitti_dir --out $out_dir"

$run_code

