#!/bin/bash

# CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/oln_box/oln_box_local.py $2 --eval bbox --eval-options jsonfile_prefix="."
# CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/mask_rcnn/mask_rcnn_r50_fpn_local.py $2 --eval bbox --eval-options jsonfile_prefix="."
# CUDA_VISIBLE_DEVICES=$1 python tools/test.py configs/oln_box/class_agn_faster_rcnn.py $2 --eval bbox --eval-options jsonfile_prefix="."

# --show-dir work_dirs/mask_rcnn/result_image


#!/usr/bin/env bash

# CONFIG=configs/mask_rcnn/mask_rcnn_r50_fpn_local.py
CONFIG=configs/oln_box/oln_box_local.py
CHECKPOINT=$1
GPUS=$2
PORT=${PORT:-29502}
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} \
    --eval bbox --eval-options jsonfile_prefix="./oln_nonVOC-result" \