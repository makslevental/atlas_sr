#!/bin/bash

EXPERIMENT_NAME=$1
UPSCALE=${2:-2}
EPOCHS=${3:-100}

export PYTHONPATH=../
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train_srgan_dali.py \
  --experiment-name=$EXPERIMENT_NAME \
  --checkpoint-dir=~/data/checkpoints \
  --channels=3 \
  --upscale-factor=$UPSCALE \
  --epochs=$EPOCHS \
  --batch-size=128 \
  --g-lr=1e-3 \
  --d-lr=1e-3 \
  --crop-size=88 \
  --workers=4 \
  --print-freq=5 \
  --use-apex \
  --train-mx-path=~/data/VOC2012/voc_train.rec \
  --train-mx-index-path=~/data/VOC2012/voc_train.idx \
  --val-mx-path=~/data/VOC2012/voc_val.rec \
  --val-mx-index-path=~/data/VOC2012/voc_val.idx \
  --tensorboard-dir=~/data/tensorboard
#  --train-data-dir=~/data/VOC2012/train \
#  --val-data-dir=~/data/VOC2012/val \
