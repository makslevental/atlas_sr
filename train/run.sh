#!/bin/bash
export PYTHONPATH=../
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train_resnet_dali.py \
  --mx-path=/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/imagenet_rec.rec \
  --mx-index-path=/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/imagenet_rec.idx
