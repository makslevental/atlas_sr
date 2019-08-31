#!/bin/bash
export PYTHONPATH=../
python -m torch.distributed.launch \
  --nproc_per_node=1 \
  train_srgan_dali.py \
  --train-mx-path=/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/imagenet_rec.rec \
  --train-mx-index-path=/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/imagenet_rec.idx \
  --val-mx-path=/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/imagenet_val.rec \
  --val-mx-index-path=/home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/imagenet_val.idx \
  --checkpoint-dir=/home/maksim/dev_projects/atlas_sr/checkpoints/srgan2 \
  --batch-size=16 \
  --lr=1e-4 \
  --crop-size=88 \
  --prof

