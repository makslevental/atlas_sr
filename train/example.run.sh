#!/bin/bash
export PYTHONPATH=../
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train_srgan_dali.py \
  --experiment-name=check_batch_sizes_distributed \
  --channels=3 \
  --train-mx-path=/home/maksim/data/VOC2012/voc_train.rec \
  --train-mx-index-path=/home/maksim/data/VOC2012/voc_train.idx \
  --val-mx-path=/home/maksim/data/VOC2012/voc_val.rec \
  --val-mx-index-path=/home/maksim/data/VOC2012/voc_val.idx \
  --checkpoint-dir=/home/maksim/dev_projects/atlas_sr/checkpoints \
  --upscale-factor=2 \
  --epochs=100 \
  --batch-size=128 \
  --g-lr=1e-4 \
  --d-lr=1e-4 \
  --crop-size=88 \
  --workers=4 \
  --use-apex
