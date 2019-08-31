#!/bin/bash

./im2rec.py imagenet_val /home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/val --list --recursive --pass-through
./im2rec.py imagenet_val.lst /home/maksim/data/ILSVRC2017_CLS-LOC/ILSVRC/Data/CLS-LOC/val --recursive --pass-through
