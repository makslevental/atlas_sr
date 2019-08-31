#!/bin/bash

./im2rec.py $1 $2 --list --recursive --pass-through
./im2rec.py $1.lst $2 --recursive --pass-through
