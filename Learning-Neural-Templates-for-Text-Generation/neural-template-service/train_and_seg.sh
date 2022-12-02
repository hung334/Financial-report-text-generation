#!/bin/sh

python train.py --config train.cfg
python segment.py --config seg.cfg