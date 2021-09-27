#!/bin/bash

python3 -W ignore tools/train.py \
configs/gustav/kungbib-cascade-mask.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/vanilla