#!/bin/bash

#
# 2 classes
#

python3 -W ignore tools/train.py \
configs/gustav/kungbib-cascade-mask.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/vanilla

python3 tools/test.py \
configs/gustav/kungbib-cascade-mask.py \
checkpoints/custom/tf/vanilla/latest.pth \
--work-dir=checkpoints/custom/tf/vanilla \
--eval segm bbox \
--show-dir checkpoints/custom/tf/vanilla/analysis

#
# 1 class
#

python3 -W ignore tools/train.py \
configs/gustav/kungbib-cascade-mask-1c.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/vanilla-1c

python3 tools/test.py \
configs/gustav/kungbib-cascade-mask-1c.py \
checkpoints/custom/tf/vanilla-1c/latest.pth \
--work-dir=checkpoints/custom/tf/vanilla-1c \
--eval segm bbox \
--show-dir checkpoints/custom/tf/vanilla-1c/analysis \
--show-score-thr 0.6