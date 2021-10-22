#!/bin/bash
DATASET=poly-dn-2010-2020-729

#
# 2 classes
#

python3 -W ignore tools/train.py \
configs/gustav/kungbib-cascade-mask-101-32x8d.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/vanilla-101-32x8d

rm -rf checkpoints/custom/tf/vanilla-101-32x8d/analysis && mkdir -p checkpoints/custom/tf/vanilla-101-32x8d/analysis

python3 tools/test.py \
configs/gustav/kungbib-cascade-mask-101-32x8d.py \
checkpoints/custom/tf/vanilla-101-32x8d/latest.pth \
--work-dir=checkpoints/custom/tf/vanilla-101-32x8d \
--eval segm bbox \
--show-dir checkpoints/custom/tf/vanilla-101-32x8d/analysis \
--show-score-thr 0.8

python3 tools/test.py \
configs/gustav/kungbib-cascade-mask-101-32x8d.py \
checkpoints/custom/tf/vanilla-101-32x8d/latest.pth \
--format-only \
--options "jsonfile_prefix=./checkpoints/custom/tf/vanilla-101-32x8d/results"

python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/vanilla-101-32x8d/results.bbox.json \
./checkpoints/custom/tf/vanilla-101-32x8d/results \
--ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
--extraplots \
--areas 80000 360000 10000000000


python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/vanilla-101-32x8d/results.segm.json \
./checkpoints/custom/tf/vanilla-101-32x8d/results \
--ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
--types='segm' \
--extraplots \
--areas 80000 360000 10000000000

#
# 1 class
#

# python3 -W ignore tools/train.py \
# configs/gustav/kungbib-cascade-mask-101-32x8d-1c.py \
# --seed=0 \
# --work-dir=checkpoints/custom/tf/vanilla-101-32x8d-1c

# rm -rf checkpoints/custom/tf/vanilla-101-32x8d-1c/analysis && mkdir -p checkpoints/custom/tf/vanilla-101-32x8d-1c/analysis

# python3 tools/test.py \
# configs/gustav/kungbib-cascade-mask-101-32x8d-1c.py \
# checkpoints/custom/tf/vanilla-101-32x8d-1c/latest.pth \
# --work-dir=checkpoints/custom/tf/vanilla-101-32x8d-1c \
# --eval segm bbox \
# --show-dir checkpoints/custom/tf/vanilla-101-32x8d-1c/analysis \
# --show-score-thr 0.8

# python3 tools/test.py \
# configs/gustav/kungbib-cascade-mask-101-32x8d-1c.py \
# checkpoints/custom/tf/vanilla-101-32x8d-1c/latest.pth \
# --format-only \
# --options "jsonfile_prefix=./checkpoints/custom/tf/vanilla-101-32x8d-1c/results"

# python3 tools/analysis_tools/coco_error_analysis.py \
# ./checkpoints/custom/tf/vanilla-101-32x8d-1c/results.bbox.json \
# ./checkpoints/custom/tf/vanilla-101-32x8d-1c/results \
# --ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
# --extraplots \
# --areas 80000 360000 10000000000


# python3 tools/analysis_tools/coco_error_analysis.py \
# ./checkpoints/custom/tf/vanilla-101-32x8d-1c/results.segm.json \
# ./checkpoints/custom/tf/vanilla-101-32x8d-1c/results \
# --ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
# --types='segm' \
# --extraplots \
# --areas 80000 360000 10000000000