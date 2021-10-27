#!/bin/bash
DATASET=poly-dn-2010-2020-729

#
# 2 classes
#

python3 -W ignore tools/train.py \
configs/gustav/kungbib-cascade-mask-101.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/vanilla-101

rm -rf checkpoints/custom/tf/vanilla-101/analysis && mkdir -p checkpoints/custom/tf/vanilla-101/analysis

python3 tools/test.py \
configs/gustav/kungbib-cascade-mask-101.py \
checkpoints/custom/tf/vanilla-101/latest.pth \
--work-dir=checkpoints/custom/tf/vanilla-101 \
--eval segm bbox \
--show-dir checkpoints/custom/tf/vanilla-101/analysis \
--show-score-thr 0.8

python3 tools/test.py \
configs/gustav/kungbib-cascade-mask-101.py \
checkpoints/custom/tf/vanilla-101/latest.pth \
--format-only \
--options "jsonfile_prefix=./checkpoints/custom/tf/vanilla-101/results"

python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/vanilla-101/results.bbox.json \
./checkpoints/custom/tf/vanilla-101/results \
--ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
--extraplots \
--areas 80000 360000 10000000000


python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/vanilla-101/results.segm.json \
./checkpoints/custom/tf/vanilla-101/results \
--ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
--types='segm' \
--extraplots \
--areas 80000 360000 10000000000

#
# 1 class
#

# python3 -W ignore tools/train.py \
# configs/gustav/kungbib-cascade-mask-101-1c.py \
# --seed=0 \
# --work-dir=checkpoints/custom/tf/vanilla-101-1c

# rm -rf checkpoints/custom/tf/vanilla-101-1c/analysis && mkdir -p checkpoints/custom/tf/vanilla-101-1c/analysis

# python3 tools/test.py \
# configs/gustav/kungbib-cascade-mask-101-1c.py \
# checkpoints/custom/tf/vanilla-101-1c/latest.pth \
# --work-dir=checkpoints/custom/tf/vanilla-101-1c \
# --eval segm bbox \
# --show-dir checkpoints/custom/tf/vanilla-101-1c/analysis \
# --show-score-thr 0.8

# python3 tools/test.py \
# configs/gustav/kungbib-cascade-mask-101-1c.py \
# checkpoints/custom/tf/vanilla-101-1c/latest.pth \
# --format-only \
# --options "jsonfile_prefix=./checkpoints/custom/tf/vanilla-101-1c/results"

# python3 tools/analysis_tools/coco_error_analysis.py \
# ./checkpoints/custom/tf/vanilla-101-1c/results.bbox.json \
# ./checkpoints/custom/tf/vanilla-101-1c/results \
# --ann=/data/gustav/datalab_data/$DATASET/test_1c_annotations.json \
# --extraplots \
# --areas 80000 360000 10000000000


# python3 tools/analysis_tools/coco_error_analysis.py \
# ./checkpoints/custom/tf/vanilla-101-1c/results.segm.json \
# ./checkpoints/custom/tf/vanilla-101-1c/results \
# --ann=/data/gustav/datalab_data/$DATASET/test_1c_annotations.json \
# --types='segm' \
# --extraplots \
# --areas 80000 360000 10000000000