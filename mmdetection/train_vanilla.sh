#!/bin/bash
DATASET=poly-dn-2010-2020-729

#
# 2 classes
#

python3 -W ignore tools/train.py \
configs/gustav/kungbib-cascade-mask.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/vanilla

rm -rf checkpoints/custom/tf/vanilla/analysis && mkdir -p checkpoints/custom/tf/vanilla/analysis

python3 tools/test.py \
configs/gustav/kungbib-cascade-mask.py \
checkpoints/custom/tf/vanilla/latest.pth \
--work-dir=checkpoints/custom/tf/vanilla \
--eval segm bbox \
--show-dir checkpoints/custom/tf/vanilla/analysis \
--show-score-thr 0.8

python3 tools/test.py \
configs/gustav/kungbib-cascade-mask.py \
checkpoints/custom/tf/vanilla/latest.pth \
--format-only \
--options "jsonfile_prefix=./checkpoints/custom/tf/vanilla/results"

python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/vanilla/results.bbox.json \
./checkpoints/custom/tf/vanilla/results \
--ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
--extraplots \
--areas 80000 360000 10000000000


python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/vanilla/results.segm.json \
./checkpoints/custom/tf/vanilla/results \
--ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
--types='segm' \
--extraplots \
--areas 80000 360000 10000000000

#
# 1 class
#

# python3 -W ignore tools/train.py \
# configs/gustav/kungbib-cascade-mask-1c.py \
# --seed=0 \
# --work-dir=checkpoints/custom/tf/vanilla-1c

# rm -rf checkpoints/custom/tf/vanilla-1c/analysis && mkdir -p checkpoints/custom/tf/vanilla-1c/analysis

# python3 tools/test.py \
# configs/gustav/kungbib-cascade-mask-1c.py \
# checkpoints/custom/tf/vanilla-1c/latest.pth \
# --work-dir=checkpoints/custom/tf/vanilla-1c \
# --eval segm bbox \
# --show-dir checkpoints/custom/tf/vanilla-1c/analysis \
# --show-score-thr 0.8

# python3 tools/test.py \
# configs/gustav/kungbib-cascade-mask-1c.py \
# checkpoints/custom/tf/vanilla-1c/latest.pth \
# --format-only \
# --options "jsonfile_prefix=./checkpoints/custom/tf/vanilla-1c/results"

# python3 tools/analysis_tools/coco_error_analysis.py \
# ./checkpoints/custom/tf/vanilla-1c/results.bbox.json \
# ./checkpoints/custom/tf/vanilla-1c/results \
# --ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
# --extraplots \
# --areas 80000 360000 10000000000


# python3 tools/analysis_tools/coco_error_analysis.py \
# ./checkpoints/custom/tf/vanilla-1c/results.segm.json \
# ./checkpoints/custom/tf/vanilla-1c/results \
# --ann=/data/gustav/datalab_data/$DATASET/test_annotations.json \
# --types='segm' \
# --extraplots \
# --areas 80000 360000 10000000000