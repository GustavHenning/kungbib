#!/bin/bash

CONFIG_NAME=$1
CHECKPOINT_DIR_NAME=$2

echo "Running training for $CONFIG_NAME found in checkpoints/custom/tf/$CHECKPOINT_DIR_NAME"

python3 -W ignore tools/train.py \
configs/gustav/$CONFIG_NAME.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/$CHECKPOINT_DIR_NAME

cd checkpoints/custom/tf/$CHECKPOINT_DIR_NAME

# remove previous test results
rm -rf ./analysis && mkdir -p ./analysis
rm -rf ./results && mkdir -p ./results

# save space by removing all epochs that are not latest.pth
ls | grep epoch | grep -v $(basename $(readlink -f latest.pth)) | xargs rm

cd -

python3 tools/test.py \
configs/gustav/$CONFIG_NAME.py \
checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/latest.pth \
--work-dir=checkpoints/custom/tf/$CHECKPOINT_DIR_NAME \
--eval segm bbox \
--show-dir checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/analysis \
--show-score-thr 0.8

python3 tools/test.py \
configs/gustav/$CONFIG_NAME.py \
checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/latest.pth \
--format-only \
--options "jsonfile_prefix=./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results"

python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results.bbox.json \
./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results \
--ann=/data/gustav/datalab_data/model/dn-2010-2020/test_annotations.json \
--extraplots \
--areas 80000 360000 10000000000


python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results.segm.json \
./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results \
--ann=/data/gustav/datalab_data/model/dn-2010-2020/test_annotations.json \
--types='segm' \
--extraplots \
--areas 80000 360000 10000000000


# 1 class

if [ ! -f configs/gustav/$CONFIG_NAME-1c.py ]; then
    echo "$CONFIG_NAME-1c.py not found! Assuming no 1 class model should be trained"
    exit 0
else 
    echo "$CONFIG_NAME-1c.py found! Training and evaluating for 1 class..."
fi


python3 -W ignore tools/train.py \
configs/gustav/$CONFIG_NAME-1c.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c

cd checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c

# remove previous test results
rm -rf ./analysis && mkdir -p ./analysis
rm -rf ./results && mkdir -p ./results

# save space by removing all epochs that are not latest.pth
ls | grep epoch | grep -v $(basename $(readlink -f latest.pth)) | xargs rm

cd -

python3 tools/test.py \
configs/gustav/$CONFIG_NAME-1c.py \
checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c/latest.pth \
--work-dir=checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c \
--eval segm bbox \
--show-dir checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c/analysis \
--show-score-thr 0.8

python3 tools/test.py \
configs/gustav/$CONFIG_NAME-1c.py \
checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c/latest.pth \
--format-only \
--options "jsonfile_prefix=./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c/results"

python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c/results.bbox.json \
./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c/results \
--ann=/data/gustav/datalab_data/model/dn-2010-2020/test_1c_annotations.json \
--extraplots \
--areas 80000 360000 10000000000


python3 tools/analysis_tools/coco_error_analysis.py \
./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c/results.segm.json \
./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME-1c/results \
--ann=/data/gustav/datalab_data/model/dn-2010-2020/test_1c_annotations.json \
--types='segm' \
--extraplots \
--areas 80000 360000 10000000000