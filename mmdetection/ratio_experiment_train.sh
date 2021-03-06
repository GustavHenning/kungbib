#!/bin/bash

CONFIG_NAME=$1
CHECKPOINT_DIR_NAME=$2
IMAGE_SCALE_MODIFIER=$3
RATIO_OF_TRAIN=$4
ANN_FILE="/data/gustav/datalab_data/model/dn-2010-2020/train_${RATIO_OF_TRAIN}_annotations.json"
CLASSLESS_ANN_FILE="/data/gustav/datalab_data/model/dn-2010-2020/train_${RATIO_OF_TRAIN}_1c_annotations.json"
echo "$ANN_FILE"

img_scales () {
IS_TRAIN=$1
SCALE=$2

SCALED_WIDTH=$(bc -l <<< "scale=0; (1333*$SCALE)/1")
SCALLED_SMALL_HEIGHT=$(bc -l <<< "scale=0; (640*$SCALE)/1")
SCALED_HEIGHT=$(bc -l <<< "scale=0; (800*$SCALE)/1")

if [ $IS_TRAIN -eq 0 ]; then
    echo "[($SCALED_WIDTH,$SCALLED_SMALL_HEIGHT),($SCALED_WIDTH,$SCALED_HEIGHT)]"
else
    echo "($SCALED_WIDTH,$SCALED_HEIGHT)"
fi
}

TRAIN_IMG_SIZES="$(img_scales 0 $IMAGE_SCALE_MODIFIER)"
TEST_IMG_SIZES="$(img_scales 1 $IMAGE_SCALE_MODIFIER)"

echo "Running training for $CONFIG_NAME found in checkpoints/custom/tf/$CHECKPOINT_DIR_NAME"

python3 -W ignore tools/train.py \
configs/gustav/$CONFIG_NAME.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/$CHECKPOINT_DIR_NAME \
--cfg-options test_pipeline.1.transforms.0.extra_dims=$DIMENSIONS \
train_pipeline.3.img_scale=$TRAIN_IMG_SIZES \
test_pipeline.1.img_scale=$TEST_IMG_SIZES \
data.train.pipeline.3.img_scale=$TRAIN_IMG_SIZES \
data.val.pipeline.1.img_scale=$TEST_IMG_SIZES \
data.test.pipeline.1.img_scale=$TEST_IMG_SIZES \
data.train.dataset.ann_file=$ANN_FILE

cd checkpoints/custom/tf/$CHECKPOINT_DIR_NAME

if [ -f analysis.zip ]; then
    rm -f analysis.zip
fi

if [ -f results.zip ]; then
    rm -f results.zip
fi

zip -r analysis.zip ./analysis
zip -r results.zip ./results

#remove previous test results
rm -rf ./analysis && mkdir -p ./analysis
rm -rf ./results && mkdir -p ./results

#save space by removing all epochs that are not latest.pth
ls | grep epoch | grep -v $(basename $(readlink -f latest.pth)) | xargs rm

cd -

python3 tools/test.py \
checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/$CONFIG_NAME.py \
checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/latest.pth \
--work-dir=checkpoints/custom/tf/$CHECKPOINT_DIR_NAME \
--eval segm bbox 
# --show-dir checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/analysis \
# --show-score-thr 0.8

#
# Produce graphs for each dataset
# 
DATASETS=(dn-2010-2020 dn-svd-2001-2004 ab-ex-2001-2004)

for dataset in "${DATASETS[@]}"
do
    echo "$dataset"
    python3 tools/test.py \
    checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/$CONFIG_NAME.py \
    checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/latest.pth \
    --cfg-options data.test.img_prefix="/data/gustav/datalab_data/model/$dataset/" \
    data.test.ann_file="/data/gustav/datalab_data/model/$dataset/test_annotations.json" \
    --format-only \
    --options "jsonfile_prefix=./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset-results"

    python3 tools/analysis_tools/coco_error_analysis.py \
    ./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset-results.bbox.json \
    ./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset \
    --ann=/data/gustav/datalab_data/model/$dataset/test_annotations.json \
    --extraplots \
    --areas 80089 360000 10000000000

    python3 tools/analysis_tools/coco_error_analysis.py \
    ./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset-results.segm.json \
    ./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset \
    --ann=/data/gustav/datalab_data/model/$dataset/test_annotations.json \
    --types='segm' \
    --extraplots \
    --areas 80089 360000 10000000000
done


ONE_CLASS_STRING="-1c"
CONFIG_NAME+=$ONE_CLASS_STRING
CHECKPOINT_DIR_NAME+=$ONE_CLASS_STRING

# 1 class

if [ ! -f configs/gustav/$CONFIG_NAME.py ]; then
    echo "$CONFIG_NAME.py not found! Assuming no 1 class model should be trained"
    exit 0
else 
    echo "$CONFIG_NAME.py found! Training and evaluating for 1 class..."
fi


python3 -W ignore tools/train.py \
configs/gustav/$CONFIG_NAME.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/$CHECKPOINT_DIR_NAME \
--cfg-options test_pipeline.1.transforms.0.extra_dims=$DIMENSIONS \
train_pipeline.3.img_scale=$TRAIN_IMG_SIZES \
test_pipeline.1.img_scale=$TEST_IMG_SIZES \
data.train.pipeline.3.img_scale=$TRAIN_IMG_SIZES \
data.val.pipeline.1.img_scale=$TEST_IMG_SIZES \
data.test.pipeline.1.img_scale=$TEST_IMG_SIZES \
data.train.dataset.ann_file=$CLASSLESS_ANN_FILE

cd checkpoints/custom/tf/$CHECKPOINT_DIR_NAME

if [ -f analysis.zip ]; then
    rm -f analysis.zip
fi

if [ -f results.zip ]; then
    rm -f results.zip
fi

zip -r analysis.zip ./analysis
zip -r results.zip ./results

# remove previous test results
rm -rf ./analysis && mkdir -p ./analysis
rm -rf ./results && mkdir -p ./results

# save space by removing all epochs that are not latest.pth
ls | grep epoch | grep -v $(basename $(readlink -f latest.pth)) | xargs rm

cd -

# Generate images

python3 tools/test.py \
checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/$CONFIG_NAME.py \
checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/latest.pth \
--work-dir=checkpoints/custom/tf/$CHECKPOINT_DIR_NAME \
--eval segm bbox 
# --show-dir checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/analysis \
# --show-score-thr 0.8

#
# Produce graphs for each dataset
# 
DATASETS=(dn-2010-2020 dn-svd-2001-2004 ab-ex-2001-2004)

for dataset in "${DATASETS[@]}"
do
    echo "$dataset"
    python3 tools/test.py \
    checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/$CONFIG_NAME.py \
    checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/latest.pth \
    --cfg-options data.test.img_prefix="/data/gustav/datalab_data/model/$dataset/" \
    data.test.ann_file="/data/gustav/datalab_data/model/$dataset/test_1c_annotations.json" \
    --format-only \
    --options "jsonfile_prefix=./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset-results"

    python3 tools/analysis_tools/coco_error_analysis.py \
    ./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset-results.bbox.json \
    ./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset \
    --ann=/data/gustav/datalab_data/model/$dataset/test_1c_annotations.json \
    --extraplots \
    --areas 80089 360000 10000000000

    python3 tools/analysis_tools/coco_error_analysis.py \
    ./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset-results.segm.json \
    ./checkpoints/custom/tf/$CHECKPOINT_DIR_NAME/results/$dataset \
    --ann=/data/gustav/datalab_data/model/$dataset/test_1c_annotations.json \
    --types='segm' \
    --extraplots \
    --areas 80089 360000 10000000000
done
