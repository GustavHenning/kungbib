#!/bin/bash

CONFIG_NAME=$1
ENCODER=$2
DIMENSIONS=$3
MODEL_NAME=$4
IMAGES_SCALE_MODIFIER=$5


BASE_CHANNELS=3
TOTAL_CHANNELS="$((DIMENSIONS+BASE_CHANNELS))"

COMMON_CONFIG_NAME="kungbib-cascade-mask"
CONFIG_NAME_IDENTIFYER=${CONFIG_NAME#"$COMMON_CONFIG_NAME"}
echo "$CONFIG_NAME_IDENTIFYER"
MODEL_DIR=$(echo "${ENCODER}_${DIMENSIONS}_${MODEL_NAME}${CONFIG_NAME_IDENTIFYER}_${IMAGES_SCALE_MODIFIER}" | tr "/" "_")

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

TRAIN_IMG_SIZES="$(img_scales 0 $IMAGES_SCALE_MODIFIER)"
TEST_IMG_SIZES="$(img_scales 1 $IMAGES_SCALE_MODIFIER)"

echo "Running training for $CONFIG_NAME found in checkpoints/custom/tf/$MODEL_DIR"
echo "Settings are encoder=$ENCODER, dimensions=$DIMENSIONS, model_name=$MODEL_NAME, total channels are $TOTAL_CHANNELS"


python3 tools/train.py \
configs/gustav/$CONFIG_NAME.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/$MODEL_DIR \
--cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
img_norm_cfg.extra_dims=$DIMENSIONS \
data.train.dataset.pipeline.2.extra_dims=$DIMENSIONS \
data.train.dataset.pipeline.6.dimensions=$DIMENSIONS \
data.train.dataset.pipeline.6.encoder=$ENCODER \
data.train.dataset.pipeline.6.model_name=$MODEL_NAME \
data.test.pipeline.1.transforms.2.extra_dims=$DIMENSIONS \
data.test.pipeline.1.transforms.4.encoder=$ENCODER \
data.test.pipeline.1.transforms.4.dimensions=$DIMENSIONS \
data.test.pipeline.1.transforms.4.model_name=$MODEL_NAME \
data.val.pipeline.1.transforms.2.extra_dims=$DIMENSIONS \
data.val.pipeline.1.transforms.4.encoder=$ENCODER \
data.val.pipeline.1.transforms.4.dimensions=$DIMENSIONS \
data.val.pipeline.1.transforms.4.model_name=$MODEL_NAME \
train_pipeline.2.extra_dims=$DIMENSIONS \
test_pipeline.1.transforms.2.extra_dims=$DIMENSIONS \
train_pipeline.3.img_scale=$TRAIN_IMG_SIZES \
test_pipeline.1.img_scale=$TEST_IMG_SIZES \
data.train.dataset.pipeline.3.img_scale=$TRAIN_IMG_SIZES \
data.val.pipeline.1.img_scale=$TEST_IMG_SIZES \
data.test.pipeline.1.img_scale=$TEST_IMG_SIZES

echo "Training done."

cd checkpoints/custom/tf/$MODEL_DIR

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

python3 -W ignore tools/test.py \
checkpoints/custom/tf/$MODEL_DIR/$CONFIG_NAME.py \
checkpoints/custom/tf/$MODEL_DIR/latest.pth \
--work-dir=checkpoints/custom/tf/$MODEL_DIR \
--eval segm bbox 
# --show-dir checkpoints/custom/tf/$MODEL_DIR/analysis \
# --show-score-thr 0.8

#
# Produce graphs for each dataset
# 
DATASETS=(dn-2010-2020 dn-svd-2001-2004 ab-ex-2001-2004)

for dataset in "${DATASETS[@]}"
do
    echo "$dataset"
    python3 tools/test.py \
    checkpoints/custom/tf/$MODEL_DIR/$CONFIG_NAME.py \
    checkpoints/custom/tf/$MODEL_DIR/latest.pth \
    --cfg-options data.test.img_prefix="/data/gustav/datalab_data/model/$dataset/" \
    data.test.ann_file="/data/gustav/datalab_data/model/$dataset/test_annotations.json" \
    --format-only \
    --options "jsonfile_prefix=./checkpoints/custom/tf/$MODEL_DIR/results/$dataset-results"

    python3 tools/analysis_tools/coco_error_analysis.py \
    ./checkpoints/custom/tf/$MODEL_DIR/results/$dataset-results.bbox.json \
    ./checkpoints/custom/tf/$MODEL_DIR/results/$dataset \
    --ann=/data/gustav/datalab_data/model/$dataset/test_annotations.json \
    --extraplots \
    --areas 80089 360000 10000000000

    python3 tools/analysis_tools/coco_error_analysis.py \
    ./checkpoints/custom/tf/$MODEL_DIR/results/$dataset-results.segm.json \
    ./checkpoints/custom/tf/$MODEL_DIR/results/$dataset \
    --ann=/data/gustav/datalab_data/model/$dataset/test_annotations.json \
    --types='segm' \
    --extraplots \
    --areas 80089 360000 10000000000
done

            #
            #   1 class
            #

ONE_CLASS_STRING="-1c"
CONFIG_NAME+=$ONE_CLASS_STRING
MODEL_DIR+=$ONE_CLASS_STRING

if [ ! -f configs/gustav/$CONFIG_NAME.py ]; then
    echo "$CONFIG_NAME.py not found! Assuming no 1 class model should be trained"
    exit 0
else 
    echo "$CONFIG_NAME.py found! Training and evaluating for 1 class..."
fi

python3 tools/train.py \
configs/gustav/$CONFIG_NAME.py \
--seed=0 \
--work-dir=checkpoints/custom/tf/$MODEL_DIR \
--cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
img_norm_cfg.extra_dims=$DIMENSIONS \
data.train.dataset.pipeline.2.extra_dims=$DIMENSIONS \
data.train.dataset.pipeline.6.dimensions=$DIMENSIONS \
data.train.dataset.pipeline.6.encoder=$ENCODER \
data.train.dataset.pipeline.6.model_name=$MODEL_NAME \
data.test.pipeline.1.transforms.2.extra_dims=$DIMENSIONS \
data.test.pipeline.1.transforms.4.encoder=$ENCODER \
data.test.pipeline.1.transforms.4.dimensions=$DIMENSIONS \
data.test.pipeline.1.transforms.4.model_name=$MODEL_NAME \
data.val.pipeline.1.transforms.2.extra_dims=$DIMENSIONS \
data.val.pipeline.1.transforms.4.encoder=$ENCODER \
data.val.pipeline.1.transforms.4.dimensions=$DIMENSIONS \
data.val.pipeline.1.transforms.4.model_name=$MODEL_NAME \
train_pipeline.2.extra_dims=$DIMENSIONS \
test_pipeline.1.transforms.2.extra_dims=$DIMENSIONS \
train_pipeline.3.img_scale=$TRAIN_IMG_SIZES \
test_pipeline.1.img_scale=$TEST_IMG_SIZES \
data.train.dataset.pipeline.3.img_scale=$TRAIN_IMG_SIZES \
data.val.pipeline.1.img_scale=$TEST_IMG_SIZES \
data.test.pipeline.1.img_scale=$TEST_IMG_SIZES

echo "Training done."

cd checkpoints/custom/tf/$MODEL_DIR

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

python3 -W ignore tools/test.py \
checkpoints/custom/tf/$MODEL_DIR/$CONFIG_NAME.py \
checkpoints/custom/tf/$MODEL_DIR/latest.pth \
--work-dir=checkpoints/custom/tf/$MODEL_DIR \
--eval segm bbox 
# --show-dir checkpoints/custom/tf/$MODEL_DIR/analysis \
# --show-score-thr 0.8

#
# Produce graphs for each dataset
# 
DATASETS=(dn-2010-2020 dn-svd-2001-2004 ab-ex-2001-2004)

for dataset in "${DATASETS[@]}"
do
    echo "$dataset"
    python3 tools/test.py \
    checkpoints/custom/tf/$MODEL_DIR/$CONFIG_NAME.py \
    checkpoints/custom/tf/$MODEL_DIR/latest.pth \
    --cfg-options data.test.img_prefix="/data/gustav/datalab_data/model/$dataset/" \
    data.test.ann_file="/data/gustav/datalab_data/model/$dataset/test_1c_annotations.json" \
    --format-only \
    --options "jsonfile_prefix=./checkpoints/custom/tf/$MODEL_DIR/results/$dataset-results"

    python3 tools/analysis_tools/coco_error_analysis.py \
    ./checkpoints/custom/tf/$MODEL_DIR/results/$dataset-results.bbox.json \
    ./checkpoints/custom/tf/$MODEL_DIR/results/$dataset \
    --ann=/data/gustav/datalab_data/model/$dataset/test_1c_annotations.json \
    --extraplots \
    --areas 80089 360000 10000000000

    python3 tools/analysis_tools/coco_error_analysis.py \
    ./checkpoints/custom/tf/$MODEL_DIR/results/$dataset-results.segm.json \
    ./checkpoints/custom/tf/$MODEL_DIR/results/$dataset \
    --ann=/data/gustav/datalab_data/model/$dataset/test_1c_annotations.json \
    --types='segm' \
    --extraplots \
    --areas 80089 360000 10000000000
done