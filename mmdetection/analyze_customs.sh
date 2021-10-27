#!/bin/bash

# https://github.com/open-mmlab/mmdetection/blob/master/docs/useful_tools.md
LOGS_DIR=checkpoints/custom/tf/

for d in $LOGS_DIR*/ ; do
    mkdir $d/analysis
    echo "$LOGS_DIR $d"
    LATEST_LOG=$(ls $d/*.log.json | sort -V | tail -n 1)

    python3 tools/analysis_tools/analyze_logs.py plot_curve $LATEST_LOG \
    --keys loss_rpn_cls loss_rpn_bbox loss_cls loss_bbox loss_mask loss \
    --legend loss_rpn_cls loss_rpn_bbox loss_cls loss_bbox loss_mask loss \
    --out $d/analysis/losses.jpg

    python3 tools/analysis_tools/analyze_logs.py plot_curve $LATEST_LOG \
    --keys acc \
    --legend acc \
    --out $d/analysis/acc.jpg

    # # create log from val because its strictly made for 1 val 1 train
    cat $LATEST_LOG | grep val > $d/latest.log.val.json

    cat $d/latest.log.val.json | grep mAP_copypaste | tail -n 1

    python3 tools/analysis_tools/analyze_logs.py plot_curve $d/latest.log.val.json \
    --keys bbox_mAP bbox_mAP_50 bbox_mAP_75 bbox_mAP_s bbox_mAP_m bbox_mAP_l segm_mAP segm_mAP_50 segm_mAP_75 segm_mAP_s segm_mAP_m segm_mAP_l \
    --legend bbox_mAP bbox_mAP_50 bbox_mAP_75 bbox_mAP_s bbox_mAP_m bbox_mAP_l segm_mAP segm_mAP_50 segm_mAP_75 segm_mAP_s segm_mAP_m segm_mAP_l \
    --out $d/analysis/performance.jpg


done
# TODO visualize topk?