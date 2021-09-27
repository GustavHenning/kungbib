#!/bin/bash

# https://github.com/open-mmlab/mmdetection/blob/master/docs/useful_tools.md
LOGS_DIR=checkpoints/custom/tf/vanilla

mkdir $LOGS_DIR/analysis

python3 tools/analysis_tools/analyze_logs.py plot_curve $LOGS_DIR/20210927_150915.log.json \
--keys loss_rpn_cls loss_rpn_bbox loss_cls loss_bbox loss_mask loss \
--legend loss_rpn_cls loss_rpn_bbox loss_cls loss_bbox loss_mask loss \
--out $LOGS_DIR/analysis/losses.jpg

python3 tools/analysis_tools/analyze_logs.py plot_curve $LOGS_DIR/20210927_150915.log.json \
--keys acc \
--legend acc \
--out $LOGS_DIR/analysis/acc.jpg

# TODO visualize topk?