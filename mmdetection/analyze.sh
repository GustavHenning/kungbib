#!/bin/bash
d=$1 # directory of checkpoint (workdir)

mkdir -p $d/analysis


if [ $(ls $d/*.log.json | wc -l) -eq 0 ]; then
    echo "No logs found, skipping..."
    exit 0
fi

echo "$d"
LATEST_LOG=$(ls $d/*.log.json | sort -V | tail -n 1)
echo "$?"
echo $LATEST_LOG

if [[ $(cat $LATEST_LOG | grep -v env_info | wc -l) -gt 0 ]]; then
    python3 tools/analysis_tools/analyze_logs.py plot_curve $LATEST_LOG \
    --keys loss_rpn_cls loss_rpn_bbox loss_cls loss_bbox loss_mask loss \
    --legend loss_rpn_cls loss_rpn_bbox loss_cls loss_bbox loss_mask loss \
    --out $d/analysis/losses.jpg
else
    echo "$d has 0 lines of loss... Maybe model errors?"
fi

if [[ $(cat $LATEST_LOG | grep -v env_info | grep acc | wc -l) -gt 0 ]]; then
    python3 tools/analysis_tools/analyze_logs.py plot_curve $LATEST_LOG \
    --keys acc \
    --legend acc \
    --out $d/analysis/acc.jpg
else
    echo "$d has 0 lines of acc.."
fi

# create log from val because its strictly made for 1 val 1 train
cat $LATEST_LOG | grep val > $d/latest.log.val.json

if [ ! -f $d/latest.log.val.json ]; then
    echo "no latest log found, skipping"
    exit 0
fi

if [ $(cat $d/latest.log.val.json | grep -v env_info | wc -l) -eq 0 ]; then
    echo "Latest log contains no valid lines, skipping..."
    exit 0
fi

if [[ $(cat $d/latest.log.val.json | grep -v env_info | grep bbox_mAP | wc -l) -eq 0 ]]; then
    echo "$d has no latest eval results..."
else
    python3 tools/analysis_tools/analyze_logs.py plot_curve $d/latest.log.val.json \
    --keys bbox_mAP bbox_mAP_50 bbox_mAP_75 bbox_mAP_s bbox_mAP_m bbox_mAP_l segm_mAP segm_mAP_50 segm_mAP_75 segm_mAP_s segm_mAP_m segm_mAP_l \
    --legend bbox_mAP bbox_mAP_50 bbox_mAP_75 bbox_mAP_s bbox_mAP_m bbox_mAP_l segm_mAP segm_mAP_50 segm_mAP_75 segm_mAP_s segm_mAP_m segm_mAP_l \
    --out $d/analysis/performance.jpg
fi