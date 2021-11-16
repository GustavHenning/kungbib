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

if [[ $(cat $d/latest.log.val.json | grep -v env_info | grep 0_bbox_mAP | wc -l) -gt 0 ]]; then
    cp $d/latest.log.val.json $d/latest.log.val.formatted.json
    sed -i -e 's/0_/in_/g' $d/latest.log.val.formatted.json
    sed -i -e 's/1_/near_/g' $d/latest.log.val.formatted.json
    sed -i -e 's/2_/out_/g' $d/latest.log.val.formatted.json

    python3 tools/analysis_tools/analyze_logs.py plot_curve $d/latest.log.val.formatted.json \
    --keys \
    in_bbox_mAP in_segm_mAP  \
    near_bbox_mAP near_segm_mAP  \
    out_bbox_mAP  out_segm_mAP  \
    --legend \
    in_bbox_mAP in_segm_mAP  \
    near_bbox_mAP near_segm_mAP  \
    out_bbox_mAP  out_segm_mAP  \
    --out $d/analysis/performance_all.jpg

    python3 tools/analysis_tools/analyze_logs.py plot_curve $d/latest.log.val.formatted.json \
    --keys \
    in_bbox_mAP in_bbox_mAP_50 in_bbox_mAP_75 in_bbox_mAP_s in_bbox_mAP_m in_bbox_mAP_l in_segm_mAP in_segm_mAP_50 in_segm_mAP_75 in_segm_mAP_s in_segm_mAP_m in_segm_mAP_l \
    --legend \
    in_bbox_mAP in_bbox_mAP_50 in_bbox_mAP_75 in_bbox_mAP_s in_bbox_mAP_m in_bbox_mAP_l in_segm_mAP in_segm_mAP_50 in_segm_mAP_75 in_segm_mAP_s in_segm_mAP_m in_segm_mAP_l \
    --out $d/analysis/performance_in.jpg

    python3 tools/analysis_tools/analyze_logs.py plot_curve $d/latest.log.val.formatted.json \
    --keys \
    near_bbox_mAP near_bbox_mAP_50 near_bbox_mAP_75 near_bbox_mAP_s near_bbox_mAP_m near_bbox_mAP_l near_segm_mAP near_segm_mAP_50 near_segm_mAP_75 near_segm_mAP_s near_segm_mAP_m near_segm_mAP_l \
    --legend \
    near_bbox_mAP near_bbox_mAP_50 near_bbox_mAP_75 near_bbox_mAP_s near_bbox_mAP_m near_bbox_mAP_l near_segm_mAP near_segm_mAP_50 near_segm_mAP_75 near_segm_mAP_s near_segm_mAP_m near_segm_mAP_l \
    --out $d/analysis/performance_near.jpg

    python3 tools/analysis_tools/analyze_logs.py plot_curve $d/latest.log.val.formatted.json \
    --keys \
    out_bbox_mAP out_bbox_mAP_50 out_bbox_mAP_75 out_bbox_mAP_s out_bbox_mAP_m out_bbox_mAP_l out_segm_mAP out_segm_mAP_50 out_segm_mAP_75 out_segm_mAP_s out_segm_mAP_m out_segm_mAP_l \
    --legend \
    out_bbox_mAP out_bbox_mAP_50 out_bbox_mAP_75 out_bbox_mAP_s out_bbox_mAP_m out_bbox_mAP_l out_segm_mAP out_segm_mAP_50 out_segm_mAP_75 out_segm_mAP_s out_segm_mAP_m out_segm_mAP_l \
    --out $d/analysis/performance_out.jpg
else
    if [[ $(cat $d/latest.log.val.json | grep -v env_info | grep bbox_mAP | wc -l) -eq 0 ]]; then
        echo "$d has no latest eval results..."
    else
        python3 tools/analysis_tools/analyze_logs.py plot_curve $d/latest.log.val.json \
        --keys bbox_mAP bbox_mAP_50 bbox_mAP_75 bbox_mAP_s bbox_mAP_m bbox_mAP_l segm_mAP segm_mAP_50 segm_mAP_75 segm_mAP_s segm_mAP_m segm_mAP_l \
        --legend bbox_mAP bbox_mAP_50 bbox_mAP_75 bbox_mAP_s bbox_mAP_m bbox_mAP_l segm_mAP segm_mAP_50 segm_mAP_75 segm_mAP_s segm_mAP_m segm_mAP_l \
        --out $d/analysis/performance.jpg
    fi
fi