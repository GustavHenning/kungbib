#!/bin/bash
# TODO average over runs

LOGS_DIR=checkpoints/custom/tf/ratio/vanilla_1_*

echo "in"
for d in $LOGS_DIR*/ ; do
    if [[ $d == *-1c* ]]; then
        continue
    fi
    #echo "$d - in"
    segm=$(tail -n 1 $d/latest.log.val.formatted.json | jq .in_segm_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
    bbox=$(tail -n 1 $d/latest.log.val.formatted.json | jq .in_bbox_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
    echo "$segm & $bbox"
done
echo "near"
for d in $LOGS_DIR*/ ; do
    if [[ $d == *-1c* ]]; then
        continue
    fi
    #echo "$d - near"
    segm=$(tail -n 1 $d/latest.log.val.formatted.json | jq .near_segm_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
    bbox=$(tail -n 1 $d/latest.log.val.formatted.json | jq .near_bbox_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
    echo "$segm & $bbox"
done
echo "out"
for d in $LOGS_DIR*/ ; do
    if [[ $d == *-1c* ]]; then
        continue
    fi
    #echo "$d - out"
    segm=$(tail -n 1 $d/latest.log.val.formatted.json | jq .out_segm_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
    bbox=$(tail -n 1 $d/latest.log.val.formatted.json | jq .out_bbox_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
    echo "$segm & $bbox"
done

echo "100%"
echo "in"
segm=$(tail -n 1 checkpoints/custom/tf/vanilla_1/latest.log.val.formatted.json | jq .in_segm_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
bbox=$(tail -n 1 checkpoints/custom/tf/vanilla_1/latest.log.val.formatted.json | jq .in_bbox_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
echo "$segm & $bbox"

echo "near"
segm=$(tail -n 1 checkpoints/custom/tf/vanilla_1/latest.log.val.formatted.json | jq .near_segm_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
bbox=$(tail -n 1 checkpoints/custom/tf/vanilla_1/latest.log.val.formatted.json | jq .near_bbox_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
echo "$segm & $bbox"

echo "out"
segm=$(tail -n 1 checkpoints/custom/tf/vanilla_1/latest.log.val.formatted.json | jq .out_segm_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
bbox=$(tail -n 1 checkpoints/custom/tf/vanilla_1/latest.log.val.formatted.json | jq .out_bbox_mAP_copypaste | cut -d' ' -f1,2,3 | tr -d '\"' | sed 's/ / \& /g')
echo "$segm & $bbox"
