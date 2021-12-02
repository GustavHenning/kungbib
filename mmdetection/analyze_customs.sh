#!/bin/bash

LOGS_DIR=checkpoints/custom/tf/

for d in $LOGS_DIR*/ ; do
    if [[ $d == *ratio* ]]; then
        continue
    fi
    bash analyze.sh $d
done