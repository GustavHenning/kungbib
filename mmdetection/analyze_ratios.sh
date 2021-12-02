#!/bin/bash

runs=(1 2 3 4 5)

for i in "${runs[@]}"
do
    LOGS_DIR=checkpoints/custom/tf/ratio/run_$i/
    for d in $LOGS_DIR*/ ; do
        bash analyze.sh $d
    done
done