#!/bin/bash

LOGS_DIR=checkpoints/custom/tf/vanilla*

for d in $LOGS_DIR*/ ; do
    bash analyze.sh $d
done