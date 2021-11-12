#!/bin/bash

LOGS_DIR=checkpoints/custom/tf/

for d in $LOGS_DIR*/ ; do
    bash analyze.sh $d
done