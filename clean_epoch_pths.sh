#!/bin/bash

cd ~/git/kungbib/mmdetection/checkpoints/custom/tf/

for file in $(find . -name "epoch*"); do
    echo $file
    rm -f $file
done

cd -