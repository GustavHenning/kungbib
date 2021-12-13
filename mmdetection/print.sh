#!/bin/bash

python3 tools/test.py \
checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75/kungbib-cascade-mask.py \
checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75/latest.pth \
--work-dir=checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75 \
--show-dir checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75/analysis \
--show-score-thr 0.8 \
--show

#--cfg-options data.test.img_prefix='/data/gustav/datalab_data/model/dn-2010-2020/' \
#data.test.ann_file='/data/gustav/datalab_data/model/dn-2010-2020/test_annotations.json' \
#/dn-2010-2020
