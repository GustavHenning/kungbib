#!/bin/bash

function confusions() {
where=$1
cfg_name=$2
ann_file=$3

# produce results.pkl
python3 tools/test.py $where/$cfg_name $where/latest.pth --work-dir=$where --eval segm bbox --out $where/results.pkl \
--cfg-options data.test.img_prefix=/data/gustav/datalab_data/model/dn-2010-2020/ \
data.test.ann_file=/data/gustav/datalab_data/model/dn-2010-2020/$ann_file


# produce confusion matrix
python3 tools/analysis_tools/confusion_matrix.py $where/$cfg_name $where/results.pkl $where/analysis \
--cfg-options data.test.img_prefix=/data/gustav/datalab_data/model/dn-2010-2020/ \
data.test.ann_file=/data/gustav/datalab_data/model/dn-2010-2020/$ann_file


}



#confusions checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75 kungbib-cascade-mask.py test_annotations.json
confusions checkpoints/custom/tf/ratio/run_1/vanilla_1_1.0 kungbib-cascade-mask.py test_annotations.json
confusions checkpoints/custom/tf/ratio/run_1/vanilla_1_0.75-1c kungbib-cascade-mask-1c.py test_1c_annotations.json
confusions checkpoints/custom/tf/ratio/run_1/vanilla_1_1.0-1c kungbib-cascade-mask-1c.py test_1c_annotations.json
