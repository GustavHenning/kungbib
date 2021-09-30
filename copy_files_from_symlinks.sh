#!/bin/bash

cp -rf /home/gush/git/kungbib/mmdetection/configs/gustav/* /home/gush/git/kungbib/mmdetection/mm_additions/configs/gustav/
cp -rf /home/gush/git/kungbib/mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-tf.py /home/gush/git/kungbib/mmdetection/mm_additions/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-tf.py
cp -rf /home/gush/git/kungbib/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco-tf.py /home/gush/git/kungbib/mmdetection/mm_additions/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco-tf.py
cp -rf /home/gush/git/kungbib/mmdetection/pipelines/text_features.py /home/gush/git/kungbib/mmdetection/mm_additions/pipelines/text_features.py
cp -rf /home/gush/git/kungbib/mmdetection/pipelines/transforms.py /home/gush/git/kungbib/mmdetection/mm_additions/pipelines/transforms.py
cp -rf /home/gush/git/kungbib/mmdetection/pipelines/encoders/* /home/gush/git/kungbib/mmdetection/mm_additions/pipelines/encoders
cp -rf /data/gustav/anaconda3/envs/openmmlab/lib/python3.8/site-packages/mmdet/apis/test.py /home/gush/git/kungbib/mmdetection/mm_additions/tools/test.py