#
#   Dimensions
#
BASE_CHANNELS=3
# TODO: mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-tf contains img_norm_cfg with extra_dims that needs to correspond to this value
# CHECK THIS FILE BEFORE RUNNING !!!!!!!!!!!!!!!!!!!!!!!1
declare -a dimensions=(384) #3 6 10 50 100 384 
#
#   Model
#
declare -a encoders=("bert") # "doc2vec"

#
#   BERT model names
#
#declare -a bert_models=("KB/bert-base-swedish-cased" "af-ai-center/bert-base-swedish-uncased")

declare -a bert_models=("multi-qa-MiniLM-L6-cos-v1" "all-mpnet-base-v2" "multi-qa-mpnet-base-dot-v1" "all-distilroberta-v1" "all-MiniLM-L12-v2" "multi-qa-distilbert-cos-v1" "all-MiniLM-L6-v2" "paraphrase-multilingual-mpnet-base-v2" "paraphrase-albert-small-v2" "paraphrase-multilingual-MiniLM-L12-v2" "paraphrase-MiniLM-L3-v2" "distiluse-base-multilingual-cased-v1" "distiluse-base-multilingual-cased-v2")

# size 768: "KB/bert-base-swedish-cased" "af-ai-center/bert-base-swedish-uncased" 
# size 384: "multi-qa-MiniLM-L6-cos-v1" "all-mpnet-base-v2" "multi-qa-mpnet-base-dot-v1" "all-distilroberta-v1" "all-MiniLM-L12-v2" "multi-qa-distilbert-cos-v1" "all-MiniLM-L6-v2" "paraphrase-multilingual-mpnet-base-v2" "paraphrase-albert-small-v2" "paraphrase-multilingual-MiniLM-L12-v2" "paraphrase-MiniLM-L3-v2" "distiluse-base-multilingual-cased-v1" "distiluse-base-multilingual-cased-v2"

for e in "${encoders[@]}"
do

    for d in "${dimensions[@]}"
    do

        for m in "${bert_models[@]}"
        do
            TOTAL_CHANNELS="$((d+BASE_CHANNELS))"
            MODEL_DIR=$(echo "${e}_dim_${d}_${m}" | tr "/" "_")

            echo "Training $MODEL_DIR"

            #
            #   2 classes
            #

            python3 -W ignore tools/train.py \
            configs/gustav/kungbib-cascade-mask-tf.py \
            --seed=0 \
            --work-dir=checkpoints/custom/tf/$MODEL_DIR \
            --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
            data.train.pipeline.6.dimensions=$d \
            data.train.pipeline.6.encoder=$e \
            data.train.pipeline.6.model_name=$m \
            data.test.pipeline.1.transforms.4.encoder=$e \
            data.test.pipeline.1.transforms.4.dimensions=$d \
            data.test.pipeline.1.transforms.4.model_name=$m \
            data.val.pipeline.1.transforms.4.encoder=$e \
            data.val.pipeline.1.transforms.4.dimensions=$d \
            data.val.pipeline.1.transforms.4.model_name=$m


            python3 -W ignore tools/test.py \
            configs/gustav/kungbib-cascade-mask-tf.py \
            checkpoints/custom/tf/$MODEL_DIR/latest.pth \
            --work-dir=checkpoints/custom/tf/$MODEL_DIR \
            --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
            data.train.pipeline.6.dimensions=$d \
            data.train.pipeline.6.encoder=$e \
            data.train.pipeline.6.model_name=$m \
            data.test.pipeline.1.transforms.4.encoder=$e \
            data.test.pipeline.1.transforms.4.dimensions=$d \
            data.test.pipeline.1.transforms.4.model_name=$m \
            data.val.pipeline.1.transforms.4.encoder=$e \
            data.val.pipeline.1.transforms.4.dimensions=$d \
            data.val.pipeline.1.transforms.4.model_name=$m \
            --eval segm bbox \
            --show-dir checkpoints/custom/tf/$MODEL_DIR/analysis \
            --show-score-thr 0.8

            python3 tools/test.py \
            configs/gustav/kungbib-cascade-mask-tf.py \
            checkpoints/custom/tf/$MODEL_DIR/latest.pth \
            --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
            data.train.pipeline.6.dimensions=$d \
            data.train.pipeline.6.encoder=$e \
            data.train.pipeline.6.model_name=$m \
            data.test.pipeline.1.transforms.4.encoder=$e \
            data.test.pipeline.1.transforms.4.dimensions=$d \
            data.test.pipeline.1.transforms.4.model_name=$m \
            data.val.pipeline.1.transforms.4.encoder=$e \
            data.val.pipeline.1.transforms.4.dimensions=$d \
            data.val.pipeline.1.transforms.4.model_name=$m \
            --format-only \
            --options "jsonfile_prefix=./checkpoints/custom/tf/$MODEL_DIR/results"

            python3 tools/analysis_tools/coco_error_analysis.py \
                ./checkpoints/custom/tf/$MODEL_DIR/results.bbox.json \
                ./checkpoints/custom/tf/$MODEL_DIR/results \
                --ann=/data/gustav/datalab_data/poly-dn-2010-2020-720/test_annotations.json \
                --extraplots \
                --areas 80000 360000 10000000000

            
            python3 tools/analysis_tools/coco_error_analysis.py \
                ./checkpoints/custom/tf/$MODEL_DIR/results.segm.json \
                ./checkpoints/custom/tf/$MODEL_DIR/results \
                --ann=/data/gustav/datalab_data/poly-dn-2010-2020-720/test_annotations.json \
                --types='segm' \
                --extraplots \
                --areas 80000 360000 10000000000

            #
            #   1 class
            #

            # python3 -W ignore tools/train.py \
            # configs/gustav/kungbib-cascade-mask-tf-1c.py \
            # --seed=0 \
            # --work-dir=checkpoints/custom/tf/$MODEL_DIR-1c \
            # --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
            # data.train.pipeline.6.dimensions=$d \
            # data.train.pipeline.6.encoder=$e \
            # data.train.pipeline.6.model_name=$m \
            # data.test.pipeline.1.transforms.4.encoder=$e \
            # data.test.pipeline.1.transforms.4.dimensions=$d \
            # data.test.pipeline.1.transforms.4.model_name=$m \
            # data.val.pipeline.1.transforms.4.encoder=$e \
            # data.val.pipeline.1.transforms.4.dimensions=$d \
            # data.val.pipeline.1.transforms.4.model_name=$m


            # python3 tools/test.py \
            # configs/gustav/kungbib-cascade-mask-tf-1c.py \
            # checkpoints/custom/tf/$MODEL_DIR-1c/latest.pth \
            # --work-dir=checkpoints/custom/tf/$MODEL_DIR-1c \
            # --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
            # data.train.pipeline.6.dimensions=$d \
            # data.train.pipeline.6.encoder=$e \
            # data.train.pipeline.6.model_name=$m \
            # data.test.pipeline.1.transforms.4.encoder=$e \
            # data.test.pipeline.1.transforms.4.dimensions=$d \
            # data.test.pipeline.1.transforms.4.model_name=$m \
            # data.val.pipeline.1.transforms.4.encoder=$e \
            # data.val.pipeline.1.transforms.4.dimensions=$d \
            # data.val.pipeline.1.transforms.4.model_name=$m \
            # --eval segm bbox \
            # --show-dir checkpoints/custom/tf/$MODEL_DIR-1c/analysis \
            # --show-score-thr 0.8
        done
    done
done

# TODO add log analysis for all models