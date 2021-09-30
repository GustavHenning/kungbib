#
#   Dimensions
#
BASE_CHANNELS=3
declare -a dimensions=(384) #3 6 10 50 100 
#
#   Model
#
declare -a encoders=("bert") # "doc2vec"

#
#   BERT model names
#
declare -a bert_models=("all-mpnet-base-v2" "multi-qa-mpnet-base-dot-v1" "all-distilroberta-v1" "all-MiniLM-L12-v2" "multi-qa-distilbert-cos-v1" "all-MiniLM-L6-v2" "multi-qa-MiniLM-L6-cos-v1" "paraphrase-multilingual-mpnet-base-v2" "paraphrase-albert-small-v2" "paraphrase-multilingual-MiniLM-L12-v2" "paraphrase-MiniLM-L3-v2" "distiluse-base-multilingual-cased-v1" "distiluse-base-multilingual-cased-v2")

for e in "${encoders[@]}"
do

    for d in "${dimensions[@]}"
    do

        for m in "${bert_models[@]}"
        do
            TOTAL_CHANNELS="$((d+BASE_CHANNELS))"
            MODEL_DIR="${e}_dim_${d}_${m}"
            echo "Training $MODEL_DIR"

            #
            #   2 classes
            #

            python3 -W ignore tools/train.py \
            configs/gustav/kungbib-cascade-mask-tf.py \
            --seed=0 \
            --work-dir=checkpoints/custom/tf/$MODEL_DIR \
            --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
            data.train.pipeline.3.dimensions=$d \
            data.train.pipeline.3.encoder=$e \
            data.train.pipeline.3.model_name=$m \
            data.test.pipeline.1.transforms.1.encoder=$e \
            data.test.pipeline.1.transforms.1.dimensions=$d \
            data.test.pipeline.1.transforms.1.model_name=$m \
            data.val.pipeline.1.transforms.1.encoder=$e \
            data.val.pipeline.1.transforms.1.dimensions=$d \
            data.val.pipeline.1.transforms.1.model_name=$m


            python3 tools/test.py \
            configs/gustav/kungbib-cascade-mask-tf.py \
            checkpoints/custom/tf/$MODEL_DIR/latest.pth \
            --work-dir=checkpoints/custom/tf/$MODEL_DIR \
            --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
            data.train.pipeline.3.dimensions=$d \
            data.train.pipeline.3.encoder=$e \
            data.train.pipeline.3.model_name=$m \
            data.test.pipeline.1.transforms.1.encoder=$e \
            data.test.pipeline.1.transforms.1.dimensions=$d \
            data.test.pipeline.1.transforms.1.model_name=$m \
            data.val.pipeline.1.transforms.1.encoder=$e \
            data.val.pipeline.1.transforms.1.dimensions=$d \
            data.val.pipeline.1.transforms.1.model_name=$m \
            --eval segm bbox \
            --show-dir checkpoints/custom/tf/$MODEL_DIR/analysis \
            --show-score-thr 0.6

            #
            #   1 class
            #

            python3 -W ignore tools/train.py \
            configs/gustav/kungbib-cascade-mask-tf-1c.py \
            --seed=0 \
            --work-dir=checkpoints/custom/tf/$MODEL_DIR-1c \
            --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
            data.train.pipeline.3.dimensions=$d \
            data.train.pipeline.3.encoder=$e \
            data.train.pipeline.3.model_name=$m \
            data.test.pipeline.1.transforms.1.encoder=$e \
            data.test.pipeline.1.transforms.1.dimensions=$d \
            data.test.pipeline.1.transforms.1.model_name=$m \
            data.val.pipeline.1.transforms.1.encoder=$e \
            data.val.pipeline.1.transforms.1.dimensions=$d \
            data.val.pipeline.1.transforms.1.model_name=$m


            python3 tools/test.py \
            configs/gustav/kungbib-cascade-mask-tf-1c.py \
            checkpoints/custom/tf/$MODEL_DIR-1c/latest.pth \
            --work-dir=checkpoints/custom/tf/$MODEL_DIR-1c \
            --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
            data.train.pipeline.3.dimensions=$d \
            data.train.pipeline.3.encoder=$e \
            data.train.pipeline.3.model_name=$m \
            data.test.pipeline.1.transforms.1.encoder=$e \
            data.test.pipeline.1.transforms.1.dimensions=$d \
            data.test.pipeline.1.transforms.1.model_name=$m \
            data.val.pipeline.1.transforms.1.encoder=$e \
            data.val.pipeline.1.transforms.1.dimensions=$d \
            data.val.pipeline.1.transforms.1.model_name=$m \
            --eval segm bbox \
            --show-dir checkpoints/custom/tf/$MODEL_DIR-1c/analysis \
            --show-score-thr 0.6
        done
    done
done

# TODO add log analysis for all models