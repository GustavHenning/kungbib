#
#   Dimensions
#
BASE_CHANNELS=3
declare -a dimensions=(3 6 10 50 100 384)
#
#   Model
#
declare -a encoders=("bert" "doc2vec")


for e in "${encoders[@]}"
do

    for d in "${dimensions[@]}"
    do
        TOTAL_CHANNELS="$((d+BASE_CHANNELS))"
        MODEL_DIR="${e}_dim_${d}"
        echo "Training $MODEL_DIR"

        python3 -W ignore tools/train.py \
        configs/gustav/kungbib-cascade-mask-tf.py \
        --seed=0 \
        --work-dir=checkpoints/custom/tf/$MODEL_DIR \
        --cfg-options model.backbone.in_channels=$TOTAL_CHANNELS \
        data.train.pipeline.3.dimensions=$d \
        data.train.pipeline.3.encoder=$e \
        data.test.pipeline.1.transforms.1.encoder=$e \
        data.test.pipeline.1.transforms.1.dimensions=$d \
        data.val.pipeline.1.transforms.1.encoder=$e \
        data.val.pipeline.1.transforms.1.dimensions=$d

    done
done

# TODO add log analysis for all models