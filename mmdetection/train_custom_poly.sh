#!/bin/bash

declare -a bert_models=( "all-mpnet-base-v2" "multi-qa-mpnet-base-dot-v1" "all-distilroberta-v1" "all-MiniLM-L12-v2" "multi-qa-distilbert-cos-v1" "all-MiniLM-L6-v2" "paraphrase-multilingual-mpnet-base-v2" "paraphrase-albert-small-v2" "paraphrase-multilingual-MiniLM-L12-v2" "paraphrase-MiniLM-L3-v2" "distiluse-base-multilingual-cased-v1" "distiluse-base-multilingual-cased-v2" "multi-qa-MiniLM-L6-cos-v1")

# size 768: "KB/bert-base-swedish-cased" "af-ai-center/bert-base-swedish-uncased" 
# size 384: "multi-qa-MiniLM-L6-cos-v1" "all-mpnet-base-v2" "multi-qa-mpnet-base-dot-v1" "all-distilroberta-v1" "all-MiniLM-L12-v2" "multi-qa-distilbert-cos-v1" "all-MiniLM-L6-v2" "paraphrase-multilingual-mpnet-base-v2" "paraphrase-albert-small-v2" "paraphrase-multilingual-MiniLM-L12-v2" "paraphrase-MiniLM-L3-v2" "distiluse-base-multilingual-cased-v1" "distiluse-base-multilingual-cased-v2"

for m in "${bert_models[@]}"; do
    bash train_tf.sh kungbib-cascade-mask-tf bert 384 $m
done