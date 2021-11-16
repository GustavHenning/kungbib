#!/bin/bash


DATA_DIR=/data/gustav/datalab_data/model/

for i in $(find "$DATA_DIR" -name "*.jpg") ; do
    echo "$(file $i | cut -d',' -f 8)"
done