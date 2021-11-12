#!/bin/bash

bash train_tf.sh kungbib-cascade-mask-tf doc2vec 1 doc2vec
bash train_tf.sh kungbib-cascade-mask-tf doc2vec 2 doc2vec
bash train_tf.sh kungbib-cascade-mask-tf doc2vec 3 doc2vec
bash train_tf.sh kungbib-cascade-mask-tf doc2vec 4 doc2vec
bash train_tf.sh kungbib-cascade-mask-tf doc2vec 16 doc2vec
bash train_tf.sh kungbib-cascade-mask-tf doc2vec 32 doc2vec
bash train_tf.sh kungbib-cascade-mask-tf doc2vec 64 doc2vec # https://link.springer.com/chapter/10.1007/978-3-319-67008-9_16