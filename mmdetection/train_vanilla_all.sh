#!/bin/bash

bash train.sh kungbib-cascade-mask vanilla_1 1

bash train.sh kungbib-cascade-mask-101 vanilla-101_1 1

bash train.sh kungbib-cascade-mask-101-32x4d vanilla-101-32x4d_1 1
bash train.sh kungbib-cascade-mask-101-32x8d vanilla-101-32x8d_1 1

bash train.sh kungbib-cascade-mask-101-64x4d vanilla-101-64x4d_1 1


bash train.sh kungbib-cascade-mask vanilla_0.5 0.5

bash train.sh kungbib-cascade-mask-101 vanilla-101_0.5 0.5

bash train.sh kungbib-cascade-mask-101-32x4d vanilla-101-32x4d_0.5 0.5
bash train.sh kungbib-cascade-mask-101-32x8d vanilla-101-32x8d_0.5 0.5

bash train.sh kungbib-cascade-mask-101-64x4d vanilla-101-64x4d_0.5 0.5