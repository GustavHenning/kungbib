#!/bin/bash
runs=(1 2 3 4 5)

for i in "${runs[@]}"
do
    bash ratio_experiment_train.sh kungbib-cascade-mask ratio/run_$i/vanilla_1_0.25 1 0.25 
    bash ratio_experiment_train.sh kungbib-cascade-mask ratio/run_$i/vanilla_1_0.5 1 0.5
    bash ratio_experiment_train.sh kungbib-cascade-mask ratio/run_$i/vanilla_1_0.75 1 0.75
    bash ratio_experiment_train.sh kungbib-cascade-mask ratio/run_$i/vanilla_1_0.9 1 0.9
    bash ratio_experiment_train.sh kungbib-cascade-mask ratio/run_$i/vanilla_1_1.0 1 1.0
done

