#!/bin/sh

FOLDS=(0 1 2 3)

for n in ${FOLDS[@]}; do
    python ./src/run_gbp.py ./dataset/NIH3T3_4foldcv/fold$n/test ./results/fold$n/gbp --model_path ./results/fold$n/model.npz -g $1
done
