#!/bin/sh

FOLDS=(0 1 2 3)

for n in ${FOLDS[@]}; do
    python ./src/run_dtd.py ./dataset_paper/NIH3T3_4foldcv/fold$n/test ./results_paper/fold$n/dtd --model_path ./results_paper/fold$n/model.npz -g $1
done
