#!/bin/sh

mkdir ./results

FOLDS=(0 1 2 3)
for n in ${FOLDS[@]}; do
    python ./src/train_test.py ./dataset/NIH3T3_4foldcv/fold$n/train ./dataset/NIH3T3_4foldcv/fold$n/test ./results/fold$n -g $1
done

python ./src/agg_results.py --results ./results



         
         
