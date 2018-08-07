#!/bin/bash

usage_exit() {
        echo "Usage: $0 -d 'dataset directory' -o 'output directory'" 1>&2
        exit 1
}

CMDNAME=`basename $0`
FLG_D="FALSE"
FLG_O="FALSE"
FLG_G="FALSE"

while getopts :d:o:g:h OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE"; VALUE_D="$OPTARG" ;;
    "o" ) FLG_O="TRUE"; VALUE_O="$OPTARG" ;;
    "g" ) FLG_G="TRUE"; VALUE_G=$OPTARG ;;
    "h" ) usage_exit ;;
    \?  ) usage_exit ;;
  esac
done
if [ $FLG_D = "TRUE" ]; then
  DATA=${VALUE_D%/}
else
  usage_exit
fi
if [ $FLG_O = "TRUE" ]; then
  OUT=${VALUE_O%/}
else
  usage_exit
fi
if [ $FLG_G = "TRUE" ]; then
  GPU_ID=$VALUE_G
else
  GPU_ID=-1
fi
mkdir $OUT

FOLDS=(0 1 2 3)
for n in ${FOLDS[@]}; do
    python ./src/train_test.py $DATA/fold$n/train $DATA/fold$n/test $OUT/fold$n -g $GPU_ID
done

python ./src/agg_results.py --results $OUT
