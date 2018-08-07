#!/bin/bash

usage_exit() {
        echo "Usage: $0 -d path/to/dataset -r path/to/results -g GPU_id" 1>&2
        exit 1
}

CMDNAME=`basename $0`
FLG_D="FALSE"
FLG_R="FALSE"
FLG_G="FALSE"

while getopts :d:r:g:h OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE"; VALUE_D="$OPTARG" ;;
    "r" ) FLG_R="TRUE"; VALUE_R="$OPTARG" ;;
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
if [ $FLG_R = "TRUE" ]; then
  RES=${VALUE_R%/}
else
  usage_exit
fi
if [ $FLG_G = "TRUE" ]; then
  GPU_ID=$VALUE_G
else
  GPU_ID=-1
fi

FOLDS=(0 1 2 3)
for n in ${FOLDS[@]}; do
    python ./src/run_dtd.py $DATA/fold$n/test $RES/fold$n/dtd --model_path $RES/fold$n/model.npz -g $GPU_ID
done
