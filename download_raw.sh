#!/bin/bash

echo "downloading the raw time-lapse phase contrast images of NIH/3T3 fibroblasts..."
if type wget > /dev/null 2>&1; then
    wget hogehoge.zip
elif type curl > /dev/null 2>&1; then
    curl -O hogehoge.zip
else
    echo "both `wget` and `curl` command were not found, please install"
fi
unzip hogehoge.zip
rm hogehoge.zip
