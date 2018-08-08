#!/bin/bash

echo "downloading the NIH/3T3 image dataset..."
if type wget > /dev/null 2>&1; then
    wget hogehoge.zip
elif type curl > /dev/null 2>&1; then
    curl -O hogehoge.zip
else
    echo "both `wget` and `curl` command were not found, please install"
fi
unzip hogehoge.zip
rm hogehoge.zip
