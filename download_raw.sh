#!/bin/bash

baseURL='https://fun.bio.keio.ac.jp/software/MDPredictor/'
zipfile='raw_images.zip'
concatURL=${baseURL}${zipfile}

echo "downloading the raw time-lapse phase contrast images..."
if type wget > /dev/null 2>&1; then
    wget $concatURL
elif type curl > /dev/null 2>&1; then
    curl -O $concatURL
else
    echo "both 'wget' and 'curl' command were not found, please install"
    exit 1
fi
unzip $zipfile
rm $zipfile
