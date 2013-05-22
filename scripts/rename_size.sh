#!/bin/bash

if [ $1 == "" ]; then
    echo "Please specify an input/output file."
    exit -1
fi

cat $1 | sed s/size1\(\)/n_rows/g | sed s/size2\(\)/n_cols/g > /tmp/rename.txt
mv /tmp/rename.txt $1
