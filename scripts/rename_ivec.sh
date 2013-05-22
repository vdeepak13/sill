#!/bin/bash

if [ $1 == "" ]; then
    echo "Please specify an input/output file."
    exit -1
fi

cat $1 | sed s/ivec/uvec/g > /tmp/rename.txt
mv /tmp/rename.txt $1
