#!/bin/bash

if [ $1 == "" ]; then
    echo "Please specify an input/output file."
    exit -1
fi

cat $1 | sed s/namespace\ prl/namespace\ sill/g | sed s/\<prl/\<sill/g | sed s/prl::/sill::/g | sed s/\ PRL\_/\ SILL\_/g > /tmp/rename.txt
mv /tmp/rename.txt $1
