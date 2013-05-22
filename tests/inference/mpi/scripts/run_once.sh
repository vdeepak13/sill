#!/bin/bash

EXPECTED_ARGS=3

if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` folder nprocs factor"
  exit -1;
fi

folder=$1
nprocs=$2
factor=$3


echo "=================================================================="
echo "Folder: $folder"
echo "NProcs: $nprocs"
echo "Factor: $factor"

pathtoparaml=/a/paraml/paraml
pathtoexec=$pathtoparaml/release/tests/inference/mpi
pathtoscripts=$pathtoparaml/tests/inference/mpi/scripts
executable=mpi_markov_logic



cd $folder


rm -f log_*
echo "Processing MLN $folder"
echo "RUNNING: mpiexec -l -n $nprocs $path/$executable " \
    "--bound=0.001 --partfactor=$factor *.out results.txt > runoutput.txt"

# actually execute mpiexec
mpiexec -l -n $nprocs $pathtoexec/$executable --partfactor=$factor \
    *.out results.txt >> runoutput.txt

# run parser logparser.py to collect log files
$pathtoscripts/logparser.py $factor

# # Send the data to the sql table
$pathtoscripts/send_to_sql.sh

# remove extraneous log files
rm log_*

# save processed results
outputfolder="res_unweighted_$[nprocs]_$[factor]"
echo "Creating folder: $folder"
mkdir $outputfolder
mv *.txt $outputfolder/.

cd ..

