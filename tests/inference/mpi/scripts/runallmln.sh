#!/bin/bash

# path="/a/paraml/paraml/release/tests/inference/mpi"
# executable="mpi_markov_logic"

# BIN="mpiexec -n $NPROC /a/paraml/paraml/release/tests/inference/mpi/mpi_markov_logic 
# --partfactor=$PFACTOR"

pathtoparaml=/a/paraml/paraml
pathtoscripts=$pathtoparaml/tests/inference/mpi/scripts



for nprocs in 120 60 30 20  
do
    for factor in 1 5 10
    do
	echo "==================================================="
	echo "Processing $nprocs cpus with factor $factor"
	for folder in `ls -d */`
	do
	    $pathtoscripts/run_once.sh $folder $nprocs $factor
	done
    done
done


# 	    cd $f
# 	    rm log_*
#             echo "Processing MLN $f"
# 	    echo "RUNNING: mpiexec -l -n $nprocs $path/$executable " \
# 		"--partfactor=$factor *.out results.txt > runoutput.txt"
# 	    # actually execute mpiexec
# 	    mpiexec -l -n $nprocs $path/$executable --partfactor=$factor \
# 		*.out results.txt >> runoutput.txt
            
#             # run parser logparser.py to collect log files
# 	    /a/paraml/paraml/tests/inference/mpi/logparser.py $factor
	    
# 	    # Send the data to the sql table
# 	    /a/paraml/paraml/tests/inference/mpi/scripts/send_to_sql.sh

# 	    # remove extraneous log files
# 	    rm log_*

# 	    # save processed results
#  	    folder="res_unweighted_$[nprocs]_$[factor]"
# 	    echo "Creating folder: $folder"
#  	    mkdir $folder
#  	    mv *.txt $folder/.
# 	    cd ..
