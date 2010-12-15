#!/bin/bash

echo "Running timing experiments"

output_filename='results.txt';
for trial in $(seq 1 3); do
#    for ncpus in 2 3 4 5 6 7 8 9 10; do
    for ncpus in 2; do
        for buffer in 10 20 30 40 50 100 500; do
            for updates in 10000 50000 100000; do
                echo "trial= $trial cpus=$ncpus buffer=$buffer updates=$updates";
                mpiexec -n $ncpus ./latency_timing --type=mpi --users=$ncpus --updates=$updates --buffer=$buffer >> $output_filename
                ./latency_timing --type=shared --users=$ncpus --updates=$updates --buffer=$buffer >> $output_filename
            done
        done
    done
done
