#!/bin/bash
constant=32
for ((i=1; i<=256; i = i * 2))
do
    srun -N $i --ntasks-per-node=$constant -t 30:00 -ppbatch -A coda \
        src/test_sparse > "../strong_scaling_output/amazon_results/first_amazon_strong_scaling_${i}_nodes.txt"
done