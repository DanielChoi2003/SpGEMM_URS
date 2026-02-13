#!/bin/bash
constant=32
scale=16
for ((i=1; i<=256; i=i*2))
do
    srun -N $i --ntasks-per-node=$constant \
         -t 30:00 -ppbatch -A coda \
         src/rmat_spgemm $scale > "../strong_scaling_output/rmat/no_cache_strong_scaling_${i}_nodes.txt"
    ((scale++))
done