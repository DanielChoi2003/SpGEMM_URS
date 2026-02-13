#!/bin/bash
constant=112
scale=17
node=1
    srun -N$node --ntasks-per-node=$constant \
         -t 1:00:00 -ppbatch -A coda \
         src/rmat_spgemm_cache $scale > "../rmat_gprof/cache_${scale}.txt"