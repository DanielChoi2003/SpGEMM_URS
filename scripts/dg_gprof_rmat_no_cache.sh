#!/bin/bash
constant=112
scale=17
node=1
    srun -N$node --ntasks-per-node=$constant \
         -t 30:00 -ppbatch -A coda \
         src/rmat_spgemm_no_cache $scale > "../rmat_gprof/no_cache_${scale}.txt"