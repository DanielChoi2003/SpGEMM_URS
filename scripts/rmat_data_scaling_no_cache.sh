#!/bin/bash
constant=100
nodes=64
for ((scale=17; scale <= 25; scale++))
do
    sbatch <<EOF
#!/bin/bash
#SBATCH -N $nodes
#SBATCH --ntasks-per-node=$constant
#SBATCH -t 1:00:00
#SBATCH -p pbatch
#SBATCH -A coda
#SBATCH -o ../data_scaling_64_nodes/no_cache_data_scaling_${scale}_scale.txt

srun src/rmat_spgemm_no_cache $scale
EOF
done