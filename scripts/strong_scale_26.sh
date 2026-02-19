#!/bin/bash
constant=112
scale=26
for ((nodes=2; nodes <= 128; nodes*=2))
do
    sbatch <<EOF
#!/bin/bash
#SBATCH -N $nodes
#SBATCH --ntasks-per-node=$constant
#SBATCH -t 1:00:00
#SBATCH -p pbatch
#SBATCH -A coda
#SBATCH -o strong_scaling_{$nodes}_nodes_2nd.txt

srun src/test_sparse $scale
EOF
done