#!/bin/bash --login

#SBATCH -p serial
#SBATCH -t 20 
#SBATCH -a 1-2

module purge
module load apps/binapps/conda/miniforge3/26.3.2-3

conda activate myenv

python3 src/fit_model.py --model 1 --ind ${SLURM_ARRAY_TASK_ID} --outdir 'test_fit_model'