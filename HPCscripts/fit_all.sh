#!/bin/bash --login

#SBATCH --job-name=test_fit
#SBATCH --time=04:00:00
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --array=1-7

module purge
module load apps/binapps/conda/miniforge3/26.3.2-3
conda activate myenv

export PYTHON_JULIACALL_SYSIMAGE='julia_hpa_sysimage.so'

python -u src/fit_model.py --model 1 --ind ${SLURM_ARRAY_TASK_ID} --outdir 'test_fit_model' > output/test_fit_model/output_${SLURM_ARRAY_TASK_ID}.txt
