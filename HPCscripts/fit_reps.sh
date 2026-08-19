#!/bin/bash --login

#SBATCH --job-name=fit_reps
#SBATCH --time=04:00:00
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --array=1-10

module purge
module load apps/binapps/conda/miniforge3/26.3.2-3
conda activate myenv

export PYTHON_JULIACALL_SYSIMAGE='julia_hpa_sysimage.so'

data_id=$1
outdir=$2
model=$3

if [ ! -d output/${outdir} ]; then
    mkdir output/${outdir}
    mkdir output/${outdir}/cmaes_output
fi

python -u src/fit_model.py --model ${model} --ind ${data_id} --outdir ${outdir} --seed ${SLURM_ARRAY_TASK_ID} > output/${outdir}/cmaes_output/output_rep${SLURM_ARRAY_TASK_ID}_model${model}_dataid${data_id}.txt
