#!/bin/bash --login

#SBATCH --job-name=ABC
#SBATCH --time=06:00:00
#SBATCH --partition=multicore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-7

module purge
module load apps/binapps/conda/miniforge3/26.3.2-3
conda activate myenv

export PYTHON_JULIACALL_SYSIMAGE='julia_hpa_sysimage.so'

outdir=$1
model=$2
reps=$3
thresh=$4

if [ ! -d output/${outdir} ]; then
    mkdir output/${outdir}
    mkdir output/${outdir}/ABC_output
fi

python -u src/run_ABC.py --model ${model} --ind ${SLURM_ARRAY_TASK_ID} --outdir ${outdir} --reps ${reps} > output/${outdir}/ABC_output/output_model${model}_dataid${SLURM_ARRAY_TASK_ID}.txt

python -u src/process_ABC.py --model ${model} --ind ${SLURM_ARRAY_TASK_ID} --outdir ${outdir} --thresh ${thresh}
