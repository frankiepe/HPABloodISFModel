#!/bin/bash

outdir='fit_models'
model=1

for i in {1..7}
do
    sbatch HPCscripts/fit_reps.sh $i $outdir $model
done
