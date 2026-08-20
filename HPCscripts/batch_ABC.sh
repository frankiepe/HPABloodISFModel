#!/bin/bash

outdir='ABC_exp'
model=1
reps=50000
thresh=2500

sbatch HPCscripts/run_ABC_all.sh $outdir $model $reps $thresh
