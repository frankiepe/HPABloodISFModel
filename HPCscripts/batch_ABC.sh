#!/bin/bash

outdir='ABC_exp'
model=1
reps=100000
thresh=2000

sbatch HPCscripts/run_ABC_all.sh $outdir $model $reps $thresh
