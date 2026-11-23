#!/bin/bash

outdir='inferBP_exp'
model='5'
reps=10000
thresh=1

sbatch HPCscripts/run_inferBP_all.sh $outdir $model $reps $thresh
