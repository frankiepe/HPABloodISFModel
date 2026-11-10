# This script is intended to infer the blood plasma hormone profiles of an individual
# based on their ISF profile. This will be done as follows:
# Firstly, we will evaluate KDE distributions for each model parameter based on the 
# model 5 fits to the BP+ISF datasets (not dataID 2 which we suspect has some significant external stressor)
# For a given ISF dataset, we will then take draws from the KDE distributios for each parameter (ensuring
# that the parameters remain within their specified bounds). We will simulate the model under these sampled
# parameters and evaluate a sum-of-squares objective with the ISF cortisol and cortisone data. Via some
# accept/reject criterion we will retain a subset of the sampled parameter sets and this will be our
# ABC-style posterior distribution conditioned on the ISF data. We can plot the retained simulations
# to illustrate the potential range of cortisol/cortisone/ACTH behaviour in blood plasma.

import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
from methods import model_dict, day_len
import methods.plotting as plotting
import methods.models as models
import methods.classes as mc
from methods import PARAMETER_BOUNDARIES as pb
import methods.data_processing as dp
import json
import scipy.stats
import pints

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = str, help='Select model: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3a: HPAModelFEInterCBGAlbSimple, ' \
                    '3b: HPAModelFEInterCBGAlb, 4: HPAModelFEInterCBGAlbBloodISF, 5: HPAModelFEInterBothCBGAlbBloodISF, ' \
                    '6: HPAModelFEInterISFOnlyCBGAlbBloodISF')
parser.add_argument('-i', '--ind', type = int, help='Select data index')
parser.add_argument('-n', '--ninds', type=int, help='Number of datasets fitted')
parser.add_argument('-r', '--nreps', type=int, help='Number of fitting reps')
parser.add_argument('-s', '--step', type=float, default=1, help='Step size for dde solver')
parser.add_argument('-w', '--warmup', type = int, default=5, help='No. of days of warmup (to reach steady state)')
parser.add_argument('-o', '--outdir', type=str, help='Directory for plotting output')
parser.add_argument('-a', '--abcreps', type=int, default=1000, help='Number of abc repetitions')
args = parser.parse_args()

def run_inference(m_n, d_n, num_inds, nreps, outdir, step, warmup, abcreps, days_to_keep=1):
    # Get the best-fitting parameters for each dataset
    best_pars = []
    for i in range(1, num_inds+1):
        if i != 2:
            min_obj = 1e9
            for j in range(1, nreps+1):
                pars_file = f"output/fit_models/{model_dict[m_n]}/fits/fit_dataid{i}_seed{j}.json"
                try:
                    # Load config
                    with open(pars_file, 'r') as f:
                        config = json.load(f)
                    score = config.get('score', {})
                    if score < min_obj:
                        pars = config.get('parameters', {})
                        min_obj = score
                except FileNotFoundError:
                    print(f"File not found: {pars_file}")
                    continue
            best_pars.append(pars)
    fitted_pars = pars.keys()
    df = pd.DataFrame(best_pars, columns=fitted_pars)

    # Get KDE distributions for each parameter
    kde = {}
    for par in fitted_pars:
        kde[par] = scipy.stats.gaussian_kde(df[par])

    # Initialise model
    pars_file_default = f'configs/{model_dict[m_n]}/test_parameters.json'
    with open(pars_file_default, 'r') as f:
        config_default = json.load(f)
    pars_init = config_default.get('parameters', {})
    ics = config_default.get('initial_conditions', {})
    num_days = warmup + days_to_keep
    model = models.HPAModelFEInterBothCBGAlbBloodISF

    # Create time array
    timesteps = day_len * num_days
    if day_len/step != int(day_len/step):
        print(f"Warning: day_len ({day_len}) is not divisible by step ({step}). This may cause issues when plotting.")
    times = np.arange(0, timesteps, step)

    dde_model = model(parameters=pars_init, fixed_pars={}, init_conds=ics, times=times, 
                        num_days=num_days, days_to_keep=days_to_keep, step=step)

    # Get ISF data and define objective
    timesISF, _, _, _, _, mCORT, mCortisone = dp.get_data(d_n)
    sISF = pd.to_datetime(pd.Series(timesISF))
    timesISF = (sISF - sISF.iloc[0]).dt.total_seconds() / 60
    full_model = mc.ModelISFFEInterCBGAlb(dde_model, pars_init, times)
    problem = pints.MultiOutputProblem(full_model, timesISF, np.array([mCORT, mCortisone]).T)
    f = pints.MeanSquaredError(problem)

    n_pars = len(pars_init)
    pars_all = np.empty((abcreps, n_pars), dtype=float)
    objs = np.empty(abcreps, dtype=float)
    i = 0
    while i < abcreps:
        # Take a draw from the KDE distributions
        pars_sample = {}
        for par in fitted_pars:
            lb = pb[m_n][par][0]
            ub = pb[m_n][par][1]
            while True:
                sample = kde[par].resample(1)[0][0]
                if lb <= sample <= ub:
                    pars_sample[par] = sample
                    break

        p = list(pars_sample.values())
        obj_i = f(p)
        if obj_i < 1e6:
            if i % 50 == 0:
                print(f"Iteration {i}")
            pars_all[i] = p
            objs[i] = obj_i
            i+=1

    o_arr = objs
    p_arr = pars_all
    indices = np.argsort(o_arr)
    sorted_o = o_arr[indices]
    sorted_p = p_arr[indices]

    # Save parameters
    df = pd.DataFrame(sorted_p, columns=fitted_pars) 
    df.insert(0, "Obj", sorted_o)
    df.to_csv("output/" + outdir + f"/{model_dict[m_n]}/pars" + f"/all_pars_model{m_n}_dataID{d_n}.csv", index=False)

    # Run the simulation
    #result = dde_model.simulate(p_save, times, fitting=False, reject=False)

    # Get CRH drive
    #crh_drive = [dde_model.crh(t, t_s=p_save[-2], lambda_a=p_save[-4], lambda_s=p_save[-3], sigma=p_save[-1]) for t in times[int((day_len*warmup)/step):]]

    # Plot the model output
    #plotting.plot_model_output(m_n, result, times[int((day_len*warmup)/step):]-day_len*warmup, crh_drive, outdir=outdir, filename=f'sample_output_{d_n}', days_to_keep=1, plot_data=True, d_n=d_n)

if __name__ == "__main__":
    # read parsed variables
    m_n = args.model
    d_n = args.ind
    num_inds = args.ninds
    nreps = args.nreps
    outdir = args.outdir
    step = args.step
    warmup = args.warmup
    abcreps = args.abcreps

    # Make directories if necessary
    if not os.path.exists(f'output/{outdir}'):
        os.makedirs(f'output/{outdir}', exist_ok=True)
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/pars'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/pars', exist_ok=True)
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/plots'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/plots', exist_ok=True)

    run_inference(m_n, d_n, num_inds, nreps, outdir, step, warmup, abcreps)
    
