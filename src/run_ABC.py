import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import pints
import pandas as pd
import numpy as np
import json
import argparse
from matplotlib import pyplot as plt
import methods.models as models
import methods.classes as mc
from methods import model_dict, day_len
import methods.data_processing as dp
import warnings

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = int, help='Select model for optimisation: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3: HPAModelFEInterCBGAlb, ' \
                    '4: HPAModelFEInterCBGAlbBloodISF, 5: HPAModelFEInterBothCBGAlbBloodISF, ' \
                    '6: HPAModelCRHSupp')
parser.add_argument('-i', '--ind', type = int, help='Select data index')
parser.add_argument('-w', '--warmup', type = int, default=5, help='No. of days of warmup (to reach steady state)')
parser.add_argument('-s', '--step', type=float, default=0.1, help='Step size for dde solver (must be sufficiently small for convergence)')
parser.add_argument('-o', '--outdir', type=str, help='Directory for plotting output')
parser.add_argument('-f', '--fixed_pars', type=str, nargs='*', help='Parameters to fix (i.e. not fit)')
args = parser.parse_args()

def run_ABC(m_n, d_n, warmup, step, outdir, fixed_pars, days_to_keep=1):
    # Get config file
    init_pars_file = f'configs/{model_dict[m_n]}/test_parameters.json'

    # Load config
    with open(init_pars_file, 'r') as f:
        config = json.load(f)

    # Get parameters from config
    init_pars = config.get('parameters', {})
    ics = config.get('initial_conditions', {})
    num_days = warmup+days_to_keep

    # Create time array
    timesteps = day_len * num_days
    if day_len/step != int(day_len/step):
        print(f"Warning: day_len ({day_len}) is not divisible by step ({step}). This may cause issues when plotting.")
    times = np.arange(0, timesteps, step)

    # Initialise model
    if m_n == 1:
        model = models.BaseHPAModel
        m_wrap = mc.ModelBlood
    elif m_n == 2:
        model = models.HPAModelFEInter
        m_wrap = mc.ModelBloodFEInter
    elif m_n == 3:
        model = models.HPAModelFEInterCBGAlb
        m_wrap = mc.ModelBloodFEInterCBGAlb
    elif m_n == 4:
        model = models.HPAModelFEInterCBGAlbBloodISF
        m_wrap1 = mc.ModelBloodFEInterCBGAlb
        m_wrap2 = mc.ModelISFFEInterCBGAlb
    elif m_n == 5:
        model = models.HPAModelFEInterBothCBGAlbBloodISF
        m_wrap1 = mc.ModelBloodFEInterCBGAlb
        m_wrap2 = mc.ModelISFFEInterCBGAlb
    elif m_n == 6:
        model = models.HPAModelCRHSupp
        m_wrap = mc.ModelBlood

    # Define model
    dde_model = model(parameters=init_pars, init_conds=ics, times=times, num_days=num_days, days_to_keep=days_to_keep, step=step)

    # Fix parameters
    if fixed_pars is not None:
        dde_model.set_fix_parameters({i:init_pars[i] for i in fixed_pars})

    # Wrap in blood or ISF model for fitting
    if m_n in [1,2,3,6]:
        full_model = m_wrap(dde_model, init_pars, times)
    else:
        full_model1 = m_wrap1(dde_model, init_pars, times)
        full_model2 = m_wrap2(dde_model, init_pars, times) 

    # Load data
    timesISF, timesBP, CORT, Cortisone, ACTH, mCORT, mCortisone = dp.get_data(d_n)
    sBP = pd.to_datetime(pd.Series(timesBP))
    timesBP = (sBP - sBP.iloc[0]).dt.total_seconds() / 60
    sISF = pd.to_datetime(pd.Series(timesISF))
    timesISF = (sISF - sISF.iloc[0]).dt.total_seconds() / 60

    # Define Pints problem for optimisation
    if m_n in [1,6]:
        problem = pints.MultiOutputProblem(full_model, timesBP, np.array([ACTH, CORT]).T)
        f = pints.MeanSquaredError(problem)
    elif m_n in [2,3]:
        problem = pints.MultiOutputProblem(full_model, timesBP, np.array([ACTH, CORT, Cortisone]).T)
        f = pints.MeanSquaredError(problem)
    else:
        problem1 = pints.MultiOutputProblem(full_model1, timesBP, np.array([ACTH, CORT, Cortisone]).T)
        problem2 = pints.MultiOutputProblem(full_model2, timesISF, np.array([mCORT, mCortisone]).T)
        f1 = pints.MeanSquaredError(problem1)
        f2 = pints.MeanSquaredError(problem2)
        f = pints.SumOfErrors([f1,f2])

    # Set up model boundaries
    bounds = dde_model.get_and_create_boundaries()

    thresh = 10000
    reps = 1000
    pars_accept = []
    for i in np.arange(0,reps):
        if i % 20 == 0:
            print(f"Iteration {i}/{reps}")
        par_i = bounds.sample(1)[0]
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            obj_i = f(list(par_i))

            # Process captured warnings
            if len(caught_warnings) > 0:
                print(f"{par_i} produced warning(s)")
                print(f"Correponds to objective of: {obj_i}")

        if obj_i < thresh:
            pars_accept.append(par_i)

    print(f"% of accepted parameters: {100*len(pars_accept)/reps}")

    if len(pars_accept) > 0:
        for i in np.arange(0, len(init_pars.keys())):
            param_name = list(init_pars.keys())[i]
            if param_name not in fixed_pars:
                param_values = [p[i] for p in pars_accept]

                if outdir is not None:
                    os.makedirs("output/" + outdir, exist_ok=True)
                    hist_file = os.path.join("output/" + outdir, f"hist_{param_name}_step{step}_dataID{d_n}.png")
                else:
                    hist_file = f"hist_{param_name}_step{step}_dataID{d_n}.png"

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(param_values, bins=20, edgecolor='black')
                ax.set_title(f"Histogram of accepted values for parameter '{param_name}'")
                ax.set_xlabel(param_name)
                ax.set_ylabel('Count')
                fig.tight_layout()
                fig.savefig(hist_file)
                plt.close(fig)
                print(f"Saved histogram for '{param_name}' to: {hist_file}")
    else:
        print("No accepted parameters were found; histogram was not created.")

if __name__ == "__main__":
    # read parsed variables
    m_n = args.model
    d_n = args.ind
    warmup = args.warmup
    step = args.step
    outdir = args.outdir
    fixed_pars = args.fixed_pars

    print("Run ABC experiment")
    run_ABC(m_n, d_n, warmup, step, outdir, fixed_pars)