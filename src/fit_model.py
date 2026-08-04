import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import pints
import pandas as pd
import numpy as np
import json
import csv
import argparse
import methods.models as models
import methods.classes as mc
from methods import model_dict, day_len
import methods.plotting as plotting
import methods.data_processing as dp

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

def get_pars(m_n, d_n, warmup, step, outdir, fixed_pars, days_to_keep=1):
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

    # Get parameter initialisation
    q0 = list(init_pars.values())

    # Define Pints optimiser
    opt = pints.OptimisationController(
            f, q0, boundaries=bounds, method=pints.CMAES)
    opt.set_max_iterations(10)
    opt.set_log_interval(iters=10, warm_up=10)
    opt.set_function_tolerance(iterations=20, threshold=1e-2)

    # Simulate initial parameterisation
    res_init = dde_model.simulate(q0, times, fitting=False)

    # Get CRH drive
    crh_drive = [dde_model.crh(t) for t in times[int((day_len*warmup)/step):]]

    # Plot initial parameterisation
    plotting.plot_model_output(m_n, res_init, times[int((day_len*warmup)/step):]-day_len*warmup, crh_drive, outdir=outdir,
                               filename=f'model_init_output_ind{d_n}_step{step}', days_to_keep=1, plot_data=True, d_n=d_n)

    # Run optimisation
    print(f"Fitting {model_dict[m_n]}...")
    p, s = opt.run()

    # Ensure fixed parameters are stored correctly
    if fixed_pars is not None:
        final_pars = init_pars.copy()
        param_keys = list(init_pars.keys())
        for i, key in enumerate(param_keys):
            if key not in fixed_pars:
                final_pars[key] = p[i]
            else:
                p[i] = init_pars[key]

    # Simulate optimised parameterisation
    res_fit = dde_model.simulate(p, times, fitting=False)

    # Plot optimised parameterisation
    plotting.plot_model_output(m_n, res_fit, times[int((day_len*warmup)/step):]-day_len*warmup, crh_drive, outdir=outdir,
                               filename=f'model_fit_output_ind{d_n}_step{step}', days_to_keep=1, plot_data=True, d_n=d_n)

    return p, s 

if __name__ == "__main__":
    # read parsed variables
    m_n = args.model
    d_n = args.ind
    warmup = args.warmup
    step = args.step
    outdir = args.outdir
    fixed_pars = args.fixed_pars

    repeats = 1 #todo

    # Make directories if necessary
    if not os.path.exists(f'output/{outdir}'):
        os.makedirs(f'output/{outdir}')
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/fits'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/fits')
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/plots'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/plots')

    # Perform fitting    
    pars, sc = get_pars(m_n, d_n, warmup, step, outdir, fixed_pars)

    # Save fitted parameters
    with open(f"output/{outdir}/{model_dict[m_n]}/fits/fit_step{step}_dataid{d_n}.csv", 'w') as f:
        f.write('"rep","pars","score"')
        f.write("\n")
        writer = csv.writer(f)
        writer.writerows(zip(np.arange(1, repeats+1, 1), [pars], [sc]))