import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import pandas as pd
import numpy as np
import json
import argparse
import methods.models as models
from methods import model_dict, day_len

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = str, help='Select model for optimisation: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3a: HPAModelFEInterCBGAlbSimple, ' \
                    '3b: HPAModelFEInterCBGAlb, 4: HPAModelFEInterCBGAlbBloodISF, 5: HPAModelFEInterBothCBGAlbBloodISF, ' \
                    '6: HPAModelCRHSupp')
parser.add_argument('-w', '--warmup', type = int, default=5, help='No. of days of warmup (to reach steady state)')
parser.add_argument('-s', '--step', type=float, default=0.1, help='Step size for dde solver (must be sufficiently small for convergence)')
parser.add_argument('-o', '--outdir', type=str, help='Directory for plotting output')
parser.add_argument('-c', '--constraint', type=str, help='Constraint to test')
parser.add_argument('-f', '--fixed', type=str, nargs='*', help='Parameters to fix (i.e. not fit)')
parser.add_argument('-r', '--reps', type=int, default=1000, help='Number of simulation repetitions')
args = parser.parse_args()

def run_pass_fail_constraint(m_n, warmup, step, outdir, fixed, reps, constraint, days_to_keep=1):
    # Get config file
    init_pars_file = f'configs/{model_dict[m_n]}/test_parameters.json'

    # Load config
    with open(init_pars_file, 'r') as f:
        config = json.load(f)

    # Get parameters from config
    init_pars = config.get('parameters', {})
    fixed_pars = init_pars.copy()
    
    # Filter fixed vs. non-fixed parameters
    all_par_keys = list(init_pars.keys())
    for par_key in all_par_keys:
        if par_key in fixed:
            init_pars.pop(par_key)
        else:
            fixed_pars.pop(par_key)

    ics = config.get('initial_conditions', {})
    num_days = warmup+days_to_keep

    # Create time array
    timesteps = day_len * num_days
    if day_len/step != int(day_len/step):
        print(f"Warning: day_len ({day_len}) is not divisible by step ({step}). This may cause issues when plotting.")
    times = np.arange(0, timesteps, step)

    # Initialise model
    if m_n == '1':
        model = models.BaseHPAModel
    elif m_n == '2':
        model = models.HPAModelFEInter
    elif m_n == '3a':
        model = models.HPAModelFEInterCBGAlbSimple
    elif m_n == '3b':
        model = models.HPAModelFEInterCBGAlb
    elif m_n == '4':
        model = models.HPAModelFEInterCBGAlbBloodISF
    elif m_n == '5':
        model = models.HPAModelFEInterBothCBGAlbBloodISF
    elif m_n == '6':
        model = models.HPAModelCRHSupp

    # Define model
    dde_model = model(parameters=init_pars, fixed_pars=fixed_pars, init_conds=ics, times=times, 
                      num_days=num_days, days_to_keep=days_to_keep, step=step)

    # Set up model boundaries
    bounds = dde_model.get_and_create_boundaries()

    # Sample pars and check corresponding fail flags
    pars_all = np.empty((reps, len(init_pars)), dtype=float)
    dict_fail_flags_all = {}
    for i in range(reps):
        if i % 20 == 0:
            print(f"Iteration {i}/{reps}")
        par_i = np.asarray(bounds.sample(1)[0], dtype=float) # sample
        fail_flags_i = dde_model.simulate(list(par_i), times, fitting=False, reject=True, fail_flags=True)
        pars_all[i] = par_i
        dict_fail_flags_all[i] = fail_flags_i[constraint]

     # Save parameters
    df = pd.DataFrame(pars_all, columns=init_pars.keys()) 
    df.insert(0, "Fail", dict_fail_flags_all)
    df.to_csv("output/" + outdir + f"/{model_dict[m_n]}/pars" + f"/all_pars_model{m_n}_{constraint}.csv", index=False)

if __name__ == "__main__":
    # Read parsed variables
    m_n = args.model
    warmup = args.warmup
    step = args.step
    outdir = args.outdir
    fixed = args.fixed if args.fixed is not None else {}
    reps = args.reps
    constraint = args.constraint

    # Make directories if necessary
    if not os.path.exists(f'output/{outdir}'):
        os.makedirs(f'output/{outdir}', exist_ok=True)
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/pars'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/pars', exist_ok=True)
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/plots'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/plots', exist_ok=True)

    # Run the pass/fail constraint function
    run_pass_fail_constraint(m_n, warmup, step, outdir, fixed, reps, constraint)
