import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import pints
import pandas as pd
import numpy as np
import json
import argparse
from methods.models import BaseHPAModel, HPAModelFEInter, HPAModelFEInterCBGAlb, HPAModelFEInterCBGAlbBloodISF
from methods import model_dict, day_len
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = int, help='Select model for optimisation: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3: HPAModelFEInterCBGAlb, 4: HPAModelFEInterCBGAlbBloodISF')
parser.add_argument('-i', '--ind', type = int, help='Select data index')
args = parser.parse_args()

def get_pars(m_n, d_n):
    print(f"Fitting {model_dict[m_n]}...")

    init_pars_file = f'configs/{model_dict[m_n]}/test_parameters.json'

    # Load config
    with open(init_pars_file, 'r') as f:
        config = json.load(f)

    # Get parameters from config
    init_pars = config.get('parameters', {})
    num_days = 6
    days_to_keep = 1

    # Create time array
    timesteps = day_len * num_days
    step = 1 # stepsize must be sufficiently small for convergence of dde solver
    if day_len/step != int(day_len/step):
        print(f"Warning: day_len ({day_len}) is not divisible by step ({step}). This may cause issues when plotting.")
    times = np.arange(0, timesteps, step)

    # Define model
    model = BaseHPAModel(parameters=init_pars, times = times, num_days=num_days, days_to_keep=days_to_keep, step=step)

    # Load data
    dfBP = pd.read_csv(f'data/processed/HABS{d_n}_BP.csv')
    dfBP['datetime'] = pd.to_datetime(dfBP['Date'] + ' ' + dfBP['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    timesBP = dfBP['datetime']
    CORT = dfBP['Cortisol']
    ACTH = dfBP['ACTH']

    s = pd.to_datetime(pd.Series(timesBP))
    timesBP = (s - s.iloc[0]).dt.total_seconds() / 60

    problem = pints.MultiOutputProblem(model, timesBP, np.array([CORT, ACTH]).T)
    f = pints.MeanSquaredError(problem)

    # Set up model boundaries
    bounds = model.get_and_create_boundaries()

    q0 = list(init_pars.values())

    opt = pints.OptimisationController(
            f, q0, boundaries=bounds, method=pints.CMAES)

    result = model.simulate(q0, timesBP)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.plot(timesBP, result.T[0])
    ax2.plot(timesBP, result.T[1])
    ax2.plot(timesBP, CORT, marker = 'x', color = 'k')
    ax1.plot(timesBP, ACTH, marker = 'x', color = 'k')
    plt.savefig('before.png')

    opt.set_max_iterations(100)
    opt.set_log_interval(iters=10, warm_up=5)
    opt.set_max_unchanged_iterations(iterations=20, threshold=1e-2)

    # Run optimisation
    p, s = opt.run()

    result = model.simulate(p, timesBP)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.plot(timesBP, result.T[0])
    ax2.plot(timesBP, result.T[1])
    ax2.plot(timesBP, CORT, marker = 'x', color = 'k')
    ax1.plot(timesBP, ACTH, marker = 'x', color = 'k')
    plt.savefig('after.png')

    return p, s 

if __name__ == "__main__":
    m_n = args.model
    d_n = args.ind

    pars, sc = get_pars(m_n, d_n)
    print(pars)
    print(sc)