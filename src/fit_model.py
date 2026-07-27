import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import pints
import pandas as pd
import numpy as np
import json
import argparse
import methods.models as models
import methods.classes as mc
from methods import model_dict, day_len
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = int, help='Select model for optimisation: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3: HPAModelFEInterCBGAlb, ' \
                    '4: HPAModelFEInterCBGAlbBloodISF, 5: HPAModelFEInterBothCBGAlbBloodISF, ' \
                    '6: HPAModelCRHSupp')
parser.add_argument('-i', '--ind', type = int, help='Select data index')
args = parser.parse_args()

def get_pars(m_n, d_n):
    # Get config file
    init_pars_file = f'configs/{model_dict[m_n]}/test_parameters.json'

    # Load config
    with open(init_pars_file, 'r') as f:
        config = json.load(f)

    # Get parameters from config
    init_pars = config.get('parameters', {})
    num_days = 4
    days_to_keep = 1

    # Create time array
    timesteps = day_len * num_days
    step = 1 # stepsize must be sufficiently small for convergence of dde solver
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
    dde_model = model(parameters=init_pars, times=times, num_days=num_days, days_to_keep=days_to_keep, step=step)

    # Wrap in blood or ISF model for fitting
    if m_n in [1,2,3,6]:
        full_model = m_wrap(dde_model, init_pars, times)
    else:
        full_model1 = m_wrap1(dde_model, init_pars, times)
        full_model2 = m_wrap2(dde_model, init_pars, times)

    # Load data
    dfBP = pd.read_csv(f'data/processed/HABS{d_n}_BP.csv')
    dfISF = pd.read_csv(f'data/processed/HABS{d_n}_ISF.csv')    
    dfBP['datetime'] = pd.to_datetime(dfBP['Date'] + ' ' + dfBP['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    dfISF['datetime'] = pd.to_datetime(dfISF['Date'] + ' ' + dfISF['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    timesBP = dfBP['datetime']
    timesISF = dfISF['datetime']
    CORT = dfBP['Cortisol']
    ACTH = dfBP['ACTH']
    Cortisone = dfBP['Cortisone']
    mCORT = dfISF['mCortisol']
    mCortisone = dfISF['mCortisone']

    sBP = pd.to_datetime(pd.Series(timesBP))
    timesBP = (sBP - sBP.iloc[0]).dt.total_seconds() / 60
    sISF = pd.to_datetime(pd.Series(timesISF))
    timesISF = (sISF - sISF.iloc[0]).dt.total_seconds() / 60

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

    q0 = list(init_pars.values())

    opt = pints.OptimisationController(
            f, q0, boundaries=bounds, method=pints.CMAES)

    result = dde_model.simulate(q0, timesBP)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    ax1.plot(timesBP, result.T[0])
    ax2.plot(timesBP, result.T[1])
    ax2.plot(timesBP, CORT, marker = 'x', color = 'k')
    ax1.plot(timesBP, ACTH, marker = 'x', color = 'k')
    plt.savefig('before.png')
    plt.close()

    opt.set_max_iterations(100)
    opt.set_log_interval(iters=10, warm_up=5)
    opt.set_function_tolerance(iterations=20, threshold=1e-2)

    # Run optimisation
    print(f"Fitting {model_dict[m_n]}...")
    p, s = opt.run()

    result = dde_model.simulate(p, timesBP)
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