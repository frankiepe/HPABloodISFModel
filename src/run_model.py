import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import numpy as np
import argparse
import json
import methods.models as models
import methods.plotting as plotting
from methods import model_dict, day_len

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = str, help='Select model for optimisation: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3a: HPAModelFEInterCBGAlbSimple, ' \
                    '3b: HPAModelFEInterCBGAlb, 4: HPAModelFEInterCBGAlbBloodISF, 5: HPAModelFEInterBothCBGAlbBloodISF, ' \
                    '6: HPAModelCRHSupp')
parser.add_argument('-s', '--step', type=float, default=0.1, help='Step size for dde solver')
parser.add_argument('-w', '--warmup', type = int, default=5, help='No. of days of warmup (to reach steady state)')
parser.add_argument('-o', '--outdir', type=str, help='Directory for plotting output')
args = parser.parse_args()

def run_model(m_n, step, warmup, days_to_keep=1):
    pars_file = f'configs/{model_dict[m_n]}/test_parameters.json'

    # Load config
    with open(pars_file, 'r') as f:
        config = json.load(f)

    # Get parameters from config
    pars = config.get('parameters', {})
    ics = config.get('initial_conditions', {})
    num_days = warmup + days_to_keep

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

    dde_model = model(parameters=pars, fixed_pars={}, init_conds=ics, times=times, num_days=num_days, days_to_keep=days_to_keep, step=step)

    # Get plotting times (some cropping may occur if day_len/step != whole number)
    plot_times = times[int((day_len/step)*(num_days-days_to_keep)):] - (day_len)*(num_days-days_to_keep)

    # Run the simulation
    print("Running simulation...")       
    result = dde_model.simulate(list(pars.values()), plot_times, fitting=False)
    print("Simulation complete.")

    # Get CRH drive
    crh_drive = [dde_model.crh(t) for t in plot_times]

    return result, plot_times, crh_drive

if __name__ == "__main__":
    m_n = args.model
    step = args.step # must be sufficiently small to ensure dde solver converges (i.e. <=0.1)
    warmup = args.warmup
    outdir = args.outdir
    if not os.path.exists(f'output/{outdir}'):
        os.makedirs(f'output/{outdir}')
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/plots')

    res, times, crh_drive = run_model(m_n, step, warmup)
    plotting.plot_model_output(m_n, res, times, crh_drive, outdir=outdir, filename=f'model_output_step{step}', days_to_keep=1)