# This script will take the median fitted parameters
# and will vary each parameter one at a time by 0.1%, 1% and 10%
# to see how much the model output changes.

import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import numpy as np
import argparse
import json
import methods.models as models
import methods.plotting as plotting
from methods import PARAMETER_BOUNDARIES as pb
from methods import model_dict, day_len

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = str, help='Select model for optimisation: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3a: HPAModelFEInterCBGAlbSimple, ' \
                    '3b: HPAModelFEInterCBGAlb, 4: HPAModelFEInterCBGAlbBloodISF, 5: HPAModelFEInterBothCBGAlbBloodISF, ' \
                    '6: HPAModelFEInterISFOnlyCBGAlbBloodISF')
parser.add_argument('-s', '--step', type=float, default=1, help='Step size for dde solver')
parser.add_argument('-w', '--warmup', type = int, default=5, help='No. of days of warmup (to reach steady state)')
parser.add_argument('-o', '--outdir', type=str, help='Directory for plotting output')
args = parser.parse_args()

def run_SA(m_n, step, warmup, days_to_keep=1):
    pars_file_default = f'configs/{model_dict[m_n]}/test_parameters.json'
    pars_file_fit = f'output/fit_models/{model_dict[m_n]}/median_pars.json'

    # Load configs
    with open(pars_file_default, 'r') as f:
        config_default = json.load(f)
    with open(pars_file_fit, 'r') as f:
        config_fit = json.load(f)

    # Get parameters from configs
    pars = config_fit.get('parameters', {})
    ics = config_default.get('initial_conditions', {})
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
        model = models.HPAModelFEInterISFOnlyCBGAlbBloodISF

    dde_model = model(parameters=pars, fixed_pars={}, init_conds=ics, times=times, 
                      num_days=num_days, days_to_keep=days_to_keep, step=step)

    # Run the median simulation
    p = list(pars.values())
    result_median = dde_model.simulate(p, times, fitting=False, reject=False)
    crh_drive = [dde_model.crh(t, t_s=p[-2], lambda_a=p[-4], lambda_s=p[-3], sigma=p[-1]) for t in times[int((day_len*warmup)/step):]]

    plotting.plot_model_output(m_n, result_median, times[int((day_len*warmup)/step):]-day_len*warmup, crh_drive, outdir=outdir, filename=f'SA_median_output', days_to_keep=1)

    for i in range(len(p)):
        param_name = list(pars.keys())[i]
        print(param_name)
        lb = pb[m_n][param_name][0]
        ub = pb[m_n][param_name][1]
        for variation in [0.001, 0.01, 0.1, 1, 10, 100]:  # 0.1%, 1%, 10%, 100%, 1000%, 10000%
            for direction in [-1, 1]:  # decrease and increase
                new_value = p[i] * (1 + direction * variation)
                if new_value < lb or new_value > ub:
                    print(f"Skipping value for {param_name} outside bounds: {new_value}")
                    continue
                p_new = p.copy()
                p_new[i] = new_value
                result_varied = dde_model.simulate(p_new, times, fitting=False, reject=False)
                crh_drive_varied = [dde_model.crh(t, t_s=p_new[-2], lambda_a=p_new[-4], lambda_s=p_new[-3], sigma=p_new[-1]) for t in times[int((day_len*warmup)/step):]]
                filename = f'SA_{param_name}_{"increase" if direction == 1 else "decrease"}_{int(variation*100)}pct_output'
                plotting.plot_model_output(m_n, result_varied, times[int((day_len*warmup)/step):]-day_len*warmup, crh_drive_varied, outdir=outdir, filename=filename, days_to_keep=1)

if __name__ == "__main__":
    m_n = args.model
    step = args.step
    warmup = args.warmup
    outdir = args.outdir
    if not os.path.exists(f'output/{outdir}'):
        os.makedirs(f'output/{outdir}')
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/plots')

    run_SA(m_n, step, warmup)
    