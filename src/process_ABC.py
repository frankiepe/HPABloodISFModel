import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import pandas as pd
import numpy as np
import json
import argparse
import methods.plotting as plotting
import methods.models as models
from methods import model_dict, day_len
import seaborn as sns
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = str, help='Select model for optimisation: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3a: HPAModelFEInterCBGAlbSimple, ' \
                    '3b: HPAModelFEInterCBGAlb, 4: HPAModelFEInterCBGAlbBloodISF, 5: HPAModelFEInterBothCBGAlbBloodISF, ' \
                    '6: HPAModelCRHSupp')
parser.add_argument('-i', '--ind', type = int, help='Select data index')
parser.add_argument('-w', '--warmup', type = int, default=5, help='No. of days of warmup (to reach steady state)')
parser.add_argument('-s', '--step', type=float, default=0.1, help='Step size for dde solver (must be sufficiently small for convergence)')
parser.add_argument('-o', '--outdir', type=str, help='Directory for plotting output')
parser.add_argument('-f', '--fixed', type=str, nargs='*', help='Parameters to fix (i.e. not fit)')
parser.add_argument('-t', '--thresh', type = int, default=2000, help='Objective threshold for ABC filter')
args = parser.parse_args()

def process_ABC(m_n, d_n, warmup, step, outdir, fixed, thresh, days_to_keep=1):
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
    dde_model = model(parameters=init_pars, fixed_pars=fixed_pars, init_conds=ics, times=times, num_days=num_days, days_to_keep=days_to_keep, step=step)

    df = pd.read_csv("output/" + outdir + f"/{model_dict[m_n]}/pars" + f"/all_pars_model{m_n}_dataID{d_n}.csv")
    df_thresh = df[df.Obj<=thresh]
    pars_accept = []
    for _, row in df_thresh.iterrows():
        pars_accept.append([float(row[i]) for i in init_pars.keys()])

    print(f"Accepted sets: {len(pars_accept)}")

    # Plot histograms of accepted parameters and correlation plot
    if len(pars_accept) > 0:
        for i in np.arange(0, len(init_pars.keys())):
            param_name = list(init_pars.keys())[i]
            param_values = [p[i] for p in pars_accept]

            if outdir is not None:
                hist_file = os.path.join("output/" + outdir + f"/{model_dict[m_n]}/plots", f"hist_{param_name}_model{m_n}_dataID{d_n}.png")
            else:
                hist_file = f"hist_{param_name}_model{m_n}_dataID{d_n}.png"

            plotting.plot_parameter_histograms(param_values, param_name, hist_file)
            print(f"Saved histogram for '{param_name}' to: {hist_file}")

        df = pd.DataFrame(pars_accept, columns=init_pars.keys())
        sns.pairplot(df, kind="kde", corner=True)
        plt.savefig("output/" + outdir + f"/{model_dict[m_n]}/plots" + f"/pairplot_model{m_n}_dataID{d_n}.png")
    else:
        print("No accepted parameters were found; histogram was not created.")

    # Plot accepted model trajectories
    plot_times = times[int((day_len/step)*(num_days-days_to_keep)):] - (day_len)*(num_days-days_to_keep)
    if outdir is not None:
        m_traj_file = os.path.join("output/" + outdir + f"/{model_dict[m_n]}/plots", f"m_traj_model{m_n}_dataID{d_n}.png")
    else:
        m_traj_file = f"m_traj_model{m_n}_dataID{d_n}.png"
    plotting.plot_model_trajectories_ABC(dde_model, plot_times, pars_accept, [], m_n, d_n, m_traj_file)

if __name__ == "__main__":
    m_n = args.model
    d_n = args.ind
    warmup = args.warmup
    step = args.step
    outdir = args.outdir
    fixed = args.fixed
    thresh = args.thresh
    if fixed is None:
        fixed = {}

    # Make directories if necessary
    if not os.path.exists(f'output/{outdir}'):
        os.makedirs(f'output/{outdir}')
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/pars'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/pars')
    if not os.path.exists(f'output/{outdir}/{model_dict[m_n]}/plots'):
        os.makedirs(f'output/{outdir}/{model_dict[m_n]}/plots')

    # Process ABC output
    process_ABC(m_n, d_n, warmup, step, outdir, fixed, thresh)
    