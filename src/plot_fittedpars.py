import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse
from methods import model_dict
import seaborn as sns
from methods import PARAMETER_BOUNDARIES as pb
import json

parser = argparse.ArgumentParser(description='Fit models to data.')
parser.add_argument('-m', '--model', type = str, help='Select model for optimisation: ' \
                    '1: BaseHPAModel, 2: HPAModelFEInter, 3a: HPAModelFEInterCBGAlbSimple, ' \
                    '3b: HPAModelFEInterCBGAlb, 4: HPAModelFEInterCBGAlbBloodISF, 5: HPAModelFEInterBothCBGAlbBloodISF, ' \
                    '6: HPAModelFEInterISFOnlyCBGAlbBloodISF')
parser.add_argument('-n', '--ninds', type=int, help='Number of datasets fitted')
parser.add_argument('-r', '--nreps', type=int, help='Number of fitting reps')
parser.add_argument('-o', '--outdir', type=str, help='Directory for plotting output')
args = parser.parse_args()

def plot_pars(m_n, num_inds, reps, outdir):
    best_pars = []
    for i in range(1, num_inds+1):
        print(f'Data: {i}')
        min_obj = 1e9
        for j in range(1, reps+1):
            pars_file = f"output/{outdir}/{model_dict[m_n]}/fits/fit_dataid{i}_seed{j}.json"
            try:
                # Load config
                with open(pars_file, 'r') as f:
                    config = json.load(f)
                score = config.get('score', {})
                if score < min_obj:
                    print(f'Rep: {j}')
                    pars = config.get('parameters', {})
                    min_obj = score
            except FileNotFoundError:
                print(f"File not found: {pars_file}")
                continue
        best_pars.append(pars)
    fitted_pars = pars.keys()
    df = pd.DataFrame(best_pars, columns=fitted_pars)

    if m_n == '1':
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    if m_n == '2':
        fig, axes = plt.subplots(3, 6, figsize=(14, 10))
    if m_n == '3a':
        fig, axes = plt.subplots(4, 6, figsize=(14, 10))
    if m_n == '3b':
        fig, axes = plt.subplots(5, 5, figsize=(14, 10))
    if m_n == '4':
        fig, axes = plt.subplots(4, 7, figsize=(14, 10))
    if m_n == '5':
        fig, axes = plt.subplots(4, 8, figsize=(14, 10))
    if m_n == '6':
        fig, axes = plt.subplots(4, 8, figsize=(14, 10))
    axes = axes.flatten()

    for i, param in enumerate(fitted_pars):
        if (m_n == 5 and param in ['gamma_f_i', 'gamma_e_i']) or param in ['k_Fon', 'k_Eon']:
            sns.violinplot(x=df[param], ax=axes[i], color='skyblue', inner='point', orient='h', log_scale=True)
        else:
            sns.violinplot(x=df[param], ax=axes[i], color='skyblue', inner='point', orient='h')
        lb = pb[m_n][param][0]
        ub = pb[m_n][param][1]
        axes[i].axvline(lb)
        axes[i].axvline(ub)
        axes[i].set_title(param)
        axes[i].set_ylabel('')
        if (m_n == 5 and param in ['gamma_f_i', 'gamma_e_i']) or param in ['k_Fon', 'k_Eon']:
            axes[i].set_xscale('log')

    plt.suptitle(f"Violin plots (Fitting model {m_n})")
    plt.tight_layout()
    plt.savefig(f'output/{outdir}/{model_dict[m_n]}/violin_plots.png')
    
    # Save median parameters and score to JSON
    pars = df.median()
    path = f"output/{outdir}/{model_dict[m_n]}/median_pars.json"
    pars_list = pars.tolist() if isinstance(pars, np.ndarray) else list(pars)
    out_data = {
        "parameters": dict(zip(fitted_pars, pars_list))
    }
    with open(path, "w") as f:
        json.dump(out_data, f, indent=4)

if __name__ == "__main__":
    # read parsed variables
    m_n = args.model
    num_inds = args.ninds
    reps = args.nreps
    outdir = args.outdir

    plot_pars(m_n, num_inds, reps, outdir)

    