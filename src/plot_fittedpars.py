import os
import sys
top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, top_level_dir)
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import re
import argparse
from methods import model_dict
import seaborn as sns
from methods import PARAMETER_BOUNDARIES as pb

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
        min_obj = 1e9
        for j in range(1, reps+1):
            fit = pd.read_csv(f"output/{outdir}/{model_dict[m_n]}/fits/fit_dataid{i}_seed{j}.csv")
            score = fit['score'][0]
            if score < min_obj:
                vals = np.fromstring(fit['values'][0].strip("[]"), sep=" ")
                min_obj = score
        best_pars.append(vals)
    fitted_pars = re.findall(r"'([^']*)'", fit['fitted_pars'][0])
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
        sns.violinplot(x=df[param], ax=axes[i], color='skyblue', inner='point', orient='h')
        lb = pb[m_n][param][0]
        ub = pb[m_n][param][1]
        axes[i].axvline(lb)
        axes[i].axvline(ub)
        axes[i].set_title(param)
        axes[i].set_ylabel('')
        if param in ['k_Fon', 'k_Eon']:
            axes[i].set_xscale('log')

    plt.suptitle(f"Violin plots (Fitting model {m_n})")
    plt.tight_layout()
    plt.savefig(f'output/{outdir}/{model_dict[m_n]}/violin_plots.png')
    
    return 0

if __name__ == "__main__":
    # read parsed variables
    m_n = args.model
    num_inds = args.ninds
    reps = args.nreps
    outdir = args.outdir

    plot_pars(m_n, num_inds, reps, outdir)

    