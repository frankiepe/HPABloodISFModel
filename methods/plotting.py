from matplotlib import pyplot as plt
import methods.data_processing as dp
from methods import PARAMETER_BOUNDARIES as pb
import pandas as pd
import numpy as np
import os
from . import day_len, model_dict

def plot_model_output(m_n, res, times, crh_drive, outdir='model_output', filename='model_output', days_to_keep=1, plot_data=False, d_n=1):
    if m_n <=3 or m_n == 6:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        axes = [ax1, ax2]
        if m_n == 1 or m_n == 6:
            ax1.plot(times, res.T[1], label='Cortisol', color='blue')
            ax2.plot(times, res.T[0], label='ACTH', color='orange')
            ax1.set_title('Cortisol in Blood Plasma')
            ax2.set_title('ACTH in Blood Plasma')
        elif m_n == 2:
            ax1.plot(times, res.T[1], label='Cortisol', color='blue')
            ax1.plot(times, res.T[2], label='Cortisone', color='red')
            ax2.plot(times, res.T[0], label='ACTH', color='orange')
            ax1.set_title('Cortisol and Cortisone in Blood Plasma')
            ax2.set_title('ACTH in Blood Plasma')
        else:
            F_tot = res.T[1]+res.T[3]+res.T[4]
            E_tot = res.T[2]+res.T[5]+res.T[6]
            ax1.plot(times, F_tot, label='Total Cortisol', color='blue')
            ax1.plot(times, res.T[1], label='Free Cortisol', color='green')
            ax1.plot(times, E_tot, label='Total Cortisone', color='red')
            ax1.plot(times, res.T[2], label='Free Cortisone', color='yellow')
            ax2.plot(times, res.T[0], label='ACTH', color='orange')
            ax1.set_title('Cortisol and Cortisone in Blood Plasma')
            ax2.set_title('ACTH in Blood Plasma')
        ax1.set_ylabel('nmol/L')
        ax2.set_ylabel('pmol/L')
        ax2.set_xlabel('Time (minutes)')
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        axes = [ax1, ax2, ax3]
        F_tot = res.T[1]+res.T[3]+res.T[4]
        E_tot = res.T[2]+res.T[5]+res.T[6]
        ax1.plot(times, F_tot, label='Total Cortisol', color='blue')
        ax1.plot(times, res.T[1], label='Free Cortisol', color='green')
        ax1.plot(times, E_tot, label='Total Cortisone', color='red')
        ax1.plot(times, res.T[2], label='Free Cortisone', color='yellow')
        ax2.plot(times, res.T[9], label='Free Cortisol', color='blue', alpha=0.5)
        ax2.plot(times, res.T[10], label='Free Cortisone', color='red', alpha=0.5)
        ax3.plot(times, res.T[0], label='ACTH', color='orange')
        ax1.set_ylabel('nmol/L')
        ax2.set_ylabel('nmol/L')
        ax3.set_ylabel('pmol/L')
        ax3.set_xlabel('Time (minutes)')
        ax1.set_title('Cortisol and Cortisone in Blood Plasma')
        ax2.set_title('Cortisol and Cortisone in ISF')
        ax3.set_title('ACTH in Blood Plasma')

    if plot_data:
        print(f"Plotting data for individual #{d_n}...")
        timesISF, timesBP, CORT, Cortisone, ACTH, mCORT, mCortisone = dp.get_data(d_n)
        sBP = pd.to_datetime(pd.Series(timesBP))
        timesBP = (sBP - sBP.iloc[0]).dt.total_seconds() / 60
        sISF = pd.to_datetime(pd.Series(timesISF))
        timesISF = (sISF - sISF.iloc[0]).dt.total_seconds() / 60

        if m_n in [1,2,6]:
            ax1.plot(timesBP, CORT, label='Cortisol data', color='blue', marker='o')
            ax2.plot(timesBP, ACTH, label='ACTH data', color='orange', marker='o')
            if m_n == 2:
                ax1.plot(timesBP, Cortisone, label='Cortisone data', color='red', marker='o')
        elif m_n in [3,4,5]:
            ax1.plot(timesBP, CORT, label='Total Cortisol data', color='blue', marker='o')
            ax1.plot(timesBP, Cortisone, label='Total Cortisone data', color='red', marker='o')
            if m_n == 3:
                ax2.plot(timesBP, ACTH, label='ACTH data', color='orange', marker='o')
            elif m_n == 4 or m_n == 5:
                ax2.plot(timesISF, mCORT, label='Free Cortisol data', color='blue', marker='o', alpha=0.5)
                ax2.plot(timesISF, mCortisone, label='Free Cortisone data', color='red', marker='o', alpha=0.5)
                ax3.plot(timesBP, ACTH, label='ACTH data', color='orange', marker='o')

    for ax in axes:
        ax.set_xlim(list(times)[0], list(times)[-1])
        for i in range(days_to_keep):
            ax.axvline(x=day_len*i, color='gray', linestyle='--') 
        ax.legend()
        axn = ax.twinx()
        axn.plot(times, crh_drive, color = 'grey', alpha = 0.4)
        axn.set_ylabel('CRH drive', color = 'grey')

    plt.suptitle('Corticosteroid and ACTH Levels Over Time')

    savedir = f'output/' + outdir + f"/{model_dict[m_n]}/plots"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.savefig(f'{savedir}/{filename}.png')
    plt.close(fig)

def plot_parameter_histograms(param_values, param_name, reps, hist_file, bins=20):
    lb = pb[param_name][0]
    ub = pb[param_name][1]
    fig, ax = plt.subplots(figsize=(8, 5))
    _, bin_edges, _ = ax.hist(param_values, bins=bins, edgecolor='black', density=True, color="tab:orange", alpha=0.5, label='Posterior')
    bin_width = bin_edges[1] - bin_edges[0]
    ax.axhline(len(param_values)/(reps * bin_width), color = 'black', alpha = 0.5)
    ax.set_title(f"Histogram of accepted values for parameter '{param_name}'")
    ax.set_xlabel(param_name)
    ax.set_ylabel('Density')
    ax.set_xlim(lb, ub)
    ax.fill_between(np.linspace(lb, ub, 100), 0, len(param_values)/(reps * bin_width) , color='tab:blue', alpha=0.4, label='Prior')
    ax.legend()
    fig.tight_layout()
    fig.savefig(hist_file)
    plt.close(fig)