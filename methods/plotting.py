from matplotlib import pyplot as plt
from . import day_len, model_dict

def plot_model_output(m_n, res, times, crh_drive, filename='model_output', days_to_keep=1):
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

    for ax in axes:
        ax.set_xlim(times[0], times[-1])
        for i in range(days_to_keep):
            ax.axvline(x=day_len*i, color='gray', linestyle='--') 
        ax.legend()
        axn = ax.twinx()
        axn.plot(times, crh_drive, color = 'grey', alpha = 0.4)
        axn.set_ylabel('CRH drive', color = 'grey')

    plt.suptitle('Corticosteroid and ACTH Levels Over Time')

    plt.savefig(f'figures/model_output/{model_dict[m_n]}/{filename}.png')
