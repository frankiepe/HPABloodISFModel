from matplotlib import pyplot as plt
import methods.data_processing as dp

for i in range(1, 8):
    # Read processed data
    timesISF, timesBP, CORT, Cortisone, ACTH, mCORT, mCortisone = dp.get_data(i)

    # Plot timeseries data
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot total ISF cortisol and cortisone
    ax1.plot(timesBP, CORT, label='Cortisol', color='blue', marker='o')
    ax1.plot(timesBP, Cortisone, label='Cortisone', color='red', marker='o')
    ax1.set_ylabel('nmol/L')
    ax1.set_title('Total Cortisol and Cortisone in Blood Plasma')

    # Plot free ISF cortisol and cortisone
    ax2.plot(timesISF, mCORT, label='mCortisol', color='blue', marker='o', alpha=0.5)
    ax2.plot(timesISF, mCortisone, label='mCortisone', color='red', marker='o', alpha=0.5)
    ax2.set_ylabel('nmol/L')
    ax2.set_title('Free Cortisol and Cortisone in ISF')

    ax3.plot(timesBP, ACTH, label='ACTH', color='orange', marker='o')
    ax3.set_ylabel('pmol/L')
    ax3.set_xlabel('Time')
    ax3.set_title('ACTH in Blood Plasma')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.suptitle('Cortisol, Cortisone and ACTH Levels Over Time')
    plt.savefig(f'figures/hormone_profiles/Cortisol_ACTH_plot_{i}.png')
