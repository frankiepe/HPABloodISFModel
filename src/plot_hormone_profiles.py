import pandas as pd
from matplotlib import pyplot as plt

for i in range(1, 8):
    # Read processes data
    dfISF = pd.read_csv(f'data/processed/HABS{i}_ISF.csv')
    dfBP = pd.read_csv(f'data/processed/HABS{i}_BP.csv')

    # Combine date and time columns
    dfISF['datetime'] = pd.to_datetime(dfISF['Date'] + ' ' + dfISF['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    dfBP['datetime'] = pd.to_datetime(dfBP['Date'] + ' ' + dfBP['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

    # Save dataframe columns as variables for plotting
    timesISF = dfISF['datetime']
    timesBP = dfBP['datetime']
    CORT = dfBP['Cortisol']
    Cortisone = dfBP['Cortisone']
    ACTH = dfBP['ACTH']
    mCORT = dfISF['mCortisol']
    mCortisone = dfISF['mCortisone']

    # Plot timeseries data
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot total ISF cortisol and cortisone
    ax1.plot(timesBP, CORT, label='Cortisol', color='blue', marker='o')
    ax1.plot(timesBP, Cortisone, label='Cortisone', color='red', marker='o')
    ax1.set_ylabel('nmol/L')
    ax1.set_title('Total Cortisol and Cortisone in Blood Plasma')

    # Plot free ISF cortisol and cortisone
    ax2.plot(timesISF, mCORT, label='mCortisol', color='blue', marker='o')
    ax2.plot(timesISF, mCortisone, label='mCortisone', color='red', marker='o')
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
