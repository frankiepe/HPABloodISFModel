import pandas as pd

def process_data():
    for i in range(1, 8):
        # Read and process BP data
        df = pd.read_csv(f'data/files/HABS{i}.csv')
        dfBP = df[['Date', 'Time', 'Cortisol', 'ACTH', 'Cortisone']].copy()
        dfBP = dfBP.dropna(subset=['Date', 'Time', 'Cortisol', 'ACTH', 'Cortisone'])
        dfBP.to_csv(f'data/processed/HABS{i}_BP.csv', index=False)

        # Read and process ISF data
        dfISF = df[['Date', 'Time', 'mCortisol', 'mCortisone']].copy()
        dfISF = dfISF.dropna(subset=['Date', 'Time', 'mCortisol', 'mCortisone'])
        dfISF.to_csv(f'data/processed/HABS{i}_ISF.csv', index=False)

def get_data(d_n):
    # Read processed data
    dfISF = pd.read_csv(f'data/processed/HABS{d_n}_ISF.csv')
    dfBP = pd.read_csv(f'data/processed/HABS{d_n}_BP.csv')

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

    return timesISF, timesBP, CORT, Cortisone, ACTH, mCORT, mCortisone