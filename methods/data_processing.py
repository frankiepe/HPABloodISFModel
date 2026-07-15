import pandas as pd

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
