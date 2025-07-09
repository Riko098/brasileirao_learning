import pandas as pd
df = pd.read_csv('C:/Users/erico/Documents/desenv/ia/brasileirao/brasileirao_ml_data.csv')
print(df['vencedor'].value_counts())