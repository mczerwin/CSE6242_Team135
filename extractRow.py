import pandas as pd

df = pd.read_csv('train.csv')

df = df[df['Id'] == '0007de18844b0dbbb5e1f607da0606e0']

df.to_csv('row.csv', index=False)