import pandas as pd
import numpy as np


# python preprocess_interactions_data.py
df = pd.read_csv('data/interactions_data.csv')
n_rows = df.shape[0]
ratings = np.array([1] * n_rows)
df['RATING'] = ratings
df = df[["USER_ID", "ITEM_ID", "RATING", "TIMESTAMP"]]
df.to_csv("benchmark/data/santander/interactions_data_preprocessed.csv", index=False, header=False)
print('created file', "benchmark/data/santander/interactions_data_preprocessed.csv")