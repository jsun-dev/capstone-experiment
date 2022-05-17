import os
import pickle
import numpy as np
import pandas as pd

ROOT_PATH = 'MIDV-2020-Text'

CSV_PATH = os.path.join(ROOT_PATH, 'MIDV-2020-Text.csv')

# Split the dataset into training and testing
df = pd.read_csv(CSV_PATH)
np.random.seed(112)
df_train, df_test = np.split(df.sample(frac=1, random_state=42),
                             [int(0.8 * len(df))])

# Save the training and testing datasets
df_train.to_csv(os.path.join(ROOT_PATH, 'train.csv'), index=False)
df_test.to_csv(os.path.join(ROOT_PATH, 'test.csv'), index=False)

# Save the labels
categories = df['category'].unique()
index = [i for i in range(len(categories))]
labels = dict(zip(categories, index))
with open(os.path.join(ROOT_PATH, 'labels.pkl'), 'wb') as f:
    pickle.dump(labels, f)
