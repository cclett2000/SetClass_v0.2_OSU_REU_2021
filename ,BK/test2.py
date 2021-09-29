# yet another previous attempt, naturally, scrapped and started anew

import os

import numpy as np
#np.set_printoptions(threshold=np.inf)

import pandas as pd

from kneed import KneeLocator

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# begin run
os.system('cls' if os.name in ('nt', 'dos', 'window') else 'clear')
print('Script started...')

min_stat = 20
test_percent = 20
sample_percent = 20
ticks = 0
label_encoder = LabelEncoder()

temp = []
profile = []

def get_file():
    with open('.data/dat_REU_MSC.csv', 'r') as file:
        data = pd.read_csv(file, header=None)
        data_identifier = data[len(data.columns) - 1]
        data_label = data[len(data.columns) - 2]

        del data[len(data.columns) - 1]
        del data[len(data.columns) - 2]

        file.close()
        return data, data_label, data_identifier

data, label, id = get_file()

data = np.array(data)
label = np.array(label)
id = np.array(id)

while ticks < 200:
    ticks += 1
    print(data[ticks - 1])



