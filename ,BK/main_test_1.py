# another previous version, scrapped and started anew

import os

import numpy as np
#np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kneed import KneeLocator

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KDTree
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


# begin run
os.system('cls' if os.name in ('nt', 'dos', 'window') else 'clear')
print('Script started...')


# init_samples = will be a sample of the data to generate anchors
# min_stat = minimum number of data points that a fingerprint needs in order
#            to be assigned to a class;
# test_percent = percent of the overall data set aside for testing, remaining will be used
#                in the training model
# ticks = 't' in algorithm, will be each iteration of loop for data?
# label_encoder = this is the variable for calling sklearn's label encoder;
min_stat = 20
test_percent = 20
sample_percent = 20
ticks = 0
label_encoder = LabelEncoder()


# temp = referred to as 'R' in algorithm, this will contain the fingerprints that do not
#        contain at least the number of data points that are specified in 'min_stat' above
# profile = referred to as 'P' in algorithm, this will contain fingerprints that surpass the
#           value specified in 'min_stat'
#           - NOTE: the fingerprint is added once the number of data points surpass 'min_stat'
#                   by one;
temp = []
profile = []


# get file
def get_file():
    with open('.data/dat_REU_MSC.csv', 'r') as file:
        data = pd.read_csv(file, header=None)
        data_identifier = data[len(data.columns) - 1]
        data_label = data[len(data.columns) - 2]

        del data[len(data.columns) - 1]
        del data[len(data.columns) - 2]

        file.close()
        return data, data_label, data_identifier


data, label, identifier = get_file()

# Note: in paper 'q' is the number of anchors,
#       here 'cluster_count' is the number of anchors;
def generate_anchors():
    enable_SC = True
    init_sample_size = round(len(data) * (sample_percent / 100))
    init_sample = data.scaled_sample(n=init_sample_size)
    init_sample_raw_labels = init_sample[len(init_sample.columns) - 2]
    init_sample_labels = label_encoder.fit_transform(init_sample_raw_labels)

    # use SSE to generate optimal number of anchors
    n_init = 10
    kmeans_kwargs = {
        'init': 'random',
        'n_init': n_init,
        'max_iter': 300,
        'random_state': 42
    }

    sse = []
    for k in range(1, (n_init)):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(init_sample)
        sse.append(kmeans.inertia_)

    # identifies elbow point; chooses best number of anchors
    elbow_detect = KneeLocator(range(1, (n_init)),
                               sse,
                               curve='convex',
                               direction='decreasing')

    if enable_SC is True:
        # silhouette coefficient; cluster cohesion
        silhouette_coefficients = []

        for k in range(2, n_init):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(init_sample)
            score = silhouette_score(init_sample, kmeans.labels_)
            silhouette_coefficients.append(score)

            plt.style.use('fivethirtyeight')
            plt.plot(range(2, n_init), silhouette_coefficients)
            plt.xticks(range(2, n_init))
            plt.xlabel('# of Clusters')
            plt.ylabel('Silhouette Coefficient')
            plt.show()

    return elbow_detect.elbow
cluster_count = generate_anchors()

train_data, test_data, train_label, test_label = train_test_split(data,
                                                                  label,
                                                                  test_size=(test_percent / 100),
                                                                  random_state=42)


# 'create_fingerprint' creates arrays each ID and labels each ID to denote that
# it is an ID, this may not be viable but the idea is to assign feature sets and
# accompanying labels to each array if the ID matches;
def create_fingerprint():
    tree = KDTree(train_data, leaf_size=2)
    dist, ind = tree.query(train_data[:1], k=3)


create_fingerprint()





# TODO: -find a way to assign feature/label sets to each array within the array
#       -use anchor cal thingy from link (SSE);

#print(testing_fingerprint)




print('Script finished.')
