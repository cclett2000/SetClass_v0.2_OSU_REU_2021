# previous attempt; scrapped and started anew

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# numpy array print options (PRINT THE ENTIRE ARRAY DAMMIT)
np.set_printoptions(threshold=np.inf)

# begin run
os.system('cls' if os.name in ('nt', 'dos', 'window') else 'clear')
print('Script started...')

# https://realpython.com/k-means-clustering-python/
# TODO:
#  create the pipeline and yeet the data through it
#  create an algorithm that selects the best cluster amount
#  calculate SSE;

min_stat = 20
test_percent = 20
init_sample_size = 200
t = 1

with open('.data/dat_REU_MSC.csv', 'r') as file:
    data = pd.read_csv(file, header=None)
    data_identifier = data[len(data.columns) - 1]
    data_label = data[len(data.columns) - 2]

    del data[len(data.columns) - 2]
    del data[len(data.columns)]

    file.close()

data_arr = np.array(data)
identifier_arr = np.array(data_identifier)
label_arr = np.array(data_label)


training_data, testing_data, training_label, testing_label = train_test_split(data_arr,
                                                                              label_arr,
                                                                              test_size=(test_percent / 100),
                                                                              random_state=42)


# create fingerprints
# use a numpy array? create an array that contains an array for each identifier;

# this generates the number of anchors
# cluster_count is the number of anchors
# - need to make this work off of an init_sample opposed to entire data set;
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(label_arr)
# this will be used for n_clusters
# (number of classes);
cluster_count = len(label_encoder.classes_)

preprocessing = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=2, random_state=42))
])

clusterer = Pipeline([
    ('kmeans', KMeans(n_clusters=cluster_count,
                      init='k-means++',
                      n_init=12,
                      max_iter=120,
                      random_state=42))
])

main_pipe = Pipeline([
    ('preprocessing', preprocessing),
    ('clusterer', clusterer)
])

main_pipe.fit(data_arr)

preprocessed_data = main_pipe['preprocessing'].transform(data_arr)
predicted_labels = main_pipe['clusterer']['kmeans'].labels_
silhouette_score(preprocessed_data, predicted_labels)

# # for plotting
# pcadf = pd.DataFrame(main_pipe['preprocessing'].transform(data_arr),
#                      columns=['x-value', 'y-value'])
# pcadf['predicted_cluster'] = main_pipe['clusterer']['kmeans'].labels_
# pcadf['encoded_labels'] = label_encoder.inverse_transform(encoded_labels)
#
# plt.style.use('fivethirtyeight')
# plt.figure(figsize=(8, 8))
#
# scatter_plot = sns.scatterplot(x='x-value',
#                                y='y-value',
#                                s=50,
#                                data=pcadf,
#                                size=2,
#                                hue='predicted_cluster',
#                                style='encoded_labels',
#                                palette='Set2')
#
# scatter_plot.set_title('Test plot for sample data')
# plt.legend(bbox_to_anchor=(1.01, 1))
# plt.show()
#
print('Script finished.')
