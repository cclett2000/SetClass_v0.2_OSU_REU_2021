import os
import tarfile
import urllib
from urllib import parse, request

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

get_file = False

os.system('cls' if os.name in ('nt', 'dos', 'window') else 'clear')
print('running...')

def get_sample_file(get_file):
    if get_file:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
        archive_name = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"

        # build url
        full_download_url = urllib.parse.urljoin(url, archive_name)

        # this downloads file from url
        download_file = urllib.request.urlretrieve(full_download_url, archive_name)

        # extract the downloaded file
        tar = tarfile.open(archive_name, 'r:gz')
        tar.extractall()
        tar.close()
get_sample_file(get_file)

data_file = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
label_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"

# generates the data from .csv files, sklearn requires NumPy array
data = np.genfromtxt(data_file,
                     delimiter=',',
                     usecols=range(1, 20532),
                     skip_header=1)

labels = np.genfromtxt(label_file,
                       delimiter=',',
                       usecols=(1,),
                       skip_header=1,
                       dtype='str')

# encodes labels for usage
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# number of clusters for kmeans
cluster_count = len(label_encoder.classes_)

# preprocessor - uses PCA to perform dimensionality reduction
preprocessor = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=2, random_state=42))
])

# kmeans clusterer
#Notes:
# init: You’ll use "k-means++" instead of "random" to ensure centroids
#       are initialized with some distance between them. In most cases,
#       this will be an improvement over "random".
#
# n_init: You’ll increase the number of initializations to ensure you find a stable solution.
#
# max_iter: You’ll increase the number of iterations per initialization to ensure that k-means will converge.;
clusterer = Pipeline([
    ('kmeans', KMeans(n_clusters=cluster_count,
                      init='k-means++',
                      n_init=50,
                      max_iter=500,
                      random_state=42))
])

# the above pipes can be used to create a larger pipe
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('clusterer', clusterer)
])

# .fit will perform all pipeline steps on data
pipe.fit(data)

# performance evaluation using silhouette coefficient
preprocessed_data = pipe['preprocessor'].transform(data)
predicted_labels = pipe['clusterer']['kmeans'].labels_
silhouette_score(preprocessed_data, predicted_labels)

# calculate ARI; no clue, look into latter
adjusted_rand_score(encoded_labels, predicted_labels)

# plotting
pcadf = pd.DataFrame(pipe['preprocessor'].transform(data),
                     columns=['component_1', 'component_2'])

pcadf['predicted_cluster'] = pipe['clusterer']['kmeans'].labels_
pcadf['encoded_labels'] = label_encoder.inverse_transform(encoded_labels)

# plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(x='component_1',
                       y='component_2',
                       s=50,
                       data=pcadf,
                       size=2,
                       hue='predicted_cluster',
                       style='encoded_labels',
                       palette='Set2')

scat.set_title("Clustering results from TCGA Pan-Cancer\nGene Expression Data")

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)

plt.show()

print('done...')