# Charles Lett Jr.
# REU 2021
# Project to learn the basics of classification (machine learning concept) and to
# implement Dr. Charu C. Aggarwal's Setwise Stream Classification Algorithm
# https://www.researchgate.net/publication/266660383_The_setwise_stream_classification_problem?enrichId=rgreq-32f5c639b9bb483b8bdb0fbc0f364ad3-XXX&enrichSource=Y292ZXJQYWdlOzI2NjY2MDM4MztBUzoyNTk5OTk3Nzk3ODI2NjBAMTQzOTAwMDE4NjU3OA%3D%3D&el=1_x_2&_esc=publicationCoverPdf;

import os
import random
import matplotlib.pyplot as plt
from timeit import default_timer as rtimer
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from math import floor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#----------------------------------------------------------------------------------------------------------
# begin run
# 'os.system' only works in command prompt (maybe linux terminal)
#            it clears the command window each run for readability;
# os.system('cls' if os.name in ('nt', 'dos', 'window') else 'clear')
print(" *** Running 'main.py' *** ")

runtime_start = rtimer()
#----------------------------------------------------------------------------------------------------------
display_fingerprint = False
show_profiler_info = False
show_profiler_info_simple = False

label_encoder = LabelEncoder()
anchor_count = 90 # a.k.a. 'k'
profile_count = 6

min_stat = 20
test_percent = 20
sample_percent = 20

# holds fp
fingerprint = []
tmp_fingerprint = []

test_fingerprint = []
test_tmp_fingerprint = []

# for fp init
id = []
fv = []
label = []
anchor = []

def get_file():
    print('>> [FILE PREPARATION]')
    file_type = 1

    if file_type == 0:
        # FIRST SAMPLE DATA
        def sample():
            with open('.data/dat_REU_MSC.csv', 'r') as file:
                data = pd.read_csv(file, header=None)
                data_feature = data[data.columns[:15]]
                data_identifier = data[len(data.columns) - 1]
                data_label = data[len(data.columns) - 2]
                data_len = len(data)

                # X = data or feature set, y = labels or targets
                X_train, X_test, y_train, y_test, train_id, test_id = train_test_split(data_feature,
                                                                                       data_label,
                                                                                       data_identifier,
                                                                                       test_size= (test_percent / 100)
                                                                                       )
            return file_type, data_len, X_train, X_test, y_train, y_test, train_id, test_id
        return sample()

    if file_type == 1:
        # UCI-HAR DATA
        def uci_har():
            def prep_train():
                # entity
                with open('.data/UCI_HAR/train/subject_train.txt') as entity_file:
                    entity_file = entity_file.readlines()
                    entities = []
                    entities = [entity.strip() for entity in entity_file]

                # header
                with open('.data/UCI_HAR/features.txt') as hdr_file:
                    hdr_file = hdr_file.readlines()
                    train_hdr = []
                    train_hdr = [line.strip() for line in hdr_file]

                # feature vector
                with open('.data/UCI_HAR/train/X_train.txt') as X_train_file:
                    df_train_x = pd.read_csv(X_train_file, sep='\s+', names=train_hdr, header=None)

                # label
                with open('.data/UCI_HAR/train/y_train.txt') as y_train_file:
                    df_train_y = pd.read_csv(y_train_file, header=None)

                return df_train_x, df_train_y, entities
            X_train, y_train, train_id = prep_train()

            def prep_test():
                # entity
                with open('.data/UCI_HAR/test/subject_test.txt') as entity_file:
                    entity_file = entity_file.readlines()
                    entities = []
                    entities = [entity.strip() for entity in entity_file]

                # header
                with open('.data/UCI_HAR/features.txt') as hdr_file:
                    hdr_file = hdr_file.readlines()
                    train_hdr = []
                    train_hdr = [line.strip() for line in hdr_file]

                # feature vector
                with open('.data/UCI_HAR/test/X_test.txt') as X_test_file:
                    df_test_x = pd.read_csv(X_test_file, sep='\s+', names=train_hdr, header=None)

                # label
                with open('.data/UCI_HAR/test/y_test.txt') as y_test_file:
                    df_test_y = pd.read_csv(y_test_file, header=None)

                return df_test_x, df_test_y, entities
            X_test, y_test, test_id = prep_test()

            data_len = len(X_test) + len(X_train)

            # show ids
            # print('labels:', set(y_train.values.ravel()))
            # print('entity_id (train):', set(train_id))
            # print('entity_id (test):', set(test_id))

            return file_type, data_len, X_train, X_test, y_train, y_test, train_id, test_id
        return uci_har()
file_type, data_len, X_train, X_test, y_train, y_test, train_id, test_id = get_file()

def prep_sample():
    print('>> [SAMPLE INITIALIZATION]')
    global X_train, y_train

    # preps the init_sample for anchor generation
    init_sample_size = round(data_len * (sample_percent / 100))

    # collect sample
    init_sample = X_train.sample(n=init_sample_size)
    init_sample_raw_labels = y_train.sample(n=init_sample_size)

    # remove collected sample from dataset
    X_train = X_train.drop(init_sample.index)
    y_train = y_train.drop(init_sample_raw_labels.index)

    # format labels
    init_sample_raw_labels = init_sample_raw_labels.values.ravel()

    if file_type == 1:
        init_sample_identifier = random.sample(train_id, init_sample_size)
    else:
        init_sample_identifier = train_id.sample(n=init_sample_size)


    # label encoding; will be more useful in the future; '.values.ravel() creates a 1D array'
    init_sample_encoded_labels = label_encoder.fit_transform(init_sample_raw_labels)

    # scales the sample (feature)
    scaler = StandardScaler()
    scaled_sample = scaler.fit_transform(init_sample)

    return init_sample, scaled_sample, init_sample_identifier, \
           init_sample_raw_labels, init_sample_encoded_labels, init_sample_size
sample, scaled_sample, sample_id, sample_label, encoded_sample_label, sample_size = prep_sample()
classes = label_encoder.classes_

def data_scaler():
    print('>> [DATA SCALING]')

    # scales data for use in kmeans, may not be needed
    scaler = StandardScaler()
    tr_data = scaler.fit_transform(X_train)
    ts_data = scaler.fit_transform(X_test)

    return tr_data, ts_data
X_train_sc, X_test_sc = data_scaler()

def generate_anchors():
    print('>> [ANCHOR GENERATION]')

    # type (1 = anchor estimation or 2 = anchor generation): selects which plot to display, any other
    # input will disable graph display
    type = 0
    model = KMeans()
    n_init = 10

    kmeans_args = {'n_clusters': anchor_count,
                   'init': 'k-means++',
                   'n_init': n_init,  # runs the kmeans algorithm (n_init) number of times with different
                                      # random centroids (anchors) to choose the model with lowest SSE;
                   'max_iter': 300,  # number of iterations, can be slow if set too high and if convergence
                                     # is never reached;
                   'tol': 1e-04,  # controls tolerance (in regard to changes in the within-cluster SSE)
                                  # 1e-04 = 0.0001;
                   }
    kmeans = KMeans(**kmeans_args)

    # clusters and predicts the closest anchor for each sample?
    X_samp = scaled_sample
    y_km = kmeans.fit_predict(X_samp)
    centers = kmeans.cluster_centers_

    def anchor_est_plot():
        from yellowbrick.cluster import KElbowVisualizer
        # generates graph to estimate optimal number of clusters
        vis = KElbowVisualizer(model, k=(2,11), timings=True)

        vis.fit(X_samp)
        vis.show()

        # ask whether to use distortion or silhouette metric

    def anchor_gen_plot():
        # link: https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c
        # 4 clusters (should create a loop for this)
        # kmeans prediction model

        plt.scatter(X_samp[y_km == 0, 0],
                    X_samp[y_km == 0, 1],
                    s=50, c='lightgreen',
                    marker='s', edgecolors='black',
                    label= 'cluster 1')
        plt.scatter(X_samp[y_km == 1, 0],
                    X_samp[y_km == 1, 1],
                    s=50, c='orange',
                    marker='s', edgecolors='black',
                    label= 'cluster 2')
        plt.scatter(X_samp[y_km == 2, 0],
                    X_samp[y_km == 2, 1],
                    s=50, c='lightblue',
                    marker='s', edgecolors='black',
                    label= 'cluster 3')
        plt.scatter(X_samp[y_km == 3, 0],
                    X_samp[y_km == 3, 1],
                    s=50, c='yellow',
                    marker='s', edgecolors='black',
                    label= 'cluster 4')
        plt.scatter(X_samp[y_km == 4, 0],
                    X_samp[y_km == 4, 1],
                    s=50, c='pink',
                    marker='s', edgecolors='black',
                    label= 'cluster 5')

        # centroids
        # plt.scatter(kmeans.cluster_centers_[:, 0],
        #             kmeans.cluster_centers_[:, 3],
        #             s=250, marker='*',
        #             c='red', edgecolors='black',
        #             label='Anchors')

        # render plots
        # plt.legend(scatterpoints=1)
        plt.grid()
        plt.show()

    if type == 1:
        anchor_est_plot()
        print('Set "k" according to the graph and rerun with graphing disabled\nExiting...')
        exit()

    elif type == 2:
        anchor_gen_plot()
        exit()

    return y_km, centers, kmeans
sample_cluster_pred, cluster_centers, kmeans = generate_anchors()

# FP Key = [ID][FV#][Class][Anchor][Class][Profile]
def init_fingerprint():
    print('>> [FINGERPRINT INITIALIZATION]')
    fp_tick = 0
    id_index = 0
    sample_id_iter = 0

    # anchor distance calculator
    NN = NearestNeighbors(n_neighbors=len(cluster_centers) + 1, algorithm='kd_tree')

    # initializes fingerprints
    if file_type == 0:
        sample_id_iter = sample_id.iloc
    if file_type == 1:
        sample_id_iter = sample_id

    for fp_tick in range(len(sample)):
        if sample_id_iter[fp_tick] not in id:
            id.append(sample_id_iter[fp_tick])
            label.append(sample_label[fp_tick] - 1)
            fv.append(0)
            anchor.append([0] * anchor_count)

    # appends cluster centers to array then compares the centers to the sample which is the
    # first element of the array;
    nn_samp = [None]
    for cluster in cluster_centers:
        nn_samp.append(cluster)

    # increments data using i, i will increment until max number of data is called, code will be executed
    # during each increment depending on conditions;
    i = 0
    while i < len(sample_id):
        nn_samp[0] = scaled_sample[i]

        # if id position equals the length of the fingerprint id array reset to zero;
        if id_index == len(id) - 1:
            id_index = 0

        # increment the id list index until the data id and fingerprint id match;
        if sample_id_iter[i] != id[id_index]:
            id_index += 1

        # if data id and fingerprint id match, execute the below code;
        if sample_id_iter[i] == id[id_index]:
            dist, ind = NN.fit(nn_samp).kneighbors(nn_samp)
            closest_anchor = ind[0][1]

            # active updater - init
            anchor[id_index] = [elem * fv[id_index] for elem in anchor[id_index]] # multiply
            anchor[id_index][closest_anchor - 1] += 1 # inc
            fv[id_index] += 1 # inc
            anchor[id_index] = [elem / fv[id_index] for elem in anchor[id_index]] # divide

            i += 1

    # assigns fingerprints to fp list
    # tmp if less than min_stat;
    for i in range(len(id)):
        if fv[i] < min_stat:
            tmp_fingerprint.append(list([id[i], fv[i], label[i], anchor[i]]))
        if fv[i] > min_stat:
            fingerprint.append(list([id[i], fv[i], label[i], anchor[i]]))

    # # display fingerprint info - used for debugging
    # if display_fingerprint:
    #     print('*** FP Initialization ***')
    #     for fp in range(len(fingerprint)):
    #         print('[APP]', fingerprint[fp], '[ AncSum:', round(sum(fingerprint[fp][3]), 4), ']')
    #
    #     for fp_tmp in range(len(tmp_fingerprint)):
    #         print('[TMP]', fp_tmp,  tmp_fingerprint[fp_tmp], '[ AncSum:', round(sum(tmp_fingerprint[fp_tmp][3]), 4), ']')
init_fingerprint()

def init_profiler():
    print('>> [PROFILE INITIALIZATION]')
    class_count = len(classes)

    def set_classes():
        # create structure for class profiles
        '''class_arr[class_index][profile_index]'''
        class_arr, class_fv = [], []
        class_arr, class_fv = [[] for i in range(len(classes))], \
                              [0 for j in range(len(classes))]

        return class_arr, class_fv
    class_arr, class_profile = set_classes()

    def set_profiles(class_arr):
        from heapq import nlargest
        global profile_count
        # *** distribute number of profiles over number of classes ***
        # use ratio for calculation, if 8 profiles and 6 classes
        # must distribute the 8 profiles into the 6 classes
        #
        # (for 2 classes with 8 profiles) if 3:1 ratio where class0 = 3/4 of init_sample
        # and class1 = 1/4 of init_sample
        #   - calculations
        #       ***formula = (num of class / total class or entity) * profile_count***
        #       > class0 = (3/4) * 8 = 6 profiles for class0
        #       > class1 = (1/4) * 8 = 2 profiles for class1
        #       > class0 + class1 = 8 **if the sum of class profiles equals
        #                               the profile_count, things are great!**;

        # fills class_profile
        # class_profile holds the total number of feature vectors per class (aka label);
        for i in range(len(fingerprint)):
            # print(i, fingerprint[i][2])
            class_profile[(fingerprint[i][2])] += fingerprint[i][1]

        # profile distribution calculation [formula = (class / total) * profile_count]
        class_total = sample_size
        for j in range(len(class_profile)):
            class_profile[j] = floor((class_profile[j] / class_total) * profile_count)

        # print('[Before Check] class_profile (Number of profiles):', class_profile)
        # checks for profile_count match, if not add profile based on nlargest
        while sum(class_profile) < profile_count:
            tmp_lst = class_profile[:] # cloned class_fv list
            tmp_max = nlargest(round(len(class_profile) * 0.12),class_profile)  # gets largest items (12% of class_fv)
            # tmp_min = nsmallest(round(len(class_profile) * 0.085), class_profile) # gets smallest items (8.5% of class_fv)
            for x in range(len(tmp_lst)):
                for y in range(len(tmp_max)):
                    if class_profile[x] == 0:
                        class_profile[x] += 1
                    if sum(class_profile) == profile_count:
                        break
                    elif tmp_lst[x] == tmp_max[y]:
                        class_profile[x] += 1
        # print('[After Check] class_profile (Number of profiles):', class_profile)

        # create empty profiles; add profiles if number not reached
        for k in range(len(class_arr)):
            class_arr[k] = [[] for x in range(class_profile[k])]

        # profile display
        # for i in range(len(class_arr)):
        #     print('(' + str(i) + ')', 'class_arr', class_arr[i], '|| class_profile:', class_profile[i])
        # print('requested profiles:', profile_count)
        # print('calculated profiles:', sum(class_profile))
        # exit()

        return class_arr, class_profile
    class_arr, class_profile = set_profiles(class_arr)

    def fit_profiles(class_arr, class_profile, class_count):
        # reset to 0 if = class_count - 1
        class_ind = 0

        # track profile pos; inc by 1 if added to, reset if = profile_max
        profile_track = [0 for i in range(class_count)]

        # profiles per class (class 0...n)
        profile_max = [p_count for p_count in class_profile]

        # for class tracker profiling:
        # - uses an array that contains '0' * number of classes (or labels)
        # - each '0' represents the profile that is currently selected to add fingerprint to
        # - every time the profile is added to, the profile is incremented by 1, if profile equals
        #   the number of profiles specified, then reset to 0
        # - can hopefully use the same algorithm to update profiles using train/test data;

        i = 0
        while i < len(fingerprint):
            if class_ind == class_count - 1:
                class_ind = 0

            if class_ind != fingerprint[i][2]:
                class_ind += 1

            if class_ind == fingerprint[i][2]:
                # print(profile_track, profile_max, class_ind)
                # ^ when using the above, profile_track = profile_max - 1; due to python starting at 0
                class_arr[class_ind][profile_track[class_ind]].append(fingerprint[i][3])
                fingerprint[i].append(class_ind)
                fingerprint[i].append(profile_track[class_ind])

                if profile_track[class_ind] == profile_max[class_ind] - 1:
                    profile_track[class_ind] = 0
                else:
                    profile_track[class_ind] += 1
                    # print(profile_track)
                i += 1

        # display fingerprint info - used for debugging
        if display_fingerprint:
            print('*** FP Update ***')
            for fp in range(len(fingerprint)):
                print('[APP]', fp, fingerprint[fp], '[ AncSum:', round(sum(fingerprint[fp][3]), 4), ']')

            for fp_tmp in range(len(tmp_fingerprint)):
                print('[TMP]', fp_tmp, fp_tmp, tmp_fingerprint[fp_tmp], '[ AncSum:',
                      round(sum(tmp_fingerprint[fp_tmp][3]), 4), ']')

        return class_arr
    class_arr = fit_profiles(class_arr, class_profile, class_count)

    def calculations():
        for class_ in range(len(classes)):
            for profile_ in range(class_profile[class_]):
                # index tracking
                index = 'Class: ' + str(class_arr.index(class_arr[class_]) + 1) + \
                        ', Profile: ' + str(profile_ + 1)
                # data storage
                data = class_arr[class_][profile_]
                data_len = len(data)

                # precheck - sum of anchor sets
                data_check = list(map(sum, class_arr[class_][profile_]))

                # sum of all anchors
                data_sum = list(map(sum, zip(*class_arr[class_][profile_])))

                # FIX average of anchor sum
                data_avg = [(x / data_len) for x in data_sum]
                data_avg_chk = sum(data_avg)

                if show_profiler_info:
                    print(index, '\n',
                          'DataArray:', data, '\n',
                          'DataCheck:', data_check, '\n',
                          'DataSum:', data_sum, '\n',
                          'DataAvg:', data_avg, '\n',
                          'DataAvgChk:', round(data_avg_chk, 4), '\n',
                          'DataLen:', data_len, '\n',)

                # setup array
                data.clear()
                data.append(data_avg)
                data.append(data_len)
    calculations()

    def debug_stuph():
        # Display - Debugging
        # for class_ in range(len(class_arr)):
        #     for profile_ in class_arr[class_]:
        #         print('Class', class_ + 1,
        #               ', Profile', class_arr[class_].index(profile_) + 1,
        #               '\n', profile_, '\n')

        # print()

        # shows class dataset # Check with mentor
        for i in range(len(class_arr)):
            print('C' + str(i + 1), 'P' + str(len(class_arr[i])), str(class_arr[i]))

        # exit()
    # debug_stuph()

    return class_arr
class_profile = init_profiler()

def train_work():
    # fp ex: ['1', 93, 1, [0.26881720430107525, 0.27956989247311825, 0.15053763440860216...]
    print('>> [TRAINING MODEL]')

    def fingerprint_updater(class_profile):
        label_ind = 1

        # training data
        data = X_train_sc
        tr_label = y_train.values.ravel()
        tr_label_len = list(set(label))[-1]
        tr_id = train_id
        tr_tic = 0

        # nearest neighbor init
        NN_clus = NearestNeighbors(n_neighbors=len(cluster_centers) + 1, algorithm='kd_tree')
        NN_prof = NearestNeighbors(n_neighbors= 2, algorithm='kd_tree')

        nn_cluster_data = [None]

        for cluster in cluster_centers:
            nn_cluster_data.append(cluster)

        # update fingerprint
        while tr_tic < len(data):
            nn_cluster_data[0] = data[tr_tic]
            # add to fingerprint if not present
            if tr_id[tr_tic] not in id:
                id.append(tr_id[tr_tic])
                fv.append(1)
                label.append(tr_label[tr_tic])
                anchor.append([0] * anchor_count)

            for fp in range(len(fingerprint)):
                # print(tr_id[tr_tic], fp, fingerprint[fp])
                # increment label
                # print(fp, fingerprint[fp])
                if tr_id[tr_tic] != fingerprint[fp][0]:
                    label_ind += 1

                # assign train sets to fingerprint
                if tr_id[tr_tic] == fingerprint[fp][0]:
                    prev_fp = fingerprint[fp].copy()
                    # print('Previous FP', prev_fp)

                    # distance between selected data point and anchors
                    clus_dist, clus_ind = NN_clus.fit(nn_cluster_data).kneighbors(nn_cluster_data)
                    closest_anchor = clus_ind[0][1]

                    # fingerprint updater - training
                    fingerprint[fp][3] = [elem * fingerprint[fp][1] for elem in fingerprint[fp][3]] # multiply
                    fingerprint[fp][3][closest_anchor - 1] += 1
                    fingerprint[fp][1] += 1
                    fingerprint[fp][3] = [elem / fingerprint[fp][1] for elem in fingerprint[fp][3]] # divide

                    # profile updater - training
                    nn_profile_data = [fingerprint[fp][3]]

                    for c in range(len(class_profile[fingerprint[fp][2]])):
                        nn_profile_data.append(class_profile[fingerprint[fp][2]][c][0]) # profile[0] = avg. arrays; profile[1] = fp count

                    # print('Class Profile BFR', class_profile[fingerprint[fp][4]][0])

                    prof_dist, prof_ind = NN_prof.fit(nn_profile_data).kneighbors(nn_profile_data)
                    prof_ind = prof_ind[0][1] - 1

                    closest_profile = class_profile[prof_ind][0][0]
                    closest_profile_len = class_profile[prof_ind][0][1]

                    fingerprint[fp][-1] = prof_ind
                    nn_profile_data.clear()
                    # print('Closest Profile BFR', class_profile[fingerprint[fp][2]][prof_ind], ', Closest IND:', prof_ind, 'Length:', closest_profile_len)
                    # print(closest_profile, closest_profile_len)
                    # print(class_profile[fingerprint[fp][2]][0][0])
                    # exit()

                    if prof_ind != fingerprint[fp][-1]:
                        class_profile[fingerprint[fp][2]][0][0] = [x * class_profile[fingerprint[fp][2]][0][1] for x in class_profile[fingerprint[fp][2]][0][0]]
                        class_profile = [class_profile[y] - fingerprint[fp][3][y] for y in range(anchor_count)]
                        class_profile[fingerprint[fp][2]][0][1] -= 1
                        class_profile = [z / class_profile[fingerprint[fp][2]][0][1] for z in class_profile[fingerprint[fp][2]][0][0]]

                        closest_profile = [x * closest_profile_len for x in closest_profile]
                        closest_profile = [closest_profile[y] + fingerprint[fp][3][y] for y in range(anchor_count)]
                        closest_profile_len += 1
                        closest_profile = [z / closest_profile_len for z in closest_profile]

                        # update fp with class/profile position

                    # print(closest_profile)
                    # for i in range(anchor_count):
                    #     closest_profile = closest_profile - prev_fp[3]
                    # print(type(int(fingerprint[3][0])))
                    # exit()
                    if prof_ind == fingerprint[fp][-1]:
                        # print(prev_fp)
                        # print(fingerprint[fp])

                        closest_profile = [x * closest_profile_len for x in closest_profile]
                        closest_profile = [closest_profile[i] - prev_fp[3][i] for i in range(anchor_count)]
                        closest_profile = [closest_profile[j] + fingerprint[fp][3][j] for j in range(anchor_count)]
                        closest_profile = [x / closest_profile_len for x in closest_profile]

                        # update fp with class/profile position

                    # print('Closest Profile AFT', class_profile[fingerprint[fp][2]][prof_ind], ', Closest IND:', prof_ind, 'Length:', closest_profile_len)

                    # update class profile
                    # print('Class Profile AFT', class_profile[fingerprint[fp][4] - 1][0])

                    # print('Updated FP', fingerprint[fp])
                    # print()


                    # store prev, curr fp
                    #  use NN to find closest profile, if same (profile - old_fp) + updated_fp
                    #  if different, (profile - fp_new) & profile_count - 1;

                    # refer to Ethan's text and update profile using NN_clus
                    #  similar approach as updating fingerprint anchor?;

                    # increment data point
                    tr_tic += 1
                    label_ind = 0
        return class_profile
    updated_class_profile = fingerprint_updater(class_profile)

    def debug_stuph():
        if display_fingerprint:
            print('***TRAIN***')
            for i in range(len(fingerprint)):
                print('[APP]', i, fingerprint[i], '[ AncSum:', round(sum(fingerprint[i][3]), 4), ']')

            for j in range(len(tmp_fingerprint)):
                print('[TMP]', j, tmp_fingerprint[j], '[ AncSum:', round(sum(tmp_fingerprint[j][3]), 4), ']')

        # Display - Debugging
        # for class_ in range(len(class_arr)):
        #     for profile_ in class_arr[class_]:
        #         print('Class', class_ + 1,
        #               ', Profile', class_arr[class_].index(profile_) + 1,
        #               '\n', profile_, '\n')

        # print()

        # shows class dataset # Check with mentor
        if show_profiler_info_simple:
            for i in range(len(class_profile)):
                print('C' + str(i + 1), 'P' + str(len(class_profile[i])), str(class_profile[i]), 'Len', len(class_profile[i]))
        # exit()
    debug_stuph()

        #   find entity_id then search for fingerprint
        #
        #   TRAIN
        #   once matching fingerprint found execute below code
        #   - update fingerprint (anchor set * fv) -> increment fv and closest anchor - refer to anchor generation
        #   - (new anchor set / fv) - this gets the new distribution
        #   - using update fingerprint use the last two elements (class & profile) to update the profiles
        #       > multiply anchor set in profile by last element in profile (number of sets used)
        #           + same as fingerprint update
        #       > class = label, using fingerprint label add to the profile its already assigned to
        #       > recalculate the sum and then the average
        #
        #   TEST
        #   same as above...ish
        #   - use NN_clus or Kmeans to predict label based on distance to anchor? ask tomorrow
        #   *** get training data through, save test until you can speak to mentor tomorrow
        #   *** should more than likely use NN_clus like in the profiling;
    return updated_class_profile
upd_class_profile = train_work()

# 0 - 5 (6) [][][] = main array,  class array,  profile array
upd_cp_splat = []

# converts updated profile into a single array and places trackers
class_ind = 0
prof_ind = 0
while class_ind < len(upd_class_profile):
   while prof_ind < len(upd_class_profile[class_ind]):
       upd_cp_splat.append(upd_class_profile[class_ind][prof_ind])
       upd_class_profile[class_ind][prof_ind][1] = [class_ind, prof_ind]

       prof_ind += 1
   prof_ind = 0
   class_ind += 1

def test_work(y_test, upd_cp_splat):
    print('>> [TESTING]')
    # fp ex: ['1', 93, 1, [0.26881720430107525, 0.27956989247311825, 0.15053763440860216...]
    test_id_arr = []
    test_fv_arr = []
    test_label_arr = []
    test_anchor_arr = []

    # creates fingerprint for test data
    def generate_fingerprint():
        fp_tick = 0
        id_index = 0
        test_id_iter = 0

        fp_tick = 0
        id_index = 0
        test_id_iter = 0

        # anchor distance calculator
        NN = NearestNeighbors(n_neighbors=len(cluster_centers) + 1, algorithm='kd_tree')

        # initializes fingerprints
        if file_type == 0:
            test_id_iter = test_id.iloc
        if file_type == 1:
            test_id_iter = test_id

        for fp_tick in range(len(X_test_sc)):
            if test_id_iter[fp_tick] not in test_id_arr:
                test_id_arr.append(test_id_iter[fp_tick])
                # print(y_test.values.ravel()[fp_tick])
                test_label_arr.append(y_test.values.ravel()[fp_tick])
                test_fv_arr.append(0)
                test_anchor_arr.append([0] * anchor_count)

                # print('test_id_arr length:', len(test_id_arr))

        # print(test_id_arr)
        # print(test_label_arr)

        # appends cluster centers to array then compares the centers to the sample which is the
        # first element of the array;
        nn_test = [None]
        for cluster in cluster_centers:
            nn_test.append(cluster)

        # increments data using i, i will increment until max number of data is called, code will be executed
        # during each increment depending on conditions;
        i = 0
        while i < len(test_id):
            # print(test_id[i], i, '/', len(test_id), len(X_test_sc))
            nn_test[0] = X_test_sc[i]

            # if id position equals the length of the fingerprint id array reset to zero;
            # print(len(test_id_arr) - 1)
            if id_index == len(test_id_arr) - 1:
                id_index = 0

            # increment the id list index until the data id and fingerprint id match;
            if test_id_iter[i] != test_id_arr[id_index]:
                id_index += 1

            # if data id and fingerprint id match, execute the below code;
            if test_id_iter[i] == test_id_arr[id_index]:
                dist, ind = NN.fit(nn_test).kneighbors(nn_test)
                closest_anchor = ind[0][1]

                # active updater - init
                test_anchor_arr[id_index] = [elem * test_fv_arr[id_index] for elem in test_anchor_arr[id_index]]  # multiply
                test_anchor_arr[id_index][closest_anchor - 1] += 1  # inc
                test_fv_arr[id_index] += 1  # inc
                test_anchor_arr[id_index] = [elem / test_fv_arr[id_index] for elem in test_anchor_arr[id_index]]  # divide

                i += 1

        # assigns fingerprints to fp list
        # tmp if less than min_stat;
        nn_profile_data = [None]
        for cp in range(len(upd_cp_splat)):
            nn_profile_data.append(upd_cp_splat[cp][0])
        for j in range(len(test_id_arr)):
            if test_fv_arr[j] < min_stat:
                test_tmp_fingerprint.append(list([test_id_arr[j], test_fv_arr[j], test_label_arr[j], test_anchor_arr[j], 0]))

            if test_fv_arr[j] > min_stat:
                test_fingerprint.append(list([test_id_arr[j], test_fv_arr[j], test_label_arr[j], test_anchor_arr[j], 0]))
                # label prediction

                def label_predict(upd_cp_splat, test_fingerprint):
                    NN_prof = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
                    nn_profile_data[0] = test_fingerprint[j][3]
                    # upd_cp_splat[1] -> [class, profile]
                    dist, ind = NN_prof.fit(nn_profile_data).kneighbors(nn_profile_data)
                    cp_index = ind[0][1] - 1

                    test_fingerprint[j][-1] = upd_cp_splat[cp_index][1][0] + 1
                    print(test_fingerprint[j])

                    # exit()
                label_predict(upd_cp_splat, test_fingerprint)
    generate_fingerprint()

    def metrics(y_test, test_id):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from collections import Counter
        test_label = y_test.values.ravel()
        y_true = []
        y_pred = []

        # fit y_true, y_pred
        i, j = 0, 0
        while i < len(test_label):
            if j == len(test_fingerprint):
                j = 0
            if test_id[i] != test_fingerprint[j][0]:
                j += 1
            if test_id[i] == test_fingerprint[j][0]:
                y_true.append(test_label[i])
                y_pred.append(test_fingerprint[j][-1])
            i += 1

        conf_matrix = confusion_matrix(y_true, y_pred)
        # table = conf_matrix.ravel()
        # print(conf_matrix)

        print('True: ', Counter(y_true))
        print('Pred: ', Counter(y_pred))

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        accuracy_score = 'accuracy: ' + str(round(accuracy * 100, 2)) + '%'
        precision_score = 'precision: ' + str(round(precision * 100, 2)) + '%'
        recall_score = 'recall: ' + str(round(recall * 100, 2)) + '%'
        f1_score = 'f1 score: ' + str(round(f1 * 100, 2)) + '%'

        # ask about specificity
        print(accuracy_score)
        print(precision_score)
        print(recall_score)
        print(f1_score)
    metrics(y_test, test_id)

    def debug_stuph():
        if display_fingerprint:
            print('***TEST***')
            for i in range(len(test_fingerprint)):
                print('[APP]', i, test_fingerprint[i], '[ AncSum:', round(sum(test_fingerprint[i][3]), 4), ']')

            for j in range(len(test_tmp_fingerprint)):
                print('[TMP]', j, test_tmp_fingerprint[j], '[ AncSum:', round(sum(test_tmp_fingerprint[j][3]), 4), ']')

        # Display - Debugging
        # for class_ in range(len(class_arr)):
        #     for profile_ in class_arr[class_]:
        #         print('Class', class_ + 1,
        #               ', Profile', class_arr[class_].index(profile_) + 1,
        #               '\n', profile_, '\n')

        # print()

        # shows class dataset # Check with mentor
        # for i in range(len(class_profile)):
        #     print('C' + str(i + 1), 'P' + str(len(class_profile[i])), str(class_profile[i]))
        # # exit()
    debug_stuph()
test_work(y_test, upd_cp_splat)

#----------------------------------------------------------------------------------------------------------
runtime_end = rtimer()
print('>> Runtime (main.py):', round(runtime_end - runtime_start, 2), 'seconds <<')

# end run
print(" *** 'main.py' Finished. *** ")
#----------------------------------------------------------------------------------------------------------