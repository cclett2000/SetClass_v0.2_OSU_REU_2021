#TODO: NOTES (I just like the yellow text in PyCharm XD)
#  silhouette coefficient:
#  clustering performance metrics ranges from -1 to 1. A silhouette coefficient of 0
#  indicates that clusters are significantly overlapping one another, and a silhouette
#  coefficient of 1 indicates clusters are well-separated.
#  ;
#  ARI:
#  ARI score of 0 indicates that cluster labels are randomly assigned, and an ARI score
#  of 1 means that the true labels and predicted labels form identical clusters
#  ;
#  ;
#  THIS IN REGARDS TO THE CLUSTER (KMEANS)
# # init: You’ll use "k-means++" instead of "random" to ensure centroids
# #       are initialized with some distance between them. In most cases,
# #       this will be an improvement over "random".
# #
# # n_init: You’ll increase the number of initializations to ensure you find a stable solution.
# #
# # max_iter: You’ll increase the number of iterations per initialization to ensure that k-means will converge.;;

# here for reference
def generate_profiles():
#     # list for class storage
#     show_profiling = True
#     global profile_count
#     profile_class = []
#     profile_index = []
#
#     def prep_profile():
#         # fills profile_class with appropriate number of lists
#         for i in classes:
#             profile_class.append([i])
#
#         # calculates upper/lower num of profiles (this is for two classes)
#         profile_lower = math.floor(profile_count / len(profile_class))
#         profile_upper = math.ceil(profile_count / len(profile_class))
#
#         # this set of loops appends empty profiles to class (Now Dynamic!...ish)
#         # TODO: implement better profile division method (utilize lower/upper profile)
#         # TODO: create array to track entity id, do this in fp generation?
#         for x in range(len(profile_class)):
#             for i in range(profile_lower):
#                 profile_class[x].append([])
#             profile_index.append(x + 1)
#
#         # separates fingerprints based on class (make dynamic later)
#         pf1_count =  0
#         pf1_info = []
#         pf1_storage = []
#         for fp in range(len(fingerprint)):
#             # PROFILE 1
#             if fingerprint[fp][fp_key.get('class')] == profile_class[0][0]:
#                 pf1_count += 1
#                 # stores fingerprint info (id, class, fv#)
#                 pf1_info.append(fingerprint[fp][:fp_key.get('anc')])
#                 # stores anchors for profiling
#                 pf1_storage.append(fingerprint[fp][fp_key.get('anc')])
#
#
#         # limits for splitting profiles
#         lower_lim = math.floor(pf1_count / len(profile_class))
#         # print(pf1_info[0])
#
#
#         # length check, better this way than retyping print everytime a check is needed
#         if show_profiling:
#             print('[Class1, Profile 1&2 Length Check: ' + str(len(pf1_storage)) + ']')
#
#         # test variables, this is just to get profiling to work PROFILE1
#         pf01 = len(pf1_storage[:lower_lim]) # length
#         pf_calc1 = [sum(x) for x in zip(*pf1_storage[:lower_lim])] # sum
#         pf_avg1 = [round(i / pf01, 3) for i in pf_calc1] # average
#
#         # fill class 1, profile 1
#         for x in pf_avg1:
#             profile_class[0][1].append(x)
#         profile_class[0][1].append(pf01)
#
#         # test variables, this is just to get profiling to work PROFILE2
#         pf02 = len(pf1_storage[lower_lim - 1: -1]) # length
#         pf_calc2 = [sum(x) for x in zip(*pf1_storage[lower_lim - 1: -1])] # sum
#         pf_avg2 = [round(i / pf02, 3) for i in pf_calc2] # average
#
#         # fill class 1, profile 2
#         for y in pf_avg2:
#             profile_class[0][2].append(y)
#         profile_class[0][2].append(pf02)
#
#         # label fp with profile
#         for test in pf1_info[:lower_lim]:
#             for i in range(len(fingerprint)):
#                 if fingerprint[i][0] == test[0]:
#                     fingerprint[i].append('profile=' + str(profile_index[0]))
#
#         for test in pf1_info[lower_lim: -1]:
#             for i in range(len(fingerprint)):
#                 if fingerprint[i][0] == test[0]:
#                     fingerprint[i].append('profile=' + str(profile_index[1]))
#
#         # display for debugging
#         if show_profiling:
#             print('[PROFILE INFO]')
#             for i in range(len(profile_class)):
#                 print(profile_class[i])
#             print('C1, P1 Sum:', sum(profile_class[0][1][0:4]))
#             print('C1, P2 Sum:', sum(profile_class[0][2][0:4]))
#             print('Key = [CLASSID, [PROFILE = ANC, NUM]]')
#
#     prep_profile()
    pass
def profile_work_test_1():
    # def set_classes():
    #     # preps classes for profiling
    #
    #     # class storage
    #     classes = []
    #     classes_index = []
    #
    #     # stores classes in list
    #     for i in fingerprint:
    #         classes.append(i[2])
    #     print(classes)
    #
    #     class_num = set(classes)
    #     classes.clear()
    #     for i in class_num:
    #         classes.append([i])
    #         classes_index.append(i)
    #     return classes, classes_index
    # class_arr, class_ind = set_classes()
    #
    # def prep_data():
    #     # runs through init_sample and stores fingerprints for profiler
    #     class_index = 0
    #     fp_storage = []
    #     for i in range(len(class_ind)):
    #         fp_storage.append([])
    #
    #     # stores fingerprint based on class
    #     for i in range(len(fingerprint)):
    #         # if class length reached, reset
    #         if class_index == len(class_ind) - 1:
    #             class_index = 0
    #
    #         # if class does not match index, increment
    #         if fingerprint[i][2] != class_ind[class_index]:
    #             class_index += 1
    #
    #         if fingerprint[i][2] == class_ind[class_index]:
    #             fp_storage[class_index].append(fingerprint[i])
    #
    #     # fp_storage check
    #     # for i in fp_storage:
    #     #     print(i)
    #
    #     # returns fingerprint storage and length
    #     return fp_storage
    # fp_storage = prep_data()
    #
    # def set_profiles():
    #     # adds profiles to classes
    #
    #     # for < 2 class, use floor then randomly add whats needed until profile_count
    #     global profile_count
    #     # profiles per class (update for odd # of profiles later)
    #     profile_num = profile_count / len((class_arr))
    #
    #     # fits profiles to classes and labels profiles
    #     for j in range(len(class_arr)):
    #         for i in range(int(profile_num)):
    #             class_arr[j].append([])
    #
    #     return class_arr, profile_num
    # class_arr, profile_num = set_profiles()
    #
    # def fit_profiles():
    #     # use randomizer to assign anchors
    #     class_index = 1
    #     for i in range(len(fingerprint)):
    #         # reset if max
    #         if class_index == len(class_arr):
    #             class_index = 1
    #
    #         # increment index if match not detected
    #         if fingerprint[i][2] != class_index:
    #             class_index += 1
    #
    #         # fit profile array and update fingerprint
    #         if fingerprint[i][2] == class_index:
    #             test = random.randint(1, int(profile_num))
    #             class_arr[class_index - 1][test].append(fingerprint[i][3])
    #             fingerprint[i].append('class=' + str(class_index))
    #             fingerprint[i].append('profile=' + str(test))
    #
    #     return class_arr
    # class_arr = fit_profiles()
    #
    # # WORKING
    # # do profile calculations here
    # # pf01 = len(pf1_storage[:lower_lim])  # length
    # # pf_calc1 = [sum(x) for x in zip(*pf1_storage[:lower_lim])]  # sum
    # # pf_avg1 = [round(i / pf01, 3) for i in pf_calc1]  # average
    #
    #
    #
    #
    # # for i in class_arr:
    # #     print(i)
    # # print('key: [CLASS, PROFILE]')
    pass
def update_fingerprint():
    #
    # display_fingerprint = False
    # print('>> [FINGERPRINT INITIALIZATION]')
    #
    # # holds fp
    # fingerprint = []
    # tmp_fingerprint = []
    #
    # # for fp init
    # id = []
    # fv = []
    # label = []
    # anchor = []
    #
    # def fit_fingerprint():
    #     # prints display in console
    #     fp_tick = 0
    #     id_index = 0
    #
    #     # anchor distance calculator
    #     NN = NearestNeighbors(n_neighbors=len(cluster_centers) + 1, algorithm='kd_tree')
    #
    #     # initializes fingerprints
    #     if file_type == 0:
    #         sample_id_iter = sample_id.iloc
    #     if file_type == 1:
    #         sample_id_iter = sample_id
    #
    #     for fp_tick in range(len(sample)):
    #         if sample_id_iter[fp_tick] not in id:
    #             id.append(sample_id_iter[fp_tick])
    #             label.append(encoded_sample_label[fp_tick])  # TODO: fix this, labels incorrectly stored in fp (stores id)
    #             fv.append(0)
    #             anchor.append([0] * anchor_count)
    #
    #     # increments data using i, i will increment until
    #     # max number of data is called, code will be executed
    #     # during each increment depending on conditions;
    #     i = 0
    #     while i < len(sample_id):
    #         # appends cluster centers to array
    #         # then compares the centers to the sample which is the
    #         # first element of the array;
    #         X_data = [scaled_sample[i]]
    #         for cluster in cluster_centers:
    #             X_data.append(cluster)
    #
    #         # if id position equals the length of the fingerprint id array
    #         # reset to zero;
    #         if id_index == len(id) - 1:
    #             id_index = 0
    #
    #         # increment the id list index until the data
    #         # id and fingerprint id match;
    #         if sample_id_iter[i] != id[id_index]:
    #             id_index += 1
    #
    #         # if data id and fingerprint id match, execute
    #         # the below code;
    #         if sample_id_iter[i] == id[id_index]:
    #             # increments feature vector array for the id index
    #             fv[id_index] += 1
    #
    #             # get distance between selected data and cluster centers
    #             # get the indices for comparisons;
    #             dist, ind = NN.fit(X_data).kneighbors(X_data)
    #
    #             # gets closest anchor
    #             # increments anchor based on closest anchor index;
    #             closest_anchor = ind[0][1]
    #             anchor[id_index][closest_anchor - 1] += 1
    #
    #             i += 1
    #
    #     # calculate anchor distribution
    #     for i in range(len(id)):
    #         for j in range(len(anchor[id_index])):
    #             anchor[i][j] = anchor[i][j] / fv[i]  # TODO: ROUNDING HERE
    #
    #     # assigns fingerprints to fp list
    #     # tmp if less than min_stat;
    #     for i in range(len(id)):
    #         if fv[i] < min_stat:
    #             tmp_fingerprint.append(list([id[i], fv[i], label[i], anchor[i]]))
    #         if fv[i] > min_stat:
    #             fingerprint.append(list([id[i], fv[i], label[i], anchor[i]]))
    #
    #     # sorts the created fingerprints for readability
    #     fingerprint.sort()
    #     tmp_fingerprint.sort()
    #
    #     # display fingerprint info - used for debugging
    #     if display_fingerprint:
    #         for fp in range(len(fingerprint)):
    #             print('[APP]', fingerprint[fp], '[ AncSum:', round(sum(fingerprint[fp][3]), 2), ']')
    #
    #         print('-' * 75)
    #
    #         for fp_tmp in range(len(tmp_fingerprint)):
    #             print('[TMP]', tmp_fingerprint[fp_tmp], '[ AncSum:', round(sum(tmp_fingerprint[fp_tmp][3]), 2), ']')
    #
    #         print('Key = [ID][FV#][Class][Anchor]')
    #
    # fit_fingerprint()
    #
    # print('Fingerprint Length (APP):', len(fingerprint))
    # print('Fingerprint Length (TMP):', len(tmp_fingerprint))
    #
    # return tmp_fingerprint, fingerprint
    pass