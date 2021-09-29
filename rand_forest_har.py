# Charles Lett Jr.
# REU 2021
# random forest classification algorithm to verify that the prediction
# accuracy is an issue in my code and not the data

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# UCI-HAR DATA
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

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train.values.ravel())

y_pred = clf.predict(X_test)

print('Accuracy', metrics.accuracy_score(y_test, y_pred) * 100)
