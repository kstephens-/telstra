import pandas as pd
import numpy as np
import xgboost as xgb

import cv
import score as sc
import base_features as bf

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC


submit = False
version = '0.3'


def xgboost_model(train, labels, test):

    params = {}
    params['objective'] = 'multi:softprob'
    params['num_class'] = 3
    params['eval_metric'] = 'mlogloss'

    params['eta'] = 0.01
    params['gamma'] = 0
    params['max_depth'] = 10
    params['min_child_weight'] = 0.02
    params['max_delta_step'] = 0
    params['subsample'] = 0.6
    params['colsample_bytree'] = 0.8
    params['lambda'] = 0.2
    params['alpha'] = 0

    params['silent'] = 1

    xgtrain = xgb.DMatrix(train, labels)
    xgtest = xgb.DMatrix(test)

    num_rounds = 600
    m = xgb.train(list(params.items()), xgtrain, num_rounds)
    return m, m.predict(xgtest)


scores = []
if not submit:

    data = pd.read_csv('../data/train.csv')

    for train, test in cv.train_test(data):

        train = bf.base_features(train)
        test = bf.base_features(test)

        train, test = bf.location_features(train, test, cutoff=5)
        train, test = bf.dangerous_location(train, test)
        train, test = bf.safe_event(train, test)

        labels = train.fault_severity.values
        answers = test.fault_severity.values

        train.drop(['id', 'fault_severity', 'location'], axis=1, inplace=True)
        test.drop(['id', 'fault_severity', 'location'], axis=1, inplace=True)

        train = train.fillna(0)
        train = train.astype(float)

        test = test.fillna(0)
        test = test.astype(float)

        # knn = KNeighborsClassifier(
        #     n_neighbors=15,
        #     weights='uniform',
        #     leaf_size=30,
        #     p=1,
        #     n_jobs=-1
        # )
        # knn.fit(train, labels)
        # # print()
        # # print('knn score')
        # # print(knn.score(test, answers))
        # # print()

        # bayes = MultinomialNB(alpha=0, fit_prior=True)
        # bayes.fit(train, labels)
        # # print()
        # # print('bayes score')
        # # print(bayes.score(test, answers))
        # # print()
        # knn_train = np.reshape(knn.predict(train), (-1, 1))
        # knn_test = np.reshape(knn.predict(test), (-1, 1))
        # print()
        # print()
        # print('knn test')
        # print(knn_test)
        # print()
        # print()

        #bayes_train = np.reshape(bayes.predict(train), (-1, 1))
        # bayes_test = np.reshape(bayes.predict(test), (-1, 1))

        # svm = LinearSVC(
        #     penalty='l2',
        #     loss='hinge',
        #     C=1.0,
        #     random_state=42,
        #     intercept_scaling=5
        # )
        # svm.fit(train, labels)

        #svm_train = np.reshape(svm.predict(train), (-1, 1))
        #svm_test = np.reshape(svm.predict(test), (-1, 1))

        # train = np.append(train, knn_train, axis=1)
        # test = np.append(test, knn_test, axis=1)
        #train_labels = np.reshape(labels, (-1, 1))

        #train = np.append(train, train_labels, axis=1)
        #test = np.append(test, bayes_test, axis=1)

        #train = np.append(train, train_labels, axis=1)
        #test = np.append(test, svm_test, axis=1)

        model, predictions = xgboost_model(train, labels, test)

        score = sc.multi_log_loss(predictions, answers)
        print('cv score:', score)

        scores.append(score)

    print()
    print('score:', np.mean(scores))


    #     zeros = np.zeros(shape=(len(test), 3))
    #     predictions = pd.DataFrame(zeros, columns=[0, 1, 2])
    #     predictions.loc[:, 1] = 1
    #     predictions.loc[:, 0] = 0.0001
    #     predictions.loc[:, 2] = 0.0001

    #     answers = test['fault_severity']

    #     score = sc.multi_log_loss(predictions, answers)
    #     print('cv score: ', score)

    #     scores.append(score)

    # print()
    # print('score: ', np.mean(scores))


else:

    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    train = bf.base_features(train)
    test = bf.base_features(test)

    train, test = bf.location_features(train, test, cutoff=5)

    labels = train.fault_severity.values
    ids = test['id']

    train.drop(['id', 'fault_severity', 'location'], axis=1, inplace=True)
    test.drop(['id', 'location'], axis=1, inplace=True)

    train = train.fillna(0)
    train = train.astype(float)

    test = test.fillna(0)
    test = test.astype(float)

    model, predictions = xgboost_model(train, labels, test)

    predictions_df = pd.DataFrame({
        'id': ids,
        'predict_0': predictions[:, 0],
        'predict_1': predictions[:, 1],
        'predict_2': predictions[:, 2]
    })
    predictions_df.to_csv('../submissions/{}_v{}.csv'.format('xgb', version),
                          index=False)
