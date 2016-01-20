import pandas as pd
import numpy as np
import xgboost as xgb

import cv
import score as sc
import base_features as bf

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif


submit = False
version = '0.25'


def xgboost_model(train, labels, test):

    params = {}
    params['objective'] = 'multi:softprob'
    params['num_class'] = 3
    params['eval_metric'] = 'mlogloss'

    params['eta'] = 0.02
    params['gamma'] = 2.0
    params['max_depth'] = 10
    params['min_child_weight'] = 0.1
    params['max_delta_step'] = 1.5
    params['subsample'] = 0.75
    params['colsample_bytree'] = 0.85
    params['lambda'] = 0.2
    params['alpha'] = 0

    params['silent'] = 1

    xgtrain = xgb.DMatrix(train, labels)
    xgtest = xgb.DMatrix(test)

    num_rounds = 1200
    m = xgb.train(list(params.items()), xgtrain, num_rounds)
    return m, m.predict(xgtest)

train_scores = []
scores = []
if not submit:

    data = pd.read_csv('../data/train.csv')

    for train, test in cv.train_test(data):

        # print(train.shape)
        # print(test.shape)
        # print()
        train = bf.base_features(train)
        test = bf.base_features(test)
        # print(train.shape)
        # print(test.shape)

        train, test = bf.location_features(train, test, cutoff=5)
        #train, test = bf.dangerous_location(train, test)

        # try:
        #     train.loc[:, 'no 313'] = \
        #         (train['feature 313'] == 0).astype(float)
        #     test.loc[:, 'no 313'] = \
        #         (test['feature 313'] == 0).astype(float)
        # except KeyError:
        #     print('not adding feature')
        #     pass

        #train, test = bf.log_feature_prob(train, test, level=1)
        #train, test = bf.log_feature_prob(train, test, level=2)
    #     print()
    #     print(train.shape)
    #     print(test.shape)
    #     print()
        #train, test = bf.event_severity(train, test)
    #    train, test = bf.danger_log(train, test)
    #     print()
    #     print(train.shape)
    #     print(test.shape)
    #     print()


        labels = train.fault_severity.values
        answers = test.fault_severity.values

        train.drop(['id', 'fault_severity', 'location'], axis=1, inplace=True)
        test.drop(['id', 'fault_severity', 'location'], axis=1, inplace=True)

        train = train.fillna(0)
        train = train.astype(float)

        test = test.fillna(0)
        test = test.astype(float)

        print()
        print(train.shape)
        print(test.shape)


        ch2 = SelectKBest(chi2, k=500)
        train = ch2.fit_transform(train, labels)
        test = ch2.transform(test)

        print(train.shape)
        print(test.shape)

        # print(train.shape)
        # print(test.shape)

        # pca = PCA(n_components=400)
        # print('transforming data')
        # train = pca.fit_transform(train)
        # test = pca.transform(test)
        # print('data transformed')
        # print(train.shape)
        # print(test.shape)

        # print()
        # print('variance ratio')
        # print(pca.explained_variance_ratio_)
        # print()

        # knn = KNeighborsClassifier(
        #     n_neighbors=15,
        #     weights='uniform',
        #     leaf_size=50,
        #     p=1,
        #     n_jobs=-1
        # )
        # knn.fit(train, labels)
        # # # print()
        # # # print('knn score')
        # print(knn.score(test, answers))
        # # # print()

        # # bayes = MultinomialNB(alpha=0, fit_prior=True)
        # # bayes.fit(train, labels)
        # # # print()
        # # # print('bayes score')
        # # # print(bayes.score(test, answers))
        # # # print()
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

        #train = np.append(train, knn_train, axis=1)
        #test = np.append(test, knn_test, axis=1)
        #train_labels = np.reshape(labels, (-1, 1))

        #train = np.append(train, train_labels, axis=1)
        #test = np.append(test, bayes_test, axis=1)

        #train = np.append(train, train_labels, axis=1)
        #test = np.append(test, svm_test, axis=1)

        #print('predicting model')
        model, predictions = xgboost_model(train, labels, test)
        #print('model predicted')
        train_score = sc.multi_log_loss(
            model.predict(xgb.DMatrix(train)), labels
        )
        print('cv train score', train_score)
        train_scores.append(train_score)

        score = sc.multi_log_loss(predictions, answers)
        print('cv score:', score)

        scores.append(score)

    print()
    print('train score:', np.mean(train_scores))
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
    #train, test = bf.dangerous_location(train, test)
    # train, test = bf.safe_event(train, test)
    #train, test = bf.event_severity(train, test)

    labels = train.fault_severity.values
    ids = test['id']

    train.drop(['id', 'fault_severity', 'location'], axis=1, inplace=True)
    test.drop(['id', 'location'], axis=1, inplace=True)

    train = train.fillna(0)
    train = train.astype(float)

    test = test.fillna(0)
    test = test.astype(float)

    ch2 = SelectKBest(chi2, k=500)
    train = ch2.fit_transform(train, labels)
    test = ch2.transform(test)

    model, predictions = xgboost_model(train, labels, test)

    train_score = sc.multi_log_loss(
        model.predict(xgb.DMatrix(train)), labels
    )
    print('train score', train_score)

    predictions_df = pd.DataFrame({
        'id': ids,
        'predict_0': predictions[:, 0],
        'predict_1': predictions[:, 1],
        'predict_2': predictions[:, 2]
    })
    predictions_df.to_csv('../submissions/{}_v{}.csv'.format('xgb', version),
                          index=False)
