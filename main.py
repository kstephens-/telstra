import pandas as pd
import numpy as np

import cv
import score as sc


submit = False
version = '0.1'


scores = []
if not submit:

    data = pd.read_csv('../data/train.csv')

    for train, test in cv.train_test(data):

        zeros = np.zeros(shape=(len(test), 3))
        predictions = pd.DataFrame(zeros, columns=[0, 1, 2])
        predictions.loc[:, 1] = 1
        predictions.loc[:, 0] = 0.0001
        predictions.loc[:, 2] = 0.0001

        answers = test['fault_severity']

        score = sc.multi_log_loss(predictions, answers)
        print('cv score: ', score)

        scores.append(score)

    print()
    print('score: ', np.mean(scores))


else:

    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
