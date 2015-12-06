import numpy as np
import pandas as pd

def multi_log_loss(predictions, answers):

    m = len(predictions)

    zeros = np.zeros(shape=(m, 3))
    y = pd.DataFrame(zeros, columns=[0, 1, 2])

    for i, c in zip(answers.index, answers.values):
        y.set_value(i, c, 1)

    return -(1/m) * np.sum(np.sum(y * np.log(predictions)))
