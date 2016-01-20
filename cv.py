import numpy as np
import pandas as pd
import itertools
import math
import random

random.seed(42)

def train_test(df, base_split=0.20, recombine=0.6):

    n_splits = math.floor(1/base_split)

    nmb_unknown = len(df['location']) * 0.10
    locations = df['location'].value_counts()
    location_labels = sorted(locations.index.tolist())

    unique_location_sets = []

    for i in range(n_splits):
        ctr = 0
        group_labels = []
        while ctr < nmb_unknown:
            group_index = random.randint(0, len(location_labels)-1)
            unknown = location_labels[group_index]

            location_labels[group_index], location_labels[-1] = \
                location_labels[-1], location_labels[group_index]
            location_labels.pop()

            group_labels.append(unknown)
            ctr += locations[unknown]

        unique_location_sets.append(group_labels)

    for unknowns in unique_location_sets:

        msk = df['location'].isin(unknowns)
        unknown_test_locations = df[msk]
        rest = df[~msk]

        train = rest.sample(frac=0.44, random_state=42)
        rest_test = rest[~rest.index.isin(train.index)]
        test = pd.concat([rest_test, unknown_test_locations], axis=0)

        yield train, test
