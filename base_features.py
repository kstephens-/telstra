import pandas as pd
import numpy as np


event_type = pd.read_csv('../data/event_type.csv')
resource_type = pd.read_csv('../data/resource_type.csv')
severity_type = pd.read_csv('../data/severity_type.csv')
log_feature = pd.read_csv('../data/log_feature.csv')


def filter_on_total(x, distribution, total=1, value='Rare'):

    try:
        if distribution[x] < total:
            return value
    except (ValueError, KeyError):
        return np.nan
    else:
        return x


def event_type_features(df):

    # d = event_type['event_type'].value_counts()
    # event_type.loc[:, 'event_type'] = event_type['event_type'] \
    #     .apply(filter_on_total, args=(d, ), total=1, value='RareEvent')

    event_type_dummies = pd.get_dummies(event_type['event_type'])
    event_dummy = pd.concat([event_type, event_type_dummies], axis=1)

    event_grpd = event_dummy \
        .groupby(event_dummy.id).sum()

    df_event = pd.merge(df, event_grpd, left_on='id', right_index=True)
    return df_event


def resource_type_features(df):

    resource_type_dummies = pd.get_dummies(resource_type['resource_type'])
    resource_dummy = pd.concat([resource_type, resource_type_dummies],
                               axis=1)

    resource_grpd = resource_dummy \
        .groupby(resource_dummy.id).sum()

    df_resource = pd.merge(df, resource_grpd, left_on='id', right_index=True)
    return df_resource.drop(['resource_type 1', 'resource_type 2',
                             'resource_type 4', 'resource_type 5',
                             'resource_type 6', 'resource_type 8',
                             'resource_type 10'], axis=1)


def severity_type_features(df):

    severity_type_dummies = pd.get_dummies(severity_type['severity_type'])
    severity_dummy = pd.concat([severity_type, severity_type_dummies],
                               axis=1)

    severity_grpd = severity_dummy \
        .groupby(severity_dummy.id).sum()

    df_severity = pd.merge(df, severity_grpd, left_on='id', right_index=True)
    return df_severity


def log_features(df):

    log_table = pd.pivot_table(
        log_feature, values='volume',
        index='id', columns='log_feature',
        aggfunc=np.sum, fill_value=0
    )
    df_log = pd.merge(df, log_table, left_on='id', right_index=True)

    # print()
    # print(df_log.shape)
    # print()
    # df_log = pd.merge(df, log_feature, on='id')
    # print()
    # print(df_log.shape)
    # print()
    return df_log


def dangerous_log(train, test, level=0):

    feature_count = \
        train.loc[
            (train['fault_severity'] == level),
            ['log_feature', 'volume']
        ].groupby(by='log_feature', sort=False).sum()

    feature_total = \
        train.loc[
            (train['fault_severity'] == level),
            'volume'
        ].sum()

    danger = feature_count / feature_total

    train.loc[
        (train['fault_severity'] == level) & (train['log_feature'].isin(danger.index)),
            'dangerous_log'
    ] = danger['volume']

    test.loc[
        (test['fault_severity'] == level) & (test['log_feature'].isin(danger.index)),
            'dangerous_log'
    ] = danger['volume']

    return train, test


def danger_log(train, test):

    train, test = dangerous_log(train, test, level=0)
    train, test = dangerous_log(train, test, level=1)
    train, test = dangerous_log(train, test, level=2)

    return train, test


def base_features(df):

    df_event = event_type_features(df)
    df_resource = resource_type_features(df_event)
    df_severity = severity_type_features(df_resource)
    df_log = log_features(df_severity)

    df_complete = df_log

    return df_complete


def location_features(train, test, cutoff=0):

    train_locations = train[['location']]
    test_locations = test[['location']]

    train_locations.loc[:, 'train'] = True
    test_locations.loc[:, 'train'] = False

    train_distribution = train_locations['location'].value_counts()

    locations = pd.concat([train_locations, test_locations])
    locations.loc[:, 'location'] = locations['location'] \
        .apply(rare_category, args=(train_distribution, ),
               cutoff=cutoff, value='RareLocation')

    locations_bin = pd.get_dummies(locations['location'])
    locations_dummy = pd.concat([locations, locations_bin], axis=1)

    msk = locations_dummy['train']
    locations_dummy.drop(['train', 'location'], axis=1, inplace=True)

    train_locs = pd.concat([train, locations_dummy[msk]], axis=1)
    test_locs = pd.concat([test, locations_dummy[~msk]], axis=1)

    return train_locs, test_locs


def dangerous_location(train, test):

    danger = \
        (train.loc[(train['fault_severity'] == 2) | (train['fault_severity'] == 1), 'location']).value_counts() / \
        len(train.loc[(train['fault_severity'] == 2) | (train['fault_severity'] == 1), 'location'])

    msk = danger >= 0.025

    train.loc[train['location'].isin(danger[msk].index), 'dangerous'] = 1
    test.loc[test['location'].isin(danger[msk].index), 'dangerous'] = 1

    return train, test


def safe_event(train, test):

    safe = \
        (train.loc[train['fault_severity'] == 0, 'location']).value_counts() / \
        len(train.loc[train['fault_severity'] == 0, 'location'])

    msk = safe >= 0.01

    train.loc[train['location'].isin(safe[msk].index), 'safe_event'] = 1
    test.loc[test['location'].isin(safe[msk].index), 'safe_event'] = 1

    return train, test


def rare_category(x, category_distribution, cutoff=1, value='Rare'):
    try:
        if category_distribution[x] < cutoff:
            return value
    except (ValueError, KeyError):
        return np.nan
    else:
        return x

