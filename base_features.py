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


def severity_high(df):

    p = severity_type[['id', 'severity_type']]

    p.loc[:, 'high log severity'] = \
        (p['severity_type'].isin(['severity_type 3',
                                   'severity_type 4',
                                   'severity_type 5'])).astype(float)
    ret = pd.merge(df, p[['id', 'high log severity']], on='id', how='left')
    return ret


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
    #df_log = pd.merge(df, log_feature, on='id')
    # print()
    # print(df_log.shape)
    # print()
    return df_log


def log_feature_volume(df):

    g = log_feature[['id', 'volume']].groupby(by='id', as_index=False).sum()
    ret_g = pd.merge(df, g, on='id', how='left')

    h = log_feature[['id', 'log_feature']].groupby(by='id', as_index=False).count()
    i = pd.merge(h, g, on='id', how='left')

    i.loc[:, 'feature_by_volume'] = i['log_feature'] * i['volume']
    ret = pd.merge(ret_g, i[['id', 'feature_by_volume']], on='id', how='left')

    return ret


def log_feature_prob(train, test, level=0):

    t = pd.merge(train[['id', 'fault_severity']], log_feature, on='id',
                 how='left').drop_duplicates()

    log_given_severity = \
       t.loc[t['fault_severity'] == level, 'log_feature'].value_counts() / \
       t['fault_severity'].value_counts()[level]

    severity_prob = \
        t['fault_severity'].value_counts()[level] / len(t)

    log_probs = \
        t['log_feature'].value_counts() / len(t)

    log_feature_probs = \
        (log_given_severity * severity_prob) / log_probs

    prob_df = pd.DataFrame({'probs': log_feature_probs})

    p = pd.merge(t, prob_df, left_on='log_feature', right_index=True, how='left')
    prob_table = pd.pivot_table(p, values='probs', index='id', columns='log_feature',
                                aggfunc=np.mean, fill_value=0)

    train = pd.merge(train, prob_table, left_on='id', right_index=True,
                     how='left')
    test = pd.merge(test, prob_table, left_on='id', right_index=True,
                    how='left')

    return train, test


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

    df_log_count = log_feature_volume(df_log)
    df_severity_high = severity_high(df_log_count)

    df_complete = df_severity_high

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


def event_type_features(df):

    # d = event_type['event_type'].value_counts()
    # event_type.loc[:, 'event_type'] = event_type['event_type'] \
    #     .apply(filter_on_total, args=(d, ), total=1, value='RareEvent')

    event_type_dummies = pd.get_dummies(event_type['event_type'])
    event_dummy = pd.concat([event_type, event_type_dummies], axis=1)

    event_grpd = event_dummy \
        .groupby(event_dummy.id).sum()

    df_event = pd.merge(df, event_grpd, left_on='id', right_index=True)
    #df_event.loc[:, 'has_event_20'] = df_event['event_type 20']
    return df_event


def event_severity_prob(train, test, level=0):
    # print()
    # print(train.shape)
    # print(test.shape)
    # print()
    t = pd.merge(train[['id', 'fault_severity']], event_type, on='id',
                 how='left').drop_duplicates()

    event_given_severity = \
        t.loc[t['fault_severity'] == level, 'event_type'].value_counts() / \
        len(t.loc[t['fault_severity'] == level])

    severity_probs = \
        len(t.loc[t['fault_severity'] == level, :]) / len(t)

    event_probs = \
        t['event_type'].value_counts() / len(t['event_type'])

    event_severity_probs = \
        (event_given_severity * severity_probs) / event_probs
    # event_probs_test = \
    #     (event_given_severity * severity_probs) / event_probs
    prob_df = pd.DataFrame({'probs': event_severity_probs})

    p = pd.merge(t, prob_df, left_on='event_type', right_index=True, how='left')
    prob_table = pd.pivot_table(p, values='probs', index='id', columns='event_type',
                                aggfunc=np.median, fill_value=0)

    train = pd.merge(train, prob_table, left_on='id', right_index=True,
                     how='left')
    test = pd.merge(test, prob_table, left_on='id', right_index=True,
                    how='left')

    # grouped_event_probs = event_probs_df \
    #     .groupby(by=event_probs_df.index).median()

    # print()
    # print('grouped_event_probs')
    # print(grouped_event_probs)
    # print()

    # print()
    # print(train.shape)
    # print(test.shape)
    # print()
    # train = pd.merge(train, event_probs_df, left_on='event_type',
    #                  right_index=True, how='left')
    # test = pd.merge(test, event_probs_df, left_on='event_type',
    #                  right_index=True, how='left')
    # print()
    # print(train.shape)
    # print(test.shape)
    # print()

    return train, test


def event_severity(train, test):

    #event_grouped = event_type.groupby(by='event_type', as_index=False)

    # print()
    # print('event grouped')
    # print(event_grouped.head(10))
    # print()

    # print()
    # print(train.shape)
    # print(test.shape)

    # train = pd.merge(train, event_type, on='id', how='left')
    # test = pd.merge(test, event_type, on='id', how='left')

    # print()
    # print(train.shape)
    # print(test.shape)

    # train, test = event_severity_prob(train, test, level=0)
    # train, test = event_severity_prob(train, test, level=1)
    # train, test = event_severity_prob(train, test, level=2)
    #train, test = event_severity_prob(train, test, level=0)
    #train, test = event_severity_prob(train, test, level=1)
    train, test = event_severity_prob(train, test, level=2)

    #train = train.groupby(by=)
    return train, test


    # feature_count = \
    #     train.loc[
    #         (train['fault_severity'] == level),
    #         ['log_feature', 'volume']
    #     ].groupby(by='log_feature', sort=False).sum()

    # # event_given_severity = \
    # #     train
    # train = pd.merge(train, event_type, on='id')
    # test = pd.merge(test, event_type, on='id')

    # safe = \
    #     (train.loc[train['fault_severity'] == 0, 'event_type']).value_counts() / \
    #     len(train.loc[train['fault_severity'] == 0, 'event_type'])

    # msk = safe >= 0.01

    # train.loc[train['event_type'].isin(safe[msk].index), 'safe_event'] = 1
    # test.loc[test['event_type'].isin(safe[msk].index), 'safe_event'] = 1

    return train, test


def rare_category(x, category_distribution, cutoff=1, value='Rare'):
    try:
        if category_distribution[x] < cutoff:
            return value
    except (ValueError, KeyError):
        return np.nan
    else:
        return x

