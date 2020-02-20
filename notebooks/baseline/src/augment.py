import pandas as pd


def augment_danish(data_train, remove_duplicates=True):
    # in danish dataset last column is wrong
    data_train = data_train[:-1]
    
    # convert ids to int
    data_train = data_train.astype({'id': int})

    # will be used when calling join
    data_labels = data_train[['id','subtask_a']].set_index('id')

    all_augm_data = {}
    for lang in 'ar el en tr'.split():
        fn = f'../../data/back-translations/Danish/offenseval-da-training-v1-{lang}.json'
        augm_data = pd.read_json(fn)
        augm_data = augm_data.rename(columns={'da': 'tweet'})
        augm_data = augm_data.join(data_labels, on='id')
        all_augm_data[lang] = augm_data
    
    all_data = pd.concat([data_train] + list(all_augm_data.values()))
    
    if remove_duplicates:
        # remove duplicates
        # (from 14800 to 12773 entries)
        all_data = all_data.drop_duplicates(subset='tweet')
    
    return all_data
