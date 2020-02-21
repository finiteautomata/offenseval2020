import pandas as pd


def augment_danish(data_train, remove_duplicates=True):
    # will be used when calling join
    data_labels = data_train[['id','subtask_a']].set_index('id')

    all_augm_data = {}
    for lang in 'ar el en tr'.split():
        fn = f'../../data/back-translations/Danish/offenseval-da-training-v1-{lang}.json'
        augm_data = pd.read_json(fn)
        augm_data = augm_data.rename(columns={'da': 'tweet'})
        augm_data = augm_data.join(data_labels, on='id')
        augm_data = augm_data.dropna()  # remove data not in data_train
        all_augm_data[lang] = augm_data
    
    all_data = pd.concat([data_train] + list(all_augm_data.values()))
    
    if remove_duplicates:
        all_data = all_data.drop_duplicates(subset='tweet')
    
    return all_data

def augment_arabic(data_train, remove_duplicates=True):
    # will be used when calling join
    data_labels = data_train[['id','subtask_a']].set_index('id')

    all_augm_data = {}
    for lang in 'da el en tr'.split():
        fn = f'../../data/back-translations/Arabic/offenseval-ar-training-v1-{lang}.json'
        augm_data = pd.read_json(fn)
        augm_data = augm_data.rename(columns={'ar': 'tweet'})
        augm_data = augm_data.join(data_labels, on='id')
        augm_data = augm_data.dropna()  # remove data not in data_train
        all_augm_data[lang] = augm_data
    
    all_data = pd.concat([data_train] + list(all_augm_data.values()))
    
    if remove_duplicates:
        all_data = all_data.drop_duplicates(subset='tweet')
    
    return all_data
