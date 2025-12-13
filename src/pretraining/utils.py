import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import pickle

def load(path, pickle_file):
    dataset = pd.read_pickle(os.path.join(path, pickle_file))
    dataset.info(memory_usage='deep')
    return dataset

def loads(data_sets_dir):
    data_sets_files = sorted([f for f in listdir(data_sets_dir) if isfile(join(data_sets_dir, f))])

    dataset = load(data_sets_dir, data_sets_files[0])
    data_sets_files.remove(data_sets_files[0])

    for ds_file in data_sets_files:
        dataset = pd.concat([dataset, load(data_sets_dir, ds_file)])

    dataset = dataset.reset_index(drop=True)
    return dataset

def load_split_indices(path, file_name='split_idx.pkl'):
    with open(os.path.join(path, file_name), 'rb') as f:
        return pickle.load(f)

def get_split_df(dataset, idxs):
    return dataset.loc[idxs].reset_index(drop=True)


def load_split_datasets(path, dataset):
    print(f"Loading split datasets from {path}")
    indices = load_split_indices(path+"/split", 'split_idx.pkl')

    train_df = get_split_df(dataset, indices['train'])
    val_df = get_split_df(dataset, indices['val'])
    test_df = get_split_df(dataset, indices['test'])
    test_short_df = get_split_df(dataset, indices['short'])
    test_long_df = get_split_df(dataset, indices['long'])

    return train_df, val_df, test_df, test_short_df, test_long_df