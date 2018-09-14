import os
import torch
import shutil
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

def save_checkpoints(args, epoch, model, optimizer, time_str):
    """Save checkpoints and optimizer"""
    model_dst = os.path.join(
        args.results_root, "malenov", time_str,
        'model_epoch_{}.pth.tar'.format(epoch)
    )
    torch.save({
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
    }, model_dst)
    shutil.copy(model_dst, os.path.join(args.results_root, "malenov", time_str, 'model_latest.pth.tar'))


def get_valid_indices_and_labels(index_mins, index_maxs, indices, labels, cube_half_size):
    valid_indices = []
    valid_labels = []
    for index, label in zip(indices, labels):
        if [(index_mins[i]+cube_half_size <= index[i] < index_maxs[i]-cube_half_size) for i in range(1,3)] == [True]*2:
            valid_indices.append(index)
            valid_labels.append(label)
    valid_indices = np.array(valid_indices)
    valid_labels = np.array(valid_labels)
    return valid_indices, valid_labels

def get_train_val_set(data_root="../data/malenov/", use_stratified_kfold=True, cube_size=32, train_size=0.8, seed=42):
    labels = np.load(data_root+"labels.npy")
    indices = np.load(data_root+"indices.npy")
    
    index_mins = indices.min(0)
    index_maxs = indices.max(0)
    
    cube_half_size = cube_size
    
    if use_stratified_kfold:
        valid_indices, valid_labels = get_valid_indices_and_labels(index_mins, index_maxs, indices, labels, cube_half_size)
    
        splitter = StratifiedShuffleSplit(n_iter=1, random_state=seed, y=valid_labels, train_size=train_size)
    
        train_index, test_index = [*splitter][0]
        print("Labels in TRAIN:", len(train_index), "Labels in TEST:", len(test_index))
        X_train, X_val = valid_indices[train_index], valid_indices[test_index]
        y_train, y_val = valid_labels[train_index], valid_labels[test_index]
    
    else:
        train_indices, train_labels = np.load("./split/train_split.npy")
        val_indices, val_labels = np.load("./split/val_split.npy")
    
        X_train, y_train = get_valid_indices_and_labels(index_mins, index_maxs, train_indices, train_labels, cube_half_size)
        X_val, y_val = get_valid_indices_and_labels(index_mins, index_maxs, val_indices, val_labels, cube_half_size)
    
    seismic = torch.FloatTensor(np.load(data_root+"seismic_cube.npy")[:, :, :, 0]).unsqueeze(0)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    return seismic, X_train, y_train, X_val, y_val
