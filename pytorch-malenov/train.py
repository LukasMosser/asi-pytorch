import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit
from tqdm import tqdm as tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MalenovDataset
from options import prepare_output_directory, get_args
from model import MalenovNet
from utils import save_checkpoints

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


def main():
    args = get_args()

    # CUDA setting
    if not torch.cuda.is_available():
        raise ValueError("Doesn't make much sense without a GPU. Expect long training times.")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = False #To combat randomness

    args, train_writer, val_writer, time_str = prepare_output_directory(args)

    labels = np.load(args.data_root+"labels.npy")
    indices = np.load(args.data_root+"indices.npy")

    index_mins = indices.min(0)
    index_maxs = indices.max(0)
        
    cube_half_size = args.cube_size

    
    if args.use_stratified_kfold:
        valid_indices, valid_labels = get_valid_indices_and_labels(index_mins, index_maxs, indices, labels, cube_half_size)

        splitter = StratifiedShuffleSplit(n_iter=1, random_state=args.seed, y=valid_labels, train_size=args.train_size)

        train_index, test_index = [*splitter][0]
        print("Labels in TRAIN:", len(train_index), "Labels in TEST:", len(test_index))
        X_train, X_val = valid_indices[train_index], valid_indices[test_index]
        y_train, y_val = valid_labels[train_index], valid_labels[test_index]

    else:
        train_indices, train_labels = np.load("./split/train_split.npy")
        val_indices, val_labels = np.load("./split/val_split.npy")

        X_train, y_train = get_valid_indices_and_labels(index_mins, index_maxs, train_indices, train_labels, cube_half_size)
        X_val, y_val = get_valid_indices_and_labels(index_mins, index_maxs, val_indices, val_labels, cube_half_size)

    seismic = torch.FloatTensor(np.load(args.data_root+"seismic_cube.npy")[:, :, :, 0]).unsqueeze(0)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)

    train_dset = MalenovDataset(seismic, X_train, y_train, args.cube_size)
    val_dset = MalenovDataset(seismic, X_val, y_val, args.cube_size)

    train_loader = DataLoader(train_dset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    model = MalenovNet()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.5, 0.9))

    for epoch in range(args.epochs):
        model.train()
        avg_acc = 0.
        avg_loss = 0.
        for i, (X, y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            y_pred = model(X.to(device))
            loss = criterion(y_pred, y.to(device))
            loss.backward()
            avg_loss += loss.detach().item()*X.size(0)
            optimizer.step()
            pred = torch.argmax(F.softmax(y_pred.detach(), 1), 1)
            avg_acc += accuracy_score(y_true=y.cpu().numpy(), y_pred=pred.cpu().numpy())*X.size(0)

        if epoch % args.log_interval == args.log_interval-1:
            print("Train: Average Loss: ", avg_loss/len(train_dset))
            print("Train: Average Accuracy: ", avg_acc/len(train_dset))

            train_writer.add_scalar('loss', avg_loss/len(train_dset), epoch)
            train_writer.add_scalar('accuracy', avg_acc/len(train_dset), epoch)

        model.eval()
        avg_acc = 0.
        avg_loss = 0.
        with torch.set_grad_enabled(False):
            for i, (X, y) in enumerate(tqdm(val_loader)):
                optimizer.zero_grad()
                y_pred = model(X.to(device))
                loss = criterion(y_pred, y.to(device))
                avg_loss += loss.detach().item()*X.size(0)
                pred = torch.argmax(F.softmax(y_pred.detach(), 1), 1)
                avg_acc += accuracy_score(y_true=y.cpu().numpy(), y_pred=pred.cpu().numpy())*X.size(0)

        if epoch % args.log_interval == args.log_interval-1:
            print("Val: Average Loss: ", avg_loss/len(val_dset))
            print("Val: Average Accuracy: ", avg_acc/len(val_dset))
            val_writer.add_scalar('loss', avg_loss/len(val_dset), epoch)
            val_writer.add_scalar('accuracy', avg_acc/len(val_dset), epoch)

        if epoch % args.checkpoint_interval == args.checkpoint_interval-1:
            save_checkpoints(args, epoch, model, optimizer, time_str)

if __name__ == '__main__':
    main()