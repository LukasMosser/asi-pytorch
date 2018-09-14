import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import StratifiedShuffleSplit
from tqdm import tqdm as tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MalenovTestDataset
from options import prepare_output_directory, get_args
from model import MalenovNet
from utils import save_checkpoints


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
    
    seismic = torch.FloatTensor(np.load(args.data_root+"seismic_cube.npy")[:, :, :, 0]).unsqueeze(0)

    test_dset = MalenovTestDataset(seismic, args.inline, args.cube_size)
    test_loader = DataLoader(test_dset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    model = MalenovNet()
    ckpt = torch.load(args.checkpoint_path)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    model.eval()
    avg_acc = 0.
    avg_loss = 0.

    np.save("indices_"+str(args.inline)+".npy", test_dset.indices)
    np.save("amplitudes_"+str(args.inline)+".npy", test_dset.amplitudes)
    preds = []
    with torch.set_grad_enabled(False):
        for i, X in enumerate(tqdm(test_loader)):
            y_pred = model(X.to(device))
            preds.extend(torch.argmax(F.softmax(y_pred.detach(), 1), 1).cpu().numpy())
    np.save("preds_"+str(args.inline)+".npy", np.array(preds))
    

if __name__ == '__main__':
    main()