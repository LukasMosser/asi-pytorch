import os
import torch
import shutil

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