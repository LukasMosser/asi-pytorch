import argparse 
import os
import datetime
import json

from tensorboardX import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    # Dataset configuration
    parser.add_argument('--data_root', type=str, default='../data/malenov/',
                        help='path to dataset root directory. default: ../data/malenov/')
    parser.add_argument('--batch_size', '-B', type=int, default=32,
                        help='mini-batch size of training data. default: 32')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for training data loader. default: 2')

    # Optimizer settings
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate of Adam. default: 0.01')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 (betas[0]) value of Adam. default: 0.5')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 (betas[1]) value of Adam. default: 0.9')

    # Training setting
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed. default: 42 (derived from Douglas Adams)')
    parser.add_argument('--epochs', '-E', type=int, default=10,
                        help='Max epoch number of training. default: 10')
    parser.add_argument('--cube_size', '-cs', type=int, default=32,
                        help='Size of input cube. default: 32')
    parser.add_argument('--train_size', '-ts', type=float, default=0.8,
                        help='Size of training split in range 0 to 1. default: 0.8')
    parser.add_argument('--use_stratified_kfold', '-strat', default=False, action='store_true',
                        help='If true use stratified kfold, else use left right split of training image. default: True')
    parser.add_argument('--num_examples', '-ne', type=int, default=10000,
                        help='Number of examples in an epoch. default: 10000')

    # Log and Save interval configuration
    parser.add_argument('--results_root', type=str, default='results',
                        help='Path to results directory. default: results')
    parser.add_argument('--checkpoint_interval', '-ci', type=int, default=1,
                        help='Interval of saving checkpoints (model and optimizer). default: 1')
    parser.add_argument('--log_interval', '-li', type=int, default=1,
                        help='Interval of showing losses. default: 100')
    # Resume training
    parser.add_argument('--args_path', default=None, help='Checkpoint args json path. default: None')
    parser.add_argument('--checkpoint_path', '-mcp', default=None,
                        help='Model and optimizer checkpoint path. default: None')
    
    #Test Time
    parser.add_argument('--inline', default=230, type=int, help='Set inline for prediction')
    args = parser.parse_args()

    args.results_root = os.path.expandvars(args.results_root)
    args.data_root = os.path.expandvars(args.data_root)
    return args

def prepare_output_directory(args):
    time_str = datetime.datetime.now().strftime('%y%m%d_%H%M')
    root = os.path.join(args.results_root, "malenov", time_str)
    os.makedirs(root, exist_ok=True)

    train_writer = SummaryWriter(root+"/train")
    val_writer = SummaryWriter(root+"/val")

    with open(os.path.join(root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))
    return args, train_writer, val_writer, time_str
