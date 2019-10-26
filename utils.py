import sys
import os
import torch
import shutil


# refer to https://github.com/xternalz/WideResNet-pytorch
def save_checkpoint(state, config, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists(config.snapshot_dir):
        os.makedirs(config.snapshot_dir)
    filename = config.snapshot_dir + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, config.snapshot_dir + 'model_best.pth.tar')