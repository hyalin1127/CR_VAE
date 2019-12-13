import os
import math
import numpy as np
from tqdm import trange
from itertools import repeat
import torch

def adjust_learning_rate(init_lr, optimizer, iteration):
    lr = max(init_lr * (0.9 ** (iteration//10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, outdir='./'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = os.path.join(outdir, 'model.pt')

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss
