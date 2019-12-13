#!/usr/bin/env python
import time
import os
import numpy as np
import pandas as pd
import scipy
from glob import glob
from scipy.io import mmread
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset

class DNaseDataset(Dataset):
    def __init__(self, input_file_name,expected_file_name,
                 low = 0,
                 transpose = False,
                 transforms=[]):

        self.input,self.output = load_data(input_file_name,expected_file_name, transpose)
        for transform in transforms:
            self.input = transform(self.input)
            self.output = transform(self.output)

        self.input = torch.from_numpy(self.input).float()
        self.output = torch.from_numpy(self.output).float()
        self.n_cells, self.n_peaks = self.output.shape

    def __len__(self):
        return self.output.shape[0]

    def __getitem__(self, index):
        input = self.input[index]
        output = self.output[index]
        return (input,output)

def load_data(input_file_name,expected_file_name, transpose=False):
    if os.path.isfile(input_file_name) and os.path.isfile(expected_file_name):
        input_data = pd.read_csv(input_file_name, sep="\t", header=0,index_col=0).T.astype('float32').values
        output_data = pd.read_csv(expected_file_name, sep="\t", header=0,index_col=0).T.astype('float32').values
    else:
        raise ValueError("File {} not exists".format(path))

    if transpose:
        input_data = input_data.transpose()
        output_data = output_data.transpose()
    return(input_data,output_data)
