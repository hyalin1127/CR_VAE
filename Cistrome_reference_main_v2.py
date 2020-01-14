#!/usr/bin/env python
"""
# Author:  Chen-Hao Chen
# File Name: Cistrome_reference.py
"""
import torch
import numpy as np
import pandas as pd
import os
import argparse

from model_v3 import CR_VAE
from app_IO import DNaseDataset
from utils import read_labels, cluster_report, estimate_k, binarization
from plot import plot_embedding

from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = "/n/scratchlfs02/xiaoleliu_lab/cchen/sc_imputation/VAE/DNase_data/"
model_path = "/n/scratchlfs02/xiaoleliu_lab/cchen/sc_imputation/VAE/model/"
#data_path = "/Users/chen-haochen/Dropbox/Cistrome_imputation/scATAC/VAE/data/"
#model_path = "/Users/chen-haochen/Dropbox/Cistrome_imputation/scATAC/VAE/model/"

def output():
    # Derive latent features
    '''
    input = pd.read_csv(expected_file_name,sep="\t",header=0,index_col=0)
    model.load_state_dict(torch.load('/%s/CR_VAE_model_%s.ckpt' %(model_path,model_name), map_location=lambda storage, loc: storage),strict=False)
    feature = model.encodeBatch(testloader, device=device, out='z')
    feature = pd.DataFrame(feature)
    feature.index = input.columns.values
    feature.to_csv(os.path.join(outdir, '%s_feature.txt' %(model_name)), sep='\t', header=False,index=True)
    '''
    '''
    # Dervie single cell features
    model.load_state_dict(torch.load('/%s/CR_VAE_model_%s.ckpt' %(model_path,model_name), map_location=lambda storage, loc: storage),strict=False)
    for type in ["BMMC_healthy","BMMC_PBMC","BMMC_CLL"]:
        expected_file_name = "/n/scratchlfs/xiaoleliu_lab/cchen/sc_imputation/VAE/sc_specific_data/%s_filtered_specific_peaks.csv" %(type)
        input = pd.read_csv(expected_file_name,sep="\t",header=0,index_col=0)
        dataset = DNaseDataset(expected_file_name,expected_file_name,transpose = False)
        loader_params = {'batch_size': batch_size, 'shuffle': False,'num_workers': 8, 'drop_last': False, 'pin_memory': True}
        testloader = DataLoader(dataset,**loader_params)
        feature = model.encodeBatch(testloader, device=device, out='z')
        feature = pd.DataFrame(feature)
        feature.index = input.columns.values
        feature.to_csv(os.path.join(outdir, '%s_%s_specific_feature.txt' %(type,model_name)), sep='\t', header=False)
    '''

def main():
    #args = prepare_optparser()
    # setting:
    #seed = args.seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    batch_size = 64
    lr = 0.0001
    weight_decay = 5e-4
    model_name = "VAE_test_v14"
    outdir = model_path

    k = 50
    latent = 10
    input_dim = 120000
    encode_dim = [2048, 128]
    decode_dim = [1024]
    max_iter = 3000

    dims = [input_dim, latent, encode_dim, decode_dim]
    model = CR_VAE(dims, n_centroids=k)

    normalizer = MaxAbsScaler()

    expected_file_name = "/%s/DNase_test_peaks.csv" %(data_path)
    dataset = DNaseDataset(expected_file_name,expected_file_name,transpose = False)
    loader_params = {'batch_size': batch_size, 'shuffle': False,'num_workers': 16, 'drop_last': False, 'pin_memory': True}
    testloader = DataLoader(dataset,**loader_params)

    # Training

    model.init_gmm_params(testloader)

    model.load_state_dict(torch.load('/%s/CR_VAE_model_VAE_test_v13.ckpt' %(model_path))) #*****

    input_files = []
    for iteration in range(0,80):
        input_file_name = "/%s/DNase_train_peaks_%s.csv" %(data_path,str(iteration))
        expected_file_name = "/%s/DNase_test_peaks.csv" %(data_path)
        input_files.append([input_file_name,expected_file_name])

    model.fit(input_files,
              lr=lr,
              batch_size = batch_size,
              weight_decay = weight_decay,
              device = device,
              max_iter= max_iter,
              name = model_name,
              outdir = outdir
              )

    torch.save(model.state_dict(), '/%s/CR_VAE_model_%s.ckpt' %(model_path,model_name))
    torch.save(model, '/%s/CR_VAE__model_%s.tmp' %(model_path,model_name)) # Save the whole model

if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        sys.stderr.write("User interrupt me! ;-) Bye!\n")
        sys.exit(0)
