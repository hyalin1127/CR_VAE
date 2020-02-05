#!/usr/bin/env python
"""
# Original Author: Xiong Lei
# Modified: Chen-Hao Chen
# File Name: model.py
# Description:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau

import time
import math
import numpy as np
from tqdm import trange
from itertools import repeat
from sklearn.mixture import GaussianMixture

from layer import Encoder, Decoder, build_mlp, DeterministicWarmup
from loss import elbo, elbo_SCALE
from app import *
from app_IO import DNaseDataset
from torch.utils.data import DataLoader

class CR_VAE(nn.Module):
    def __init__(self, dims, n_centroids, bn=False, dropout=0.2, binary=True):
        super(CR_VAE, self).__init__()

        [x_dim, z_dim, encode_dim, decode_dim] = dims
        self.binary = binary
        if binary:
            decode_activation = nn.Sigmoid()
        else:
            decode_activation = None

        self.encoder = Encoder([x_dim, encode_dim, z_dim], bn=bn, dropout=dropout)
        self.decoder = Decoder([z_dim, decode_dim, x_dim], bn=bn, dropout=dropout, output_activation=decode_activation)

        self.reset_parameters()

        self.n_centroids = n_centroids
        z_dim = dims[1]

        # init c_params
        self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(z_dim, n_centroids)) # mu
        self.var_c = nn.Parameter(torch.ones(z_dim, n_centroids)) # sigma^2

    def loss_function(self, x,y):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
        likelihood, kld = elbo_SCALE(recon_x, x, gamma, (mu_c, var_c, pi), (mu, logvar), binary=self.binary)

        return -likelihood, kld

    def get_gamma(self, z):
        """
        Inference c from z
        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N,1) # NxK
        mu_c = self.mu_c.repeat(N,1,1) # NxDxK
        var_c = self.var_c.repeat(N,1,1) # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, dataloader, device='cpu'):
        """
        Init SCALE model with GMM model parameters
        """
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        z = self.encodeBatch(dataloader, device)
        gmm.fit(z)
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))

    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.
        :param x: input data
        :return: reconstructed input
        """
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x

    def predict(self, dataloader, device='cpu', method='kmeans'):
        """
        Predict assignments applying k-means on latent feature
        Input:
            x, data matrix
        Return:
            predicted cluster assignments
        """

        if method == 'kmeans':
            from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
            feature = self.encodeBatch(dataloader, device)
            kmeans = KMeans(n_clusters=self.n_centroids, n_init=20, random_state=0)
            pred = kmeans.fit_predict(feature)
        elif method == 'gmm':
            logits = self.encodeBatch(dataloader, device, out='logit')
            pred = np.argmax(logits, axis=1)

        return pred

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def fit(self,
            files,
            batch_size,
            lr=0.002,
            weight_decay=5e-4,
            device='cpu',
            beta = 1,
            n = 200,
            max_iter=30000,
            verbose=True,
            name='',
            patience=10,
            outdir='./'
        ):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        Beta = DeterministicWarmup(n=n, t_max=beta)

        iteration = 0
        early_stopping = EarlyStopping(patience=patience, outdir=outdir)
        loss_record = open("/%s/%s_loss_record.txt" %(outdir,name),'wt')
        epoch_loss = []

        for i in range(10):
            for input_file_name,expected_file_name in files:
                dataset = DNaseDataset(input_file_name,expected_file_name,transpose = False)
                loader_params = {'batch_size': batch_size, 'shuffle': True,'num_workers': 16, 'drop_last': True, 'pin_memory': True}
                dataloader = DataLoader(dataset,**loader_params)
                file_loss = 0

                for x,y in dataloader:
                    epoch_lr = adjust_learning_rate(lr, optimizer, iteration)
                    x = x.float().to(device)
                    y = y.float().to(device)
                    optimizer.zero_grad()

                    recon_loss, kl_loss = self.loss_function(x,y)
                    loss = (recon_loss + next(Beta) * kl_loss)/len(x);
                    loss.backward()
                    optimizer.step()

                    file_loss += loss.item()
                    iteration+=1

                loss_record.write("%s\n" %str(file_loss))
                epoch_loss.append(file_loss)

            if len(epoch_loss)>2400 and (np.sum(epoch_loss[-50:])/np.sum(epoch_loss[-100:-50]))>0.9999:
                loss_record.write('EarlyStopping: run {} epochs'.format(len(epoch_loss)))
                break
                        #early_stopping(epoch_loss, self)
                        #if early_stopping.early_stop:
                        #    loss_record.write('EarlyStopping: run {} iteration'.format(iteration))
                        #    break
                        #continue

    def encodeBatch(self, dataloader, device='cpu', out='z', transforms=None):
        output = []
        for inputs,outputs in dataloader:
            x = inputs
            x = x.view(x.size(0), -1).float().to(device)
            z, mu, logvar = self.encoder(x)

            if out == 'z':
                output.append(z.detach().cpu())
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach())

        output = torch.cat(output).numpy()
        if out == 'x':
            for transform in transforms:
                output = transform(output)
        return output
