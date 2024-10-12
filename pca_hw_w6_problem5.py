# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:18:29 2024

@author: seand
"""

import numpy as np



def pca(dat, npc):
    mean = np.mean(dat,axis=0)
#1. subtract mean from data
    mean_sub = dat - mean
#2. compute covariance matrix
    cov = np.matmul(mean_sub.T,mean_sub)
#3. eigendecomposition
    eig_val, eig_vec = np.linalg.eigh(cov) 
#4. sort eigenvalues and eigenvectors large to small
    indices = np.argsort(eig_val)[::-1]
    eig_val = eig_val[indices]
    eig_vec = eig_vec[:,indices]
#5. get explained variance
    sum_eig_val = np.sum(eig_val)
    explained_variance = eig_val / sum_eig_val
#6. select number of PCs to compute
    n_comp = npc
    eig_vec = eig_vec[:,:n_comp]
    eig_val = eig_val[:n_comp]
#7. compute pca
    pca_data = mean_sub.dot(eig_vec)
    return pca_data, explained_variance













data=np.array([[7., 4., 3.],
               [4., 1., 8.],
               [6., 3., 5.],
               [8., 6., 1.],
               [8., 5., 7.],
               [7., 2., 9.],
               [5., 3., 3.],
               [9., 5., 8.],
               [7., 4., 5.],
               [8., 2., 2.]])

print('hello world')
