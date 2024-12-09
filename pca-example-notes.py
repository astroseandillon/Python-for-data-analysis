# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 20:32:36 2024

@author: seand
"""





import numpy as np
import numpy.random as rnd

import matplotlib.pyplot as plt

#def runme(npca=5,plotrelpc=True,plotorigvspc=True):

npca=5
    
#####Setting Up
ndim = 10
mu = np.array([3]+np.random.rand(ndim)) # Mean
sigma=np.full((ndim,ndim),5+np.random.rand(ndim*ndim).reshape(ndim,ndim))

#some trickery to adjust the relative PCs to an interesting range of values
sigarr=5+np.arange(0,ndim)*0.5
sigarr[0:2]=np.array([15,7.5])
np.fill_diagonal(sigma,sigarr)

# print("Mu ", mu.shape)
# print("Sigma ", sigma.shape)

 # Create 1000 samples using mean and sigma
org_data = rnd.multivariate_normal(mu, sigma, size=(1000))
print("Data shape ", org_data.shape)


 ##### Step 1 -  Subtract mean from data
mean = np.mean(org_data, axis= 0)
# print("Mean ", mean.shape)

mean_data = org_data - mean
# print("Data after subtracting mean ", org_data.shape, "\n")


 #cov = np.cov(mean_data.T)
 #cov = np.cov(mean_data,rowvar=False)

#### Step 2 - Computing the Covariance Matrix
cov=np.matmul(mean_data.T,mean_data)

 #print("Covariance matrix ", cov.shape, "\n")


#### Step 3 - Eigendecomposition
eig_val, eig_vec = np.linalg.eigh(cov)

# print("Eigen vectors ", eig_vec.shape)
# print("Eigen values ", eig_val.shape, "\n")


#### Step 4 - Sort the Eigenvalues and Eigenvectors, large to small
indices = np.argsort(eig_val)[::-1]

eig_val = eig_val[indices]
eig_vec = eig_vec[:,indices]


###some analysis
 # Get explained variance
sum_eig_val = np.sum(eig_val)
explained_variance = eig_val/ sum_eig_val

print("Explained variance ", explained_variance)
cumulative_variance = np.cumsum(explained_variance)
print("Cumulative variance ", cumulative_variance)

plotrelpc=True
if plotrelpc:
  # Plot explained variance
  plt.plot(np.arange(0, len(explained_variance), 1), cumulative_variance,marker='o')
  plt.title("Explained variance vs number of components")
  plt.xlabel("Number of components")
  plt.ylabel("Explained variance")
  plt.show()

####end some analysis

































