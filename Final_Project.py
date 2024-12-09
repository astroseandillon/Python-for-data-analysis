#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:31:24 2024

@author: physics
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
from sklearn.decomposition import PCA

# from astropy.io import ascii
# from astropy.table import Table
# I tried to use astropy but it didn't want to work as well as pandas

import pandas as pd

df = pd.read_csv('pca_csv.csv')
# Loading the csv file into a pandas dataframe

dmatrix = df[['Kinematic Age (yr)', 
                'Central Star Temp (K)', 
                'Aspect Ratio (Overall)',
                'PG Mass (Solar M)',
                'C(e-4)',
                'O(e-4)',
                'C/O',
                'Metallicity [O/H](e-4)',
                'Radial Velocity (km/s)',
                'Magnititude',
                'MAT',
                'Mass/Temp',
                'MA',
                'A/T',
                'TA']]
# Specifying which columns we're interested in


data = dmatrix.values
# making an array of the data points we care about

# evals, evecs = np.linalg.eigh(data)

data = np.nan_to_num(data, copy=True, nan=-1.0)
data=data

pca = PCA(n_components=5)
pca.fit(data)


print(pca.explained_variance_)

print('')




mean = np.mean(data,axis=0)

mean_data = data-mean

cov = np.matmul(mean_data.T, mean_data)

eig_val, eig_vec = np.linalg.eigh(cov)

indices = np.argsort(eig_val)[::-1]

eig_val = eig_val[indices]
eig_vec = eig_vec[:,indices]

sum_eig_val = np.sum(eig_val)
explained_variance = eig_val/ sum_eig_val

print("Explained variance ", explained_variance)
cumulative_variance = np.cumsum(explained_variance)
print("Cumulative variance ", cumulative_variance)


























# data = np.where(data, data==np.nan,-1)







# <Table length=136>
#          name           dtype     class     n_bad
# ---------------------- ------- ------------ -----
#                   Name   str19       Column     0
#                     RA   str11       Column     0
#                    Dec   str12       Column     0
#            Other names   str80 MaskedColumn     1
#     Kinematic Age (yr)   int64 MaskedColumn    25
#  Central Star Temp (K)   int64 MaskedColumn    42
# Structure (Literature)   int64       Column     0
# Aspect Ratio (Overall) float64       Column     0
#    Opening Angle (Avg)    str5 MaskedColumn   100
#      PG Mass (Solar M) float64 MaskedColumn    78
#                 C(e-4) float64 MaskedColumn    83
#                 O(e-4) float64 MaskedColumn    70
#                    C/O float64 MaskedColumn    68
# Metallicity [O/H](e-4) float64 MaskedColumn    52
# Radial Velocity (km/s) float64       Column     0
#            Magnititude float64 MaskedColumn    31
#                    MAT float64 MaskedColumn    80
#              Mass/Temp float64 MaskedColumn    79
#                     MA   int64 MaskedColumn    78
#                    A/T float64 MaskedColumn    46
#                     TA   int64 MaskedColumn    47


# fig, ax = plt.subplots()
# ax.scatter(data['Magnititude'], data['Mass/Temp'])












