#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:31:24 2024

@author: physics
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
# data = np.loadtxt('pca_csv.csv', dtype=str)
df = pd.read_csv('pca_csv.csv')

print(df.columns)




data = df.iloc[:,4:-1]














