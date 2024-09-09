#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:20:55 2024

@author: physics
"""
from sys import argv


def fwhm(lam, d_tel):
    fwhm = 1.028 * 206265 * float(lam) * 1e-6 / float(d_tel)
    return fwhm

script, lam, d_tel = argv

print('The FWHM is %f' %(fwhm(lam, d_tel)))
