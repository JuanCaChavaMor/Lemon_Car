#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 00:39:08 2023

@author: juancarl
"""
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

def oversample_data(X, y):
    '''
    Perform oversampling on the input data (X) and target labels (y).
    X: Input features
    y: Target labels (0 or 1)
    Returns the oversampled X and y.
    '''
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled