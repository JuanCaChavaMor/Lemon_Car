#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:23:16 2023

@author: juancarl
"""
import pandas as pd


def split_categ_contin_cols(df):
    '''
    Auxiliar function to separate continuos from object and category columns of df
    '''
    categ_cols = []
    contin_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            categ_cols.append(col)
        else:
            contin_cols.append(col)
    
    return categ_cols, contin_cols