#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:31:30 2023

@author: juancarl
"""
import pandas as pd
import split_categ_contin_cols 

def findNullVals(df): 
    
    null_categcol = []
    null_contincol = []
    
    null_vals = df.isnull().sum().sort_values()
    
    df_null = pd.DataFrame({'nullcols' : null_vals.index, 'countval' : null_vals.values})
    df_null = df_null[df_null.countval > 0]
    
    print ("Null variables with values :", df_null)
    print ("Duplicateged values :", df_null.duplicated().sum())
    
    nullcolumns = list(df_null.nullcols)
    null_categcol, null_contincol = split_categ_contin_cols(df)
    
    return null_categcol, null_contincol