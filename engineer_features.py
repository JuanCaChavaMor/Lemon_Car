#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 00:32:44 2023

@author: juancarl
"""
import pandas as pd
import numpy as np
import split_categ_contin_cols 
import rename
import fill_NAN_mo_me
import findNullVals
import clean_data 
import encode_categorical_columns 

def engineer_features(df_train, df_test):
    ''' Prepares df for modeling
    Arguments: 
    df_train: Training features
    df_test: Test features
    '''
    # Delete redundant and ethical cols
    df_train = df_train.drop(
    ['AUCGUART', 'PRIMEUNIT', 'BYRNO', 'VNZIP1', 'VehYear', 'WheelTypeID', 'VNST', 'Nationality'],
               axis=1)
    df_test = df_test.drop(
    ['AUCGUART', 'PRIMEUNIT', 'BYRNO', 'VNZIP1', 'VehYear', 'WheelTypeID', 'VNST', 'Nationality'],
               axis=1)
    # Category Columns: Both df have the same cat_cols or you are in trouble!
    cat_cols, con_cols = split_categ_contin_cols(df_train) # we split the columns in order to transform cat_cols
    # Cleaning
    df_train = clean_data(df_train)
    df_test = clean_data(df_test)
    #  encoding: fit transform encode for df_train and just transform for df_test
    df_train, df_test = encode_categorical_columns(df_train, df_test, cat_cols)
    # Remove the original categorical columns
    #df = df.drop(cat_cols, axis=1)
    
    return df_train, df_test