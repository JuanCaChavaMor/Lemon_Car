#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 00:00:26 2023

@author: juancarl
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(df_train, df_test, cat_cols):
    '''
    Performs label encoding and returns a fitted and transformed encoded df_train and a transformed df_test.
    
    Parameters:
    df_train (pd.DataFrame): Training features to be fit and transform.
    df_test (pd.DataFrame): Test features to be transformed.
    cat_cols (list): Categorical features. You should have the same categorical columns in both DF.

    Returns:
    pd.DataFrame, pd.DataFrame: Fitted and transformed training DataFrame, transformed test DataFrame.
    '''
    # Create a copy of the DataFrames
    encoded_df_train = df_train.copy()
    encoded_df_test = df_test.copy()
    
    encoder = LabelEncoder() 
    for col in cat_cols:
        encoded_df_train[col] = encoder.fit_transform(encoded_df_train[col])
        encoded_df_test[col] = encoder.transform(encoded_df_test[col])
    
    return encoded_df_train, encoded_df_test