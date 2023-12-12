#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:38:00 2023

@author: juancarl
"""
import pandas as pd
import numpy as np 

def fill_NAN_mo_me(df, categcols, contincols):
    '''
    - Fill missing values (NaN) in specified categorical columns using the mode
    and continuous columns with the median.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    categcols (list): List of categorical columns to fill missing values.
    contincols (list): List of continuous columns to fill missing values.
    
    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    '''
    # Copy the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Fill missing values with the mode for categorical columns
    for column in categcols:
        if column in df.columns:
            mode_value = df[column].mode().iloc[0]
            df[column].fillna(mode_value, inplace=True)
        else:
            print(f"Warning: Column '{column}' not found in the DataFrame.")
    
    # Fill missing values with the median for continuous columns
    for column in contincols:
        if column in df.columns:
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
        else:
            print(f"Warning: Column '{column}' not found in the DataFrame.")
    
    return df