#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:43:02 2023

@author: juancarl
"""
import pandas as pd
import rename 
import split_categ_contin_cols
import fill_NAN_mo_me 
import findNullVals 


def clean_data(df):
    '''
    Cleans the data df in order to use later for modelation. it will be use in Features Engineering
    '''
    # Convert columns to string if they are not already
    string_columns = ['Transmission', 'TopThreeAmericanName', 'Color']
    for col in string_columns:
        df[col] = df[col].astype(str)
    # Rename 
    df = rename(df) # That includes the binarization of Purch_Year
    ## Separate continuous from category and object columns
    null_categcols, null_contincols = split_categ_contin_cols(df)
    # Fill the NaN values with column's mode and media
    df = fill_NAN_mo_me(df, null_categcols, null_contincols)
    
    # Checking Null and duplicates
    findNullVals(df)
    return df