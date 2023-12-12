#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:35:22 2023

@author: juancarl
"""
import pandas as pd
import numpy as np

def rename(df):
    '''Rename columns to ease the treatment of the features'''
    # Mapping of sizes to categories
    '''
    size_mapping = {
        'COMPACT': 'small',
        'SMALL TRUCK': 'small',
        'SMALL SUV': 'small',
        'SPORTS': 'small',
        'MEDIUM': 'medium',
        'MEDIUM SUV': 'medium',
        'CROSSOVER': 'medium',
        'LARGE': 'large',
        'LARGE SUV': 'large',
        'LARGE TRUCK': 'large',
        'VAN': 'large',
        'SPECIALTY': 'large'
    }
    
    # Use the map function to replace values in the 'Size' column
    df['Size'] = df['Size'].map(size_mapping)
    '''
    # Mapping of nationalities
    '''df['Nationality'] = df['Nationality'].replace({
        'TOP LINE ASIAN': 'JAPANESE',
        'OTHER': 'EUROPEAN'
    })
    '''
    # Additional renaming operations
    # Check the data type before applying .str methods
    if df['Transmission'].dtype == 'object':
        df['Transmission'] = df['Transmission'].str.strip()
        df['Transmission'].replace({'Manual': 'MANUAL'}, inplace=True)
        df['Transmission'].replace('nan', np.nan, inplace=True)
    if df['Color'].dtype == 'object':
        df['Color'] = df['Color'].str.strip()
        df['Color'].replace({'NOT AVAIL': 'nan', 'OTHER': 'nan'}, inplace=True)

    if df['TopThreeAmericanName'].dtype == 'object':
        df['TopThreeAmericanName'] = df['TopThreeAmericanName'].str.strip()
        df['TopThreeAmericanName'].replace({'OTHER': 'nan'}, inplace=True)
        df['TopThreeAmericanName'].replace('nan', np.nan, inplace=True)
    df['IsOnlineSale'] = df['IsOnlineSale'].astype('int8')
    df['WarrantyCost'] = df['WarrantyCost'].astype('float64')
    df['VehOdo']=df['VehOdo'].astype('float64')
    # We are interested in the year
    # df.loc[:, 'Purch_Year'] = df['PurchDate'].dt.year (deleted for probe test)
    # Drop the original 'PurchDate' 
    #df = df.drop('PurchDate', axis =1 ) 
    # Purch_Year to Binary
    #df['Purch_Year'] = np.where(df['Purch_Year'] == 2009, 1, 0)
    return df