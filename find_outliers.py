#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 00:35:39 2023

@author: juancarl
"""
import pandas as pd
import numpy as np
def find_outliers(df, target):
    '''
    Create a dictionary outlier_counts to store the number of outliers per column.
    Find outliers and delete cells containing them.
    Also, count how many cells in the target with value 1 are deleted.
    '''
    outlier_counts = {}  # Dictionary to store outlier counts for each column
    deleted_positive_targets = 0  # Count of target cells with value 1 that are deleted
    
    # Create a copy of the original target
    target_copy = target.copy()
    
    # Iterate through all columns of data type 'float64' with more than two unique values
    for col in df.select_dtypes(include=['float64']).columns:
        if len(df[col].unique()) <= 2:
            continue  # Skip binary & integer columns
        
        # Get variable stats
        stats = df[col].describe()
        
        IQR = stats['75%'] - stats['25%']
        upper = stats['75%'] + 1.5 * IQR
        lower = stats['25%'] - 1.5 * IQR
        
        print('The upper and lower bounds of {} for candidate outliers are {} and {}.'.format(col, upper, lower))
        
        # Identify rows with outliers
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        num_upper_outliers = outlier_mask.sum()
        outlier_counts[col] = num_upper_outliers  # Store outlier count
        
        print("Values greater than upper bound : ", num_upper_outliers)
        
        # Create a second mask to identify rows with values below the lower bound
        lower_outlier_mask = (df[col] < lower)
        
        # Count how many target cells with value 1 are deleted using the second mask
        deleted_positive_targets += target_copy[lower_outlier_mask].sum()
        
        # Use the second mask to remove rows with values below the lower bound
        df = df[~lower_outlier_mask]
        target_copy = target_copy[~lower_outlier_mask]
    
    print("Outlier Counts:")
    for col, count in outlier_counts.items():
        print(f"{col}: {count} outliers")
    
    print("Deleted positive target cells with value 1:", deleted_positive_targets)
                
    return df, target_copy
