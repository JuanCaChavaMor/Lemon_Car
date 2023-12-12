#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 23:19:08 2023

@author: juancarl
"""

# import basic modules 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Modules
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier

from imblearn.pipeline import Pipeline #our new pipeline builer

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix, classification_report
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score

np.set_printoptions(suppress=True) 
pd.options.display.float_format = '{:.2f}'.format
## Used created libraries
import split_categ_contin_cols
import rename 
import fill_NAN_mo_me 
import findNullVals 
import clean_data 
import encode_categorical_columns 
import engineer_features 
import find_outliers 
import oversample_data 

def predict_rf(csv_file_path='data_train.csv'):
    '''
    Reads a csv file, performs split, features engineering. Just on trainset: sampling, outlier removal,
    PCA and modelling (best model).
    Args:     csv_file_path (str): Path to the CSV file.
    Returns : A tuple containing predictions and a DataFrame of scores and the data sets for posterior work.
    
    '''
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file_path)
    # Remove noise columns
    noise_cols = ['Model','Trim','SubModel','PurchDate']
    data=data.drop(noise_cols, axis=1)
    #train_test_split

    # Set a random seed for reproducibility
    random_seed = 42
    # Define the target variable ('IsBadBuy')
    target = data['IsBadBuy']
    # Define the features (all columns except 'IsBadBuy')
    features = data.drop(columns=['IsBadBuy'])
    # Split the data into training and testing sets (test size: 10%, random seed: 42)
    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.10,
                                                                            random_state=random_seed)
    aux_encoding = features_train.copy()      ## Save a copy of features_train without encoding to use as fit pattern for aim features
    # Features Engineering & data cleaning
    features_train, features_test = engineer_features(features_train, features_test)
    
    # Remove outliers from training features
    print('Outliers in train set after oversampling')
    features_train, target_train = find_outliers(features_train, target_train)
    # Resampling training Features
    features_train, target_train = oversample_data(features_train, target_train)
      
    # PCA (Optional)
    
    #features_train, features_test= apply_pca(features_train, features_test, 2)
    
    
    # Model
    # Initialize the model with the best hyperparameters
    best_rf_model = RandomForestClassifier(max_depth= 20, max_features= 2,
                                       min_samples_split= 50, n_estimators= 150, random_state=42)
    # Create a pipeline with standardization
    pipeline_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("model_rf", best_rf_model)
    ])
    # Train the Model
    # Fit the pipeline on the standardized training data
    pipeline_rf.fit(features_train, target_train)
    predictions = pipeline_rf.predict(features_test)
    # Check the dimensions of the train and test sets
    print("features_train shape:", features_train.shape)
    print("features_test shape:", features_test.shape)
    print("target_train shape:", target_train.shape)
    print("target_test shape:", target_test.shape)
    print('------------------------------------------- :)')
    # Goodness of fit
    # Evaluate the best model
    accuracy = accuracy_score(target_test, predictions)
    recall = recall_score(target_test, predictions)
    precision = precision_score(target_test, predictions)
    f1 = f1_score(target_test, predictions)

    # Show the scores
    scores_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1'],
    'Score': [accuracy, recall, precision, f1]})
    
    return predictions, scores_df, features_train, features_test, target_train, target_test, aux_encoding

predictions, scores, df_train, df_test, target_train, target_test, aux_encoding = predict_rf('data_train.csv')


# Generate the confusion matrix
cm = confusion_matrix(target_test, predictions)
x_labels = ['IsBadBuy = 1', 'IsBadBuy = 0']
y_labels = ['IsBadBuy = 1', 'IsBadBuy = 0']
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=x_labels, yticklabels=y_labels, cbar = True)
plt.xlabel('Predicted')
plt.title('Confusion Matrix WITHOUT PCA')
plt.ylabel('True')
plt.show()




# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(target_test, predictions)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_score(target_test, predictions))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic WITHOUT PCA')
plt.legend(loc="lower right")
plt.show()

# Model Interpretation

# Model
# Initialize the model with the best hyperparameters (you used rf_model before, not best_rf_model)
rf_model = RandomForestClassifier(max_depth= 20, max_features= 2,
                                       min_samples_split= 50, n_estimators= 150, random_state=42)

# Create a pipeline with standardization
pipeline_rf_fe = Pipeline([
    ("scaler", StandardScaler()),
    ("model_rf", rf_model)  # Use rf_model here, not best_rf_model
])

# Train the Model
# Fit the pipeline on the standardized training data (you used df_train instead of features_train)
pipeline_rf_fe.fit(df_train, target_train)
predictions_fe = pipeline_rf_fe.predict(df_test)  # Use pipeline_rf_fe, not pipeline_rf

print('number of features:', df_train.shape[1])

# Get feature importances from the trained model
feature_importances = rf_model.feature_importances_  # Use rf_model, not pipeline_rf_fe

# Create a DataFrame to store the feature names and their importances
importance_df = pd.DataFrame({'Feature': df_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by feature importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the top N most important features (adjust N as needed)
N = 10
top_features = importance_df.head(N)
print("Top", N, "most important features:")
print(top_features)

# Visualize the feature importances

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top Feature Importances')
plt.show()