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

# read data
df_train = pd.read_csv('data_train.csv')
# Display all columns
pd.set_option('display.max_columns', None)

# Check first five rows of data
df_train.head()

df_train.info()

df_train.loc[:, 'PurchDate'] = pd.to_datetime(df_train.loc[:, 'PurchDate'], format='%m/%d/%Y')
df_train.head()

## Check unbalanced target variable
distrib_target = pd.crosstab(index=df_train.loc[:,"IsBadBuy"],
                             columns="count", 
                             normalize="columns")
distrib_target


# Check number of positive cases

df_train.loc[:,"IsBadBuy"].sum()

# Check the percentage of the missing values
percent_missing = df_train.isnull().sum() * 100 / len(df_train)
missing_value_df = pd.DataFrame({'percent_missing (%)': percent_missing})
missing_value_df.sort_values('percent_missing (%)', ascending=False)


# Drop columns with too many entries. They bring noise in our features, but we can recover them to see
# if valuable information arise

print('I will remove the columns Model, Trim & Submodel. They have information not important at this point')
print('Also Purchdate is erase: At this point is enough to know how old is our car. We can use this info for other analysis')
noise_cols = ['Model','Trim','SubModel', 'PurchDate']
df_train=df_train.drop(noise_cols, axis=1)

# perform train-test-split

# Set a random seed for reproducibility
random_seed = 42
# Define the target variable ('IsBadBuy')
target = df_train['IsBadBuy']
# Define the features (all columns except 'IsBadBuy')
features = df_train.drop(columns=['IsBadBuy'])
# Split the data into training and testing sets (test size: 10%, random seed: 42)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.10,
                                                                            random_state=random_seed)
# Check the dimensions of the train and test sets
print("features_train shape:", features_train.shape)
print("features_test shape:", features_test.shape)
print("target_train shape:", target_train.shape)
print("target_test shape:", target_test.shape)


# Create a copy of features_train and target_train for Feature Engineering
data_train = features_train.copy()
data_test = features_test.copy()
ziel_train = target_train.copy()

# Check if our test data is incomplete
features_test.isna().sum()

# save features_test as 'features_test.csv'
features_test.to_csv('features_test.csv', index=False)

print('Notice that Purch_Year-VehYear=VehicleAge')
print('search for unique values')
for column in features_train.columns:
    # Get the unique values in the column
    unique_values = features_train[column].unique()
    
    # Print the column name and its unique values
    print(f"Column: {column}")
    print("Unique Values:", unique_values)
    print("=" * 40)
features_train.head()

print("Size of the data: ",len(features_train))
print('Missing values in each column: \n', features_train.isnull().sum())

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
features_train= rename(features_train)
features_test = rename(features_test)
print(features_train.shape)
print(features_test.shape)

features_train['TopThreeAmericanName'].unique()

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

null_categcols, null_contincols =split_categ_contin_cols(features_train)
null_categcols

## Imputation

#features_test=fill_NAN_mode(features_test, null_categcols, null_contincols)
features_test.isnull().sum()

# find null and duplicate values from the dataframe.

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

## I MUSS HAVE AN IMPUTER STRATEGY for test data!!!!!

findNullVals(features_test)

# The next step is later explain

print('is explain later')
features_train=features_train.drop(['AUCGUART', 'PRIMEUNIT', 'BYRNO', 'VNST', 'Nationality', 'VNZIP1', 'VehYear', 'WheelTypeID'], axis=1)
features_test=features_test.drop(['AUCGUART', 'PRIMEUNIT', 'BYRNO', 'VNST', 'Nationality', 'VNZIP1', 'VehYear', 'WheelTypeID'], axis=1)
print(features_train.shape)
features_test.shape

print(features_test.isnull().sum())
print(100*features_test.isnull().sum().sum()/len(features_test))
len(features_test)

# What to do with missing values in the data set?
# Create a binary DataFrame indicating missing values
missing_binary = features_test.isnull().astype(int)

# Calculate the sum of missing values for each row
missing_row_counts = missing_binary.sum(axis=1)

# Check which rows have more than zero missing values
rows_with_missing = missing_row_counts[missing_row_counts > 0]

# Sort the rows by the number of missing values in descending order
rows_with_most_missing = rows_with_missing.sort_values(ascending=False)

print(100*len(rows_with_most_missing)/len(features_test))
# Calculate the number of positive cases in rows to be deleted
positive_cases_to_delete = target_test[rows_with_most_missing.index].sum()
print(positive_cases_to_delete/target_test.sum())
target_test.sum()

print('Delete is not an option: I will lose 23% of the positive cases in the test set')

features_test['WheelType'].unique()

play_test = features_test.copy()
play_test['IsBadBuy'] = target_test  # Add the 'IsBadBuy' column to test data
# Replace missing values in 'WheelType' with 'Missing'
play_test['WheelType'].fillna('Missing', inplace=True)

# Plot 'WheelType' vs. 'IsBadBuy' with 'hue' for positive cases
sns.countplot(data=play_test, x='WheelType', hue='IsBadBuy')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

plt.show()

# Create a mask to filter rows with "WheelType" equal to "Special" and "IsBadBuy" equal to 1
mask = (features_test['WheelType'] == 'Special') & (target_test == 1)

# Count the number of positive cases for "WheelType Special"
positive_cases_special = mask.sum()

print("Number of positive cases in 'WheelType Special' in features_test:", positive_cases_special)

# Create a mask to filter rows with "WheelType" equal to "Special" and "IsBadBuy" equal to 1
mask = (play_test['WheelType'] == 'Missing') & (target_test == 1)

# Count the number of positive cases for "WheelType Special"
positive_cases_special = mask.sum()

print("Number of positive cases in 'WheelType Missing' in features_test:", positive_cases_special)

print('I will impute therefore using the mode strategy')

## Imputer function

### Let's redefine
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

categcols, contincols = split_categ_contin_cols(features_train)
features_train=fill_NAN_mo_me(features_train, categcols, contincols)

# Imputation of Missing values on Test Set: Fill with mode on cat_cols, median on con:cols
categcols, contincols = split_categ_contin_cols(features_test)
features_test=fill_NAN_mo_me(features_test, categcols, contincols)

## Encoding
# Transform Columns
# Category Columns
cat_cols, con_cols = split_categ_contin_cols(features_train) # we split the columns in order to transform cat_cols

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

# Transform!
features_train, features_test = encode_categorical_columns(features_train, features_test, cat_cols)


### puedes cambiar esta wea por una one-hot 

print(features_train.shape)
print(features_test.shape)

# Outliers

## Looking in detail the continuous columns, aka con_cols

# Create subplots
fig, axes = plt.subplots(nrows=len(con_cols), ncols=2, figsize=(12, 6 * len(con_cols)))

# Loop through each continuous column
for i, col in enumerate(con_cols):
    # Box plot
    sns.boxplot(x= features_train[col], ax=axes[i, 0])
    axes[i, 0].set_title(f'Box Plot of {col}')
    
    # Distribution plot (histogram)
    sns.histplot(data=features_train, x=col, kde=True, ax=axes[i, 1], hue = target_train)
    axes[i, 1].set_title(f'Distribution Plot of {col}')
    
# Adjust subplot layout
plt.tight_layout()

# Show the plots
plt.show()

# Check for outliers
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
sns.scatterplot(data=df_train, x = 'MMRAcquisitionAuctionAveragePrice',
                 y  = 'MMRAcquisitionRetailAveragePrice',hue="IsBadBuy",ax=ax[0]);
sns.scatterplot(x='VehBCost', y='VehOdo', data=df_train,  hue='IsBadBuy',ax=ax[1]);

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_train, x='WarrantyCost', y='VehicleAge',hue='IsBadBuy');
plt.xlabel('WarrantyCost', fontsize=16);
plt.ylabel('VehicleAge', fontsize=16);
plt.legend(title='IsBadBuy', fontsize=16, loc='lower right');
plt.title( 'Vehicle age vs WarrantyCost');

print('Moral: Do not erase high values. They contain valuable information about the training data')

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

features_train, target_train = find_outliers(features_train, target_train)

# Check for outliers
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))
sns.scatterplot(data=features_train, x = 'MMRAcquisitionAuctionAveragePrice',
                 y  = 'MMRAcquisitionRetailAveragePrice',hue=target_train,ax=ax[0]);
sns.scatterplot(x='VehBCost', y='VehOdo', data=features_train,  hue=target_train ,ax=ax[1]);

plt.figure(figsize=(8,6))
sns.scatterplot(data=features_train, x='WarrantyCost', y='VehicleAge',hue=target_train);
plt.xlabel('WarrantyCost', fontsize=16);
plt.ylabel('VehicleAge', fontsize=16);
plt.legend(title='IsBadBuy', fontsize=16, loc='lower right');
plt.title( 'Vehicle age vs WarrantyCost');

## Data Clean

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

# Resampling

# Get the indices from features_train
indices_to_keep = features_train.index

# Filter target_train to keep only the corresponding indices
target_train = target_train.loc[indices_to_keep]


features_train = features_train
features_test = features_test

# initialize
undersampler = RandomUnderSampler(random_state=42)
oversampler = RandomOverSampler(random_state=42)
smotesampler = SMOTE(random_state=42)
# Model for evaluation
tree_clf = DecisionTreeClassifier(random_state=42)
#create search_space for Gridsearch (same parameters als Not.5 Kap III, Mod 2)
search_space = {'estimator__max_depth': range(2, 20, 2),
                'estimator__class_weight': [None, 'balanced']}

#Different Samplers,to iterate through
samplers = [('oversampling', oversampler),
            ('undersampling', undersampler),
            ('class_weights', 'passthrough'),
            ('SMOTE', smotesampler)
           ]
# storage container for results
results = []

# go through every sampler
for name, sampler in samplers:
    #sampling
    imb_pipe = Pipeline([#('transformer',col_transformer),
                         ('sampler', sampler),
                         ('estimator', tree_clf)
                        ])
    
    #gridsearch and CV
    grid = GridSearchCV(estimator=imb_pipe, 
                        param_grid=search_space,
                        n_jobs=5,
                        cv=5,
                        scoring='f1')
    
    grid.fit(features_train, target_train)
    
    #evaluation
    model = grid.best_estimator_.named_steps['estimator']
    target_pred = model.predict(features_test)
    recall = recall_score(target_test,target_pred )
    precision = precision_score(target_test,target_pred )
    accuracy = accuracy_score(target_test,target_pred)
    #save
    scores = {'name': name,
              'precision': precision,
              'recall': recall,
              'F1':grid.best_score_ ,
              'Accuracy': accuracy
             }
    results.append(scores)
    
#show results
pd.DataFrame(results)

# Oversampler function

print('All the scores are similar, but the Oversampling alternative has better precision/F1 ranking')
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

# Basic MOdel

data2 = df_train.copy()
# Create a figure with a 2x2 subplot layout
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: VehicleAge vs. IsBadBuy
data2.groupby('VehicleAge').agg([np.mean, np.size])['IsBadBuy'].query('size > 250')['mean'].plot(ax=axes[0, 0], title="VehicleAge Vs IsBadBuy")

# Plot 2: VehOdo vs. IsBadBuy
data2_sorted = data2.sort_values(by='VehOdo')
sns.histplot(data=data2_sorted, x='VehOdo', hue='IsBadBuy', bins=20, element='step', common_norm=False, ax=axes[0, 1])
axes[0, 1].set_xlabel("VehOdo")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Count of Cells vs. VehOdo by IsBadBuy")

# Plot 3: VehYear vs. IsBadBuy
data2.groupby("VehYear").mean()["IsBadBuy"].plot.bar(ax=axes[1, 0], title="VehYear Vs IsBadBuy")

# Plot 4: WarrantyCost vs. IsBadBuy
data2_sorted = data2.sort_values(by='WarrantyCost')
sns.histplot(data=data2_sorted, x='WarrantyCost', hue='IsBadBuy', bins=20, element='step', common_norm=False, ax=axes[1, 1])
axes[1, 1].set_xlabel("WarrantyCost")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("Count of Cells vs. WarrantyCost by IsBadBuy")

# Adjust subplot layout
plt.tight_layout()

# Show the plot
plt.show()

# define num_cols and cat_cols

# Subset the features_train DataFrame
selected_features = ['VehicleAge', 'VehOdo', 'WarrantyCost']
subset_features_train = features_train[selected_features]
subset_features_test= features_test[selected_features]
# Apply resampling to the subset
subset_features_train_resampled, target_train_resampled = oversample_data(subset_features_train, target_train)

# instantiate model
# Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression(random_state=42)


# Create a pipeline with feature scaling and Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('classifier', LogisticRegression(random_state=42))
])

# fit pipeline on cleaned (and filtered) training set

pipeline.fit(subset_features_train_resampled, target_train_resampled)

# predict and evaluate on test set
# Predict using the pipeline on the validation set
pipeline_predictions = pipeline.predict(subset_features_test)
# Evaluate the model's performance
precision = precision_score(target_test, pipeline_predictions)
recall = recall_score(target_test, pipeline_predictions)
f1 = f1_score(target_test, pipeline_predictions)
accuracy = accuracy_score(target_test, pipeline_predictions)

# Display the evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)
print('------------------------------------')
print('I choose this features just based on intuition. It is clear, that we need to add more features into our analysis in order to obtain better scores')

# Interpretation of Model Performance
print("Interpretation of Model Performance:")
print()

# Precision and Recall Analysis
print("Precision and Recall Analysis:")
print("High recall but low precision: meaning it tends to identify many cases as positive but makes a lot of false positive predictions.")
print()

# F1-score and Accuracy Analysis
print("F1-score and Accuracy Analysis:")
print("The low F1-score and accuracy indicate that the model's performance is not satisfactory,")
print("and it might need further improvement or a different approach, such as ")
print("feature engineering, or trying different algorithms.")
print()

# Features Engineering

## Delete columns which are redundant or contain sensible Client information
Col_to_delete={'Column': ['AUCGUART', 'PRIMEUNIT',  'BYRNO', 'VNZIP1', 'VehYear', 'WheelTypeID','VNST', 'Nationality'], 
                'Reason_to_delete': ['More than 95% of data is missing' , 'More than 95% of data is missing',
                'Ethical',  'Ethical', 'Redundant with VehicleAge', 'Redundant with WheelType', 'Ethical', 'Redundant with Make' ]}
to_delete = pd.DataFrame(Col_to_delete)
to_delete.head(8)

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

data_train, data_test =engineer_features(data_train, data_test)

# Test shape depending on encode used

data_train.shape[1]
data_test.shape[1]

# # Clean outliers from train set
data_train, ziel_train = find_outliers(data_train, ziel_train)

train_columns = set(data_train.columns)
test_columns = set(data_test.columns)

columns_only_in_train = train_columns - test_columns

print("Columns present only in features_train:")
print(columns_only_in_train)
print('Sanity check: both sets are ready to modeling!')

print("Length of data_train:", len(data_train))
print("Length of ziel_train:", len(ziel_train))
print('-------------------------------')
# Sampling
data_train_resampled, target_train_resampled = oversample_data(data_train, ziel_train)

print("Length of data_train after sampling:", len(data_train_resampled))
print("Length of ziel_train after sampling:", len(target_train_resampled))
print('Total positive IsBadBuy:', target_train_resampled.sum())

#Split
cat_cols, con_cols = split_categ_contin_cols(data_train_resampled)

print("Length of data_train:", len(data_train_resampled))
print("Length of target_train_resampled:", len(target_train_resampled))
print("We deleted 903/51800 positive cases. 1.74%")
target_train_resampled.sum()


scaler = StandardScaler()

#PCA

# Calculate the correlation matrix
correlation_matrix = data_train_resampled.corr()

# Create a heatmap to visualize the correlations
plt.figure(figsize=(16, 10))
mask_matrix=np.triu(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds,mask=mask_matrix);
#plt.annotate("As we can see, all the MMR features are highly correlated", xy=(0.5, -1), fontsize=12, color="black", ha="center")
plt.title("Correlation Matrix")
plt.xticks([])  # Hide x-axis labels (ticks)
plt.annotate("As we can see, all the MMR & VehBCost features are highly correlated", xy=(0.5, -0.1),
             xycoords=("axes fraction", "axes fraction"), fontsize=12, color="black", ha="center")
plt.show()


# "MMRAcquisitionRetailAveragePrice", "MMRAcquisitonRetailCleanPrice", "MMRCurrentRetailAveragePrice" , MMRCurrentRetailCleanPrice
# Define the selected columns for PCA
selected_columns = ['MMRAcquisitionAuctionAveragePrice', "MMRAcquisitionAuctionCleanPrice", "MMRAcquisitionRetailAveragePrice",
                    "MMRAcquisitonRetailCleanPrice", "MMRCurrentAuctionAveragePrice",
                    "MMRCurrentAuctionCleanPrice", "MMRCurrentRetailAveragePrice",
                    "MMRCurrentRetailCleanPrice", "VehBCost"]

# Extract the selected columns from your dataset
data_for_pca = data_train_resampled[selected_columns]

# Standardize the data

data_for_pca_standardized = scaler.fit_transform(data_for_pca)

# Perform PCA transformation on the standardized data
num_components = 2  # You can adjust this as needed
pca = PCA(n_components=num_components)
data_pca = pca.fit_transform(data_for_pca_standardized)

# Get the loadings of the first component
loadings = pca.components_[0]

# Create a DataFrame to associate the loadings with feature names
loadings_df = pd.DataFrame({'Feature': data_for_pca.columns, 'Loading': loadings})

# Calculate the Absolute Loading in percentage
loadings_df['Absolute_Loading(%)'] = (loadings_df['Loading'].abs() * 100).round(2)

# Sort the DataFrame by absolute loading values to see which features contribute the most
loadings_df = loadings_df.sort_values(by='Absolute_Loading(%)', ascending=False)

# Display the DataFrame
loadings_df.head(9)

# Initialize and fit the PCA model
pca2 = PCA()
pca2.fit(data_for_pca)  # Fit PCA on the original correlated features

# Get the explained variance ratio
explained_variance = pca2.explained_variance_ratio_

# Plot the explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio for Principal Components')
plt.grid()
plt.show()

## From the previous plot is clear that we need just 2 principal componets to capture the variance of this 9 columns

def apply_pca(df_train, df_test, num_components):
    """Perform PCA on correlated Features. Fit & transform for train set, just transform for test set

    Args:
        df_train (pd.DataFrame): Training Dataframe.
        df_test (pd.DataFrame): Test Dataframe.
        num_components (int): Number of PCA components.

    Returns:
        data_train_pca (pd.DataFrame): Training Dataframe with PCA features.
        data_test_pca (pd.DataFrame): Test Dataframe with PCA features.

    """
    # Correlated columns
    selected_columns = ['MMRAcquisitionAuctionAveragePrice',
                        'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice',
                        'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
                        'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
                        'MMRCurrentRetailCleanPrice', 'VehBCost']

    # Initialize PCA transformer and fit on the training data
    pca_transformer = PCA(n_components=num_components)
    data_train_pca = pd.DataFrame(pca_transformer.fit_transform(df_train.loc[:, selected_columns]),
                                  columns=["price1", "price2"],
                                  index=df_train.index)

    # Transform both training and test data
    df_train = df_train.drop(selected_columns, axis="columns")
    df_train = pd.concat([df_train, data_train_pca], axis=1)

    data_test_pca = pd.DataFrame(pca_transformer.transform(df_test.loc[:, selected_columns]),
                                 columns=["price1", "price2"],
                                 index=df_test.index)
    df_test = df_test.drop(selected_columns, axis="columns")
    df_test = pd.concat([df_test, data_test_pca], axis=1)

    return df_train, df_test

data_train_resampled_pca, data_test_pca = apply_pca(data_train_resampled,data_test, 2)
data_train_resampled_pca.shape

correlation_matrix = data_train_resampled_pca.corr()

# Create a heatmap to visualize the correlations
plt.figure(figsize=(16, 10))
mask_matrix=np.triu(correlation_matrix)

sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds,mask=mask_matrix);
#plt.annotate("As we can see, all the MMR features are highly correlated", xy=(0.5, -1), fontsize=12, color="black", ha="center")
plt.title("Correlation Matrix after PCA")
plt.show()

# Modules



#warnings.filterwarnings("ignore", category=FitFailedWarning)


features_train_index = data_train_resampled.index
target_train_index = target_train_resampled.index

# Check if indices coincide
indices_match = features_train_index.equals(target_train_index)
print(f"Do the indices match? {indices_match}")

features_test_index = data_test.index
target_test_index = target_test.index

# Check if indices coincide
indices_match = features_test_index.equals(target_test_index)
print(f"Do the indices match? {indices_match}")

# build unoptimized model
# Initiliaze the model

model_rf = RandomForestClassifier(random_state=42)
#fit
data_train_resampled_std = scaler.fit_transform(data_train_resampled_pca)
data_test_std = scaler.transform(data_test_pca)                        ### Corrected!
model_rf.fit(data_train_resampled_std,target_train_resampled)


#predict
target_test_pred = model_rf.predict(data_test_std)

#evaluate
accuracy = accuracy_score(target_test,target_test_pred)
recall = recall_score(target_test,target_test_pred)
precision = precision_score(target_test,target_test_pred)
f1 = f1_score(target_test,target_test_pred)
#Show
scores_df= pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1'],
    'Score': [accuracy, recall, precision, f1]
})
scores_df.head()

print('number of features:', data_train_resampled_std.shape)

# Get feature importances from the trained model
feature_importances = model_rf.feature_importances_

# Create a DataFrame to store the feature names and their importances
importance_df = pd.DataFrame(
    {'Feature': data_train_resampled_pca.columns, 'Importance': feature_importances}
    )

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

# Sort features by importance
importances = model_rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
# Select the top N most important features
selected_features = sorted_idx[:10]

# Modify your datasets to include only the selected features
data_train_selected = data_train_resampled_std[:, selected_features]
data_test_selected = data_test_std[:, selected_features]

# Train a new Random Forest model using the reduced dataset
model_rf_selected = RandomForestClassifier(random_state=42)
model_rf_selected.fit(data_train_selected, target_train_resampled)

# Predict using the model with selected features
target_test_pred_selected = model_rf_selected.predict(data_test_selected)

# Evaluate the model with selected features
accuracy_selected = accuracy_score(target_test, target_test_pred_selected)
recall_selected = recall_score(target_test, target_test_pred_selected)
precision_selected = precision_score(target_test, target_test_pred_selected)
f1_selected = f1_score(target_test, target_test_pred_selected)

# Show the evaluation scores
scores_df_selected = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1'],
    'Score': [accuracy_selected, recall_selected, precision_selected, f1_selected]
})
print("Evaluation Scores with Selected Features:")
print(scores_df_selected)


# tune hyperparameters
search_space_rf  = {"max_depth": [20],
                    "max_features":[2],
                        "min_samples_split": [50],
             'n_estimators': [150]}

grid_search = GridSearchCV(estimator=model_rf,
                           param_grid=search_space_rf,
                           cv=5,  # Number of cross-validation folds
                           n_jobs=5,  # Do not use all available CPU cores
                           scoring='f1'#,  # Use F1 score as the evaluation metric
                           #verbose=0  # Increase verbosity for progress
                           )

grid_search.fit(data_train_resampled_std, target_train_resampled)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Hyperparameters:")
print(best_params)

# Now you can use the best_model for predictions
target_test_pred = best_model.predict(data_test_std)

# Evaluate the best model
accuracy = accuracy_score(target_test, target_test_pred)
recall = recall_score(target_test, target_test_pred)
precision = precision_score(target_test, target_test_pred)
f1 = f1_score(target_test, target_test_pred)

# Show the scores
scores_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1'],
    'Score': [accuracy, recall, precision, f1]
})
scores_df.head()

print('After a lot of trial & error I found that the best hyper parameter are')

best_hp_dict={'max_depth': 20, 'max_features': 2, 'min_samples_split': 50, 'n_estimators': 150}
# Convert the dictionary to a DataFrame
params_df = pd.DataFrame.from_dict(best_hp_dict, orient='index', columns=['Value'])

# Reset the index to have parameter names as a regular column
params_df.reset_index(inplace=True)
params_df = params_df.rename(columns={'index': 'Parameter'})
params_df

# Ignore FutureWarnings and FitFailedWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
# select model
print('Logistic Regression')
model_logreg = LogisticRegression(random_state=42)

# Create Pipeline
pipeline_logreg = Pipeline([("scaler", StandardScaler()),
                           ("model_logreg", model_logreg)])

# Fit pipeline on cleaned training set, the pipeline will stadardize!
pipeline_logreg.fit(data_train_resampled_pca,target_train_resampled)

# Define hyperparameters range
# parameter grid, # l1 lasso l2 ridge
search_space_logreg = {
            'model_logreg__penalty' : ['l1','l2'], 
            'model_logreg__C'       : np.geomspace(0.001,1000,10),
            'model_logreg__solver'  : ['saga'],
              }

# Perform grid search
grid_logreg = GridSearchCV(estimator=pipeline_logreg,
                            param_grid=search_space_logreg,
                            scoring='f1',
                            cv=3, n_jobs=4)
grid_logreg.fit(data_train_resampled_pca, target_train_resampled)

# Model scores
print(f'Training F1-scores for {type(model_logreg)}:\n')
print(f'Mean (F1): {grid_logreg.best_score_}')
print(grid_logreg.best_estimator_)
print('---')
print("tuned hpyerparameters :(best parameters) ",grid_logreg.best_params_)

# Calculate and show other scores: Accuracy, Recall, and Precision
predicted_labels = grid_logreg.predict(data_test_pca)  ### CORRECTED!!

accuracy = accuracy_score(target_test, predicted_labels)
recall = recall_score(target_test, predicted_labels)
precision = precision_score(target_test, predicted_labels)

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')

report = classification_report(target_test, predicted_labels, target_names=['Montagsauto 0', 'Montagsauto 1'])

print('Classification Report:\n', report)


# Generate the confusion matrix
cm = confusion_matrix(target_test, predicted_labels)
x_labels = ['IsBadBuy = 1', 'IsBadBuy = 0']
y_labels = ['IsBadBuy = 1', 'IsBadBuy = 0']
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=x_labels, yticklabels=y_labels, cbar = True)
plt.xlabel('Predicted')
plt.title('Confusion Matrix LogReg')
plt.ylabel('True')
plt.show()

# KNN
# Initiate model
model_knn = KNeighborsClassifier()

# Make Pipeline
pipeline_knn= Pipeline([("scaler",StandardScaler()),
                           ("model_knn", model_knn)])

# Define hyperparameters range
k = [2,5,10,20,30]
search_space_knn = {'model_knn__n_neighbors': k,  
                    'model_knn__weights': ['uniform', 'distance'],
                     'model_knn__metric': ['euclidean','manhattan']}

# Perform grid search
grid_knn = GridSearchCV(estimator=pipeline_knn,
                        param_grid=search_space_knn,
                        scoring='f1',
                        cv=3,n_jobs = 4)
grid_knn.fit(data_train_resampled_pca, target_train_resampled)
# Calculate and show other scores for the KNeighborsClassifier
predicted_labels_knn = grid_knn.predict(data_test_pca)

accuracy_knn = accuracy_score(target_test, predicted_labels_knn)
recall_knn = recall_score(target_test, predicted_labels_knn)
precision_knn = precision_score(target_test, predicted_labels_knn)

# Model scores
print(f'Training F1-scores for {type(model_knn)}:\n')
print(f'Mean (F1): {grid_knn.best_score_}')
print(f'Accuracy (KNeighbors): {accuracy_knn}')
print(f'Recall (KNeighbors): {recall_knn}')
print(f'Precision (KNeighbors): {precision_knn}')

print(grid_knn.best_estimator_)
print('---')
print("tuned hpyerparameters :(best parameters) ",grid_knn.best_params_)
report_knn = classification_report(target_test, predicted_labels_knn, target_names=['Montagsauto 0', 'Montagsauto 1'])

print('Classification Report (KNeighbors):\n', report_knn)



# Generate the confusion matrix
cm = confusion_matrix(target_test, predicted_labels_knn)
x_labels = ['IsBadBuy = 1', 'IsBadBuy = 0']
y_labels = ['IsBadBuy = 1', 'IsBadBuy = 0']
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=x_labels, yticklabels=y_labels, cbar = True)
plt.xlabel('Predicted')
plt.title('Confusion Matrix KKN')
plt.ylabel('True')
plt.show()

print('Random Forest')

# Initiate model, number of estimators was already optimized

# Initialize the model with the best hyperparameters
best_rf_model = RandomForestClassifier(max_depth= 20, max_features= 2,
                                       min_samples_split= 50, n_estimators= 150, random_state=42)
# Create a pipeline with standardization
pipeline_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("model_rf", best_rf_model)
])

# Fit the pipeline on the standardized training data
pipeline_rf.fit(data_train_resampled_pca, target_train_resampled)

# Make predictions on the training data
predicted_labels_rf = pipeline_rf.predict(data_test_pca)

# Generate a classification report
report = classification_report(target_test, predicted_labels_rf, target_names=['Montagsauto 0', 'Montagsauto 1'])

# Calculate accuracy, recall, and precision
accuracy = accuracy_score(target_test, predicted_labels_rf)
recall = recall_score(target_test, predicted_labels_rf)
precision = precision_score(target_test, predicted_labels_rf)

print(f'Training F1-scores for {type(best_rf_model)}:\n')
print(f'Mean (F1): {grid_search.best_score_}')
print(f'Accuracy (Random Forest): {accuracy}')
print(f'Recall (Random Forest): {recall}')
print(f'Precision (Random Forest): {precision}')
print(pipeline_rf)
print('---')
print("Tuned hyperparameters (best parameters):", grid_search.best_params_)
print('Classification Report (Random Forest):\n', report)


# Generate the confusion matrix
cm = confusion_matrix(target_test, predicted_labels_rf)
x_labels = ['IsBadBuy = 1', 'IsBadBuy = 0']
y_labels = ['IsBadBuy = 1', 'IsBadBuy = 0']
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=x_labels, yticklabels=y_labels, cbar = True)
plt.xlabel('Predicted')
plt.title('Confusion Matrix RF')
plt.ylabel('True')
plt.show()

# Final Pipeline



def predict_rf(csv_file_path):
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
N = 15
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


# features_aim = pd.read_csv('features_aim.csv')
# # Show data
# features_aim.head()


# # Prepare the data to make predictions
# df_aim = features_aim.copy()
# # Remove noise columns
# delete_cols = ['Model','Trim','SubModel','PurchDate', 'AUCGUART', 'PRIMEUNIT', 'BYRNO', 'VNZIP1', 'VehYear', 'WheelTypeID', 'VNST', 'Nationality']
# df_aim=df_aim.drop(delete_cols, axis=1)
# # Features Engineering & data cleaning
# df_aim.shape[1]
# # PCA (replace Acquisition costs)
# #df_aim = apply_pca(df_aim, 2)
# #df_aim.shape

# print('look for missing values:')
# print('____________________________')
# findNullVals(df_aim)

# # impute the missing values
# df_aim = clean_data(df_aim)

# aux_encoding=clean_data(aux_encoding)

# aux_encoding.shape[1]
# # encode categorical features. Remember that the encoder function use a training set to fit & transform and a test set (aka aim set) to transform
# # 1. Split the columns
# cat_cols, _ = split_categ_contin_cols(df_aim)
# # 2.Encode
# _, df_aim = encode_categorical_columns(aux_encoding[cat_cols], df_aim, cat_cols)


# predictions_aim = pipeline_rf_fe.predict(df_aim)

# print('Wir sollten', predictions_aim.sum(), 'Käufe nicht machen')

# # Create a DataFrame from the predictions_aim array
# predictions_aim_df = pd.DataFrame({'IsBadBuy': predictions_aim})
# # Save the DataFrame to a CSV file
# predictions_aim_df.to_csv('predictions_aim.csv', index=False)