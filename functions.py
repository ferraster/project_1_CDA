from __future__ import division
import numpy as np
import pandas as pd

from sklearn.preprocessing import  OneHotEncoder

import missingno as msno
from scipy.stats import chi2 
from scipy.stats import chisquare
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#-----------------------------------[MAIN PREPROCESSING FUNCTION]-----------------------------------#


def preprocessing(df):
    df = handle_missing_values(df)
    df = handle_categorical_values(df)
    return df 


#-----------------------------------[REESCALING]-----------------------------------#

def center(data):   
    mu = np.mean(data,axis=0)
    data = data - mu
    return data, mu

def normalize(data): 
    distances = np.linalg.norm(data,axis=0,ord=2)
    distances[distances==0] = 1
    return data/distances,distances




#-----------------------------------[ HANDLE MISSING VALUES]-----------------------------------#

def handle_missing_values(df):
    is_MCAR = little_mcar_test(df)
    if(is_MCAR):
        return handle_missing_values_MCAR(df)
    else:
        return handle_missing_values_MAR(df)



def little_mcar_test(data, alpha=0.05):
    """
    Performs Little's MCAR (Missing Completely At Random) test on a dataset with missing values.
    """
    data = pd.DataFrame(data)
    data.columns = ['x' + str(i) for i in range(data.shape[1])]
    data['missing'] = np.sum(data.isnull(), axis=1)
    n = data.shape[0]
    k = data.shape[1] - 1
    df = k * (k - 1) / 2
    chi2_crit = chi2.ppf(1 - alpha, df)
    if n == k:
        k = k +1
    if k == 1:
     k = k + 1
    chi2_val = ((n - 1 - (k - 1) / 2) ** 2) / (k - 1) / ((n - k) * np.mean(data['missing']))
    p_val = 1 - chi2.cdf(chi2_val, df)
    if chi2_val > chi2_crit:
        #print('Reject null hypothesis: Data is not MCAR (p-value={:.4f}, chi-square={:.4f})'.format(p_val, chi2_val))
        return False
    else:
        #print('Do not reject null hypothesis: Data is MCAR (p-value={:.4f}, chi-square={:.4f})'.format(p_val, chi2_val))
        return True
 

def handle_missing_values_MCAR(data):
    df = data.copy()
    cols_with_missing_values = df.columns[df.isna().any()].tolist()

    for var in cols_with_missing_values:

        # extract a random sample
        random_sample_df = df[var].dropna().sample(df[var].isnull().sum(),
                                                    random_state=0, replace = True)
        # re-index the randomly extracted sample
        random_sample_df.index = df[df[var].isnull()].index

        # replace the NA
        df.loc[df[var].isnull(), var] = random_sample_df
    
    return df

def handle_missing_values_MAR(data, num_iterations=10, random_state=None):
    """
    Perform multiple imputation on missing values using scikit-learn's IterativeImputer.

    Parameters:
    - data: Input DataFrame with missing values
    - num_iterations: Number of imputation iterations
    - random_state: Seed for reproducibility

    Returns:
    - List of imputed DataFrames
    """
    imputed_dfs = []

    for _ in range(num_iterations):
        # Create a copy of the original DataFrame to avoid modifying the original data
        imputed_data = data.copy()

        # Initialize IterativeImputer
        imputer = IterativeImputer(max_iter=10, random_state=random_state)

        # Perform imputation
        imputed_data.iloc[:, :] = imputer.fit_transform(imputed_data)

        # Append the imputed DataFrame to the list
        imputed_dfs.append(imputed_data)

    mean_df = pd.concat(imputed_dfs).groupby(level=0).mean()

    return mean_df



#-----------------------------------[ HANDLE CATEGORICAL VALUES]-----------------------------------#

def handle_categorical_values(df):
    categories = df.keys()[-5:].to_list()

    # Reshape the data to a 2D array
    data = np.array(categories).reshape(-1, 1)

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder()

    # Fit and transform the data
    one_hot_encoded = pd.get_dummies(df[categories])
    one_hot_encoded = one_hot_encoded.astype(int)


    df = df.drop(categories,axis=1)
    df = pd.concat([df,one_hot_encoded],axis= 1)

    return df




#-----------------------------------[RESIDUALS]-----------------------------------#

def MSE(y,estimated_y):
    MSE = 0
    for i in range(len(y)):
        MSE = MSE + (y[i]-estimated_y[i])**2
    return MSE/len(y)

def TSS(y): 
    TSS = 0
    y_mean = np.mean(y)
    for i in range(len(y)):
        TSS = TSS + (y_mean - y[i])**2
    return TSS 

def RSS(y,estimated_y):
    RSS = 0
    for i in range(len(y)):
        RSS = RSS + (y[i]-estimated_y[i])**2
    return RSS

def R2(y,estimated_y):
    return 1 - RSS(y,estimated_y)/TSS(y)


