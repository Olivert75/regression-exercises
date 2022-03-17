import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer



def scaled_mimmax2 (columns, train_df , validate_df, test_df):
    '''
    Take in a train_df and a list of columns that you want to scale (MinMaxScaler). 
    Fit a scaler only in train and tramnsform in train, validate and test.
    add new scaled columns to train, validate and test
    Example
    p.scaled_mimmax(columns, train , validate , test)
    
    '''
    for col in columns: 
        # create our scaler
        scaler = MinMaxScaler()
        # fit our scaler
        scaler.fit(train_df[[col]])
        # use it
        name = col + '_minmax'
        train_df[name] = scaler.transform(train_df[[col]])
        validate_df[name]= scaler.transform(validate_df[[col]])
        test_df[name]= scaler.transform(test_df[[col]])
    return

def scaled_mimmax (columns, train_df , validate_df, test_df):
    '''
    Take in a 3 df and a list of columns that you want to scale (MinMaxScaler). 
    Fit a scaler only in train and tramnsform in train, validate and test.
    returns a new df with the scaled columns.
    Example
    p.scaled_mimmax(columns, train , validate , test)
    
    '''

    # create our scaler
    scaler = MinMaxScaler()
    # fit our scaler
    scaler.fit(train_df[columns])
    # get our scaled arrays
    train_scaled = scaler.transform(train_df[columns])
    validate_scaled= scaler.transform(validate_df[columns])
    test_scaled= scaler.transform(test_df[columns])

    # convert arrays to dataframes
    train_scaled_df = pd.DataFrame(train_scaled, columns=columns).set_index([train_df.index.values])
    validate_scaled_df = pd.DataFrame(validate_scaled, columns=columns).set_index([validate_df.index.values])
    test_scaled_df = pd.DataFrame(test_scaled, columns=columns).set_index([test_df.index.values])

    return train_scaled_df, validate_scaled_df, test_scaled_df

def plot_scaled_mimmax (columns, train_df ):
    '''
    Take in a train_df and a list of columns that you want to scale (MinMaxScaler). 
    Fit a scaler only in train 
    return plots of original and scaled columns and add new scaled columns to train
    Example
    p.plot_scaled_mimmax(columns, train)
    '''
    for col in columns: 
        # create our scaler
        scaler = MinMaxScaler()
        # fit our scaler
        scaler.fit(train_df[[col]])
        # use it
        name = col + '_minmax'
        t[name] = scaler.transform(train_df[[col]])
        plt.figure(figsize=(13, 6))
        plt.subplot(121)
        plt.hist(t[col], ec='black')
        plt.title('Original')
        plt.xlabel(col)
        plt.ylabel("counts")
        plt.subplot(122)
        plt.hist(t[name],  ec='black')
        plt.title('Scaled')
        plt.xlabel(name)
        plt.ylabel("counts")
    return