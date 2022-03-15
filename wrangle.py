import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import acquire


from sklearn.model_selection import train_test_split


# ********************************************  TELCO   ***********************************************************


def clean_telco(df):
    ''''
    This function will get customer_id, monthly_charges, tenure, and total_charges 
    from the previously acquired telco df, for all customers with a 2-year contract.
    drop any duplicate observations, 
    conver total_charges to a float type.
    return cleaned telco DataFrame
    '''
    #getting only the customers who have 2 year contract using the condition df.contract_type_id == 3
    telco_df = df[['customer_id', 'monthly_charges', 'tenure', 'total_charges']][df.contract_type_id == 3]
    #drop duplicates
    telco_df = telco_df.drop_duplicates()
    # add a '0' only to the columns that have " "
    telco_df[telco_df['total_charges']== ' '] = telco_df[telco_df['total_charges']== ' '].replace(' ','0')
    # convert total_charges to float
    telco_df['total_charges']= telco_df['total_charges'].astype('float')
        
    return telco_df



def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')                                  
    return train, validate, test

def split_Xy (train, validate, test, target):
    '''
    This function takes in three dataframe (train, validate, test) and a target  and splits each of the 3 samples
    into a dataframe with independent variables and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    Example:
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_Xy (train, validate, test, 'Fertility' )
    '''
    
    #split train
    X_train = train.drop(columns= [target])
    y_train= train[target]
    #split validate
    X_validate = validate.drop(columns= [target])
    y_validate= validate[target]
    #split validate
    X_test = test.drop(columns= [target])
    y_test= test[target]

    print(f'X_train -> {X_train.shape}               y_train->{y_train.shape}')
    print(f'X_validate -> {X_validate.shape}         y_validate->{y_validate.shape} ')        
    print(f'X_test -> {X_test.shape}                  y_test>{y_test.shape}') 
    return  X_train, y_train, X_validate, y_validate, X_test, y_test

def wrangle_telco():
    ''''
    This function will acquire telco db using get_telco function. then it will use another
    function named  clean_telco that create a new df only with  customer_id, monthly_charges, tenure, and total_charges 
    from the previously acquired telco df, this new df will contain only customers with a 2-year contract.
    drop any duplicate observations, 
    conver total_charges to a float type.
    return cleaned telco DataFrame
    '''
    df = acquire.get_telco()
    telco_df = clean_telco(df)
    return telco_df



# ******************************************** ZILLOW ********************************************

def clean_zillow (df):
    '''
    Takes in a df and drops duplicates,  nulls, all houses that do not have bedrooms and bathrooms,
    houses that calculatedfinishedsquarefeet < 800, and bedroomcnt, yearbuilt, fips are changed to
    int.
    Return a clean df
    '''
    
    # drop duplicates
    df = df.drop_duplicates()
    #drop nulls
    df = df.dropna(how='any',axis=0)

    #drop all houses with bath = 0 and bedromms = 0
    #get the index to drop the rows
    ind = list(df[(df.bedroomcnt == 0) & (df.bathroomcnt == 0)].index)
    #drop
    df.drop(ind, axis=0, inplace= True)


    #drop all houses calculatedfinisheedsqf <800
    #get the index to drop
    lis =list(df[df['calculatedfinishedsquarefeet'] < 800].index)
    #drop the rows
    df.drop(lis, axis=0, inplace = True)

    #bedrooms, yearbuilt and fips can be converted to int
    df[['bedroomcnt', 'yearbuilt', 'fips']] = df[['bedroomcnt', 'yearbuilt', 'fips']].astype(int)
    return df



def wrangle_zillow():
    ''''
    This function will acquire zillow db using get_new_zillow function. then it will use another
    function named  clean_zillwo that drops duplicates,  nulls, all houses that do not have bedrooms and bathrooms,
    houses that calculatedfinishedsquarefeet < 800.
     bedroomcnt, yearbuilt, fips are changed to int.
    return cleaned zillow DataFrame
    '''
    df = acquire.get_new_zillow()
    zillow_df = clean_zillow(df)
    return zillow_df




# Function for acquiring and prepping my student_grades df.

def wrangle_grades():
    '''
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''
    # Acquire data from csv file.
    grades = pd.read_csv('student_grades.csv')
    
    # Replace white space values with NaN values.
    grades = grades.replace(r'^\s*$', np.nan, regex=True)
    
    # Drop all rows with NaN values.
    df = grades.dropna()
    
    # Convert all columns to int64 data types.
    df = df.astype('int')
    
    return df
