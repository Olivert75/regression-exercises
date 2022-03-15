import pandas as pd
import numpy as np
import os
from env import host, user_name, password
# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")


# *************************************  connection url **********************************************

# Create helper function to get the necessary connection url.
def get_connection(db_name):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user_name}:{password}@{host}/{db_name}'

# ************************************ generic acquire function***************************************************************


def get_data_from_sql(db_name, query):
    """
    This function takes in a string for the name of the database I want to connect to
    and a query to obtain my data from the Codeup server and return a DataFrame.
    db_name : df name in a string type
    query: aalready created query that was named as query 
    Example:
    query = '''
    SELECT * 
    FROM table_name;
    '''
    df = get_data_from_sql('telco_churn', query)
    """
    df = pd.read_sql(query, get_connection(db_name))
    return df


# *******************************************telco_db********************************************************



#acquire data for the first time
def get_new_telco_churn():
    '''
    This function reads in the telco_churn data from the Codeup db
    and returns a pandas DataFrame with all columns and it was joined with other tables.
    '''
    sql_query = '''
    SELECT * FROM customers
    JOIN contract_types USING (contract_type_id)
    JOIN internet_service_types USING (internet_service_type_id)
    JOIN payment_types USING (payment_type_id)
    '''
    return pd.read_sql(sql_query, get_connection('telco_churn'))

#acquire data main function 
def get_telco():
    '''
    This function reads in telco_churn data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('telco_churn.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('telco_churn.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = get_new_telco_churn()
        
        # Write DataFrame to a csv file.
        df.to_csv('telco_churn.csv')
        
    return df


# ***************************************************************************************************
#                                     ZILLOW DB
# ***************************************************************************************************

#acquire data for the first time
def get_new_zillow():
    '''
    This function reads in the zillow data from the Codeup db
    and returns a pandas DataFrame with columns :
     bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips 
    '''
    sql_query = '''
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261
    '''
    return pd.read_sql(sql_query, get_connection('zillow'))

#acquire data main function 
def get_zillow():
    '''
    This function reads in telco_churn data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = get_new_zillow()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    return df