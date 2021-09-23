# acquire.py for zillow data 

# imports

import numpy as np
import pandas as pd
import os

from env import host, user, password

# sets up a secure connection to the Codeup db using my login infor
def get_db_url(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# assigns the zillow url to the variable name 'url' so it can be used in additional functions
url = get_db_url('zillow')

# creating functions to use in my notebook
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my env file to create a connection url to access
    the Codeup database. '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function get the zillow data from the codeup database
    '''

    sql_query = '''select prop.parcelid
        , pred.logerror
        , bathroomcnt
        , bedroomcnt
        , calculatedfinishedsquarefeet
        , fips
        , latitude
        , longitude
        , lotsizesquarefeet
        , propertylandusetypeid
        , regionidcity
        , regionidcounty
        , regionidzip
        , yearbuilt
        , structuretaxvaluedollarcnt
        , taxvaluedollarcnt
        , landtaxvaluedollarcnt
        , taxamount
    from properties_2017 prop
    inner join predictions_2017 pred on prop.parcelid = pred.parcelid
    where propertylandusetypeid = 261;'''
    
    return pd.read_sql(sql_query, get_connection('zillow'))


def get_zillow_data():
    '''
    Reading Zillow data from codeup database and creates a csv file into a dataframe
    '''
    if os.path.isfile('zillow_df.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_df.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
     
        # Cache data
        df.to_csv('zillow_df.csv')
        
    return df

