'''
ETL script

This script takes the file paths of the messages and categories datasets,
merges and cleans them, then stores the clean data into a SQLite database in the
specified database file path.
'''

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This function loads messages and categories datasets from csv files
    and merge them as a single dataset.

    inputs:
    messages_filepath - path to real disaster messages csv file
    categories_filepath - path to categories csv file

    outputs:
    df - dataframe,  messages and categories merged
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id',how='inner')

    return df


def clean_data(df):
    '''
    This function cleans and re-organizes the dataframe: expands the category
    column into 36 individual category columns and names them accordingly;
    convert values in all columns into numeric values (0 or 1); replace old
    category column with the new columns.

    inputs:
    df - original dataframe

    outputs:
    df - cleaned dataframe
    '''
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x[:-2]))
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])

    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1,keys=df.index)
    dup_idx = df[df.duplicated()==True].index
    df = df.drop(dup_idx,axis=0)
    return df




def save_data(df, database_filename):
    '''
    This function stores the clean data into a SQLite database.

    inputs:
    df - cleaned data
    database_filename - name of the db file

    outputs:
    db file saved in SQLite database; nothing to return
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False)


def main():
    '''
    This main function loads the datasets, cleans the data and then save the
    cleaned data into database.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
