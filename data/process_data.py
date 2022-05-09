import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    '''
    Load in the messages and categories datasets and merge them into one dataframe.

    Args:
        messages_filepath (str): path to the messages csv file
        categories_filepath (str): path to the categories csv file

    Returns:
        (Pandas dataframe) merged data
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='left')
    return df


def clean_data(df):

    '''
    Clean the merged dataset.

    Args:
        df (Pandas dataframe): merged data

    Returns:
        (Pandas dataframe) clean data
    '''

    # Expand categories into separate columns
    categories = df.categories.str.split(';', expand=True)
    colnames = categories.iloc[0].str.split('-', expand=True)[0].tolist()
    categories.columns = colnames
    
    # Clean values and convert to numeric if the category is not constant
    for column in categories.columns:
        if categories[column].nunique() > 1:
            categories[column] = categories[column].apply(lambda r: r[-1]).astype(int)
        else:
            categories.drop(column, axis=1, inplace=True)
        
    # Combine original df and expanded categories
    return pd.concat([df.drop('categories', axis=1), categories], axis=1).drop_duplicates()
    

def save_data(df, database_filepath):

    '''
    Save the clean data in a SQLite database.

    Args:
        df (Pandas dataframe): clean data
        database_filepath (str): path to the SQLite database

    Returns:
        (SQLAlchemy engine) SQLite engine connected to the database
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('messages', engine, index=False, if_exists='replace')
    return engine


def main():

    '''
    This file is the ETL pipeline that cleans the data and stores it in a SQLite database.

    From this project's root directory, run this file with:
    python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/messages.db
    '''

    messages_filepath = 'data/disaster_messages.csv'
    categories_filepath = 'data/disaster_categories.csv'
    database_filepath = 'messages.db'

    print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print(f'Saving data...\n    DATABASE: {database_filepath}')
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')




if __name__ == '__main__':
    main()