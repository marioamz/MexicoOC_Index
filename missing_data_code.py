import pandas as pd
import math

def reading_in(excelname):
    '''
    This file reads in an excel dataset that has already been
    cleaned and compiled elsewhere.
    '''

    df = pd.read_excel(excelname)
    return df


def find_missing_cols(df):
    '''
    This function takes a dataframe and finds the columns with
    missing variables

    It returns a list of all columns with missing variables
    '''

    # empty list to fill in columns that are nulls
    null_cols = []
    # creating a series that counts missing values in column
    null_series = df.isnull().sum()

    for ind, val in null_series.items():
        if val != 0:
            null_cols.append((ind, val))

    return null_cols


def find_missing_rows(df, row_id, cols=[]):
    '''
    This function takes a dataframe, finds the rows that have columns
    with null values, and adds an identifier (typically the index) to
    see which rows have missing values.

    It returns sets with rows for missing values, segregated by columns
    '''

    missing_rows = set()

    for tup in cols:
        for ind, row in df.iterrows():
            if math.isnan(row[tup[0]]):
                missing_rows.add((row[row_id], tup[0]))

    return missing_rows


def impute_zero(df, dict):
    '''
    This file takes a dataframe with missing columns and imputes
    missing values for some columns with zeroes

    It returns a new dataframe with values filled in place
    '''

    new_df = df.fillna(value=dict, inplace=True)
    return new_df


def case_deletion(df):
    '''
    This file takes a dataframe and deletes all the rows or columns
    with missing values

    It returns a new dataframe with the deletions
    '''
