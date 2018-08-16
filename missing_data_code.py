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

    # empty set to fill in with row, column pairs
    missing_rows = set()
    # the cols in this instance will always be the null_cols above
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


def case_deletion(df, axis):
    '''
    This file takes a dataframe and deletes all the rows or columns
    with missing values

    It returns a new dataframe with the deletions
    '''

    if axis == 'row':
        new_df = df.dropna(axis=0)
    else:
        new_df = df.dropna(axis=1)

    return new_df


def single_imputation(df, types, columns=[]):
    '''
    This file takes a dataframe with missing values and imputes those
    missing values with the median/mode/mean (type) of the column it's in

    It returns the new dataframe with missing values filled in place
    '''

    values_dict = {}

    if types == 'mean':
        for col in columns:
            values_dict[col] = df[col].mean()

    elif types == 'median':
        for col in columns:
            values_dict[col] = df[col].median()

    elif types == 'mode':
        for col in columns:
            values_dict[col] = df[col].mode()[0]

    new_df = df.fillna(value=values_dict, inplace=True)
    return new_df
