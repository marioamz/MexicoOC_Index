import pandas as pd
import missingno as mno
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

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


def visualizing_nulls(df, graph):
    '''
    This function visualizes nulls using the missingno package. It
    takes in a dataframe and the type of graph we want, and then
    returns the graph
    '''

    if graph == 'nullity':
        mno.matrix(df)
    elif graph == 'bar':
        mno.bar(df, color='purple', log='True', figsize=(30,18))
    elif graph == 'corr':
        mno.heatmap(df, figsize=(20,20))

    plt.show()


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


def linear_regression_imputation(df, dep_vars, missrow, index):
    '''
    This function takes in a dataframe and imputes it's missing values
    using linear regression

    It returns a dataframe with the values imputed
    '''

    x_train, y_train, x_test, y_test = train_test(df)


def train_test(df):
    '''
    A helper function that divides a dataframe into test-train splits
    that can help us impute missing variables using methods like linear
    regression and k-nearest neighbors

    This is divided into two steps:
        - Create two dataframes, one without the rows with missing
          variables (train) and the other only with the rows with
          the missing variables (test)
        - Divide the train frame into two: one drops the columns that
          we know have missing variables (x_train) and the other only
          keeps columns that we have missing variables (y_train)
        - Divide the test frame into two: one drops the columns that
          we know have missing variables (x_test), the other keeps only
          those variables (y_test)

    The beauty is that we can use case_deletion here for most of the work

    Returns x_train, y_train, x_test, y_test
    '''

    # Step 1
    train = case_deletion(df, 'row')
    test = df[df.isnull().any(1)]

    # Step 2
    x_train = train.drop(find_missing_cols(df), axis=1)
    y_train = train(find_missing_cols(df))

    x_test = test.drop(find_missing_cols(df), axis=1)
    y_test = test(find_missing_cols(df))

    return x_train, y_train, x_test, y_test


def explore_potential_correlations(df):
    '''
    This function explores potential correlations between variables

    This is to find instances of multicollinearity.
    '''

    axis = plt.axes()
    sns.heatmap(df.corr(), square=True, cmap='PiYG')
    axis.set_title('Correlation Matrix')
    plt.show()