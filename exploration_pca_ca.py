"""
Created on Mon Apr  9 17:47:05 2018

@author: mariomoreno
"""

import pandas as pd
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition


####CLEAN DATA####

def dummify(df, to_dummy):
    '''
    This function takes a list of columns to turn from categorical
    to dummy variable features
    '''

    new_df = pd.get_dummies(df, columns = to_dummy, dummy_na=True)

    return new_df


def clean_data(df, discretize, binary, to_del):
    '''
    This function cleans the data by removing variables that have
    multicollinearity, filling any nulls, and discretizing/turning
    variables to binary.
        - df = dataframe
        - discretize = a list of variables to discretize
        - binary = a list of variables to turn into binary
        - to_del = a list of variables to be dropped

    '''

    #Discretize
    for var in discretize:
        to_discretize(df, var)

    #Binary
    for binary_var in binary:
        to_binary(df, binary_var)

    #Delete
    for delvar in to_del:
        del df[delvar]

    return df

def to_discretize(df, var):
    '''
    This function discretizes variables into more workable ranges.

    The ranges are not automated.
    '''


    student_bins = range(0, 1000, 250)
    price_bins = range(0, 3500, 500)

    if var == 'students_reached':
        df['students_reached_groups'] = pd.cut(df[var], student_bins, labels=False)
        del df[var]
    elif var == 'total_price_excluding_optional_support':
        df['tp_exclude'] = pd.cut(df[var], price_bins, labels=False)
        del df[var]
    elif var == 'total_price_including_optional_support':
        df['tp_include'] = pd.cut(df[var], price_bins, labels=False)
        del df[var]


def remove_outliers(df, cols):
    '''
    This function removes anything outside of three standard deviations within
    a column
    '''

    df = df[((df[cols] - df[cols].mean()) / df[cols].std()).abs() < 3]
    return df


def to_binary(df, var):
    '''
    This function turns a column of continous variables into binary
    for ease of processing
    '''

    new_col = str(var) + '_binary'
    df[new_col] = (df[var] >= df[var].mean()).astype(int)

    del df[var]


def true_to_false(df, cols):
    '''
    This function takes the dataframe and a list of columns with just
    True or False values, and turns True to 1 and False to 0
    '''

    label_enc = preprocessing.LabelEncoder()

    for c in cols:
        label_enc.fit(df[c])
        new_var = label_enc.transform(df[c])
        new_name = c + '_new'
        df[new_name] = new_var
        del df[c]


####EXPLORATION####

def explore_data(df):
    '''
    This function runs a series of basic exploratory commands, including:
        - Tail and Head
        - Summary Stats
        - Null Values
        - Histograms

    It then calls functions that impute null values, remove instances of
    multicollinearity, and discretize or turn varibales into binary numbers
    '''

    # Summary stats for each column
    for i in df:
        print()
        print('Summary stats for', i)
        print(df[i].describe())
        print()

    #Histograms for every column
    plot_data(df, 'hist', None, None)


def explore_potential_correlations(df):
    '''
    This function explores potential correlations between variables

    This is to find instances of multicollinearity.
    '''

    plt.figure(figsize=(20,20))
    sns.heatmap(df.corr(), square=True, cmap='PiYG')
    plt.show()


def count_values(df, col, sort_by, ascending=False):
    '''
    Find the values that make up a particular column of interst
    '''
    groupby = df.groupby(col, sort=False).count()
    return groupby.sort_values(by=sort_by, ascending=False)


#### GRAPHS #####

def box_plot(df):
    '''
    This function creates box plots to better identify which columns
    have outliers
    '''

    for col in df:
        if type(df[col][0]) is not str:
            plt.boxplot(df[col])
            plt.title('Outliers in' + ' ' + col)
            plt.show()

# Generic plot function
def plot_data(df, plot_type, var1, var2):
    '''
    This function builds a few simple plots to visualize and start to
    understand the data. Can be called on the cleaned data, or on the
    original dataframe.
    '''

    if plot_type == 'hist':
        print('Histograms of all columns')
        df.hist(figsize=(20,20))
        plt.show()

    elif plot_type == 'bar':
        print('Bar Chart for', var1)
        df[var1].value_counts().plot('bar', figsize=(20,10))
        plt.show()

    elif plot_type == 'scatter':
        print('Scatter plot between', var1, 'and', var2)
        plt.scatter(df[var1], df[var2])
        plt.show()

    elif plot_type == 'line':
        print('Line graph between', var1, 'and', var2)
        df[[var1, var2]].groupby(var1).mean().plot(figsize=(20,10))
        plt.show()


# Pie plots
def plot_pies(values, labels, colors = ['violet', 'yellow']):
    '''
    Plots a pie chart
    '''

    plt.pie(values, labels = labels, shadow=False, colors= colors, startangle=90, autopct='%1.1f%%')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def pie_vals(df, cols1, ordered='projectid', labels = '', colors = ["violet", "yellow", "green", "blue", "orange"]):
    '''
    Pie chart for the top values in any given column
    '''

    df_to_plot = count_values(df, cols1)
    df_to_plot = df_to_plot[ordered]
    if labels == '':
        labels = tuple(df_to_plot.index)

    plot_pies(tuple(df_to_plot), labels, colors)


# Crosstabs
def graph_crosstab(df, col1, col2):
    '''
    Graph crosstab of two discrete variables

    Inputs: Dataframe, column names (strings)
    '''

    pd.crosstab(df[col1], df[col2]).plot(kind='bar')
    plt.title(col2 + " " + "distribution by" + " " + col1)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()


### PRINCIPAL COMPONENT ANALYSIS ###

def standardize(df):
    '''
    This function takes in a dataframe and standardizes all
    the variables, which is a necessary step for deploying
    and optimizing machine learning algorithms like PCA

    It returns a list of arrays that are the standardized
    columns
    '''

    std_array = preprocessing.StandardScaler().fit_transform(df)
    return std_array


def pca(array, components):
    '''
    This function takes in the arrays that have been standardized
    and the number of components we want. From there, it calculates
    the principal component vectors and how much of the variation
    they explain.
    '''

    pc = decomposition.PCA(n_components = components)
    y_pca = pc.fit_transform(array)

    print('Variance explained for first' + ' ' + str(components) + \
    ' ' + 'components:')
    print(pc.explained_variance_ratio_)

    return y_pca
