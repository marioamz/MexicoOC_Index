import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition


### PCA Weighting ###


def pca(df, components):
    '''
    This function takes in a dataframe and the number of principal
    components we want to find in that dataframe, and returns a PCA
    class which is our decomposed dataframe with the number of
    components and an array of all the principal components as
    vectors
    '''

    std_array = preprocessing.StandardScaler().fit_transform(df)
    pc = decomposition.PCA(n_components = components)
    y_pca = pc.fit_transform(std_array)

    return pc, y_pca


def pca_df(df, pc, y_pca, components, var1, var2, name):
    '''
    This function takes in a dataframe without strings, the principal
    components class we created above, the array of vectors, the
    number of components already established, and our target variable(s).

    It returns a  pandas dataframe where the columns are the principal
    component vectors
    '''

    # graphing and printing the explained variance
    explained_variance_graph(pc)
    print('Variance explained for first' + ' ' + str(components) + \
    ' ' + 'components' + ' ' + 'in' + ' ' + name)
    print(np.cumsum(pc.explained_variance_ratio_) * 100)
    print()
    
    
    #print('Eigenvalues for first' + ' ' + str(components) + \
    #     ' ' + 'components' + ' ' + 'in' + ' ' + name)
    #print(pc.explained_variance_)
    #print()
    #print()

    # creating the dataframe of PC vectors
    pcaDF = pd.DataFrame(data = y_pca, columns = \
    ['PC' + str(i+1) for i in range(components)])

    # creating final dataframe
    final = pd.concat([pcaDF, df[var1], df[var2]],  axis=1)

    return final


def explained_variance_graph(vectors):
    '''
    This function takes decomposed vectors and returns
    a cumulative explained variance graph for the number
    of components specified above
    '''

    plt.plot(np.cumsum(vectors.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def factor_loadings(pc, df_vars, df_pca, var1, var2):
    '''
    This function takes in PCA object created by a sklearn
    decomposition, a dataframe that has all the variables,
    a dataframe that has all the principal component vectors,
    target variables, and returns a dataframe that explains
    the factor loadings for each of the principal component
    vectors created
    '''

    loadings_array = pc.components_.T * np.sqrt(pc.explained_variance_)

    loadings = pd.DataFrame(data=loadings_array,
        index= [c for c in df_vars if c != var1 and c != var2],
        columns= [p for p in df_pca if p != var1 and p != var2])

    return loadings     


def ss_loadings(df):
    '''
    This function takes in a loadings dataframe, converts it to
    a sum of squares dataframe by 1) squaring all values in that
    dataframe; and 2) adding all those values per column and dividing 
    each individual values by the sum of all the squares.
    
    It also creates a list that stores how much variance each column
    explains in the sum of squares total.
    '''
    
    # Create a squares dataframe
    ss_df = df.apply(np.square)
    
    # Update dataframe to sum of squares percentage
    # Create sum of squares row
    sum_of_squares = []
    for c in ss_df:
        sum_of_sqr = np.sum(ss_df[c])
        sum_of_squares.append(sum_of_sqr)
        ss_df[c] = ss_df[c] / sum_of_sqr
        
    # Percent explained variance row
    perc_explained_var = []
    sums = sum(sum_of_squares)
    for s in sum_of_squares:
        perc_explained_var.append(s/sums)
        
    return ss_df, perc_explained_var  


def calculate_weights(ss_df, perc_list):
    '''
    This function takes in a sum of squares dataframe and a list of how much
    variance each column explains the sum of squares total.
    
    It uses numpy to calculate the weights from this information by:
        - pulling out the max values for each row
        - adding all the maxes in each column to get a total (in list)
        - for each column, multiply the max total by the percent
        of variance that column represents
        - for each max stored in nonmax_zero_df, multiply by percent of
        variance its column explains, and divide by the multiplied variable
    
    This function returns an array of all the weights in order.
    '''
    
    # pull out max values for each row
    array_ss = ss_df.values
    max_values = array_ss.max(axis=1).reshape(-1, 1)
    nonmax_zero = np.where(array_ss == max_values, max_values, 0)
    nonmax_zero_df = pd.DataFrame(nonmax_zero)
    
    # add all maxes in column for total
    sum_of_maxes = []
    for col in nonmax_zero_df:
        sum_of_maxes.append(nonmax_zero_df[col].sum())
        
    # multiply max total by percent of variance
    multiplied = np.multiply(perc_list, sum_of_maxes)
    
    # for each max in df, multiply by % of variance in column and 
    # divide by multiplied...make the change in place
    count = 0
    for col in nonmax_zero_df:
        nonmax_zero_df[col] = (nonmax_zero_df[col] * perc_list[count]) / multiplied[count]
        count += 1
        
    weights = nonmax_zero_df.max(axis=1).values
    
    ss_df['weights'] = weights
    
    return ss_df, weights
    
    
def apply_weights(norm_df, weights):
    '''
    This function calculates the weighted values for each variable.
    It takes in our normalized dataframe and the weights array calculated
    above, and multiplies each column with it's appropriate weight
    '''
    
    