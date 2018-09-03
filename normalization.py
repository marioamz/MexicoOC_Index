import pandas as pd
from scipy import stats


def z_score(df):
    '''
    This function takes in a dataframe and returns a dataframe
    where values have been normalized according to z-score.
    '''
    
    return df.apply(stats.zscore)
    
        
def ranking(df):
    '''
    This function takes in a dataframe and returns a new dataframe
    where values have been normalized according to ranking.
    
    It does it in descending order and with equal values taking
    on the minimum rank for that group.
    '''
    
    return df.rank(ascending=False, method='min')

def categorical(df):
    '''
    This function takes in a dataframe and returns a new dataframe
    where values have been normalized according to categorical values.
    
    It does by binning, but there might be a better way to do it.
    '''
    
    for c in df:
        df[c] = pd.cut(df[c], 5, labels=[1, 2, 3, 4, 5])
    
    return df