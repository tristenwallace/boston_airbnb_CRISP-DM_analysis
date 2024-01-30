import statsmodels.api as sm
import pandas as pd

def is_linear_relationship(df, parameters, target, thresh=.5):
    '''
    INPUT
        df (pandas dataframe)
        parameters (list of str): List of column names to predict target
        target (str): target column name 
        thresh (float): minimum r-squared value to consider imputing w/ linear reg
    
    OUTPUT
        Prints r squared value and whether to use linear regression to impute missing values.
        Does not return a value
    '''

    # drop NaN values    
    df = df.dropna()

    y = df[target]
    X = df[parameters]
    X['intercept'] = 1

    lm = sm.OLS(y, X)
    r = lm.fit().rsquared_adj
    
    if r>thresh:
        print("R-squared of {} is {}. Impute na w/ linear regression.".format(target, r))
    else:
        print("R-squared of {} is {}. Impute na w/ median or drop column.".format(target, r))