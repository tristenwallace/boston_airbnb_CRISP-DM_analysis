import statsmodels.api as sm
import pandas as pd
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt

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



def boot_matrix(z, B):
    """Bootstrap sample
    
    Returns all bootstrap samples in a matrix
    """
    
    bootsamples = []
    for _ in range(B):
        bootsamples.append(z.sample(200, replace = True))
    return bootsamples


def  split_hypothesis(df, group, test):
    '''
    '''
    x = df.loc[df[group] == True, test]
    y = df.loc[df[group] == False, test]
    
    return x,y


def bootstrap_t_pvalue(df, group, test, equal_var=False, B=10000, plot=False):
    """Bootstrap p values for two-sample t test
    
    Returns boostrap p value, test statistics and parametric p value
    """
    
    x, y = split_hypothesis(df, group, test)
    
    # Compute the t-statistic in your data and store it
    orig = stats.ttest_ind(x, y, equal_var=equal_var)

    # Generate boostrap distribution of t statistic
    overall_mean = (x.shape[0]/df.shape[0])*x.mean() + (y.shape[0]/df.shape[0])*y.mean()
    xboot = boot_matrix(x - x.mean() + overall_mean, B=B) # Change the data such that the null-hypothesis is true.
    yboot = boot_matrix(y - y.mean() + overall_mean, B=B)
    
    # Compute the t-statistic in each of these bootstrap samples.
    sampling_distribution = stats.ttest_ind(xboot, yboot, axis=1, equal_var=equal_var)[0]

    # Calculate proportion of bootstrap samples with at least as strong evidence against null    
    p = np.mean(sampling_distribution >= orig[0])
    statistic = orig[0]
    
    # RESULTS
    return test, p, statistic
    
    
    # Plot bootstrap distribution
    if plot:
        plt.figure()
        plt.hist(sampling_distribution, bins="fd") 