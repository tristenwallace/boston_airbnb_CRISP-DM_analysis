import statsmodels.api as sm
import pandas as pd
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

######################################################################

### functions for imputing missing values with linear regression
def is_linear_relationship(df, parameters, target, thresh=.5):
    ''' Checks if target variable has linear relationship with params
    
    INPUT
        df (pandas dataframe)
        parameters (list of str): List of column names to predict target
        target (str): target column name 
        thresh (float): minimum r-squared value to consider imputing w/ linear reg
    
    OUTPUT
        Prints r squared value and whether to use linear regression to impute
        missing values
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


######################################################################


### functions for bootstrapping T-Test 
def boot_matrix(z, B):
    """ Create matrix of bootstrap samples
    
    INPUT
        z (series): pandas series pre-filtered by split_hypothesis
        B (int): Number of times to bootstrap
    
    OUTPUT
        Returns bootstraped samples in a matrix
    """
    
    bootsamples = []
    for _ in range(B):
        bootsamples.append(z.sample(200, replace = True))
    return bootsamples


def  split_hypothesis(df, group, test):
    ''' Creates series to use in bootstrapping 
    
    INPUT
        df (dataframe): dataframe with sample data
        group (str): column name for group feature
        test (str): column name for feature being tested
    
    OUTPUT
        x (series): pandas series filtered from dataframe by test column 
                    for which the group column is true
        y (series): pandas series filtered from dataframe by test column
                    for which the group column is false
    '''
    
    x = df.loc[df[group] == True, test]
    y = df.loc[df[group] == False, test]
    
    return x,y


def bootstrap_t_pvalue(df, group, test, equal_var=False, B=10000, plot=False):
    """ Bootstrap p values for two-sample t test
    
    INPUT
        df (dataframe): dataframe with sample data
        group (str): column name for group feature
        test (str): column name for feature being tested
        equal_var (bool): default performs Welch’s t-test. Setting to 'True'
                            performs standard independent 2 sample test that 
                            assumes equal population variances
        B (int): Number of times to bootstrap
        plot (bool): Whether or not to plot results
        
    
    OUTPUT
        test (str): feature name being tested
        p (float): boostrapped p value 
        statistic (float): T-Test statistic from original sample
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
        

######################################################################


# Functions for analyzing feature importance
def create_regression_mod(X, y, test_size=.3, rand_state=42):
    '''
    INPUT:
        X (pandas dataframe): feature matrix
        y (pandas dataframe): target variable
        test_size - a float between [0,1] about what proportion of data should
                    be in the test dataset
        rand_state - an int that is provided as the random state for splitting
                        the data into training and test 
    
    OUTPUT:
        reg - model object from sklearn
        X_train, X_test, y_train, y_test - output from sklearn train test 
                                            split used for optimal model
    '''

    # Split data
    X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=test_size, random_state=rand_state)
    
    ### Standardize data
    scaler = StandardScaler()
    
    num_cols = X_train.select_dtypes('number').columns.tolist()
    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
    
    X_train_numeric_norm = pd.DataFrame(data=scaler.fit_transform \
        (X_train[num_cols]), columns=num_cols)
    X_train_norm = pd.merge(X_train_numeric_norm, X_train[cat_cols]. \
        reset_index().drop('index', axis=1), left_index=True, right_index=True)
    
    X_test_numeric_norm = pd.DataFrame(data=scaler.fit_transform \
                                        (X_test[num_cols]), columns=num_cols)
    X_test_norm = pd.merge(X_test_numeric_norm, X_test[cat_cols] \
        .reset_index().drop('index', axis=1), left_index=True, right_index=True)
    
    

    # Fit Model
    reg = RidgeCV()
    reg.fit(X_train_norm, y_train)

    return reg, X_train_norm, X_test_norm, y_train, y_test


def coef_weights(coefficients, X_train):
    '''
    INPUT:
        coefficients - the coefficients of the linear model 
        X_train - the training data, so the column names can be used
    
    OUTPUT:
        coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = coefficients
    coefs_df['abs_coefs'] = np.abs(coefficients)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    
    return coefs_df