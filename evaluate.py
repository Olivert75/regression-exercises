import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from math import sqrt

def plot_residuals ( df, y, yhat  ):
    '''
    Takes in a dataframe , y = column with actual_values and yhat= name of the columns with predicted_values
    and creates a residual plot
    
    Example:
    plot_residuals(df, 'tip', 'yhat')
    '''

    #baseline
    df['yhat_baseline'] = df[y].mean()


    # residuals
    df['residuals'] = df[yhat] - df[y]
    df['residuals_baseline'] = df ['yhat_baseline'] - df[y]

    # plot
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.hist(df.residuals_baseline, label='baseline residuals', alpha=.6)
    ax.hist(df.residuals, label='model residuals', alpha=.6)
    ax.legend()
    plt.show()
    return




def regression_errors(df, y, yhat):
    '''
    Takes in a dataframe , y = column with actual_values and yhat= name of the columns with predicted_values
    and calculate:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
    Returns a dictionary with all these values.
    Example:
    plot_residuals(df, 'tip', 'yhat')
    '''
    #import
    from sklearn.metrics import  mean_squared_error
    from math import sqrt
    
    
    #calculate SSE using sklearn
    SSE = mean_squared_error(df[y], df[yhat])*len(df)
    #explained sum of squares (ESS)
    ESS = ((df[yhat] - df[y].mean())**2).sum()
    #total sum of squares (TSS)
    TSS = ((df[y] - df[y].mean())**2).sum()
    #mean squared error (MSE)
    MSE = mean_squared_error(df[y], df[yhat])
    #root mean squared error (RMSE)
    RMSE = sqrt(MSE)
    
    #create a dictionary
    m= {
        'sse': SSE,
        'ess': ESS,
        'rmse': RMSE,
        'tss': TSS,
        'mse': MSE,
        'r2': ESS/TSS,
    }

    return m



def baseline_mean_errors(df, y):
    '''
    Takes in a dataframe , y = column with actual_values 
    and calculate:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
    Returns a dictionary with all these values
    Example:
    baseline_mean_errors(y_train, 'actual')
    '''
    #import
    from sklearn.metrics import  mean_squared_error
    from math import sqrt

    #baseline
    df['yhat_baseline'] = df[y].mean()

    #calculate SSE using sklearn
    SSE_baseline = mean_squared_error(df[y], df['yhat_baseline'])*len(df)
    #explained sum of squares (ESS)
    ESS_b = ((df.yhat_baseline - df[y].mean())**2).sum()
    #total sum of squares (TSS)
    TSS_b = ((df[y] - df[y].mean())**2).sum()
    #mean squared error (MSE)
    MSE_baseline = mean_squared_error(df[y], df.yhat_baseline)
    #root mean squared error (RMSE)
    RMSE_baseline = sqrt(MSE_baseline)
    
    #create dicc
    b ={
        'sse': SSE_baseline,
        'mse': MSE_baseline,
        'rmse': RMSE_baseline,
         'tss': TSS_b,
        'ess' : ESS_b,
        'mse': MSE_baseline,
        'r2': ESS_b/TSS_b,       
    }

    return b




def better_than_baseline(df, y, yhat):
    '''
    Takes in a df, column with actual values,  and predicted values
    and returns true if your model performs better than the baseline, otherwise false
    '''    
    from IPython.display import display, HTML
    #baseline 
    b = baseline_mean_errors(df,y)
    #Model
    m = regression_errors (df, y, yhat)
    
    df1 = pd.DataFrame(m, index= ['model'])
    df2 =pd.DataFrame(b, index= ['baseline'])
    table =pd.concat([df2, df1]).T
    display(HTML(table.to_html()))
    print('   ')

    if m['rmse']< b['rmse']:
        print ('Model performs better than the baseline: ',m['rmse']< b['rmse'] )
    else:
        print ('Model performs better than the baseline: ',m['rmse']< b['rmse'] )
    
    return