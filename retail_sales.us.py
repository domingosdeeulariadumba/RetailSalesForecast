# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:08:27 2023

@author: domingosdeeulariadumba
"""

            # IMPORTING REQUIRED LIBRARIES
            
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from scipy import stats
from scipy.signal import periodogram
from scipy.stats import linregress
import seaborn as sb
import matplotlib.pyplot as plt
plt.style.use('ggplot')



            # EXPLORATORY DATA ANALYSIS


# Importing the dataset
df = pd.read_excel("ers_dataset.xlsx")


# Checking the info of the dataset
df.info()


# Passing the date as the index of the dataframe
df.index = df['Date']

# Dropping the date column
df = df.drop('Date', axis = 1)


# Water Vs Alcohol weekly sales
sb.lineplot(data = df, legend = True)
plt.ylabel('Unit Sales')
plt.xlabel('')
plt.xticks(rotation = 25)
plt.title('Weekly Retail Sales (2019-2003)')


# Let us see when there was more sales of alcohol
alch_high = df[df['Alcohol_UnitSales'] > df['Water_UnitSales']]

sb.lineplot(data = alch_high, legend = True)
plt.ylabel('Unit Sales')
plt.xlabel('')
plt.xticks(rotation = 38)
plt.title('Weekly Retail Sales (2019-2003)\n Alcohol Higher Consumption')

    '''
    The prior plots summarize the dominance of water sales over the years 
    considered for this analysis. It is noticed a seasonal effect. around july,
    the water consumption tends to be relatively higher. This is the hottest
    season of the year, and may explain this tendency.
    When it comes to alcohol consumption, there are two main points: Also
    around july the sales was higher (this period is marked by holidays and
    is the longest school break); and secondly, compared to summer, around the
    end and the beginnig of every year the alcohol unit sales was higher than 
    the water's. This magnitude is noticed only in summer, and may be explained
    by special calendar events (such as July, Thanksgiving, 4 and christmas).
    
    From this point on, we will focus our analysis in alcohol consumption. As
    most of the times is observed higher unit sales of water.
    '''

# Checking stationarity for alcohol consumpiton
    
    '''
    To check the stationarity of alcohol consumption, we'll firts create a 
    separate dataframe for this product. And then will be applied the 
    Augmented Dickey-Fuller (ADF) Test. I have created a function, which output
    is the result of the test.
    '''

df_alch = df.iloc[:, -1:]

def check_stationarity(df):
    
    adft = adf(df_alch,autolag="AIC")
    
    scores_df = pd.DataFrame({"Values":[adft[0],adft[1],adft[2],adft[3],
                                    adft[4]['1%'], adft[4]['5%'],
                                    adft[4]['10%']]  ,
                          "Metric":["Test Statistics",
                                    "p-value","No. of lags used",
                                    "Number of observations used", 
                                    "critical value (1%)",
                                    "critical value (5%)",
                                    "critical value (10%)"]})
     
    if scores_df['Values'][1]  > 0.05 and scores_df['Values'][0]>scores_df['Values'][5]:
        print('This serie is not stationary!!!')
        
    return

check_stationarity(df_alch)

    '''
    So, the series is not stationary, as is also clear by looking to the plots.
    '''

        # SERIES COMPONENTS ANALYSIS

# Let as analyse the trend of alcohol sales

plt.plot(df_alch, color = "r",
         label = "Weekly Sales")
plt.plot(df_alch.rolling(4).mean(), color = "k",
         label = "Monthly Sales")
plt.plot(df_alch.rolling(17).mean(), color = "b",
         label = "Semiannual Sales")
plt.plot(df_alch.rolling(52).mean(), color = "g",
         label = "Annual Sales")
plt.legend(fontsize = 8)
plt.xticks(rotation = 35)
plt.ylabel('Unit Sales')
plt.title('Alcohol Retail Sales Trend')

    '''
    From the plot above, it is noticed a subtle yearly decrease trend for
    the alcohol retail sales in US.
    '''

# Let us now analyse the seasonality component

    '''
    This will be done using periodogram and seasonal plot. I have also created
    two functions for this goal.
    '''
        
    
        '''Function to generate periodogram'''
        
def plot_periodogram(data, ax=None, period_type='yearly', color='b'):
    
    N, data_1d = len(data), data.values.ravel()
    
    # Setting the observations period and frequencies
    additional_freqs = [1, 1/4, 1/13, 1/26, 1/52]
    additional_labels = ['Weekly', 'Monthly', 'Quarterly', 'Semiannual',
                         'Yearly']
    
    # Computing periodogram
    frequencies, spectrum = periodogram(data_1d,
                                        detrend = 'linear',
                                        scaling='spectrum')
    
    # Plotting the periodogram
    if ax is None:
        _, ax = plt.subplots()
    ax.step(frequencies, spectrum, color=color)
    ax.set_xscale("log")
    ax.set_xticks(additional_freqs)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_xticklabels(additional_labels, rotation=35, fontsize=10)
    ax.set_ylabel("Variance", fontsize=12)
    ax.set_title("Alcohol Retail Sales Periodogram",
                 fontsize=14)
    return ax


        '''Displaying the periodogram'''
        
plot_periodogram(df_alch)


        '''creating a dataframe to compute the seasonal plot parameters'''
        
dates = pd.DataFrame({'date':
                      df_alch.index})
    
y_season, X_season = df_alch['Alcohol_UnitSales'].reset_index(drop=True), pd.DataFrame({'Year': dates['date'].dt.year,
                                                                                        'Month': dates['date'].dt.month,
                                                                                        'Day': dates['date'].dt.day})
        '''Function to display the seasonal plot'''
        
def seasonal_plot(X_season, y_season, period, freq, ax = None):
    
    if ax is None:
        _, ax = plt.subplots()
    palette = sb.color_palette("rocket",
                               n_colors = X_season[period].nunique())
    ax = sb.lineplot(x = freq, 
                     y = y_season, hue = period, data = X_season,
                     ax = ax, palette = palette)
    ax.set_title('Yearly Alcohol Consumption',
                 fontsize = 16)
    plt.legend(fontsize = 8, loc = 'upper right')
        
    return ax


        '''Presenting the seasonal plot'''
        
seasonal_plot(X_season, y_season, period = 'Year',
              freq = 'Month')


    '''
    For this case, at first glance, it looks not necessary to display the 
    seasonal plot, since the Periodogram already captures the yearly frequency
    of alcohol consumption in a clear way. The main problem is that the 
    periodogram frequency is not precise, despite of the clear yearly variance
    spikes during some periods. For this reason we used the seasonal plot.
    it is displayed after the periodogram, and shows the specific periods of 
    the alcohol consumpion peaks throughout every analysed year (in summer 
    and christmas).
    '''


# Now, let's analyse serial depedency    
    
    '''
    We'll now analyse the relationship between the alcohol unit sales against
    its previous records. It we'll be adopted the same apporach: we'll create
    functions to visualize the lagged versions of the alcohol unit sales series
    and its Partial Autocorrelation Frequency Plot (PAFC Plot).    
    '''

def plot_lags(data, n_lags = 10):
    
    fig, axes = plt.subplots(2, n_lags//2, figsize=(14, 10), sharex=False, sharey=True,
                             dpi=240)
    
    for i, ax in enumerate(axes.flatten()[:n_lags]):
        lag_data = pd.DataFrame({'x': data.squeeze(),
                                 'y': data.squeeze().shift(i+1)}).dropna()
        
        x, y = lag_data['x'], lag_data['y']
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        regression_line = [(slope * xi) + intercept for xi in x]   
        ax.scatter(x, y, c = 'k', alpha = 0.6)
        ax.plot(x, regression_line, color='m', label=f'{r_value**2:.2f}')
        ax.set_title(f'Lag {i+1}')
        ax.legend()
        ax.grid(True) 
    plt.tight_layout()
    plt.suptitle('Lag Plots for Alcohol Unit Sales in US',
                 fontsize = 16, fontweight = 'bold', y = 1.05)
    plt.show()
    
    return



    '''Plotting the lagged versions'''

plot_lags(df_alch, 6)


    ''' Function to generate correlogram'''
    
def plot_correlogram(data, n_lags = 10):
    
    plot_pacf(data, color = 'b', lags = n_lags)
    plt.title("PACF of Alcohol Weekly Sales", fontsize=12,
              fontweight='bold', y = 1.05)
    plt.show()

    return 


    ''' Presenting the correlogram'''
    
plot_correlogram(df_alch)

    '''
    The correlogram illustrates two lagged versions of the alcohol unit sales
    out of the non-correlation area, the two first. The others, even appearing
    out of the non-correlation area, it is more likely that may be false
    positives.
    '''


# SERIES DECOMPOSITION
    
    '''
    The outcome of the below function are two plots. Being the first the 
    additive decomposition of the series, whereas the second is the 
    multiplicative.
    '''
def dec_series(data):
    
    add_dec, mltp_dec = seasonal_decompose(
        data, model='additive'), seasonal_decompose(data,
                                              model = 'multiplicative')
                                                    
    add_dec.plot().suptitle('Additive Decomposition',
                            y = 1.05, fontsize = 16)
    mltp_dec.plot().suptitle('Multiplicative Decomposition',
                           y = 1.05, fontsize = 16) 
    plt.show()
   
    return  
    
dec_series(df_alch)

    '''
    As the outcome of the above function illustrates, the additive 
    decomposition is bettar that in terms of capturing more residual records.
    The multiplicative decomposition discards considerable amount of this 
    errors.
    '''


# FORECASTING
    
 # SARIMA    
    
    '''
    The forecast step will be divided in to parts, firstly, well use the 
    SARIMA Model and then Prophet.
    For the former, we'll first split the data to train the model.
    '''    

def split_data(data, train_proportion = 0.8):
    
    train_size = round(len(data)*train_proportion)
    train_set = pd.DataFrame({'train_set': data.iloc[:train_size].
                              squeeze()})
    test_set = pd.DataFrame({'test_set': data.iloc[train_size:].
                             squeeze()})
    
    return train_set, test_set

train_data, test_data = split_data(df_alch)[0], split_data(df_alch)[1]


    '''
    grid searching the optimal combination for SARIMA
    '''
auto_select = auto_arima(train_data, seasonal = True, m = 52,
                   stepwise = True, trace = True)

     '''
     Training and fitting the model
     '''
model_sarima = SARIMAX(train_data, order=auto_select.order,
                       seasonal_order = auto_select.seasonal_order).fit()

     '''
     Forecasts for SARIMA
     '''
steps_sarima = model_sarima.get_forecast(steps = len(test_data))

forecast_sarima = steps_sarima.predicted_mean


     '''
     Setting forecast period 56 weeks ahead
     '''
steps_sarima_56 = model_sarima.get_forecast(steps = len(test_data)+56)
forecast_sarima_56 = steps_sarima_56.predicted_mean
  
     '''
     Computing congidence intervals
     '''  
c_i = steps_sarima_56.conf_int()
lower_bound = c_i['lower train_set']
upper_bound = c_i['upper train_set']
period = c_i.index.values

    '''
    Displaying the results.
    '''
    
plt.plot(figsize=(16,10), dpi= 240)
plt.fill_between(period, y1=upper_bound, y2=lower_bound,
                 alpha=0.5, linewidth=2, color='thistle')
plt.plot(df_alch, color = 'b', label = 'Actual')
plt.plot(forecast_sarima_56, color = 'r', label = 'Forecast',linestyle = '--')
plt.title("Alcohol Unit Sales Forecast Until June 2024",
          fontsize = 12)
plt.legend(fontsize = 10)
plt.show()


    # PROPHET  
    
    '''
    Now, using Prophet, on contrary to what we did with SARIMA, it is not
    required to split de data explicitly into train and test sets.
    Prophet uses a concept called "changepoints" wich automatically detects 
    potential points in the time series data where the underlying 
    data-generating process might change. It then splits the data into two 
    sets in order to train the model. By default, it uses a certain percentage
    (usually 80%, as we did with SARIMA) of the data for training and the 
    remaining portion for testing.
    Since Prophet by default needs a dependent variable labelled 'y' and the
    independent (time entries) as 'ds', we'll first create a dataframe 
    according to this 
    '''   

final_data = pd.DataFrame({'ds': df_alch.index,
                           'y': df_alch.squeeze()})

    '''
    Training and fitting the model for Prophet
    '''
model_prophet = Prophet().fit(final_data)

    '''
    Creating forecasts dataframe for 56 in advance
    '''
steps_prophet_56 = model_prophet.make_future_dataframe(periods
                                                     = len(df_alch)+56)

    '''
    Generating forecasts
    '''
forecast_prophet = model_prophet.predict(steps_prophet_56)


    '''
    Displaying the forecasts result
    '''
model_prophet.plot(forecast_prophet)
plt.title("Alcohol Unit Sales Forecast Until June 2024",
          fontsize = 12)
plt.legend(fontsize = 10)


    '''
    Inspecting results
    '''
print(forecast_prophet[['ds', 'yhat', 'yhat_lower',
                        'yhat_upper']].tail())


    # Performance Evaluation
    
    '''
    We'll next compare the metrics (MAE and RMSE) of both models. The output
    of the below function displays a table that stores the performance results
    for each model.
    '''

def check_performance():
    
    # Extracting prophet forecasts for actual unit sales (20% of the data)
    prophet_forecast = forecast_prophet['yhat'
                                  ].iloc[round(len(df_alch)*0.8):188]
    
    # Prophet metrics
    RMSE_prophet = mean_squared_error(test_data, prophet_forecast,
                                      squared = False)
    MAE_prophet = mean_absolute_error(test_data,
                                     prophet_forecast)
    
    # SARIMA metrics
    RMSE_sarima = mean_squared_error(test_data,
                                 forecast_sarima, squared = False)
    MAE_sarima = mean_absolute_error(test_data, forecast_sarima)
    
    # Creating the metrics dataframe
    df_metrics = pd.DataFrame({'Prophet': [RMSE_prophet, MAE_prophet],
                               'SARIMA': [RMSE_sarima, MAE_sarima]}, 
                               index = ['RMSE', 'MAE'])
    # Displaying the dataframe
    display(df_metrics)
    
    return

check_performance()

    '''
    Running the above command, it is noticed that SARIMA presentes a better
    forecasting of alcohol weekly retail unit sales in US when it comes to RMSE
    and MAE.
    '''
-----------------------------------END-----------------------------------------
