# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:08:27 2023

@author: domingosdeeulariadumba
"""
# %%
            # SETTING UP
            
            
## Importing libraries
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller as adf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from pylab import rcParams

## Setting global parameters for plotting
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'

## Supressing all warnings   
import warnings
warnings.filterwarnings('ignore')


## Dataset path
path = 'ers_dataset.xlsx'

## Reading the dataset xls file as a pandas dataframe
retail = pd.read_excel(path)

# %%
            # EXPLORATORY DATA ANALYSIS

                  
## First five records
retail.head()

## Passing the date column to be the index and presenting the first five rows
retail.set_index(retail.columns[0], inplace = True)
retail.head()

## Comparing water and alchohol sales
sns.lineplot(data = retail, legend = True)
plt.ylabel('Unit Sales')
plt.xlabel('')
plt.xticks(rotation = 25)
plt.title(f'Weekly Retail Sales ({retail.index.min().year} - {retail.index.max().year})')
plt.show()

## Cases when alchol sales were higher than water's

alc_higher_than_water = retail[retail['Alcohol_UnitSales'
                                      ] > retail['Water_UnitSales']]

## Plotting cases where alcohol sales were higher water's
sns.lineplot(data = alc_higher_than_water, legend = True)
plt.ylabel('Unit Sales')
plt.xlabel('')
plt.xticks(rotation = 25)
plt.title(f'Sales of Alcohol Higher than Water ({alc_higher_than_water.index.min().year} - {alc_higher_than_water.index.max().year})')
plt.show()


    '''
    The prior plots summarize the dominance of water sales over the years 
    considered for this analysis. It is noticed some seasonal effects. 
    Around july, the water consumption tends to be relatively higher. 
    This is the hottest season of the year, and may explain this tendency.
    When it comes to alcohol consumption, there are two main points: also
    around july the sales was higher (this period is marked by holidays and
    is the longest school break); and secondly, compared to summer, around the
    end and the beginnig of every year the alcohol unit sales was higher than 
    the water's. This magnitude is noticed only in summer, and may be explained
    by special calendar events (such as 4th July, Thanksgiving, and christmas).
    
    From this point on, we will focus our analysis in alcohol consumption. As
    most of the times it is observed higher unit sales of water.
    '''

        ## Classic Decomposition

### Dataset for alchohol unit sales
df_alc = retail.iloc[:, -1:]

### Additive decomposition of the series
add_dec = seasonal_decompose(df_alc, model = 'additive', period = 52)
add_dec.plot().suptitle('Additive Decomposition', y = 1.05, fontsize = 16)
plt.xticks(rotation = 25)
plt.show()

        ## Alternative Decomposition

    ### Trend Analysis
#### Trend analysis using the moving average method
plt.plot(df_alc, label = 'Original Data')
plt.plot(df_alc.rolling(52).mean(), label = 'Moving Average')
plt.ylabel('Unit Sales')
plt.xticks(rotation = 25)
plt.legend()
plt.title(f'Alcohol Sales ({df_alc.index.min().year} - {df_alc.index.max().year})')
plt.show()

    ### Seasonality Detection
    
    '''
    This will be done using the periodogram and seasonal plots. I have already
    created two functions for this purpose. It is shown below.
    '''
##### Funtion to show the periodogram
def show_periodogram(data, detrend = 'linear',
                     ax = None, fs = 365 , color = 'brown'):
    
        series = data.squeeze()
        
        # Number of observations
        n_obs = len(data)
         
        # Computing frequencies and spectrum
        frequencies, spectrum = periodogram(series, fs, window = 'boxcar', detrend = detrend,
                                            scaling = 'spectrum')

        # Plotting the periodogram
        if ax is None:
            _, ax = plt.subplots()


        # Frequency adjustment
        freqs = [1, 2, 3, 4, 6, 12, 26, 52, 104]
        freqs_labels = ['Annual (1)', 'Semiannual (2)', 'Triannual (3)',
                        'Quarterly (4)', 'Bimonthly (6)', 'Monthly (12)',
                        'Biweekly (26)', 'Weekly (52)', 'Semiweekly (104)']

        ax.step(frequencies, spectrum, color = color)
        ax.set_xscale('log')
        ax.set_xticks(freqs)
        ax.set_xticklabels(freqs_labels, rotation = 55, fontsize = 11)
        ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))
        ax.set_ylabel('Variance', fontsize = 12)
        ax.set_title(f'Periodogram ({n_obs} Observations)')
        plt.show()
        
        return

#### Presenting the periodogram for a cycle of 52 weeks a year
show_periodogram(df_alc, fs = 52)

    '''
    From the periodogram, it is noticed that the dominant cycle is the annual.
    Let us seee in seasonal plots which periods of year specifically the 
    seasonal effects stand out.
    '''


#### Funtion to show seasonal plots
def show_seasonal(df, period, freq, 
                  ax = None, title = None, x_label = None, y_label = None):
    
    # Copying the dataset
    data = df.copy()
    
    # Passing the date column as the index of the dataframe
    data.set_index(data.index.values, inplace = True)
    
    # Extracting date elements from the index
    data['year'] = data.index.year
    data['month'] = data.index.strftime('%b')
    data['week of year'] = data.index.isocalendar().week.astype(int)  
    data['day of month'] = data.index.day    
    data['day of week'] = data.index.strftime('%a')
    data['day of year'] = data.index.dayofyear
       
    
    # Plotting seasonal effects
    if ax is None:
               
        _, ax = plt.subplots()
    palette = sns.color_palette('rocket', n_colors = data[period].nunique())
    ax = sns.lineplot(
        x = freq, y = data.columns[0], hue = period, data = data, ax = ax, palette = palette)
    
    # Setting options for the plot title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Seasonal Plot ({period}/{freq})')
        
    # Setting label options for x axis    
    if x_label is None:
        plt.xlabel(freq)    
    elif x_label == 'hide':
        plt.xlabel('')    
    else:
        plt.xlabel(x_label)
        
    # Setting label options for y axis     
    if y_label is None:
        plt.ylabel(data.columns[0])    
    else:
        plt.ylabel(y_label)
    
    plt.legend(fontsize = 8, loc = 'best')
    plt.show()
    
    
    return

#### Months vs. year seasonal analysis (annual period)
show_seasonal(df_alc, 'year', 'month', x_label = 'hide', y_label = 'Alcohol Unit Sales')


#### Year vs. day of year seasonal plot (annual period)
show_seasonal(df_alc, 'year', 'day of year', y_label = 'Alcohol Unit Sales')

    '''
    It is clear crystal now that, annually, it is comon to observe alcohol 
    sales spikes around the 150th and 200th day of each year. More precisely,
    between june and august.
    '''
 
    ### Serial Dependency   
    
    '''
    We'll now analyse the relationship between the alcohol unit sales against
    its previous records. It we'll be adopted the same apporach: we'll create
    a function to visualize the lagged versions of the alcohol called 
    'show_lags'. An alternative way to so is through the correlograms, which 
    will be presented during in the forecasting section.    
    '''

def show_lags(data, n_lags = 10, title = 'Lag Plots'):
    
    fig, axes = plt.subplots(2, n_lags//2, figsize = (10, 6)
                             , sharex = False, sharey = True, dpi = 240)
    
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
    plt.suptitle(title, y = 1.05)
    plt.show()
    
    return
 
show_lags(df_alc, 6, 'Lag Plots for Alcohol Unit Sales')    

# %%    
            # FORECAST
    
        
        ## SARIMA   

    '''
    ARIMA models and its extensions consider stationarity as pre-requisite, as-
    suming that stationary data are more interpretable.To check the 
    stationarity of alcohol consumption series, we'll use the Augmented 
    Dickey-Fuller (ADF) Test. Below, I have created a function, which output
    is the result of the test and the dataframe of the statistics results.
    '''  
       
### Function to check stationarity by using the Augmented Dickey-Fuller Test
def adf_test(df):
     
     # Performinf the Augmented Dickey-Fuller Test    
     adft = adf(df, autolag = 'AIC')
     
     # Dataframe to store the results of the test    
     scores_df = pd.DataFrame({'Scores':[adft[0], adft[1], 
                                         adft[2], adft[3], adft[4]['1%'],
                                         adft[4]['5%'], adft[4]['10%']]},
                             index = ['Test Statistic',
                                     'p-value','Lags Used',
                                     'Observations Used', 
                                     'Critical Value (1%)',
                                     'Critical Value (5%)',
                                     'Critical Value (10%)'])
     
     
     # Printing the result of the test    
     if adft[1] > 0.05 and abs(adft[0]) > adft[4]['5%']:
         print('\033[1mThis series is not stationary!\033[0m')
     else:
         print('\033[1mThis series is stationary!\033[0m')
         
     print('\nResults of Dickey-Fuller Test\n' + '=' * 29)
      
     # Displaying the dataframe with the statistics parameters of the ADF test    
     display(scores_df)
     
     return

### Test result
adf_test(df_alc)

    '''
    The series is not stationary. In this case, we'll apply differecing to
    make it stationary
    ''' 
        ### Differencing    

#### First order differencing to make the series stationary
df_alc_diff = df_alc.diff()

#### Dropping missing values
df_alc_diff.dropna(inplace = True)

#### Plotting the Differenced data
sns.lineplot(data = df_alc_diff, legend = False)
plt.title(f'Alcohol Sales Data After Differencing')
plt.xlabel('')
plt.xticks(rotation = 25)
plt.show()

#### ADF test on the differenced series
adf_test(df_alc_diff)

    '''
    The series is now stationary. But let us now first apply log transformation
    to stabilize the variance and then use differencing to remove the trend.
    ''' 
        
        ### Logarithmic Transformation and differencing

#### Logarithmic Transformation
df_alc_log = np.log(df_alc)

# Differencing and removal of any missing values
df_alc_log_diff = df_alc_log.diff().dropna()

#### Plotting the differenced data after log transformation
sns.lineplot(data = df_alc_log_diff, legend = False)
plt.title(f'Data Differenced after Logarithmic Transformation')
plt.xlabel('')
plt.xticks(rotation = 25)
plt.show()

# Performing the ADF test again
adf_test(df_alc_log_diff)

   '''
   As the data is now stationary, in order to fit the model, we'll first split
   the series into train and test sets so we can then make performance compari-
   son. It is done down below.
   ''' 
   
### Funtion to split the series into train and test sets
def split_data(df, train_proportion = 0.8):
    
    train_size = round(len(df)*train_proportion)
    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]
    
    return train_set, test_set

### Splitting the data
train_data, test_data = split_data(df_alc) # Original scale
train_data_diff, test_data_diff = split_data(df_alc_log_diff) # Data Differenced in log scale

   '''
   From here, we can now train the ARIMA model. But we need to firstly find the
   bes order of the model to make accurate forecast, that is, the (pdq) for 
   trend and (PDQ)m for the seasonal effect. We'll use the correlogram to have 
   an estimation and then use the auto_aurima method to make the grid search of
   the optimal order automatically.
   ''' 

### Checking the Autoregressive and Moving Average orders on correlograms (ACF e PACF) with a customized function
def show_correlogram(data, ACF = True, PACF = True):
    
    # Transforming the dataframe to a pandas series 
    series = data.squeeze()
    
    
    # Setting the plots for correlograms
    fig, axes = plt.subplots(1, 2, figsize = (15, 6))
    
    if ACF:       
        # ACF Plot
        plot_acf(series, lags=6, ax=axes[0])
        axes[0].set_title('ACF Plot')
        
    if PACF:
        # PACF plot
        plot_pacf(series, lags=6, ax=axes[1])  
        axes[1].set_title('PACF Plot')  
    plt.tight_layout()
    plt.show()
    
    return

#### Displaying the correlograms
show_correlogram(train_data_diff)

#### Selecting automatically the order of the ARIMA model
auto_select = auto_arima(train_data_diff, seasonal = True,
                         m = 52, trace = True, stepwise = True)

#### Fitting the model with SARIMA
sarima_model = ARIMA(train_data_diff, order = auto_select.order, seasonal_order = auto_select.seasonal_order)
sarima_model_fit = sarima_model.fit()

#### Forecast for the next 23 weeks
steps_diff = len(test_data_diff) + 23
sarima_results = sarima_model_fit.get_forecast(steps = steps_diff)
sarima_forecast_log_diff = sarima_results.predicted_mean

#### Reversing the differenced operation for the forecast results: cumulative sum
sarima_forecast_cumsum = sarima_forecast_log_diff.cumsum()

#### Reversing the differenced operation for the forecast results in confidence intervals: cumulative sum
sarima_conf_int_cumsum = sarima_results.conf_int().cumsum()
lower_bound_sarima_cumsum = sarima_conf_int_cumsum.iloc[:,0]
upper_bound_sarima_cumsum = sarima_conf_int_cumsum.iloc[:,1]

#### setting the base value
base_value = df_alc_log.squeeze().iloc[len(train_data)]
series_base_value = pd.Series(base_value, index = pd.date_range(start = test_data_diff.index[0],
                                                                    periods = steps_diff, freq = 'W'))

#### Reversing the forecast value: summing with the base value
sarima_forecast_log = series_base_value.add(sarima_forecast_cumsum, fill_value = 0)

#### Reversing the forecast value in confidence intervalos: summing with the base value
lower_bound_sarima_log = series_base_value.add(lower_bound_sarima_cumsum, fill_value = 0)
upper_bound_sarima_log = series_base_value.add(upper_bound_sarima_cumsum, fill_value = 0)

#### Reversing the forecast values to the original scale: exponential transformation
sarima_forecast = np.exp(sarima_forecast_log)

#### Reversing the forecast values in confidence intervals: exponential transformation
lower_bound_sarima = np.exp(lower_bound_sarima_log)
upper_bound_sarima = np.exp(upper_bound_sarima_log)

#### Plot in Logarithmic Scale: SARIMA
plt.plot(df_alc_log, color = 'c', label = 'Actual')
plt.plot(sarima_forecast_log, color = 'm', label = 'Forecast', linestyle = '-.')
plt.fill_between(lower_bound_sarima_log.index, upper_bound_sarima_log, lower_bound_sarima_log,
                 alpha = 0.5, linewidth = 2, color = 'thistle', label = 'Confidence Interval: 95%')
plt.title(f"Forecast for {sarima_forecast_log.index.max().strftime('%B')} {sarima_forecast_log.index.max().year} (SARIMA)")
plt.text(df_alc_log.index[0], 22.9, 'Log Scale', fontsize = 12, color='red', fontweight = 'bold')
plt.ylabel('Alcohol Unit Sales')
plt.xticks(rotation = 25)
plt.legend()
plt.show()

#### Plot in original scale: SARIMA 
plt.plot(df_alc, color = 'c', label = 'Actual')
plt.plot(sarima_forecast, color = 'm', label = 'Forecast', linestyle = '-.')
plt.title(f"Forecast for {sarima_forecast.index.max().strftime('%B')} {sarima_forecast.index.max().year} (SARIMA)")
plt.ylabel('Alcohol Unit Sales')
plt.xticks(rotation = 25)
plt.legend()
plt.show()


        ## SARIMAX 

        
### Fitting the SARIMAX model
sarimax_model = SARIMAX(train_data_diff, order = auto_select.order, seasonal_order = auto_select.seasonal_order)
sarimax_model_fit = sarimax_model.fit()

### Forecasting the next 23 weeks
sarimax_results = sarimax_model_fit.get_forecast(steps = steps_diff)
sarimax_forecast_log_diff = sarimax_results.predicted_mean
        
### Reversing the forecasted data: cumulative sum
sarimax_forecast_cumsum = sarimax_forecast_log_diff.cumsum()

### Reversion for confidence intervals: cumulative sum
sarimax_conf_int_cumsum = sarimax_results.conf_int().cumsum()
lower_bound_sarimax_cumsum = sarimax_conf_int_cumsum.iloc[:,0]
upper_bound_sarimax_cumsum = sarimax_conf_int_cumsum.iloc[:,1]       
        
### Reversing the forecast value: summing with the base value
sarimax_forecast_log = series_base_value.add(sarimax_forecast_cumsum, fill_value = 0)

### Reversion for confidence intervals: sum with base value
lower_bound_sarimax_log = series_base_value.add(lower_bound_sarimax_cumsum, fill_value = 0)
upper_bound_sarimax_log = series_base_value.add(upper_bound_sarimax_cumsum, fill_value = 0)       
        
### Reversing the forecasted data to the original scale: exponentiation
sarimax_forecast = np.exp(sarimax_forecast_log)

### Reversing the forecasted data to the original scale for confidence intervals: exponentiation
lower_bound_sarimax = np.exp(lower_bound_sarimax_log)
upper_bound_sarimax = np.exp(upper_bound_sarimax_log)        
        
### Plot in logarithmic scale: SARIMAX   
plt.plot(df_alc_log, color = 'c', label = 'Actual')
plt.plot(sarimax_forecast_log, color = 'm', label = 'Forecast', linestyle = '-.')
plt.fill_between(lower_bound_sarimax_log.index, upper_bound_sarimax_log, lower_bound_sarimax_log,
                 alpha = 0.5, linewidth = 2, color = 'thistle', label = 'Confidence Interval: 95%')
plt.title(f"Forecast for {sarimax_forecast_log.index.max().strftime('%B')} {sarimax_forecast_log.index.max().year} (SARIMAX)")
plt.text(df_alc_log.index[0], 22.9, 'Log Scale', fontsize = 12, color = 'r', fontweight = 'bold')
plt.ylabel('Alcohol Unit Sales')
plt.xticks(rotation = 25)
plt.legend()
plt.show()       
        
### Plot in original scale: SARIMAX   
plt.plot(df_alc, color = 'c', label = 'Actual')
plt.plot(sarimax_forecast, color = 'm', label = 'Forecast', linestyle = '-.')
plt.title(f"Forecast for {sarimax_forecast.index.max().strftime('%B')} {sarimax_forecast.index.max().year} (SARIMAX)")
plt.ylabel('Alcohol Unit Sales')
plt.xticks(rotation = 25)
plt.legend()
plt.show()     
        
        
        ## Exponential Smoothing


   '''
   We'll now use the Triple Exponentioal Smoothing Method. Since stationarity
   is not mandatory for this case  (as with the next model, Prophet), it'll be
   used the train set in its original scale.
   '''  
   
### Fitting the data with Exponential Smothing (TES) 
tes_model = ES(train_data, trend = 'add', damped_trend = True, seasonal = 'add', seasonal_periods = 52).fit()
        
### Optimal parameters
alpha = tes_model.model.params['smoothing_level']
beta = tes_model.model.params['smoothing_slope'] if 'smoothing_slope' in tes_model.model.params else None
gamma = tes_model.model.params['smoothing_seasonal'] if 'smoothing_seasonal' in tes_model.model.params else None
phi = tes_model.model.params['damping_slope'] if 'damping_slope' in tes_model.model.params else None

### Printing the optimal parameters scores
print(f"α (Smoothing Level): {alpha:.2f}")
print(f"β (Smoothing Slope): {beta:.2f}" if beta is not None else "β (Smoothing Slope): Not Applicable")
print(f"γ (Smoothing Seasonal): {gamma:.2f}" if gamma is not None else "γ (Smoothing Seasonal): Not Applicable")
print(f"ϕ (Damping Slope): {phi:.2f}" if phi is not None else "ϕ (Damping Slope): Not Applicable")        
        
### Forecasting with TES for the next 'steps_' weeks
steps_ = steps_diff + 1
tes_forecast = tes_model.forecast(steps_).iloc[1:]        

### Computing confidence intervals
residuals = tes_model.resid
sigma = np.std(residuals)
z_score = 1.96  
lower_bound_tes = tes_forecast - (z_score * sigma)
upper_bound_tes = tes_forecast + (z_score * sigma)      
        
### Plot with TES forecast
plt.plot(df_alc, label = 'Actual', color = 'c')
plt.plot(tes_forecast, label = 'Forecast', color = 'm', linestyle = '-.')
plt.title(f"Forecast for {tes_forecast.index.max().strftime('%B')} {tes_forecast.index.max().year} (TES)")
plt.fill_between(tes_forecast.index, lower_bound_tes, upper_bound_tes, color = 'grey',
                 alpha = 0.3, label = 'Confidence Interval: 95%')
plt.ylabel('Alcohol Unit Sales')
plt.xticks(rotation = 25)
plt.legend()
plt.show()    
        
         ## Prophet        
        
        
### Setting the dataframe to fit the model with Prophet
prophet_df = pd.DataFrame({'ds': train_data.index,
                           'y': train_data.squeeze()})        
        
### Fitting the model
model_prophet = Prophet().fit(prophet_df)       
                
### Forecast for next 'steps_' weeks
model_prophet_steps_df = model_prophet.make_future_dataframe(periods = steps_, freq = 'W')        
        
### Parameters from Prophet forecasts
forecasts = model_prophet.predict(model_prophet_steps_df)      
        
### Taking the data forecasted for the 'steps' periods
prophet_forecast = forecasts.iloc[151:]       
        
### plot
model_prophet.plot(prophet_forecast)
prophet_line_forecast = plt.gca().get_lines()[1]
prophet_line_forecast.set_color('m')
prophet_line_forecast.set_linestyle('-.')
prophet_datapoints = plt.gca().get_lines()[0]
prophet_datapoints.set_color('b')
plt.title(f"Forecast for {tes_forecast.index.max().strftime('%B')} {tes_forecast.index.max().year} (Prophet)")
plt.plot(df_alc, label = 'Actual', color = 'c')
plt.ylabel('Alcohol Unit Sales')
plt.xlabel('')
plt.xticks(rotation = 25)
plt.legend()
plt.show()       
        
        
        
            # Performance Assessment     
        
## List of candidate models
sarima = sarima_forecast.iloc[:-23]
sarimax = sarimax_forecast.iloc[:-23]
tes = tes_forecast.iloc[:-23]
prophet = prophet_forecast.iloc[:-23, -1:]
forecast_models_list = [('SARIMA', sarima),
                        ('SARIMAX', sarimax), ('TES', tes),
                        ('Prophet', prophet)]       
        
## Setting the initial data point of the test set to be equal to the forecasted  
if test_data.index[0] == test_data_diff.index[0]:
    eval_data = test_data   
else:
    eval_data = test_data.iloc[1:]         
        
## Re-setting the data in prophet forecast dataframe 
prophet.index = eval_data.index       
        
##Funtion to assess the performance across the candidate models
def evaluate(models_forecast, test_set, view = 'results'):
    
    
    # Empty list of all models' metrics
    df_metrics_global = []
    test_set.rename(columns = {test_set.columns[0]: 'Actual'}, inplace = True)
    
    # Empty variable to receive the forecast sets and data set to be compared
    df_comp = pd.DataFrame()
    
    for model_name, forecast_data in models_forecast:
        # Calculate the parameters
        mse = mean_squared_error(test_set, forecast_data)
        rmse = mean_squared_error(test_set, forecast_data, squared = False)
        mae = mean_absolute_error(test_set, forecast_data)
        mape = mean_absolute_percentage_error(test_set, forecast_data)
    
        # Create a dataframe of metrics
        metrics_list = [mse, rmse, mae, mape]
        metrics_names = ['MSE', 'RMSE', 'MAE', 'MAPE']
        df_metrics = pd.DataFrame({model_name: metrics_list}, index = metrics_names)
    
        # Add the dataframe to the list
        df_metrics_global.append(df_metrics)
        
        # Add the forecast to the comparison dataframe
        df_comp[model_name] = forecast_data 
    
    # Concatenate all dataframes of metrics and results comparison
    metrics = pd.concat(df_metrics_global, axis = 1)
    df_comp_final = pd.concat([test_set, df_comp], axis = 1)
    
    
    # Option to display the transposed performance metrics dataframe 
    if view == 'metrics':
        display(metrics.T)
        
    # Option to display the transposed comparison dataframe 
    elif view == 'results':
        display(df_comp_final.head(10).T)
    else:
        print("This option is not available! :(\nPlease choose whether you want to display the performance inserting "
              "\033[1m'metrics'\033[0m or \033[1m'results'\033[0m in case of forecast comparison.")                        
    
    return      
        
## Comparing the forecast results with the actual values       
evaluate(forecast_models_list, eval_data, 'results')        
        
# Comparing the performance metrics across the candidate models        
evaluate(forecast_models_list, eval_data, 'metrics')        
                
        
    '''
    In terms of performance metrics, from the last line output it is noticed 
    that SARIMAX autperforms all other vandidate models.
    '''
# %%      
                  ________  ________   _______   ______ 
                 /_  __/ / / / ____/  / ____/ | / / __ \
                  / / / /_/ / __/    / __/ /  |/ / / / /
                 / / / __  / /___   / /___/ /|  / /_/ / 
                /_/ /_/ /_/_____/  /_____/_/ |_/_____/  

# %% -
