# Initial imports
import streamlit as st
import time
import numpy as np
import pandas as pd

import hvplot
import hvplot.pandas
import holoviews as hv
hv.extension('bokeh', logo=False)

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral10
from bokeh.models import Legend

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

import os
import requests
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import matplotlib.pyplot as plt
from pathlib import Path

# %matplotlib inline

#Load .env environment variables
load_dotenv()

faangm_closing_return_df_=pd.read_csv(Path("AMZN_min.csv"))
faangm_closing_return_df_.rename(columns={"timestamp": "Timestamp"}, inplace=True)
faangm_closing_return_df_.set_index("Timestamp", inplace=True)
faangm_closing_return_df_.head()

#################################################################
# Stochastic Oscillator + SMA
#################################################################
k_period = 16
d_period = 7
short_window = 10
long_window = 33
k_80 = 93
k_20 = 24


faangm_closing_return_df = faangm_closing_return_df_.copy()
# faangm_closing_return_df["k_period_min"] = faangm_closing_return_df["AMZN"].rolling(window=k_period).min()
# faangm_closing_return_df["k_period_max"] = faangm_closing_return_df["AMZN"].rolling(window=k_period).max()
faangm_closing_return_df["k_period_min"] = faangm_closing_return_df.AMZN.rolling(window=k_period).min()
faangm_closing_return_df["k_period_max"] = faangm_closing_return_df.AMZN.rolling(window=k_period).max()
faangm_closing_return_df["Stoch_K"] =  100 * ((faangm_closing_return_df.AMZN - faangm_closing_return_df.k_period_min) / (faangm_closing_return_df.k_period_max - faangm_closing_return_df.k_period_min))
faangm_closing_return_df["Stoch_D"] = faangm_closing_return_df.Stoch_K.rolling(window=d_period).mean()
faangm_closing_return_df["SMA_Fast"] = faangm_closing_return_df.AMZN.rolling(window=short_window).mean()
faangm_closing_return_df["SMA_Slow"] = faangm_closing_return_df.AMZN.rolling(window=long_window).mean()
faangm_closing_return_df = faangm_closing_return_df.dropna()

def implement_stoch_strategy(prices, k, d, SMA_Fast, SMA_Slow):    
    buy_price = []
    sell_price = []
    stoch_signal = []
    signal = 0
    

    for i in range(len(prices)):
        if (k[i] < k_20 and d[i] < k_20 and k[i] < d[i])  or ((SMA_Fast[i] > SMA_Slow[i]) and (SMA_Fast[i-1] <= SMA_Slow[i-1])) :
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1                
                stoch_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                stoch_signal.append(0)
        elif (k[i] > k_80 and d[i] > k_80 and k[i] > d[i]) or ((SMA_Fast[i] < SMA_Slow[i]) and (SMA_Fast[i-1] >= SMA_Slow[i-1])):
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1                
                stoch_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                stoch_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            stoch_signal.append(0)
            
    return buy_price, sell_price, stoch_signal
            
buy_price, sell_price, stoch_signal = implement_stoch_strategy(faangm_closing_return_df["AMZN"], 
                                                               faangm_closing_return_df["Stoch_K"], 
                                                               faangm_closing_return_df["Stoch_D"],
                                                               faangm_closing_return_df["SMA_Fast"],
                                                               faangm_closing_return_df["SMA_Slow"])

faangm_closing_return_df["stoch_oscl_exit"]= stoch_signal
faangm_closing_return_df["stoch_oscl_signal"]= faangm_closing_return_df.stoch_oscl_exit.cumsum()
faangm_closing_return_df["Overbought"]= k_80
faangm_closing_return_df["Oversold"]= k_20
faangm_closing_return_df = faangm_closing_return_df.dropna()
faangm_closing_return_df.to_csv("buysell.csv")



# Set initial capital
initial_capital = float(1000000)

# Set the share size
share_size = 500

# Take a 500 share position where the dual moving average crossover is 1 (sma_fast is greater than sma_slow)
faangm_closing_return_df['AMZN_position'] = share_size * faangm_closing_return_df['stoch_oscl_signal']

# Find the points in time where a 500 share position is bought or sold
faangm_closing_return_df['AMZN_entry_exit_position'] = faangm_closing_return_df['AMZN_position'].diff()

# Multiply share price by entry/exit positions and get the cumulatively sum
faangm_closing_return_df['AMZN_portfolio_holdings'] = faangm_closing_return_df['AMZN'] * faangm_closing_return_df['AMZN_entry_exit_position'].cumsum()

# Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio
faangm_closing_return_df['AMZN_portfolio_cash'] = initial_capital - (faangm_closing_return_df['AMZN'] * faangm_closing_return_df['AMZN_entry_exit_position']).cumsum()

# Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments)
faangm_closing_return_df['AMZN_portfolio_total'] = faangm_closing_return_df['AMZN_portfolio_cash'] + faangm_closing_return_df['AMZN_portfolio_holdings']

# Calculate the portfolio daily returns
faangm_closing_return_df['AMZN_portfolio_daily_returns'] = faangm_closing_return_df['AMZN_portfolio_total'].pct_change()

# Calculate the cumulative returns
faangm_closing_return_df['AMZN_portfolio_cumulative_returns'] = (1 + faangm_closing_return_df['AMZN_portfolio_daily_returns']).cumprod() - 1

# Drop all NaN values from the DataFrame
faangm_closing_return_df = faangm_closing_return_df.dropna()

faangm_closing_return_df.to_csv("backtest_stoch.csv")


# Prepare DataFrame for metrics
metrics = [
    'annual_return',
    'cumulative_returns',
    'annual_volatility',
    'sharpe_ratio'#,
    #'sortino_ratio'
]

columns = ['backtest']

# Initialize the DataFrame with index set to evaluation metrics and column as `Backtest` (just like PyFolio)
#portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)
AMZN_portfolio_evaluation_df = pd.DataFrame({ 'backtest': [np.nan] * 4 }, index=metrics)

# Calculate cumulative return
AMZN_portfolio_evaluation_df.loc['cumulative_returns'] = faangm_closing_return_df['AMZN_portfolio_cumulative_returns'][-1]

# Calculate annualized return
AMZN_portfolio_evaluation_df.loc['annual_return'] = (
    faangm_closing_return_df['AMZN_portfolio_daily_returns'].mean() * 252 * 8 * 60
)

# Calculate annual volatility
AMZN_portfolio_evaluation_df.loc['annual_volatility'] = (
    faangm_closing_return_df['AMZN_portfolio_daily_returns'].std() * np.sqrt(252 * 8 * 60)
)

# Calculate Sharpe Ratio
AMZN_portfolio_evaluation_df.loc['sharpe_ratio'] = (
    faangm_closing_return_df['AMZN_portfolio_daily_returns'].mean() * 252 * 8 * 60) / (
    faangm_closing_return_df['AMZN_portfolio_daily_returns'].std() * np.sqrt(252 * 8 * 60)
)


# # Calculate Downside Return
# sortino_ratio_df = faangm_closing_return_df[['AMZN_portfolio_daily_returns']].copy()
# sortino_ratio_df.loc[:,'downside_returns'] = 0

# target = 0
# mask = sortino_ratio_df['AMZN_portfolio_daily_returns'] < target
# sortino_ratio_df.loc[mask, 'downside_returns'] = sortino_ratio_df['AMZN_portfolio_daily_returns']**2

# # Calculate Sortino Ratio
# down_stdev = np.sqrt(sortino_ratio_df['downside_returns'].mean()) * np.sqrt(252 * 8 * 60) # Annualizing
# expected_return = sortino_ratio_df['AMZN_portfolio_daily_returns'].mean() * 252 * 8 * 60 # Annualizing
# sortino_ratio = expected_return/down_stdev

# AMZN_portfolio_evaluation_df.loc['sortino_ratio'] = sortino_ratio

#################################################################
# RSI
#################################################################

def rsi(df, periods = 13, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['AMZN'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

faangm_closing_return_df_rsi = faangm_closing_return_df_.copy()
faangm_closing_return_df_rsi['RSI']= rsi(faangm_closing_return_df_rsi)
faangm_closing_return_df_rsi.dropna(inplace=True)

df_1=faangm_closing_return_df_rsi.copy()

def implement_rsi_strategy(prices, rsi):    
    rsi_signal = []
    signal = 0

    for i in range(len(prices)):
        if (rsi[i] < 31) :
            if signal != 1:
                signal = 1                
                rsi_signal.append(signal)
            else:
                rsi_signal.append(0)
        elif (rsi[i] > 62):
            if (signal == 1):
                signal = -1                
                rsi_signal.append(signal)
            else:
                rsi_signal.append(0)
        else:
            rsi_signal.append(0)
            
    return rsi_signal
            
rsi_signal = implement_rsi_strategy(df_1["AMZN"], df_1["RSI"])

df_1["AMZN_rsi_entry_exit"]= rsi_signal
df_1["AMZN_rsi_balance"]= df_1["AMZN_rsi_entry_exit"].cumsum()



# Drop all NaN values from the DataFrame
df_1 = df_1.dropna()

df_1.to_csv("buysell_rsi.csv")


# Set initial capital
initial_capital = float(1000000)

# Set the share size
share_size = 500

# Take a 500 share position where the dual moving average crossover is 1 (sma_fast is greater than sma_slow)
df_1['AMZN_position'] = share_size * df_1['AMZN_rsi_entry_exit']

# Find the points in time where a 500 share position is bought or sold
df_1['AMZN_entry_exit_position'] = df_1['AMZN_position'].diff()

# Multiply share price by entry/exit positions and get the cumulatively sum
df_1['AMZN_portfolio_holdings'] = df_1['AMZN'] * df_1['AMZN_entry_exit_position'].cumsum()

# Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio
df_1['AMZN_portfolio_cash'] = initial_capital - (df_1['AMZN'] * df_1['AMZN_entry_exit_position']).cumsum()

# Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments)
df_1['AMZN_portfolio_total'] = df_1['AMZN_portfolio_cash'] + df_1['AMZN_portfolio_holdings']

# Calculate the portfolio daily returns
df_1['AMZN_portfolio_daily_returns'] = df_1['AMZN_portfolio_total'].pct_change()

# Calculate the cumulative returns
df_1['AMZN_portfolio_cumulative_returns'] = (1 + df_1['AMZN_portfolio_daily_returns']).cumprod() - 1

# Drop all NaN values from the DataFrame
df_1 = df_1.dropna()

df_1.to_csv("backtest_rsi.csv")


# Prepare DataFrame for metrics
metrics = [
    'annual_return',
    'cumulative_returns',
    'annual_volatility',
    'sharpe_ratio'#,
    #'sortino_ratio'
]

columns = ['backtest']

# Initialize the DataFrame with index set to evaluation metrics and column as `Backtest` (just like PyFolio)
#portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)
AMZN_portfolio_evaluation_df_rsi = pd.DataFrame({ 'backtest': [np.nan] * 4 }, index=metrics)

# Calculate cumulative return
AMZN_portfolio_evaluation_df_rsi.loc['cumulative_returns'] = df_1['AMZN_portfolio_cumulative_returns'][-1]

# Calculate annualized return
AMZN_portfolio_evaluation_df_rsi.loc['annual_return'] = (
    df_1['AMZN_portfolio_daily_returns'].mean() * 252 * 8 * 60
)

# Calculate annual volatility
AMZN_portfolio_evaluation_df_rsi.loc['annual_volatility'] = (
    df_1['AMZN_portfolio_daily_returns'].std() * np.sqrt(252 * 8 * 60)
)

# Calculate Sharpe Ratio
AMZN_portfolio_evaluation_df_rsi.loc['sharpe_ratio'] = (
    df_1['AMZN_portfolio_daily_returns'].mean() * 252 * 8 * 60) / (
    df_1['AMZN_portfolio_daily_returns'].std() * np.sqrt(252 * 8 * 60)
)

# # Calculate Downside Return
# sortino_ratio_df = df_1[['AMZN_portfolio_daily_returns']].copy()
# sortino_ratio_df.loc[:,'downside_returns'] = 0

# target = 0
# mask = sortino_ratio_df['AMZN_portfolio_daily_returns'] < target
# sortino_ratio_df.loc[mask, 'downside_returns'] = sortino_ratio_df['AMZN_portfolio_daily_returns']**2

# # Calculate Sortino Ratio
# down_stdev = np.sqrt(sortino_ratio_df['downside_returns'].mean()) * np.sqrt(252 * 8 * 60) # Annualizing
# expected_return = sortino_ratio_df['AMZN_portfolio_daily_returns'].mean() * 252 * 8 * 60 # Annualizing
# sortino_ratio = expected_return/down_stdev

# AMZN_portfolio_evaluation_df_rsi.loc['sortino_ratio'] = sortino_ratio

#################################################################
# Bollinger Bands
#################################################################

df_boll = faangm_closing_return_df_.copy()

df_boll['SMA'] = df_boll.AMZN.rolling(window=17).mean()
df_boll['stddev'] = df_boll.AMZN.rolling(window=17).std()
df_boll['Upper'] = df_boll.SMA +2* df_boll.stddev
df_boll['Lower'] = df_boll.SMA -2* df_boll.stddev
df_boll.dropna(inplace=True)

def implement_bollinger_strategy(prices, lower, upper):    
    bollinger_signal = []
    signal = 0
    

    for i in range(len(prices)):
        if (lower[i] > prices[i]) :
            if signal != 1:
                signal = 1                
                bollinger_signal.append(signal)
            else:
                bollinger_signal.append(0)
        elif (upper[i] < prices[i]):
            if signal ==1:
                signal = -1                
                bollinger_signal.append(signal)
            else:
                bollinger_signal.append(0)
        else:
            bollinger_signal.append(0)
            
    return bollinger_signal
            
bollinger_signal = implement_bollinger_strategy(df_boll["AMZN"],df_boll["Lower"], df_boll["Upper"])

df_boll["AMZN_bollinger_entry_exit"]= bollinger_signal
df_boll["AMZN_balance"]= df_boll["AMZN_bollinger_entry_exit"].cumsum()

df_boll.dropna(inplace=True)

df_boll.to_csv("buysell_boll.csv")


# Set initial capital
initial_capital = float(1000000)

# Set the share size
share_size = 500

# Take a 500 share position where the dual moving average crossover is 1 (sma_fast is greater than sma_slow)
df_boll['AMZN_position'] = share_size * df_boll['AMZN_bollinger_entry_exit']

# Find the points in time where a 500 share position is bought or sold
df_boll['AMZN_entry_exit_position'] = df_boll['AMZN_position'].diff()

# Multiply share price by entry/exit positions and get the cumulatively sum
df_boll['AMZN_portfolio_holdings'] = df_boll['AMZN'] * df_boll['AMZN_entry_exit_position'].cumsum()

# Subtract the initial capital by the portfolio holdings to get the amount of liquid cash in the portfolio
df_boll['AMZN_portfolio_cash'] = initial_capital - (df_boll['AMZN'] * df_boll['AMZN_entry_exit_position']).cumsum()

# Get the total portfolio value by adding the cash amount by the portfolio holdings (or investments)
df_boll['AMZN_portfolio_total'] = df_boll['AMZN_portfolio_cash'] + df_boll['AMZN_portfolio_holdings']

# Calculate the portfolio daily returns
df_boll['AMZN_portfolio_daily_returns'] = df_boll['AMZN_portfolio_total'].pct_change()

# Calculate the cumulative returns
df_boll['AMZN_portfolio_cumulative_returns'] = (1 + df_boll['AMZN_portfolio_daily_returns']).cumprod() - 1

# Drop all NaN values from the DataFrame
df_boll = df_boll.dropna()

df_boll.to_csv("backtest_boll.csv")

# Prepare DataFrame for metrics
metrics = [
    'annual_return',
    'cumulative_returns',
    'annual_volatility',
    'sharpe_ratio'#,
    #'sortino_ratio'
]

columns = ['backtest']

# Initialize the DataFrame with index set to evaluation metrics and column as `Backtest` (just like PyFolio)
#portfolio_evaluation_df = pd.DataFrame(index=metrics, columns=columns)
AMZN_portfolio_evaluation_df_boll = pd.DataFrame({ 'backtest': [np.nan] * 4 }, index=metrics)

# Calculate cumulative return
AMZN_portfolio_evaluation_df_boll.loc['cumulative_returns'] = df_boll['AMZN_portfolio_cumulative_returns'][-1]

# Calculate annualized return
AMZN_portfolio_evaluation_df_boll.loc['annual_return'] = (
    df_boll['AMZN_portfolio_daily_returns'].mean() * 252 * 8 * 60
)

# Calculate annual volatility
AMZN_portfolio_evaluation_df_boll.loc['annual_volatility'] = (
    df_boll['AMZN_portfolio_daily_returns'].std() * np.sqrt(252 * 8 * 60)
)

# Calculate Sharpe Ratio
AMZN_portfolio_evaluation_df_boll.loc['sharpe_ratio'] = (
    df_boll['AMZN_portfolio_daily_returns'].mean() * 252 * 8 * 60) / (
    df_boll['AMZN_portfolio_daily_returns'].std() * np.sqrt(252 * 8 * 60)
)

# # Calculate Downside Return
# sortino_ratio_df = df_boll[['AMZN_portfolio_daily_returns']].copy()
# sortino_ratio_df.loc[:,'downside_returns'] = 0

# target = 0
# mask = sortino_ratio_df['AMZN_portfolio_daily_returns'] < target
# sortino_ratio_df.loc[mask, 'downside_returns'] = sortino_ratio_df['AMZN_portfolio_daily_returns']**2

# # Calculate Sortino Ratio
# down_stdev = np.sqrt(sortino_ratio_df['downside_returns'].mean()) * np.sqrt(252 * 8 * 60) # Annualizing
# expected_return = sortino_ratio_df['AMZN_portfolio_daily_returns'].mean() * 252 * 8 * 60 # Annualizing
# sortino_ratio = expected_return/down_stdev

# AMZN_portfolio_evaluation_df_boll.loc['sortino_ratio'] = sortino_ratio



#################################################################
#################################################################
#################################################################
#################################################################
# Trading Dashboard
#################################################################
#################################################################
#################################################################

st.set_page_config(layout="wide")

st.header("Trading Dashboard")
st.write('The following plots provide insight into analysis done using Prices of a stock to determine the Portfolio gains/losses when different Algorithmic Trading Strategies are used.')

#################################################################
# Plots for Tradign indicators and signals
################################################################# 

st.subheader("Trading Indicators and Signals")

col1, col2 = st.columns(2)

with col1:
    
    #################################################################
    # Stoch + SMA
    #################################################################

    with st.expander("Stoch + SMA"):

        df=faangm_closing_return_df.copy()
        
        #################################################################
        # Plot Indicators and Signals
        #################################################################

        fig = px.line(df, x=df.index, y=['AMZN','Stoch_K','Stoch_D','SMA_Fast','SMA_Slow','Overbought','Oversold'], title='Stochastic Oscillator and Simple Moving Average',width=1200, height=500)

        exit_line= go.Scatter(x=df[df['stoch_oscl_exit'] == -1.0]['AMZN'].index, y=df[df['stoch_oscl_exit'] == -1.0]['AMZN'], mode="markers", marker=dict(size=10, color="Red"), marker_symbol='triangle-down',name="Sell")
        fig.add_trace(exit_line)

        entry_line= go.Scatter(x=df[df['stoch_oscl_exit'] == 1.0]['AMZN'].index, y=df[df['stoch_oscl_exit'] == 1.0]['AMZN'], mode="markers", marker=dict(size=10, color="Green"), marker_symbol='triangle-up',name="Buy")
        fig.add_trace(entry_line)  

        #fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))


        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1))
        fig.update_layout(legend_title_text='Indicators/Signals')
        
        fig.update_layout(
            #title="Plot Title", 
            xaxis_title="Time",  
            yaxis_title="Price"#, 
            #legend_title="Legend Title",  
            #font=dict( family="Courier New, monospace", size=18, color="RebeccaPurple") 
        )


        st.plotly_chart(fig, use_container_width=True)
        
       
        
        
        
    #################################################################
    # Bollinger Bands
    #################################################################

    with st.expander("Bollinger"):
    #with st.container():
        fig = px.line(df_boll, x=df_boll.index, y=['AMZN','Lower', 'Upper'], title='Bollinger Bands',width=1200, height=500)
        exit_line= go.Scatter(x=df_boll[df_boll['AMZN_bollinger_entry_exit'] == -1.0]['AMZN'].index, y=df_boll[df_boll['AMZN_bollinger_entry_exit'] == -1.0]['AMZN'], mode="markers", marker=dict(size=10, color="Red"), marker_symbol='triangle-down',name="Sell")
        fig.add_trace(exit_line)
        entry_line= go.Scatter(x=df_boll[df_boll['AMZN_bollinger_entry_exit'] == 1.0]['AMZN'].index, y=df_boll[df_boll['AMZN_bollinger_entry_exit'] == 1.0]['AMZN'], mode="markers", marker=dict(size=10, color="Green"), marker_symbol='triangle-up',name="Buy")
        fig.add_trace(entry_line)  

        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1))
        fig.update_layout(legend_title_text='Indicators/Signals')
        
        fig.update_layout(
            #title="Plot Title", 
            xaxis_title="Time",  
            yaxis_title="Price"#, 
            #legend_title="Legend Title",  
            #font=dict( family="Courier New, monospace", size=18, color="RebeccaPurple") 
        )

        st.plotly_chart(fig, use_container_width=True)


        
        
        

with col2:
    
    
    #################################################################
    # RSI
    #################################################################
    with st.expander("RSI"):
        fig = px.line(df_1, x=df_1.index, y=['AMZN','RSI'], title='Relative Strength Index',width=1200, height=500)
        exit_line= go.Scatter(x=df_1[df_1['AMZN_rsi_entry_exit'] == -1.0]['AMZN'].index, y=df_1[df_1['AMZN_rsi_entry_exit'] == -1.0]['AMZN'], mode="markers", marker=dict(size=10, color="Red"), marker_symbol='triangle-down',name="Sell")
        fig.add_trace(exit_line)
        entry_line= go.Scatter(x=df_1[df_1['AMZN_rsi_entry_exit'] == 1.0]['AMZN'].index, y=df_1[df_1['AMZN_rsi_entry_exit'] == 1.0]['AMZN'], mode="markers", marker=dict(size=10, color="Green"), marker_symbol='triangle-up',name="Buy")
        fig.add_trace(entry_line)  

        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1))
        fig.update_layout(legend_title_text='Indicators/Signals')
        
        fig.update_layout(
            #title="Plot Title", 
            xaxis_title="Time",  
            yaxis_title="Price"#, 
            #legend_title="Legend Title",  
            #font=dict( family="Courier New, monospace", size=18, color="RebeccaPurple") 
        )

        st.plotly_chart(fig, use_container_width=True)

        
#################################################################
# Plots for Portfolio Performance and Backtesting
#################################################################       
        
        
st.subheader("Portfolio Performance and Backtesting")

col1, col2 = st.columns(2)

with col1:
    
    
    #################################################################
    # Stoch + SMA
    #################################################################

    with st.expander("Stoch + SMA"):
         #################################################################
        # Plot total portfolio value
        #################################################################       
        
        fig = px.line(df, x=df.index, y=['AMZN_portfolio_total'], title='Stochastic Oscillator and Simple Moving Averge',width=1200, height=370)

        exit_line= go.Scatter(x=df[df['stoch_oscl_exit'] == -1.0]['AMZN_portfolio_total'].index, y=df[df['stoch_oscl_exit'] == -1.0]['AMZN_portfolio_total'], mode="markers", marker=dict(size=10, color="Red"), marker_symbol='triangle-down',name="Sell")
        fig.add_trace(exit_line)

        entry_line= go.Scatter(x=df[df['stoch_oscl_exit'] == 1.0]['AMZN_portfolio_total'].index, y=df[df['stoch_oscl_exit'] == 1.0]['AMZN_portfolio_total'], mode="markers", marker=dict(size=10, color="Green"), marker_symbol='triangle-up',name="Buy")
        fig.add_trace(entry_line)  

        fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1))
        fig.update_layout(legend_title_text='Portfolio Insights')

        fig.update_layout(
                #title="Plot Title", 
                xaxis_title="Time",  
                yaxis_title="Portfolio Value"#, 
                #legend_title="Legend Title",  
                #font=dict( family="Courier New, monospace", size=18, color="RebeccaPurple") 
            )


        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(AMZN_portfolio_evaluation_df)
        
    #################################################################
    # Bollinger Bands
    #################################################################

    with st.expander("Bollinger"):
        fig = px.line(df_boll, x=df_boll.index, y=['AMZN_portfolio_total'], title='Bollinger Bands',width=1200, height=370)
        exit_line= go.Scatter(x=df_boll[df_boll['AMZN_bollinger_entry_exit'] == -1.0]['AMZN_portfolio_total'].index, y=df_boll[df_boll['AMZN_bollinger_entry_exit'] == -1.0]['AMZN_portfolio_total'], mode="markers", marker=dict(size=10, color="Red"), marker_symbol='triangle-down',name="Sell")
        fig.add_trace(exit_line)
        entry_line= go.Scatter(x=df_boll[df_boll['AMZN_bollinger_entry_exit'] == 1.0]['AMZN_portfolio_total'].index, y=df_boll[df_boll['AMZN_bollinger_entry_exit'] == 1.0]['AMZN_portfolio_total'], mode="markers", marker=dict(size=10, color="Green"), marker_symbol='triangle-up',name="Buy")
        fig.add_trace(entry_line)  

        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1))
        fig.update_layout(legend_title_text='Portfolio Insights')
        
        fig.update_layout(
            #title="Plot Title", 
            xaxis_title="Time",  
            yaxis_title="Portfolio Value"#, 
            #legend_title="Legend Title",  
            #font=dict( family="Courier New, monospace", size=18, color="RebeccaPurple") 
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(AMZN_portfolio_evaluation_df_boll)
        
with col2:
    #################################################################
    # RSI
    #################################################################
    with st.expander("RSI"):
        fig = px.line(df_1, x=df_1.index, y=['AMZN_portfolio_total'], title='Relative Strength Index',width=1200, height=370)
        exit_line= go.Scatter(x=df_1[df_1['AMZN_rsi_entry_exit'] == -1.0]['AMZN_portfolio_total'].index, y=df_1[df_1['AMZN_rsi_entry_exit'] == -1.0]['AMZN_portfolio_total'], mode="markers", marker=dict(size=10, color="Red"), marker_symbol='triangle-down',name="Sell")
        fig.add_trace(exit_line)
        entry_line= go.Scatter(x=df_1[df_1['AMZN_rsi_entry_exit'] == 1.0]['AMZN_portfolio_total'].index, y=df_1[df_1['AMZN_rsi_entry_exit'] == 1.0]['AMZN_portfolio_total'], mode="markers", marker=dict(size=10, color="Green"), marker_symbol='triangle-up',name="Buy")
        fig.add_trace(entry_line)  

        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1))
        fig.update_layout(legend_title_text='Portfolio Insights')
        
        fig.update_layout(
            #title="Plot Title", 
            xaxis_title="Time",  
            yaxis_title="Portfolio Value"#, 
            #legend_title="Legend Title",  
            #font=dict( family="Courier New, monospace", size=18, color="RebeccaPurple") 
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(AMZN_portfolio_evaluation_df_rsi)
        
    
            
            
# chart = st.line_chart(last_rows)    
# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()

# for i in range(1, 101):
#     new_rows= faangm_closing_return_df_.iloc[i:i+1,:]
#     #hvplot_chart_new_rows= new_rows.hvplot()
#     #last_rows.add_rows(new_rows)
#     status_text.text("%i%% Complete" % (i))
#     chart.add_rows(last_rows=new_rows)
#     #st.bokeh_chart(hv.render((new_rows).hvplot(), backend='bokeh'))
#     progress_bar.progress(i)
#     last_rows = new_rows
#     time.sleep(0.01)

# progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.


st.button("Re-run")