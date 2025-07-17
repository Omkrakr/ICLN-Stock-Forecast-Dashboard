import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import probplot
from plotly.subplots import make_subplots
import plotly.subplots as sp


st.set_page_config(page_title="ICLN Stock Forecast Dashboard", layout="wide")
st.title("📈 Time Series Forecasting Dashboard – ICLN Stock")

uploaded_file = st.sidebar.file_uploader("Upload ICLN CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.date_range(start='2008-01-01', periods=len(df), freq='D')
    df.set_index('Date', inplace=True)

    # -------------------- Sidebar --------------------
    st.sidebar.header("🔧 Controls")
    start_date = st.sidebar.date_input("Start Date", df.index.min())
    end_date = st.sidebar.date_input("End Date", df.index.max())
    df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]

    st.subheader("1. Close Price with Moving Averages")
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA30'] = df['close'].rolling(window=30).mean()
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['close'], name='close'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['MA7'], name='7-day MA'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['MA30'], name='30-day MA'))
    fig1.update_layout(title="Close Price with Moving Averages")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2. Volume vs Close Price")
    fig2 = sp.make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=df.index, y=df['close'], name="Close Price"), secondary_y=False)
    fig2.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume"), secondary_y=True)
    fig2.update_layout(title="Close vs Volume")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. Volatility Zones using Standard Deviation")
    mean = df['close'].mean()
    std = df['close'].std()
    upper = mean + 2 * std
    lower = mean - 2 * std
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df['close'], name='close'))
    fig3.add_trace(go.Scatter(x=df.index, y=[upper] * len(df), name='Upper Band', line=dict(dash='dot')))
    fig3.add_trace(go.Scatter(x=df.index, y=[lower] * len(df), name='Lower Band', line=dict(dash='dot')))
    fig3.update_layout(title="Volatility Zones")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("4. Seasonal Decomposition (Trend/Seasonality/Residual)")
    result = seasonal_decompose(df['close'], model='additive', period=30)
    st.line_chart(result.trend.dropna(), use_container_width=True)
    st.line_chart(result.seasonal.dropna(), use_container_width=True)
    st.line_chart(result.resid.dropna(), use_container_width=True)

    st.subheader("5. ADF Test (Stationarity Check)")
    result_adf = adfuller(df['close'].dropna())
    st.write(f"ADF Statistic: {result_adf[0]:.4f}")
    st.write(f"p-value: {result_adf[1]:.4f}")
    st.success(" Series is Stationary" if result_adf[1] < 0.05 else " Series is Non-Stationary")

    st.subheader("6. Rolling Statistics")
    df['Rolling Mean'] = df['close'].rolling(30).mean()
    df['Rolling STD'] = df['close'].rolling(30).std()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=df['close'], name='close'))
    fig4.add_trace(go.Scatter(x=df.index, y=df['Rolling Mean'], name='Rolling Mean'))
    fig4.add_trace(go.Scatter(x=df.index, y=df['Rolling STD'], name='Rolling Std'))
    fig4.update_layout(title="Rolling Mean & Standard Deviation")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("7. OHLCV Correlation Heatmap")
    st.write("Correlation between Open, High, Low, Close, and Volume")
    fig6, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df[['open', 'high', 'low', 'close', 'volume']].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig6)

    st.subheader("8. Histogram of Daily Returns")
    df['returns'] = df['close'].pct_change()
    st.bar_chart(df['returns'])

    st.subheader("9.Outlier Detection (Boxplot & KDE)")
    fig5, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(df['volume'], ax=ax[0])
    sns.kdeplot(df['close'], ax=ax[1])
    st.pyplot(fig5)
else:
    st.warning("Please upload your cleaned stock CSV to begin visualization")

    st.subheader("10. Forecast Comparison Across Models")
    forecast_file = st.sidebar.file_uploader("Upload your Forecast CSV file", type=["csv"], key="forecast")
    if forecast_file:
        forecast_df = pd.read_csv(forecast_file)
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in ['ARIMA', 'SARIMA', 'PROPHET', 'LSTM']:
            ax.plot(forecast_df['Date'], forecast_df[col], label=col)
        ax.plot(forecast_df['Date'], forecast_df['Actual'], label='Actual', color='black', linewidth=2)
        ax.legend()
        ax.set_title("Forecasts vs Actuals")
        st.pyplot(fig)









