import streamlit as st
import pandas as pd
from datetime import date ,timedelta
import yfinance as yf
from prophet import Prophet
import prophet.plot as plot_plotly
from plotly import graph_objs as go
import pmdarima as pm
from pmdarima.arima import auto_arima
#from statsmodels.tsa.arima.model import ARIMA
from  pandas import to_datetime

start = "2020-01-01"
today = date.today().strftime("%Y-%m-%d")
st.title("Stock price prediction")
stock = ("BTC-USD","NFLX","TSLA","GC=F")
selected_stocks = st.selectbox("Select dataset for Prediction",stock)
n_years = st.slider("Years of prediction",1,5)
period = n_years * 365
@st.cache_data
def loaddata(ticker):
    df = yf.download(ticker,start,today)
    df.reset_index(inplace = True)
    return df
df = loaddata(selected_stocks)
st.subheader("Raw data")
st.write(df.tail(5))
def plot_raw_data_graph():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"],y=df["Open"],name = "stock_open"))
    fig.add_trace(go.Scatter(x=df["Date"],y=df["Close"],name = "stock_close"))
    fig.layout.update(title_text = "Time Series data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data_graph()

#Arima model
df_train = df[["Date","Close"]]
future_date_df = pd.DataFrame({'Date': pd.date_range(start=today, periods=period)})
auto_arima = pm.auto_arima(df_train["Close"], stepwise=False, seasonal=True)
forecast = auto_arima.predict(n_periods=len(future_date_df))
forecast_values = pd.DataFrame()
forecast_values["Forecast"] = forecast
forecast_values.reset_index(drop=True, inplace=True)
forecast_df = pd.DataFrame()
forecast_df["Date"] = future_date_df
forecast_df["Forecast"] = forecast_values
st.subheader("Forecast data")
st.write(forecast_df.tail(5))
st.write("Forecast Data (Arima Model)")
def plot_arima():
    fig = go.Figure()

    # Add the train data to the figure
    fig.add_trace(go.Scatter(x=df_train["Date"], y=df_train['Close'],name='Train Data'))

    # Add the forecast data to the figure
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode='lines', name='Forecast', line=dict(dash='dash')))
    fig.layout.update(title_text = "Time Series data (Forecast)",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_arima()

#Prophet model
df_train = df[["Date","Close"]]
df_train_prophet = df_train.rename(columns = {"Date":"ds","Close":"y"})
df_train_prophet['ds']= to_datetime(df_train_prophet['ds'])
m = Prophet(changepoint_prior_scale=0.5,seasonality_prior_scale=0.5)
m.fit(df_train_prophet)
future_prophet = m.make_future_dataframe(periods=period)
forecast_prophet = m.predict(future_prophet)
st.subheader("Forecast data (Prophet)")
st.write(forecast_prophet.tail(5))
def plot_prophet():
    fig = go.Figure()

    # Add the train data to the figure
    fig.add_trace(go.Scatter(x=df_train["Date"], y=df_train['Close'],name='Train Data'))

    # Add the forecast data to the figure
    fig.add_trace(go.Scatter(x=forecast_prophet["ds"], y=forecast_prophet["yhat"], mode='lines', name='Forecast', line=dict(dash='dash')))
    fig.layout.update(title_text = "Time Series data (Forecast)",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_prophet()