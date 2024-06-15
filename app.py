import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize session state to store variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'simulated_paths' not in st.session_state:
    st.session_state.simulated_paths = None
if 'drifts' not in st.session_state:
    st.session_state.drifts = None

# Function to load data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data = data.interpolate(method='linear')
    data['LogReturn'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    period = 14
    data['SMA'] = ta.trend.sma_indicator(data['Adj Close'], window=period)
    data['EMA'] = ta.trend.ema_indicator(data['Adj Close'], window=period)
    data['RSI'] = ta.momentum.rsi(data['Adj Close'], window=period)
    data = data.dropna()
    features = ['SMA', 'EMA', 'RSI']
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    return data, features

# Function to train model
def train_model(data, features):
    X = data[features]
    y = data['LogReturn']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Function to simulate GBM
def gbm_sim(spot_price, volatility, steps, model, features, data):
    dt = 1 / 252  # Daily
    paths = [spot_price]
    drift = model.predict(data[features])
    for i in range(len(drift)):
        random_shock = np.random.normal() * np.sqrt(dt)
        new_price = paths[-1] * np.exp((drift[i] - 0.5 * volatility**2) * dt + volatility * random_shock)
        paths.append(new_price)
    return paths[:-1], drift

# Streamlit interface
st.title("Simulasi Prediksi Harga Saham")
st.write("Nama Kelompok: Kelompok 1")

# Input parameters for data
ticker = st.text_input("Masukkan kode saham (contoh: BBCA.JK)", value="BBCA.JK")
start_date = st.date_input("Tanggal mulai", value=pd.to_datetime('2022-01-01'))
end_date = st.date_input("Tanggal akhir", value=pd.to_datetime('2023-12-31'))

# Default parameters for simulation
steps = 252  # Default time period (days)

# Load data button
if st.button("Load Data"):
    try:
        st.session_state.data, st.session_state.features = load_data(ticker, start_date, end_date)
        st.write("Data Loaded")
        st.write(st.session_state.data.head())
    except Exception as e:
        st.write("Error loading data:", e)
        st.session_state.data = None
        st.session_state.features = None

# Train model button
if st.session_state.data is not None and st.session_state.features is not None and st.button("Train Model"):
    try:
        st.session_state.model = train_model(st.session_state.data, st.session_state.features)
        st.write("Model trained successfully")
    except Exception as e:
        st.write("Error training model:", e)

# Simulate GBM button
if st.session_state.model is not None and st.button("Simulate GBM"):
    try:
        volatility = st.session_state.data["LogReturn"].std() * np.sqrt(252)  # Annualized volatility
        spot_price = st.session_state.data["Adj Close"].iloc[0]
        st.session_state.simulated_paths, st.session_state.drifts = gbm_sim(spot_price, volatility, steps, st.session_state.model, st.session_state.features, st.session_state.data)

        # Plot simulated paths
        st.subheader("Simulated Stock Price Paths")
        plt.figure(figsize=(10, 6))
        index = st.session_state.data.index[:len(st.session_state.simulated_paths)]
        plt.plot(index, st.session_state.simulated_paths, label='Predicted')
        plt.plot(index, st.session_state.data['Adj Close'][:len(st.session_state.simulated_paths)], label='Actual')
        plt.xlabel("Time Step")
        plt.ylabel("Stock Price")
        plt.title("Simulated vs Actual Stock Price Paths")
        plt.legend()
        st.pyplot()

        # Plot drift values
        st.subheader("Drift Values")
        labels = ['Predicted Drift', 'Actual Log Returns', 'Absolute Error']
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(st.session_state.drifts[:len(index)])
        ax[0].set_title(labels[0])
        ax[1].plot(st.session_state.data['LogReturn'][:len(index)])
        ax[1].set_title(labels[1])
        ax[2].plot([abs(i - j) for (i, j) in zip(st.session_state.drifts, st.session_state.data['LogReturn'][:len(index)])])
        ax[2].set_title(labels[2])
        plt.tight_layout()
        st.pyplot()

    except Exception as e:
        st.write("Error simulating GBM:", e)
