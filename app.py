import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to load data
def load_data(ticker="BBCA.JK", start="2022-01-01", end="2023-12-31"):
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
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

# Load data button
if st.button("Load Data"):
    try:
        data, features = load_data()
        st.write("Data Loaded")
        st.write(data.head())
    except Exception as e:
        st.write("Error loading data:", e)
        data = None
        features = None

# Train model button
if data is not None and features is not None and st.button("Train Model"):
    try:
        model = train_model(data, features)
        st.write("Model trained successfully")
    except Exception as e:
        st.write("Error training model:", e)
        model = None

# Simulate GBM button
if model is not None and st.button("Simulate GBM"):
    try:
        steps = len(data)
        spot_price = data["Adj Close"].iloc[0]
        volatility = data["LogReturn"].std() * np.sqrt(252)  # Annualized volatility
        simulated_paths, drifts = gbm_sim(spot_price, volatility, steps, model, features, data)

        # Plot simulated paths
        plt.figure(figsize=(10, 6))
        index = data.index
        plt.plot(index, simulated_paths[:len(index)], label='Predicted')
        plt.plot(index, data['Adj Close'].values, label='Actual')
        plt.xlabel("Time Step")
        plt.ylabel("Stock Price")
        plt.title("Simulated Stock Price Paths")
        plt.legend()
        st.pyplot()

        # Plot drift values
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(drifts[:len(index)])
        ax[0].set_title('Predicted Drift')
        ax[1].plot(data['LogReturn'].values[:len(index)])
        ax[1].set_title('Actual Log Returns')
        ax[2].plot([abs(i - j) for (i, j) in zip(drifts, data['LogReturn'].values[:len(index)])])
        ax[2].set_title('Absolute Error between Predicted Drift and Actual Log Returns')
        st.pyplot(fig)

    except Exception as e:
        st.write("Error simulating GBM:", e)
