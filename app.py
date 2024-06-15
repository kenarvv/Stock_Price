import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import date

# Initialize session state to store variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.session_state.model_trained = True
    return model

# Function to calculate volatility
def calculate_volatility(data):
    daily_returns = np.log(data["Adj Close"].pct_change() + 1)
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    return annualized_volatility

# Function to simulate GBM
def gbm_sim(spot_price, volatility, steps, model, features, data):
    dt = 1 / 252
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
start_date = st.date_input("Tanggal mulai", value=date(2022, 1, 1))
end_date = st.date_input("Tanggal akhir", value=date(2023, 12, 31))

# Default parameters for simulation
steps = 252  # Default time period (days)
n_simulations = 100  # Default number of simulations

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
if st.session_state.model_trained and st.button("Simulate GBM"):
    try:
        volatility = calculate_volatility(st.session_state.data)
        spot_price = st.session_state.data["Adj Close"].iloc[0]
        simulated_paths = []
        for _ in range(n_simulations):
            paths, drifts = gbm_sim(spot_price, volatility, steps, st.session_state.model, st.session_state.features, st.session_state.data)
            simulated_paths.append(paths)
        simulated_df = pd.DataFrame(simulated_paths).transpose()

        # Plot results
        st.subheader("Hasil Simulasi")
        fig, ax = plt.subplots(figsize=(10, 6))
        index = st.session_state.data.index[:steps]
        for i in range(n_simulations):
            ax.plot(index, simulated_df.iloc[:, i], color='blue', alpha=0.1)
        ax.plot(index, st.session_state.data['Adj Close'][:steps], color='red', label='Actual')
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Stock Price")
        ax.set_title("Simulated Stock Price Paths")
        ax.legend()
        st.pyplot(fig)

        # Display simulated paths in table
        st.subheader("Tabel Hasil Simulasi")
        st.dataframe(simulated_df)
    except Exception as e:
        st.write("Error simulating GBM:", e)
