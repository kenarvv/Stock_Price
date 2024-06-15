import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

# Fungsi untuk memuat data
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

# Fungsi untuk melatih model
def train_model(data, features):
    X = data[features]
    y = data['LogReturn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

# Fungsi untuk menghitung volatilitas tahunan
def calculate_volatility(data):
    daily_returns = np.log(data["Adj Close"].pct_change() + 1)
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    return annualized_volatility

# Fungsi untuk simulasi GBM
def gbm_sim(spot_price, volatility, steps, model, features, data):
    dt = 1 / 252
    paths = [spot_price]
    drift = model.predict(data[features])

    for i in range(len(drift)):
        random_shock = np.random.normal() * np.sqrt(dt)
        new_price = paths[-1] * np.exp((drift[i] - 0.5 * volatility**2) * dt + volatility * random_shock)
        paths.append(new_price)

    return paths[:-1], drift

# Main Function
if __name__ == "__main__":
    st.title("Stock Price Prediction and Simulation")
    st.subheader("Tugas Besar Pemodelan dan Simulasi\n - Ken Arvian Narasoma(1301213387)\n - Anggi Rodesa Sasabella(1301193161)")

    ticker = st.text_input("Enter the stock ticker:", "BBCA.JK")
    start_date = st.date_input("Start date", value=pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End date", value=pd.to_datetime("2023-12-31"))

    data, features = load_data(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    model, mse, r2 = train_model(data, features)

    st.write(f"Model Evaluation: Mean Squared Error: {mse}, R-squared: {r2}")

    spot_price = data["Adj Close"].iloc[0]
    volatility = calculate_volatility(data)
    steps = len(data)
    simulated_paths, drifts = gbm_sim(spot_price, volatility, steps, model, features, data)

    st.subheader("Simulated Stock Price Paths")
    plt.figure(figsize=(10, 6))
    index = data.index
    plt.plot(index, simulated_paths[:len(index)], label='Predicted')
    plt.plot(index, data['Adj Close'].values, label='Actual')
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.title("Simulated Stock Price Paths")
    plt.legend()
    st.pyplot(plt)

    st.subheader("Drift Comparison")
    labels = ['Predicted Drift', 'Actual Drift', 'Absolute Error']
    fig, ax = plt.subplots(1, 3, figsize=(10, 2))
    ax[0].plot(drifts[:len(index)])
    ax[1].plot(data['LogReturn'].values[:len(index)])
    ax[2].plot([abs(i - j) for (i, j) in zip(drifts, data['LogReturn'].values[:len(index)])])
    _ = [ax[i].set_title(j) for i, j in enumerate(labels)]
    st.pyplot(fig)
