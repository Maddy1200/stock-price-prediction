import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App.')
st.sidebar.info("Created by Vikas Sharma")

# ------------------ DOWNLOAD DATA ------------------
@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    df = df.reset_index()
    return df

# ------------------ INPUT ------------------
option = st.sidebar.text_input('Enter Stock Symbol', value='SPY').upper()

today = datetime.date.today()
duration = st.sidebar.number_input('Enter duration (days)', value=3000)

start_date = st.sidebar.date_input('Start Date', today - datetime.timedelta(days=duration))
end_date = st.sidebar.date_input('End Date', today)

data = download_data(option, start_date, end_date)

scaler = StandardScaler()

# ------------------ MAIN ------------------
def main():
    choice = st.sidebar.selectbox('Menu', ['Visualize', 'Recent Data', 'Predict'])

    if choice == 'Visualize':
        tech_indicators()
    elif choice == 'Recent Data':
        show_data()
    else:
        predict()

# ------------------ TECH INDICATORS ------------------
def tech_indicators():
    st.header('Technical Indicators')

    option = st.radio('Choose Indicator',
                      ['Close', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    df = data.copy()

    # Ensure 1D data
    close = df['Close'].squeeze()

    # Bollinger Bands
    bb_indicator = BollingerBands(close=close)
    df['bb_h'] = bb_indicator.bollinger_hband()
    df['bb_l'] = bb_indicator.bollinger_lband()
    bb = df[['Close', 'bb_h', 'bb_l']]

    # Other Indicators
    macd = MACD(close=close).macd()
    rsi = RSIIndicator(close=close).rsi()
    sma = SMAIndicator(close=close, window=14).sma_indicator()
    ema = EMAIndicator(close=close).ema_indicator()

    # Display
    if option == 'Close':
        st.line_chart(close)

    elif option == 'BB':
        st.line_chart(bb)

    elif option == 'MACD':
        st.line_chart(macd)

    elif option == 'RSI':
        st.line_chart(rsi)

    elif option == 'SMA':
        st.line_chart(sma)

    else:
        st.line_chart(ema)

# ------------------ SHOW DATA ------------------
def show_data():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

# ------------------ PREDICT ------------------
def predict():
    st.header('Prediction')

    model_name = st.radio('Choose Model',
                          ['LinearRegression', 'RandomForest', 'ExtraTrees',
                           'KNN', 'XGBoost'])

    num = int(st.number_input('Forecast Days', value=5))

    if st.button('Predict'):
        if model_name == 'LinearRegression':
            model = LinearRegression()
        elif model_name == 'RandomForest':
            model = RandomForestRegressor()
        elif model_name == 'ExtraTrees':
            model = ExtraTreesRegressor()
        elif model_name == 'KNN':
            model = KNeighborsRegressor()
        else:
            model = XGBRegressor()

        run_model(model, num)

# ------------------ MODEL ENGINE ------------------
def run_model(model, num):
    df = data[['Close']].copy()

    # Create prediction column
    df['preds'] = df['Close'].shift(-num)

    # Drop NaN
    df.dropna(inplace=True)

    x = df[['Close']].values
    y = df['preds'].values

    # Scale
    x = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=7)

    model.fit(x_train, y_train)

    preds = model.predict(x_test)

    st.write(f"R2 Score: {r2_score(y_test, preds)}")
    st.write(f"MAE: {mean_absolute_error(y_test, preds)}")

    # Forecast
    x_forecast = scaler.transform(data[['Close']].values)[-num:]
    forecast = model.predict(x_forecast)

    st.subheader("Future Predictions")
    for i, val in enumerate(forecast, 1):
        st.write(f"Day {i}: {val}")

# ------------------ RUN ------------------
if __name__ == "__main__":
    main()