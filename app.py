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

st.set_page_config(page_title="Stock Prediction", layout="wide")

st.title('📈 Stock Price Prediction App')
st.sidebar.info('Stock Prediction using Machine Learning')

# ------------------ DOWNLOAD DATA ------------------
@st.cache_data
def download_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    return df

# ------------------ INPUT ------------------
symbol = st.sidebar.text_input('Enter Stock Symbol', value='SPY').upper()

today = datetime.date.today()
duration = st.sidebar.number_input('Enter duration (days)', value=3000)

start_date = st.sidebar.date_input('Start Date', today - datetime.timedelta(days=duration))
end_date = st.sidebar.date_input('End Date', today)

data = download_data(symbol, start_date, end_date)

if data.empty:
    st.error("No data found. Try another symbol.")
    st.stop()

scaler = StandardScaler()

# ------------------ MAIN ------------------
def main():
    menu = st.sidebar.selectbox('Menu', ['Visualize', 'Recent Data', 'Predict'])

    if menu == 'Visualize':
        tech_indicators()
    elif menu == 'Recent Data':
        show_data()
    else:
        predict()

# ------------------ TECH INDICATORS ------------------
def tech_indicators():
    st.header('📊 Technical Indicators')

    option = st.radio('Choose Indicator',
                      ['Close', 'Bollinger Bands', 'MACD', 'RSI', 'SMA', 'EMA'])

    df = data.copy()
    df = df[['Close']].dropna()

    # ✅ FIX: Ensure 1D series
    close = df['Close'].values.flatten()
    close = pd.Series(close)

    # Bollinger Bands
    bb_indicator = BollingerBands(close=close)
    bb_h = bb_indicator.bollinger_hband()
    bb_l = bb_indicator.bollinger_lband()

    bb_df = pd.DataFrame({
        'Close': close,
        'BB High': bb_h,
        'BB Low': bb_l
    })

    # Other indicators
    macd = MACD(close=close).macd()
    rsi = RSIIndicator(close=close).rsi()
    sma = SMAIndicator(close=close, window=14).sma_indicator()
    ema = EMAIndicator(close=close).ema_indicator()

    # Display
    if option == 'Close':
        st.line_chart(close)

    elif option == 'Bollinger Bands':
        st.line_chart(bb_df)

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
    st.header('📋 Recent Data')
    st.dataframe(data.tail(10))

# ------------------ PREDICT ------------------
def predict():
    st.header('🔮 Stock Prediction')

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

    df['preds'] = df['Close'].shift(-num)
    df.dropna(inplace=True)

    X = df[['Close']].values
    y = df['preds'].values

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    st.success(f"R2 Score: {r2_score(y_test, preds):.2f}")
    st.success(f"MAE: {mean_absolute_error(y_test, preds):.2f}")

    # Forecast
    X_forecast = scaler.transform(data[['Close']].values)[-num:]
    forecast = model.predict(X_forecast)

    st.subheader("📅 Future Predictions")
    for i, val in enumerate(forecast, 1):
        st.write(f"Day {i}: {round(val, 2)}")

# ------------------ RUN ------------------
if __name__ == "__main__":
    main()
    