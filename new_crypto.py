from pytrends.request import TrendReq
import streamlit as st
import datetime
import pandas as pd
import numpy as np
import ta
import pickle
from datetime import timedelta
import yfinance as yf
from matplotlib import pyplot as plt


def get_data(ticker, start="1900-01-01", end=datetime.datetime.today()):
    data = yf.download(ticker, start=None, end=None)
    data.reset_index(inplace=True)
    data.index = pd.to_datetime(data.Date)
    data = data.drop("Date", axis=1)

    return data


def get_all(tickers):
    all_tickers = {}
    for enum, ticker in enumerate(tickers):
        all_tickers[ticker] = get_data(ticker)
    return all_tickers


def new_features(df_, ticker, target):
    df = df_.copy()
    # tidsintervall i dagar för rullande medelvärden
    # skulle helst ha med upp till 4 år men ETH har för få värden
    horizons = [2, 5, 60, 250]
    new_predictors = []
    df['stoch_k'] = ta.momentum.stochrsi_k(df[ticker], window=10)

    # Target
    # tomorrow's close price - alltså nästa dag
    df['Tomorrow'] = df[ticker].shift(-1)
    # after tomorrow's close price - alltså om två dagar
    df['After_tomorrow'] = df[ticker].shift(-2)
    df['y1'] = (df['Tomorrow'] > df[ticker]).astype(int)
    df['y2'] = (df['After_tomorrow'] > df[ticker]).astype(int)
    df.dropna(inplace=True)

    for horizon in horizons:
        rolling_averages = df.rolling(horizon, 1).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df[ticker] / rolling_averages[ticker]

        trend_column = f"Trend_{horizon}"
        df[trend_column] = df.shift(1).rolling(horizon, 1).sum()[target]

        new_predictors += [ratio_column, trend_column]

    new_predictors.append('stoch_k')
    df = df.dropna()
    return df, new_predictors


def latest_is_up(dict):
    yesterday = dict.iloc[-2].Close
    today = dict.iloc[-1].Close

    return today > yesterday


tickers = ['BTC-USD', 'ETH-USD', 'BCH-USD', 'XRP-USD', 'ZRX-USD']
ticker_names= ['Bitcoin', 'Ether', 'Bitcoin Cash', 'Ripple', '0x']

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Crypto Currency Price App')

# get google trends data
from pytrends.request import TrendReq
# st.write(tickers)
st.sidebar.title('Ticker')

pytrends = TrendReq(hl='en-US', tz=360)
pytrends.build_payload(kw_list=['Bitcoin'], timeframe='now 7-d')
df_trend = pytrends.interest_over_time()
# plot the data
fig, ax = plt.subplots()
df_trend.plot(ax=ax)
ax.set_title('Google Trends')
st.pyplot(fig)

if st.button('Refresh'):
    # delete all_tickers from sesion_state
    del st.session_state.all_tickers

choice = 'Graph...'
# create a streamlit checkbox
choice = st.sidebar.radio('Vad vill du se', ('Graph...', 'Prognos'), index=0)

if 'all_tickers' not in st.session_state:
    # st.write('not loaded')
    st.session_state.all_tickers = get_all(tickers)

all_tickers = st.session_state.all_tickers

if choice == 'Graph...':

    if 'start_date' not in st.session_state:
        st.session_state.start_date = st.date_input(
            'Start date', datetime.date(2022, 1, 1))
    else:
        st.session_state.start_date = st.date_input(
            'Start date', st.session_state.start_date)

    start_date = st.session_state.start_date

    BTC = all_tickers['BTC-USD'].query('index > @start_date')
    ETH = all_tickers['ETH-USD'].query('index > @start_date')
    BCH = all_tickers['BCH-USD'].query('index > @start_date')
    XRP = all_tickers['XRP-USD'].query('index > @start_date')
    ZRX = all_tickers['ZRX-USD'].query('index > @start_date')
    fig, ax = plt.subplots()

    BTC = BTC.rolling(60).mean()
    ETH = ETH.rolling(60).mean()
    BCH = BCH.rolling(60).mean()
    XRP = XRP.rolling(60).mean()
    ZRX = ZRX.rolling(60).mean()

    # compute relative development
    BTC['rel_dev'] = BTC.Close / BTC.Close.shift(1) - 1
    BTC.dropna(inplace=True)
    just = BTC.rel_dev.head(1).values[0]
    BTC.rel_dev -= just

    ETH['rel_dev'] = ETH.Close / ETH.Close.shift(1) - 1
    ETH.dropna(inplace=True)
    just = ETH.rel_dev.head(1).values[0]
    ETH.rel_dev -= just

    BCH['rel_dev'] = BCH.Close / BCH.Close.shift(1) - 1
    BCH.dropna(inplace=True)
    just = BCH.rel_dev.head(1).values[0]
    BCH.rel_dev -= just

    XRP['rel_dev'] = XRP.Close / XRP.Close.shift(1) - 1
    XRP.dropna(inplace=True)
    just = XRP.rel_dev.head(1).values[0]
    XRP.rel_dev -= just

    ZRX['rel_dev'] = ZRX.Close / ZRX.Close.shift(1) - 1
    ZRX.dropna(inplace=True)
    just = ZRX.rel_dev.head(1).values[0]
    ZRX.rel_dev -= just

    ax.plot(BTC.index, BTC.rel_dev, label='BTC')
    ax.plot(ETH.index, ETH['rel_dev'], label='ETH')
    ax.plot(BCH.index, BCH['rel_dev'], label='BCH')
    ax.plot(XRP.index, XRP['rel_dev'], label='XRP')
    ax.plot(ZRX.index, ZRX['rel_dev'], label='ZRX')
    ax.legend()
    ax.set(ylabel="Price US$", title='Crypto relativ utveckling från 0')
    # bigger graph sizes
    fig.set_size_inches(14, 12)
    # set theme
    plt.style.use('fivethirtyeight')
    st.pyplot(fig)

    exp = st.expander('Crypto Förkortningar')
    exp.write("""BTC = Bitcoin   
              ETH = Ethereum   
              BCH = Bitcoin Cash  
              XRP = Ripple  
              ZRX = 0x   
              """
              )

# %%


def load_and_predict(file, data, predictors):
    # pickle load the file
    loaded_model = pickle.load(open(file, 'rb'))
    return loaded_model.predict(data.iloc[-1:, :][predictors])


if choice == 'Prognos':
    # day name today
    today = datetime.datetime.today().strftime("%A")
    tomorrow = (datetime.date.today() +
                datetime.timedelta(days=1)).strftime("%A")
    day_after = (datetime.date.today() +
                 datetime.timedelta(days=2)).strftime("%A")
    """Priser i US$
    Prognos för i morgon och övermorgon"""
    # all_tickers = get_all(tickers)

    col1, col2, col3 = st.columns(3)

    col1.markdown('## Bitcoin')
    BTC = all_tickers['BTC-USD']
    dagens = round(BTC.iloc[-1].Close, 1)
    latest = "+ " if latest_is_up(BTC) else "- "
    col1.metric("Dagens pris $", str(dagens), latest)

    BTC_data1, new_predictors = new_features(BTC, 'Close', 'y1')
    tomorrow_up = load_and_predict('BTC_y1.pkl', BTC_data1, new_predictors)
    BTC_data2, new_predictors = new_features(BTC, 'Close', 'y2')
    two_days_upp = load_and_predict('BTC_y2.pkl', BTC_data2, new_predictors)
    col1.metric(tomorrow, "", "+ " if tomorrow_up else "- ")
    col1.metric(day_after, "", "+ " if two_days_upp else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col2.markdown('## Ether')
    ETH = all_tickers['ETH-USD']
    dagens = round(ETH.iloc[-1].Close, 1)
    latest = "+ " if latest_is_up(ETH) else "- "
    col2.metric("Dagens pris $", str(dagens), latest)

    ETH_data1, new_predictors = new_features(ETH, 'Close', 'y1')
    tomorrow_up = load_and_predict('ETH_y1.pkl', ETH_data1, new_predictors)
    ETH_data2, new_predictors = new_features(ETH, 'Close', 'y2')
    two_days_upp = load_and_predict('ETH_y2.pkl', ETH_data2, new_predictors)
    col2.metric(tomorrow, "", "+ " if tomorrow_up else "- ")
    col2.metric(day_after, "", "+ " if two_days_upp else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col3.markdown('## BCH')
    BCH = all_tickers['BCH-USD']
    dagens = round(BCH.iloc[-1].Close, 2)
    latest = "+ " if latest_is_up(BCH) else "- "
    col3.metric("Dagens pris $", str(dagens), latest)

    BCH_data1, new_predictors = new_features(BCH, 'Close', 'y1')
    tomorrow_up = load_and_predict('BCH_y1.pkl', BCH_data1, new_predictors)
    BCH_data2, new_predictors = new_features(BCH, 'Close', 'y2')
    two_days_upp = load_and_predict('BCH_y2.pkl', BCH_data2, new_predictors)
    col3.metric(tomorrow, "", "+ " if tomorrow_up else "- ")
    col3.metric(day_after, "", "+ " if two_days_upp else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col4, col5, col6 = st.columns(3)
    col4.markdown('## ZRX')
    ZRX = all_tickers['ZRX-USD']
    dagens = round(ZRX.iloc[-1].Close, 3)
    latest = "+ " if latest_is_up(ZRX) else "- "
    col4.metric("Dagens pris $", str(dagens), latest)

    ZRX_data1, new_predictors = new_features(ZRX, 'Close', 'y1')
    tomorrow_up = load_and_predict('ZRX_y1.pkl', ZRX_data1, new_predictors)
    ZRX_data2, new_predictors = new_features(ZRX, 'Close', 'y2')
    two_days_upp = load_and_predict('ZRX_y2.pkl', ZRX_data2, new_predictors)
    col4.metric(tomorrow, "", "+ " if tomorrow_up else "- ")
    col4.metric(day_after, "", "+ " if two_days_upp else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col5.markdown('## 0x')
    XRP = all_tickers['XRP-USD']
    dagens = round(XRP.iloc[-1].Close, 3)
    latest = "+ " if latest_is_up(XRP) else "- "
    col5.metric("Dagens pris $", str(dagens), latest)

    XRP_data1, new_predictors = new_features(XRP, 'Close', 'y1')
    tomorrow_up = load_and_predict('XRP_y1.pkl', XRP_data1, new_predictors)
    XRP_data2, new_predictors = new_features(XRP, 'Close', 'y2')
    two_days_upp = load_and_predict('XRP_y2.pkl', XRP_data2, new_predictors)

    col5.metric(tomorrow, "", "+ " if tomorrow_up else "- ")
    col5.metric(day_after, "", "+ " if two_days_upp else "- ")
    # st.info(f'{tomorrow_up}  {two_days_upp}')

    col6.markdown(""" """)
