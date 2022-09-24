import streamlit as st
import datetime
import pandas as pd
import numpy as np
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


def latest_is_up(dict):
    yesterday = dict.iloc[-2].Close
    today = dict.iloc[-1].Close

    return today > yesterday


tickers = ['BTC-USD', 'ETH-USD', 'BCH-USD', 'XRP-USD', 'ZRX-USD']

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Crypto Currency Price App')

if st.button('Refresh'):
    st.session_state.clear()

choice = 'Graph...'
# create a streamlit checkbox
choice = st.sidebar.radio('Vad vill du se', ('Graph...', 'Prognos'), index=0)

if 'start_date' not in st.session_state:
    st.session_state.start_date = datetime.date(2022, 1, 1)

if 'all_tickers' not in st.session_state:
    # st.write('not loaded')
    st.session_state.all_tickers = get_all(tickers)

all_tickers = st.session_state.all_tickers

if choice == 'Graph...':
    start_date = st.date_input('Start date', st.session_state.start_date)
    st.session_state.start_date = start_date
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
    ETH['rel_dev'] = ETH.Close / ETH.Close.shift(1) - 1
    BCH['rel_dev'] = BCH.Close / BCH.Close.shift(1) - 1
    XRP['rel_dev'] = XRP.Close / XRP.Close.shift(1) - 1
    ZRX['rel_dev'] = ZRX.Close / ZRX.Close.shift(1) - 1

    ax.plot(BTC.index, BTC.rel_dev, label='BTC')
    ax.plot(ETH.index, ETH['rel_dev'], label='ETH')
    ax.plot(BCH.index, BCH['rel_dev'], label='BCH')
    ax.plot(XRP.index, XRP['rel_dev'], label='XRP')
    ax.plot(ZRX.index, ZRX['rel_dev'], label='ZRX')
    ax.legend()
    ax.set(ylabel="Price US$", title='Crypto')
    # bigger graph sizes
    fig.set_size_inches(14, 12)
    # set theme
    plt.style.use('fivethirtyeight')
    st.pyplot(fig)

# %%

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
    # st.write(BTC.iloc[-2:].Close)
    col1.metric("Dagens pris $", str(dagens), latest)
    col1.metric(tomorrow, "", "+ ")
    col1.metric(day_after, "", "+ ")

    col2.markdown('## Ether')
    ETH = all_tickers['ETH-USD']
    dagens = round(ETH.iloc[-1].Close, 1)
    latest = "+ " if latest_is_up(ETH) else "- "
    # st.write(ETH.iloc[-2:].Close)
    col2.metric("Dagens pris $", str(dagens), latest)
    col2.metric(tomorrow, "", "- ")
    col2.metric(day_after, "", "+ ")

    col3.markdown('## BCH')
    BCH = all_tickers['BCH-USD']
    dagens = round(BCH.iloc[-1].Close, 2)
    latest = "+ " if latest_is_up(BCH) else "- "
    # st.write(BCH.iloc[-2:].Close)
    col3.metric("Dagens pris $", str(dagens), latest)
    col3.metric(tomorrow, "", "+ ")
    col3.metric(day_after, "", "+ ")

    col4, col5, col6 = st.columns(3)
    col4.markdown('## ZRX')
    ZRX = all_tickers['ZRX-USD']
    dagens = round(ZRX.iloc[-1].Close, 3)
    latest = "+ " if latest_is_up(ZRX) else "- "
    # st.write(ZRX.iloc[-2:].Close)
    col4.metric("Dagens pris $", str(dagens), latest)
    col4.metric(tomorrow, "", "+")
    col4.metric(day_after, "", "+ ")

    col5.markdown('## 0x')
    XRP = all_tickers['XRP-USD']
    dagens = round(XRP.iloc[-1].Close, 3)
    latest = "+ " if latest_is_up(XRP) else "- "
    # st.write(XRP.iloc[-2:].Close)
    col5.metric("Dagens pris $", str(dagens), latest)
    col5.metric(tomorrow, "", "+ ")
    col5.metric(day_after, "", "+ ")
    col6.markdown(""" """)
