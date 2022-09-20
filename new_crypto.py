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


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Crypto Currency Price App')
choice='Graph...'
# create a streamlit checkbox
choice = st.sidebar.radio('Vad vill du se',('Graph...', 'Prognose'), index=0)
if choice == 'Graph...':
    tickers = ['BTC-USD','ETH-USD','BCH-USD','XRP-USD','ZRX-USD']
    ticker='ETH-USD'
    st.write('Graph here...')
    BTC=get_data(ticker)
    fig, ax = plt.subplots()

    ax.plot(BTC.index, BTC['Close'], label=ticker[:3])
    ax.legend()
    ax.set( ylabel="Price US$", title=ticker) 
    st.pyplot(fig)
    
if choice == 'Prognose':
    # day name today
    today = datetime.datetime.today().strftime("%A")
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%A")
    day_after = (datetime.date.today() + datetime.timedelta(days=2)).strftime("%A")
    """Prognos för i morgon och övermorgon"""
    col1, col2, col3 = st.columns(3)
    col1.markdown('## Bitcoin')
    col1.metric("Dagens pris", "123 $", "- ")
    col1.metric(tomorrow, "", "+ ")
    col1.metric(day_after, "", "+ ")
    col2.markdown('## Ether')
    col2.metric("Dagens pris", "123 $", "- ")
    col2.metric(tomorrow, "", "- ")
    col2.metric(day_after, "", "+ ")
    col3.markdown('## BCH')
    col3.metric("Dagens pris", "123 $", "- ")
    col3.metric(tomorrow, "", "+ ")
    col3.metric(day_after, "", "+ ")

    col4, col5, col6 = st.columns(3)
    col4.markdown('## ZRX')
    col4.metric("Dagens pris", "123 $", "- ")
    col4.metric(tomorrow, "", "+")
    col4.metric(day_after, "", "+ ")
    col5.markdown('## 0x')
    col5.metric("Dagens pris", "123 $", "- ")
    col5.metric(tomorrow, "", "+ ")
    col5.metric(day_after, "", "+ ")
    col6.markdown("""Nothing to see here,  
                Move along!""")