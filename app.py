import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer

st.title("📈 BTST Call Generator")

# --- Step 1: Load Stock List
try:
    stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
except FileNotFoundError:
    st.error("Error: stocklist.xlsx file not found. Please make sure it's in the same directory.")
    st.stop()

selected_sheet = st.selectbox("Select Sheet", stock_sheets)

df_stocks = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
stock_symbols = df_stocks['Symbol'].tolist()

# --- Step 2: Fetch Market & Stock Data
today = datetime.today()
yesterday = today - timedelta(days=5)  # weekends adjustment

@st.cache_data(ttl=3600)
def fetch_yfinance_data(symbols):
    data = {}
    for sym in symbols:
        df = yf.download(sym + ".NS", start=yesterday.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'), interval='1d')
        if not df.empty:
            data[sym] = df
    return data

data_all = fetch_yfinance_data(stock_symbols + ["^NSEI"])

nsei_data = data_all.pop("^NSEI")
if nsei_data.empty:
    st.warning("Could not fetch NSE Index (^NSEI) data.")

# --- Step 3: Fetch News Data
@st.cache_data(ttl=3600)
def fetch_news_data():
    url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=500"
    response = requests.get(url)
    return response.json()['data']

news_data = fetch_news_data()
news_df = pd.DataFrame(news_data)

# --- Sentiment Scorer
def get_sentiment(headlines):
    sia = SentimentIntensityAnalyzer()
    score = 0
    count = 0
    for hl in headlines:
        s = sia.polarity_scores(hl)['compound']
        score += s
        count += 1
    return score / count if count else 0

# --- Step 4: Feature Engineering
btst_candidates = []

for symbol, df in data_all.items():
    try:
        today_close = df['Close'][-1]
        today_open = df['Open'][-1]
        today_high = df['High'][-1]
        today_low = df['Low'][-1]
        today_volume = df['Volume'][-1]
        
        # price & volume momentum
        price_change_pct = ((today_close - today_open) / today_open) * 100
        close_near_high = 1 if (today_close >= today_high * 0.98) else 0

        avg_volume_10 = df['Volume'][-10:].mean()
        volume_ratio = today_volume / avg_volume_10 if avg_volume_10 > 0 else 0
        
        vwap_proxy = (df['Close'][-5:] * df['Volume'][-5:]).sum() / df['Volume'][-5:].sum()
        vwap_diff = today_close - vwap_proxy
        
        # news sentiment
        stock_news = news_df[news_df['headline'].str.contains(symbol, case=False, na=False)]
        sentiment = get_sentiment(stock_news['headline'].tolist())
        
        # score
        score = (
            price_change_pct * 0.4 +
            volume_ratio * 0.2 +
            close_near_high * 10 +
            vwap_diff * 0.1 +
            sentiment * 10
        )

        btst_candidates.append({
            'Symbol': symbol,
            'Price Change %': round(price_change_pct, 2),
            'Volume/Avg10': round(volume_ratio, 2),
            'Close Near High': close_near_high,
            'VWAP Diff': round(vwap_diff, 2),
            'Sentiment': round(sentiment, 2),
            'BTST Score': round(score, 2)
        })
    except Exception as e:
        st.warning(f"Error processing {symbol}: {e}")

# --- Step 5: Show Results
result_df = pd.DataFrame(btst_candidates)
result_df = result_df.sort_values(by='BTST Score', ascending=False).reset_index(drop=True)

st.subheader("📊 Recommended BTST Candidates for Tomorrow")
st.dataframe(result_df)

# Optionally: download
st.download_button("Download as CSV", result_df.to_csv(index=False), file_name="btst_candidates.csv")
