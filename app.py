import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime
import ta
import re

# ========== Technical Indicators ========== #
def calculate_technical_indicators(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df['price_change_pct'] = df['Close'].pct_change().fillna(0) * 100

    vol_window = min(10, len(df) - 1)
    if vol_window > 1:
        vol_avg = df['Volume'].rolling(vol_window, min_periods=1).mean()
        df['volume_change_pct'] = ((df['Volume'] / vol_avg) - 1).fillna(0) * 100
    else:
        df['volume_change_pct'] = 0

    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['vwap'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['vwap_diff'] = ((df['Close'] - df['vwap']) / (df['vwap'] + 1e-8)).fillna(0) * 100
    except:
        df['vwap'] = df['Close']
        df['vwap_diff'] = 0

    try:
        range_val = df['High'] - df['Low'] + 1e-8
        df['close_position'] = (df['Close'] - df['Low']) / range_val
        df['close_position'] = df['close_position'].fillna(0.5)
    except:
        df['close_position'] = 0.5

    try:
        rsi = ta.momentum.RSIIndicator(close=df['Close'], window=min(14, len(df) - 1))
        df['rsi'] = rsi.rsi().fillna(50)
    except:
        df['rsi'] = 50

    try:
        macd = ta.trend.MACD(close=df['Close'])
        df['macd_diff'] = macd.macd_diff().fillna(0)
    except:
        df['macd_diff'] = 0

    try:
        if len(df) > 20:
            bb = ta.volatility.BollingerBands(close=df['Close'])
            df['bb_width'] = bb.bollinger_wband().fillna(0)
        else:
            df['bb_width'] = 0
    except:
        df['bb_width'] = 0

    return df

# ========== Scoring Function ========== #
def calculate_btst_score(row):
    score = 0

    def get_scalar(value, default=0):
        try:
            if isinstance(value, pd.Series):
                return float(value.iloc[-1])
            return float(value)
        except:
            return default

    price_change = get_scalar(row.get('price_change_pct', 0))
    vol_change = get_scalar(row.get('volume_change_pct', 0))
    rsi = get_scalar(row.get('rsi', 50))
    macd_diff = get_scalar(row.get('macd_diff', 0))
    bb_width = get_scalar(row.get('bb_width', 0))
    close_pos = get_scalar(row.get('close_position', 0.5))
    vwap_diff = get_scalar(row.get('vwap_diff', 0))

    if price_change > 3:
        score += 30
    elif price_change > 2:
        score += 20
    elif price_change > 1:
        score += 10

    if vol_change > 150:
        score += 20
    elif vol_change > 100:
        score += 15
    elif vol_change > 50:
        score += 10

    if 55 < rsi < 70:
        score += 10

    if macd_diff > 0:
        score += 10

    if bb_width > 0.1:
        score += 5

    if close_pos > 0.8:
        score += 20
    elif close_pos > 0.7:
        score += 15
    elif close_pos > 0.6:
        score += 10

    if vwap_diff > 1:
        score += 10
    elif vwap_diff > 0.5:
        score += 5

    return min(score, 100)

# ========== Streamlit App ========== #
st.set_page_config(page_title="Momentum Overnight Pro", layout="wide")
st.title('🚀 Momentum Overnight Pro')
st.subheader('AI-Powered BTST Opportunity Scanner')

# Load Excel Sheets
try:
    stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
except FileNotFoundError:
    st.error("Error: stocklist.xlsx file not found.")
    st.stop()

selected_sheet = st.selectbox("Select Sheet", stock_sheets)
market_choice = st.radio("Market Exchange", ['NSE', 'BSE'], index=0)

if st.button("🔍 Scan BTST Opportunities"):
    with st.spinner(f"Scanning {selected_sheet}..."):
        try:
            symbols_df = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
            if 'Symbol' not in symbols_df.columns:
                st.error("Sheet must contain a 'Symbol' column.")
                st.stop()

            symbols = symbols_df['Symbol'].astype(str).str.strip().str.upper().tolist()
            suffix = '.NS' if market_choice == 'NSE' else '.BO'
            benchmark = '^NSEI' if market_choice == 'NSE' else '^BSESN'

            # Check market strength
            try:
                nifty = yf.download(benchmark, period='2d', progress=False)
                market_strength = "Bullish" if len(nifty) >= 2 and nifty['Close'][-1] > nifty['Close'][-2] else "Bearish"
            except:
                market_strength = "Unknown"

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, symbol in enumerate(symbols):
                clean_symbol = re.sub(r'\.(NS|BO|NSE|BSE)$', '', symbol, flags=re.IGNORECASE)
                yf_symbol = clean_symbol + suffix

                try:
                    data = yf.download(yf_symbol, period='100d', progress=False, auto_adjust=True)
                    if data is None or data.empty or len(data) < 20:
                        continue

                    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if col not in data.columns:
                            data[col] = data['Close']

                    data = calculate_technical_indicators(data)
                    latest = data.iloc[-1]
                    score = calculate_btst_score(latest)

                    if len(data) >= 2:
                        prev_close = data['Close'].iloc[-2]
                        day_change = (latest['Close'] - prev_close) / prev_close * 100
                    else:
                        day_change = 0

                    results.append({
                        'Symbol': clean_symbol,
                        'Score': score,
                        'Price': latest['Close'],
                        'Change (%)': round(day_change, 2),
                        'Volume Spike (%)': round(latest.get('volume_change_pct', 0), 2),
                        'RSI': round(latest.get('rsi', 50), 2),
                        'Position': "Near High" if latest.get('close_position', 0) > 0.7 else "Mid" if latest.get('close_position', 0) > 0.5 else "Near Low",
                        'VWAP Diff (%)': round(latest.get('vwap_diff', 0), 2)
                    })

                except Exception as e:
                    st.warning(f"Error processing {symbol}: {str(e)}")

                progress_bar.progress((i + 1) / len(symbols))
                status_text.text(f"Processed {i+1}/{len(symbols)}: {clean_symbol} | Market: {market_strength}")

            # Display Results
            if results:
                df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
                df['Recommendation'] = pd.cut(df['Score'], bins=[0, 40, 65, 85, 100], labels=["Avoid", "Neutral", "Watch", "Strong Buy"])

                st.success(f"Scan Complete! Market: {market_strength}")
                top_picks = df[df['Recommendation'].isin(['Strong Buy', 'Watch'])]

                st.subheader("🔥 Top BTST Picks")
                if not top_picks.empty:
                    st.dataframe(top_picks.head(10).style.background_gradient(subset=['Score'], cmap='RdYlGn'))
                else:
                    st.info("No strong picks found today.")

                csv = df.to_csv(index=False)
                st.download_button("📥 Download Full Results", data=csv, file_name="btst_results.csv", mime="text/csv")

                st.subheader("📊 Analysis Dashboard")
                col1, col2 = st.columns(2)
                with col1:
                    if not top_picks.empty:
                        st.bar_chart(top_picks.set_index("Symbol")["Score"].head(10))
                with col2:
                    st.bar_chart(df['Recommendation'].value_counts())

                st.write("Feature Correlations")
                if len(df) > 1:
                    corr_df = df[['Score', 'Change (%)', 'Volume Spike (%)', 'RSI', 'VWAP Diff (%)']].corr()
                    st.dataframe(corr_df.style.background_gradient(cmap='coolwarm', axis=None))
            else:
                st.warning("No valid stock data found.")

        except Exception as e:
            st.error(f"Error: {e}")
