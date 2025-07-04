import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import ta  # Technical Analysis library

# Feature Engineering Functions
def calculate_technical_indicators(data):
    # Price and Volume Features
    data['price_change_pct'] = data['Close'].pct_change() * 100
    data['volume_change_pct'] = (data['Volume'] / data['Volume'].rolling(10).mean() - 1) * 100
    
    # VWAP Calculation
    data['vwap'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
    data['vwap_diff'] = (data['Close'] - data['vwap']) / data['vwap'] * 100
    
    # Close position in today's range
    data['close_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-8)
    
    # Technical Indicators
    data['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
    macd = ta.trend.MACD(data['Close'])
    data['macd_diff'] = macd.macd_diff()
    
    # Moving Averages
    data['ema20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
    data['ema50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
    data['ema200'] = ta.trend.EMAIndicator(data['Close'], window=200).ema_indicator()
    data['ema_cross'] = np.where(data['ema20'] > data['ema50'], 1, 0)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(data['Close'])
    data['bb_width'] = bb.bollinger_wband()
    
    return data.dropna()

def calculate_btst_score(row):
    """Rule-based scoring system for BTST potential"""
    score = 0
    
    # Price Momentum (Max 30 points)
    if row['price_change_pct'] > 3:
        score += 30
    elif row['price_change_pct'] > 2:
        score += 20
    elif row['price_change_pct'] > 1:
        score += 10
    
    # Volume Spike (Max 20 points)
    if row['volume_change_pct'] > 150:
        score += 20
    elif row['volume_change_pct'] > 100:
        score += 15
    elif row['volume_change_pct'] > 50:
        score += 10
    
    # Technical Indicators (Max 30 points)
    if 55 < row['rsi'] < 70:
        score += 10
    if row['macd_diff'] > 0:
        score += 10
    if row['ema_cross'] == 1:
        score += 5
    if row['bb_width'] > 0.1:
        score += 5
    
    # Closing Position (Max 20 points)
    if row['close_position'] > 0.8:
        score += 20
    elif row['close_position'] > 0.7:
        score += 15
    elif row['close_position'] > 0.6:
        score += 10
    
    # VWAP Position (Max 10 points)
    if row['vwap_diff'] > 1:
        score += 10
    elif row['vwap_diff'] > 0.5:
        score += 5
    
    # Trend Alignment (Max 10 points)
    if row['ema20'] > row['ema50'] > row['ema200']:
        score += 10
    
    return min(score, 100)  # Cap score at 100

# Streamlit App
st.title('🚀 BTST (Buy Today Sell Tomorrow) Call Generator')
st.markdown("Identify high-potential stocks for overnight trading using momentum and technical indicators")

# Load stock list from Excel
try:
    stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
except FileNotFoundError:
    st.error("Error: stocklist.xlsx file not found. Please make sure it's in the same directory.")
    st.stop()

selected_sheet = st.selectbox("Select Sheet", stock_sheets)

if st.button("🔍 Scan BTST Opportunities"):
    with st.spinner(f"Loading {selected_sheet} sheet..."):
        try:
            symbols_df = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
            if 'Symbol' not in symbols_df.columns:
                st.error("Selected sheet must contain 'Symbol' column")
                st.stop()
            
            symbols = symbols_df['Symbol'].astype(str).tolist()
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Market strength check (Nifty 50)
            nifty = yf.download('^NSEI', period='2d')['Close']
            market_strength = "Bullish" if nifty[-1] > nifty[-2] else "Bearish"
            
            for i, symbol in enumerate(symbols):
                try:
                    # Download data
                    data = yf.download(symbol, period='60d', progress=False)
                    if len(data) < 20:  # Insufficient data
                        continue
                    
                    # Calculate indicators
                    data = calculate_technical_indicators(data)
                    latest = data.iloc[-1]
                    
                    # Calculate score
                    score = calculate_btst_score(latest)
                    
                    # Get additional info
                    prev_close = data['Close'].iloc[-2]
                    day_change = (latest['Close'] - prev_close) / prev_close * 100
                    
                    results.append({
                        'Symbol': symbol,
                        'Score': score,
                        'Price': latest['Close'],
                        'Change (%)': day_change,
                        'Volume Spike (%)': latest['volume_change_pct'],
                        'RSI': latest['rsi'],
                        'Position': "Near High" if latest['close_position'] > 0.7 else "Mid" if latest['close_position'] > 0.5 else "Near Low",
                        'VWAP Diff (%)': latest['vwap_diff'],
                        'Trend': "Bullish" if latest['ema20'] > latest['ema50'] > latest['ema200'] else "Neutral"
                    })
                    
                except Exception as e:
                    st.warning(f"Error processing {symbol}: {str(e)}")
                
                # Update progress
                progress = (i + 1) / len(symbols)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i+1}/{len(symbols)}: {symbol} | Market: {market_strength}")
            
            # Create results dataframe
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Score', ascending=False)
                results_df = results_df.reset_index(drop=True)
                
                # Categorize recommendations
                results_df['Recommendation'] = pd.cut(
                    results_df['Score'],
                    bins=[0, 40, 65, 85, 100],
                    labels=['Avoid', 'Neutral', 'Watch', 'Strong Buy']
                )
                
                # Display results
                st.success(f"BTST Scan Completed! Market Condition: {market_strength}")
                
                # Top recommendations
                st.subheader("🔥 Top BTST Opportunities")
                top_picks = results_df[results_df['Recommendation'].isin(['Strong Buy', 'Watch'])]
                st.dataframe(top_picks.head(10))
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Results",
                    data=csv,
                    file_name=f'btst_scan_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )
                
                # Visualizations
                st.subheader("📊 Analysis Dashboard")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.bar_chart(top_picks.set_index('Symbol')['Score'].head(10))
                
                with col2:
                    st.write("Score Distribution")
                    st.bar_chart(results_df['Recommendation'].value_counts())
                
                # Feature correlations
                st.write("Feature Correlations with BTST Score")
                corr_df = results_df[['Score', 'Change (%)', 'Volume Spike (%)', 'RSI', 'VWAP Diff (%)']].corr()
                st.dataframe(corr_df.style.background_gradient(cmap='coolwarm'))
                
            else:
                st.warning("No valid stocks found with sufficient data")
                
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
