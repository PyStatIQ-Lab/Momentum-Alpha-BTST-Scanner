import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import ta
import time
import re

# Feature Engineering Functions
def calculate_technical_indicators(df):
    # Clone to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Ensure we have datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date ascending for proper calculations
    df = df.sort_index(ascending=True)
    
    # Price and Volume Features
    df['price_change_pct'] = df['Close'].pct_change().fillna(0) * 100
    
    # Volume change calculation
    vol_window = min(10, len(df) - 1)
    if vol_window > 1:
        vol_avg = df['Volume'].rolling(vol_window, min_periods=1).mean()
        df['volume_change_pct'] = ((df['Volume'] / vol_avg) - 1).fillna(0) * 100
    else:
        df['volume_change_pct'] = 0
    
    # VWAP Calculation
    try:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_tp = (typical_price * df['Volume']).cumsum()
        cumulative_volume = df['Volume'].cumsum()
        df['vwap'] = cumulative_tp / cumulative_volume
        df['vwap_diff'] = ((df['Close'] - df['vwap']) / (df['vwap'] + 1e-8)).fillna(0) * 100
    except:
        df['vwap'] = df['Close']
        df['vwap_diff'] = 0
    
    # Close position in today's range
    try:
        range_val = df['High'] - df['Low'] + 1e-8
        df['close_position'] = (df['Close'] - df['Low']) / range_val
        df['close_position'] = df['close_position'].fillna(0.5)
    except:
        df['close_position'] = 0.5
    
    # Technical Indicators
    try:
        df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=min(14, len(df)-1)).rsi().fillna(50)
    except:
        df['rsi'] = 50
    
    try:
        macd = ta.trend.MACD(close=df['Close'])
        df['macd_diff'] = macd.macd_diff().fillna(0)
    except:
        df['macd_diff'] = 0
    
    # Moving Averages
    for window in [20, 50]:
        try:
            if len(df) >= window:
                df[f'ema{window}'] = ta.trend.EMAIndicator(close=df['Close'], window=window).ema_indicator().fillna(df['Close'])
            else:
                df[f'ema{window}'] = df['Close']
        except:
            df[f'ema{window}'] = df['Close']
    
    # EMA crossover check
    if 'ema20' in df and 'ema50' in df:
        df['ema_cross'] = np.where(df['ema20'] > df['ema50'], 1, 0)
    else:
        df['ema_cross'] = 0
    
    # Bollinger Bands
    try:
        if len(df) > 20:
            bb = ta.volatility.BollingerBands(close=df['Close'])
            df['bb_width'] = bb.bollinger_wband().fillna(0)
        else:
            df['bb_width'] = 0
    except:
        df['bb_width'] = 0
    
    return df

def calculate_btst_score(row):
    """Rule-based scoring system for BTST potential"""
    # Convert row to dictionary if it's a Series
    if isinstance(row, pd.Series):
        row = row.to_dict()
    
    score = 0
    
    # Extract values safely
    price_change = row.get('price_change_pct', 0)
    vol_change = row.get('volume_change_pct', 0)
    rsi = row.get('rsi', 50)
    macd_diff = row.get('macd_diff', 0)
    ema_cross = row.get('ema_cross', 0)
    bb_width = row.get('bb_width', 0)
    close_pos = row.get('close_position', 0.5)
    vwap_diff = row.get('vwap_diff', 0)
    ema20 = row.get('ema20', 0)
    ema50 = row.get('ema50', 1)  # Default to 1 to avoid division by zero
    
    # Price Momentum (Max 30 points)
    if price_change > 3:
        score += 30
    elif price_change > 2:
        score += 20
    elif price_change > 1:
        score += 10
    
    # Volume Spike (Max 20 points)
    if vol_change > 150:
        score += 20
    elif vol_change > 100:
        score += 15
    elif vol_change > 50:
        score += 10
    
    # Technical Indicators (Max 30 points)
    if 55 < rsi < 70:
        score += 10
    
    if macd_diff > 0:
        score += 10
    
    if ema_cross == 1:
        score += 5
    
    if bb_width > 0.1:
        score += 5
    
    # Closing Position (Max 20 points)
    if close_pos > 0.8:
        score += 20
    elif close_pos > 0.7:
        score += 15
    elif close_pos > 0.6:
        score += 10
    
    # VWAP Position (Max 10 points)
    if vwap_diff > 1:
        score += 10
    elif vwap_diff > 0.5:
        score += 5
    
    # Trend Alignment (Max 10 points)
    if ema20 > ema50:
        score += 10
    
    return min(score, 100)  # Cap score at 100

# Streamlit App
st.set_page_config(page_title="Momentum Overnight Pro", layout="wide")
st.title('🚀 Momentum Overnight Pro')
st.subheader('AI-Powered BTST Opportunity Scanner')
st.markdown("Identify high-potential stocks for overnight trading using advanced momentum analysis")

# Load stock list from Excel
try:
    stock_sheets = pd.ExcelFile('stocklist.xlsx').sheet_names
except FileNotFoundError:
    st.error("Error: stocklist.xlsx file not found. Please make sure it's in the same directory.")
    st.stop()

selected_sheet = st.selectbox("Select Sheet", stock_sheets)
market_choice = st.radio("Market Exchange", ['NSE', 'BSE'], index=0)

if st.button("🔍 Scan BTST Opportunities"):
    with st.spinner(f"Loading {selected_sheet} sheet..."):
        try:
            symbols_df = pd.read_excel('stocklist.xlsx', sheet_name=selected_sheet)
            if 'Symbol' not in symbols_df.columns:
                st.error("Selected sheet must contain 'Symbol' column")
                st.stop()
            
            # Clean and format symbols
            symbols = symbols_df['Symbol'].astype(str).str.strip().str.upper().tolist()
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Market strength check - using ^NSEI
            benchmark = '^NSEI'
            market_strength = "Unknown"
            try:
                nifty_data = yf.download(benchmark, period='2d', progress=False)
                if not nifty_data.empty and len(nifty_data) >= 2:
                    # Get the last two closing values
                    closes = nifty_data['Close'].values
                    if closes[-1] > closes[-2]:
                        market_strength = "Bullish"
                    else:
                        market_strength = "Bearish"
                else:
                    st.warning("Insufficient market index data")
            except Exception as e:
                st.warning(f"Couldn't fetch market index data: {str(e)}")
            
            total_symbols = len(symbols)
            suffix = '.NS' if market_choice == 'NSE' else '.BO'
            
            for i, symbol in enumerate(symbols):
                try:
                    # Remove any existing suffix
                    clean_symbol = re.sub(r'\.(NS|BO|NSE|BSE)$', '', symbol, flags=re.IGNORECASE)
                    yf_symbol = clean_symbol + suffix
                    
                    # Download data
                    data = yf.download(yf_symbol, period='100d', progress=False, auto_adjust=True)
                    
                    if data is None or data.empty or len(data) < 20:
                        status_text.text(f"Skipped {symbol}: insufficient data")
                        continue
                    
                    # Ensure we have the correct columns
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_cols:
                        if col not in data.columns:
                            if 'Close' in data.columns:
                                data[col] = data['Close']
                            else:
                                status_text.text(f"Skipped {symbol}: missing columns")
                                continue
                    
                    # Calculate indicators
                    data = calculate_technical_indicators(data)
                    
                    # Get the latest data point
                    latest = data.iloc[-1]
                    
                    # Calculate score
                    score = calculate_btst_score(latest)
                    
                    # Get additional info
                    if len(data) >= 2:
                        prev_close = data['Close'].iloc[-2]
                        day_change = (latest['Close'] - prev_close) / prev_close * 100
                    else:
                        day_change = 0
                    
                    results.append({
                        'Symbol': clean_symbol,
                        'Score': score,
                        'Price': latest['Close'],
                        'Change (%)': day_change,
                        'Volume Spike (%)': latest.get('volume_change_pct', 0),
                        'RSI': latest.get('rsi', 50),
                        'Position': "Near High" if latest.get('close_position', 0) > 0.7 else "Mid" if latest.get('close_position', 0) > 0.5 else "Near Low",
                        'VWAP Diff (%)': latest.get('vwap_diff', 0),
                        'Trend': "Bullish" if latest.get('ema20', 0) > latest.get('ema50', 1) else "Neutral"
                    })
                    
                except Exception as e:
                    st.warning(f"Error processing {symbol}: {str(e)}")
                
                # Update progress
                progress = (i + 1) / total_symbols
                progress_bar.progress(progress)
                status_text.text(f"Processed {i+1}/{total_symbols}: {clean_symbol} | Market: {market_strength}")
            
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
                if not results_df.empty:
                    top_picks = results_df[results_df['Recommendation'].isin(['Strong Buy', 'Watch'])]
                    if not top_picks.empty:
                        st.dataframe(top_picks.head(10).style.background_gradient(subset=['Score'], cmap='RdYlGn'))
                    else:
                        st.info("No strong BTST candidates found today")
                else:
                    st.warning("No valid stocks found with sufficient data")
                
                # Download option
                if not results_df.empty:
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results",
                        data=csv,
                        file_name=f'btst_scan_{datetime.now().strftime("%Y%m%d")}.csv',
                        mime='text/csv'
                    )
                
                # Visualizations
                st.subheader("📊 Analysis Dashboard")
                col1, col2 = st.columns(2)
                
                with col1:
                    if not results_df.empty and 'top_picks' in locals() and not top_picks.empty:
                        st.bar_chart(top_picks.set_index('Symbol')['Score'].head(10))
                    else:
                        st.info("No strong candidates to visualize")
                
                with col2:
                    st.write("Recommendation Distribution")
                    if not results_df.empty:
                        st.bar_chart(results_df['Recommendation'].value_counts())
                    else:
                        st.info("No recommendations to show")
                
                # Feature correlations
                st.write("Feature Correlations with BTST Score")
                if len(results_df) > 1:
                    corr_df = results_df[['Score', 'Change (%)', 'Volume Spike (%)', 'RSI', 'VWAP Diff (%)']].corr()
                    st.dataframe(corr_df.style.background_gradient(cmap='coolwarm', axis=None))
                else:
                    st.info("Insufficient data for correlation analysis")
                
            else:
                st.warning("No valid stocks found with sufficient data")
                
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
