import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import ta
import time

# Feature Engineering Functions
def calculate_technical_indicators(data):
    # Clone to avoid SettingWithCopyWarning
    df = data.copy()
    
    # Price and Volume Features
    df['price_change_pct'] = df['Close'].pct_change() * 100
    
    # Volume change calculation with minimum period check
    vol_window = min(10, len(df) - 1)
    if vol_window > 0:
        df['volume_change_pct'] = (df['Volume'] / df['Volume'].rolling(vol_window).mean() - 1) * 100
    else:
        df['volume_change_pct'] = 0
    
    # VWAP Calculation
    try:
        df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        df['vwap_diff'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-8) * 100
    except:
        df['vwap'] = df['Close']
        df['vwap_diff'] = 0
    
    # Close position in today's range
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
    
    # Technical Indicators with safe calculation
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=min(14, len(df)-1)).rsi()
    except:
        df['rsi'] = 50
    
    try:
        macd = ta.trend.MACD(df['Close'])
        df['macd_diff'] = macd.macd_diff()
    except:
        df['macd_diff'] = 0
    
    # Moving Averages
    for window in [20, 50]:
        if len(df) >= window:
            df[f'ema{window}'] = ta.trend.EMAIndicator(df['Close'], window=window).ema_indicator()
        else:
            df[f'ema{window}'] = df['Close']
    
    # EMA crossover check
    if 'ema20' in df and 'ema50' in df:
        df['ema_cross'] = np.where(df['ema20'] > df['ema50'], 1, 0)
    else:
        df['ema_cross'] = 0
    
    # Bollinger Bands
    if len(df) > 20:
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_width'] = bb.bollinger_wband()
    else:
        df['bb_width'] = 0
    
    return df.dropna(subset=['price_change_pct'], how='all')

def calculate_btst_score(row):
    """Rule-based scoring system for BTST potential"""
    score = 0
    
    # Price Momentum (Max 30 points)
    price_change = row.get('price_change_pct', 0)
    if price_change > 3:
        score += 30
    elif price_change > 2:
        score += 20
    elif price_change > 1:
        score += 10
    
    # Volume Spike (Max 20 points)
    vol_change = row.get('volume_change_pct', 0)
    if vol_change > 150:
        score += 20
    elif vol_change > 100:
        score += 15
    elif vol_change > 50:
        score += 10
    
    # Technical Indicators (Max 30 points)
    rsi = row.get('rsi', 50)
    if 55 < rsi < 70:
        score += 10
    
    if row.get('macd_diff', 0) > 0:
        score += 10
    
    if row.get('ema_cross', 0) == 1:
        score += 5
    
    if row.get('bb_width', 0) > 0.1:
        score += 5
    
    # Closing Position (Max 20 points)
    close_pos = row.get('close_position', 0.5)
    if close_pos > 0.8:
        score += 20
    elif close_pos > 0.7:
        score += 15
    elif close_pos > 0.6:
        score += 10
    
    # VWAP Position (Max 10 points)
    vwap_diff = row.get('vwap_diff', 0)
    if vwap_diff > 1:
        score += 10
    elif vwap_diff > 0.5:
        score += 5
    
    # Trend Alignment (Max 10 points)
    if 'ema20' in row and 'ema50' in row:
        if row['ema20'] > row['ema50']:
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
            
            # Add exchange suffix to symbols
            suffix = '.NS' if market_choice == 'NSE' else '.BO'
            symbols_df['yf_symbol'] = symbols_df['Symbol'].astype(str) + suffix
            
            symbols = symbols_df['yf_symbol'].tolist()
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Market strength check
            benchmark = '^NSEI' if market_choice == 'NSE' else '^BSESN'
            try:
                nifty = yf.download(benchmark, period='2d')['Close']
                market_strength = "Bullish" if len(nifty) >= 2 and nifty[-1] > nifty[-2] else "Bearish"
            except:
                market_strength = "Unknown"
                st.warning("Couldn't fetch market index data")
            
            total_symbols = len(symbols)
            for i, symbol in enumerate(symbols):
                try:
                    # Download data with retry logic
                    data = None
                    for attempt in range(3):
                        try:
                            data = yf.download(symbol, period='100d', progress=False)
                            if len(data) > 10:  # Valid data check
                                break
                        except:
                            time.sleep(0.5)  # Brief pause before retry
                    
                    if data is None or len(data) < 20:
                        status_text.text(f"Skipped {symbol}: insufficient data")
                        continue
                    
                    # Calculate indicators
                    data = calculate_technical_indicators(data)
                    if len(data) == 0:
                        continue
                    
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
                        'Symbol': symbol.replace(suffix, ''),
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
                status_text.text(f"Processed {i+1}/{total_symbols}: {symbol} | Market: {market_strength}")
            
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
                st.dataframe(top_picks.head(10).style.background_gradient(subset=['Score'], cmap='RdYlGn')
                
                # Download option
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
                    if not top_picks.empty:
                        st.bar_chart(top_picks.set_index('Symbol')['Score'].head(10))
                    else:
                        st.info("No strong candidates found")
                
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
            st.exception(e)
