import streamlit as st
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import matplotlib.pyplot as plt
import time

# =========================
# AI Trade Agent - Fixed & Improved Pro Version
# =========================
class AITradeAgent:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)

    def fetch_data(self, ticker, timeframe="15min"):
        multiplier = 15 if timeframe == "15min" else 1
        timespan = "minute" if timeframe == "15min" else "day"
        
        try:
            # Fetch recent data (last ~2 months to ensure enough bars)
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_="2025-11-01",
                to="2026-01-05",
                limit=5000
            )
            
            if not aggs or len(aggs) == 0:
                st.warning(f"No data returned for {ticker}. Check ticker symbol or market hours.")
                return None

            # Properly construct DataFrame
            df = pd.DataFrame([{
                "timestamp": a.timestamp,
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume
            } for a in aggs])

            # Convert timestamp (ms) to datetime in ET
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms") \
                .dt.tz_localize('UTC') \
                .dt.tz_convert('America/New_York')
            
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            st.error(f"Polygon API Error: {e}")
            return None

    def calculate_indicators(self, df):
        if df.empty or len(df) < 30:
            return df

        # RSI (14)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        df['ema12'] = df["close"].ewm(span=12, adjust=False).mean()
        df['ema26'] = df["close"].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal_line']

        # Relative Volume (20-period average)
        df['vol_avg'] = df['volume'].rolling(window=20).mean()
        df['rvol'] = df['volume'] / df['vol_avg'].replace(0, np.nan)

        return df

    def get_signal(self, ticker, timeframe="15min"):
        df = self.fetch_data(ticker, timeframe)
        if df is None or len(df) < 30:
            return None

        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Signal Logic
        price_up = latest['close'] > prev['close']
        macd_bullish = latest['macd'] > latest['signal_line']
        volume_spike = latest['rvol'] > 1.2
        rsi_overbought = latest['rsi'] > 75

        if macd_bullish and price_up:
            if volume_spike:
                signal = "STRONG BUY (Volume Spike)"
            elif latest['rsi'] < 65:
                signal = "BUY (Momentum Confirmed)"
            else:
                signal = "BULLISH HOLD"
        elif rsi_overbought or (not price_up and not macd_bullish):
            signal = "SELL / CAUTION"
        else:
            signal = "NEUTRAL / HOLD"

        return {
            "ticker": ticker.upper(),
            "price": round(latest['close'], 2),
            "change": round(latest['close'] - prev['close'], 2),
            "pct": round(((latest['close'] - prev['close']) / prev['close']) * 100, 2),
            "rsi": round(latest['rsi'], 2) if not pd.isna(latest['rsi']) else "N/A",
            "rvol": round(latest['rvol'], 2) if not pd.isna(latest['rvol']) else "N/A",
            "macd": round(latest['macd'], 4),
            "signal": signal,
            "df": df
        }

# =========================
# Streamlit Interface
# =========================
st.set_page_config(page_title="AI Trade Agent Pro", layout="wide")

# Live Clock
now_et = datetime.now(ZoneInfo("America/New_York"))
st.title("🚀 AI Trade Agent Pro")
st.markdown(f"**Market Time (ET):** {now_et.strftime('%A, %B %d, %Y | %I:%M:%S %p')}")

# API Key Check
api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    st.error("⚠️ Please set your POLYGON_API_KEY in Streamlit Secrets or environment variables.")
    st.stop()

agent = AITradeAgent(api_key)

# Input
ticker = st.text_input("Enter Ticker Symbol", value="TSLA", help="e.g., TSLA, AAPL, NVDA").upper().strip()

# Auto-refresh every 30 seconds (only if button pressed or auto mode)
col1, col2 = st.columns([1, 4])
with col1:
    auto_refresh = st.checkbox("Auto Refresh", value=True)
with col2:
    if st.button("🔄 Run Analysis Now"):
        st.session_state.force_refresh = True

# Trigger analysis
should_run = st.session_state.get('force_refresh', False) or auto_refresh

if should_run or st.button:  # Also trigger on initial load
    with st.spinner(f"Fetching real-time data for {ticker}..."):
        data = agent.get_signal(ticker)

    if data:
        df = data['df']

        # Key Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${data['price']}", f"{data['change']} ({data['pct']}%)")
        c2.metric("RSI (14)", data['rsi'])
        c3.metric("Relative Volume", f"{data['rvol']}x")
        
        # Signal Display
        sig = data['signal']
        if "STRONG BUY" in sig:
            color = "#00ff00"
        elif "BUY" in sig:
            color = "#2ecc71"
        elif "SELL" in sig or "CAUTION" in sig:
            color = "#e74c3c"
        else:
            color = "#f1c40f"

        c4.markdown(f"""
        <div style="background-color:{color}; color:white; padding:20px; border-radius:12px; text-align:center; font-size:18px; font-weight:bold;">
            {sig}
        </div>
        """, unsafe_allow_html=True)

        # Chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, 
                                     gridspec_kw={'height_ratios': [3, 1]})

        # Price line
        ax1.plot(df['date'], df['close'], color='#3498db', linewidth=2, label='Close Price')
        ax1.fill_between(df['date'], df['close'], alpha=0.1, color='#3498db')
        ax1.set_title(f"{ticker} - Price & Volume Analysis", fontsize=16)
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Volume bars
        bar_colors = ['green' if c >= o else 'red' 
                     for c, o in zip(df['close'], df['open'])]
        ax2.bar(df['date'], df['volume'], color=bar_colors, alpha=0.7, width=0.6)
        ax2.plot(df['date'], df['vol_avg'], color='orange', linewidth=2, label='20-Period Avg Volume')
        ax2.set_ylabel("Volume")
        ax2.legend()
        ax2.grid(alpha=0.3)

        st.pyplot(fig)

        # Optional: Show recent data table
        with st.expander("View Recent Data Table"):
            display_df = df[['date', 'close', 'volume', 'rsi', 'rvol', 'macd']].tail(10).copy()
            display_df['date'] = display_df['date'].dt.strftime('%m/%d %I:%M %p')
            st.dataframe(display_df.round(2))

    else:
        st.error("Failed to retrieve or process data. Check ticker and try again.")

    # Auto-refresh countdown
    if auto_refresh:
        placeholder = st.empty()
        for i in range(30, 0, -1):
            placeholder.info(f"🔄 Auto-refreshing in {i} seconds...")
            time.sleep(1)
        placeholder.empty()
        st.rerun()

    # Clear force refresh flag
    if st.session_state.get('force_refresh'):
        del st.session_state.force_refresh
