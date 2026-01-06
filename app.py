import streamlit as st
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import matplotlib.pyplot as plt
import time

# =========================
# AI Trade Agent - Revised Pro Version (Dynamic Dates)
# =========================
class AITradeAgent:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)

    def fetch_data(self, ticker, timeframe="15min"):
        multiplier = 15 if timeframe == "15min" else 1
        timespan = "minute" if timeframe == "15min" else "day"

        # Dynamic date range: last 90 days up to today (ET)
        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        from_date = today_et - timedelta(days=90)
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = today_et.strftime("%Y-%m-%d")

        try:
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_str,
                to=to_str,
                limit=5000
            )

            if not aggs or len(aggs) == 0:
                st.warning(f"No data returned for {ticker}. Check symbol or try during market hours.")
                return None

            df = pd.DataFrame([{
                "timestamp": a.timestamp,
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume
            } for a in aggs])

            df["date"] = pd.to_datetime(df["timestamp"], unit="ms") \
                .dt.tz_localize('UTC') \
                .dt.tz_convert('America/New_York')

            df = df.sort_values("date").reset_index(drop=True)

            # Display latest bar time
            latest_time = df.iloc[-1]["date"].strftime('%b %d, %Y at %I:%M %p ET')
            st.caption(f"📊 Latest data as of: **{latest_time}**")

            return df

        except Exception as e:
            st.error(f"Polygon API Error: {e}")
            return None

    def calculate_indicators(self, df):
        if len(df) < 30:
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

        # Relative Volume (20-period)
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

        price_up = latest['close'] > prev['close']
        macd_bullish = latest['macd'] > latest['signal_line']
        volume_spike = latest['rvol'] > 1.2
        rsi_val = latest['rsi']

        if macd_bullish and price_up:
            if volume_spike:
                signal = "STRONG BUY (Volume Spike)"
            elif rsi_val < 65:
                signal = "BUY (Momentum Confirmed)"
            else:
                signal = "BULLISH HOLD"
        elif rsi_val > 75 or (not price_up and not macd_bullish):
            signal = "SELL / CAUTION"
        else:
            signal = "NEUTRAL / HOLD"

        return {
            "ticker": ticker.upper(),
            "price": round(latest['close'], 2),
            "change": round(latest['close'] - prev['close'], 2),
            "pct": round(((latest['close'] - prev['close']) / prev['close']) * 100, 2),
            "rsi": round(rsi_val, 2) if not pd.isna(rsi_val) else "N/A",
            "rvol": round(latest['rvol'], 2) if not pd.isna(latest['rvol']) else "N/A",
            "macd": round(latest['macd'], 4),
            "signal": signal,
            "df": df
        }

# =========================
# Streamlit Interface
# =========================
st.set_page_config(page_title="AI Trade Agent Pro", layout="wide")

# Live Clock (ET)
now_et = datetime.now(ZoneInfo("America/New_York"))
st.title("🚀 AI Trade Agent Pro")
st.markdown(f"**Market Time (ET):** {now_et.strftime('%A, %B %d, %Y | %I:%M:%S %p')}")

# API Key
api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    st.error("⚠️ Please set POLYGON_API_KEY in Streamlit Secrets or environment.")
    st.stop()

agent = AITradeAgent(api_key)

# User Input
ticker = st.text_input("Enter Ticker Symbol", value="TSLA", help="e.g., AAPL, NVDA, SPY").upper().strip()

# Controls
col1, col2 = st.columns([1, 4])
with col1:
    auto_refresh = st.checkbox("Auto Refresh Every 30s", value=True)
with col2:
    if st.button("🔄 Run Analysis Now"):
        st.session_state.force_run = True

# Run analysis
if auto_refresh or st.session_state.get("force_run", False):
    with st.spinner(f"Analyzing {ticker}..."):
        data = agent.get_signal(ticker)

    if data:
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${data['price']}", f"{data['change']} ({data['pct']}%)")
        c2.metric("RSI (14)", data['rsi'])
        c3.metric("Relative Volume", f"{data['rvol']}x")

        # Signal
        sig = data['signal']
        if "STRONG BUY" in sig:
            color = "#00ff88"
        elif "BUY" in sig:
            color = "#2ecc71"
        elif "SELL" in sig or "CAUTION" in sig:
            color = "#e74c3c"
        else:
            color = "#f1c40f"

        c4.markdown(f"""
        <div style="background-color:{color}; color:white; padding:25px; border-radius:12px; text-align:center; font-size:20px; font-weight:bold;">
            {sig}
        </div>
        """, unsafe_allow_html=True)

        # Chart
        df = data['df']
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(df['date'], df['close'], color='#3498db', linewidth=2.5, label='Close Price')
        ax1.fill_between(df['date'], df['close'], alpha=0.1, color='#3498db')
        ax1.set_title(f"{ticker} - Price & Volume (Latest Data)", fontsize=16)
        ax1.set_ylabel("Price ($)")
        ax1.legend()
        ax1.grid(alpha=0.3)

        bar_colors = ['green' if c >= o else 'red' for c, o in zip(df['close'], df['open'])]
        ax2.bar(df['date'], df['volume'], color=bar_colors, alpha=0.7)
        ax2.plot(df['date'], df['vol_avg'], color='orange', linewidth=2, label='20-Period Avg Volume')
        ax2.set_ylabel("Volume")
        ax2.legend()
        ax2.grid(alpha=0.3)

        st.pyplot(fig)

        # Optional data table
        with st.expander("View Recent Bars"):
            display = df[['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'rvol']].tail(15).copy()
            display['date'] = display['date'].dt.strftime('%m/%d %I:%M %p')
            st.dataframe(display.round(2))

    else:
        st.info("No valid data received. Try a different ticker or wait for market open.")

    # Auto-refresh countdown
    if auto_refresh:
        placeholder = st.empty()
        for i in range(30, 0, -1):
            placeholder.info(f"🔄 Next refresh in {i} seconds...")
            time.sleep(1)
        placeholder.empty()
        st.rerun()

    # Clear manual trigger
    if st.session_state.get("force_run"):
        del st.session_state.force_run
