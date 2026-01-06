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
# AI Trade Agent - Final Pro Version
# =========================
class AITradeAgent:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)

    def fetch_data(self, ticker, timeframe="15min"):
        multiplier = 15 if timeframe == "15min" else 1
        timespan = "minute" if timeframe == "15min" else "day"
        
        try:
            # Fetch enough data for 20-period moving averages
            aggs = self.client.get_aggs(
                ticker, multiplier, timespan, "2025-11-01", "2026-01-05", limit=5000
            )
            if not aggs: return None
            
            df = pd.DataFrame([{
                "timestamp": a.timestamp,
                "open": a.open, "high": a.high, "low": a.low, "close": a.close, "volume": a.volume
            } for a in aggs])
            
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            return df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            st.error(f"API Error: {e}")
            return None

    def calculate_indicators(self, df):
        # RSI 14
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))

        # MACD (12, 26, 9)
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal_line']

        # Volume: Relative Volume (RVOL)
        df['vol_avg'] = df['volume'].rolling(window=20).mean()
        df['rvol'] = df['volume'] / df['vol_avg']
        
        return df

    def get_signal(self, ticker, timeframe="15min"):
        df = self.fetch_data(ticker, timeframe)
        if df is None or len(df) < 30: return None
        
        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # REVISED SIGNAL LOGIC:
        # 1. Momentum: Price up + MACD bullish
        # 2. Confirmation: Relative Volume > 1.2x
        price_action_bullish = latest['close'] > prev['close']
        macd_bullish = latest['macd'] > latest['signal_line']
        volume_spike = latest['rvol'] > 1.2 
        
        rsi_val = latest['rsi']

        if macd_bullish and price_action_bullish:
            if volume_spike:
                overall = "STRONG BUY (Vol Spike)"
            elif rsi_val < 65: # Catching moves even if not "oversold"
                overall = "BUY (Momentum)"
            else:
                overall = "BULLISH HOLD"
        elif rsi_val > 75 or (latest['macd'] < latest['signal_line'] and not price_action_bullish):
            overall = "SELL / CAUTION"
        else:
            overall = "NEUTRAL / HOLD"

        return {
            "ticker": ticker.upper(),
            "price": round(latest['close'], 2),
            "change": round(latest['close'] - prev['close'], 2),
            "pct": round(((latest['close'] - prev['close']) / prev['close']) * 100, 2),
            "rsi": round(rsi_val, 2),
            "rvol": round(latest['rvol'], 2),
            "macd": round(latest['macd'], 4),
            "overall": overall,
            "df": df
        }

# =========================
# Streamlit Interface
# =========================
st.set_page_config(page_title="AI Trade Agent Pro", layout="wide")

# Live Clock Fix
now_et = datetime.now(ZoneInfo("America/New_York"))
st.title("🚀 AI Trade Agent Pro")
st.markdown(f"**Market Time:** {now_et.strftime('%A, %b %d, %Y | %I:%M:%S %p ET')}")

api_key = os.getenv("POLYGON_API_KEY") 
if not api_key:
    st.error("Please add POLYGON_API_KEY to your Environment/Secrets.")
    st.stop()

agent = AITradeAgent(api_key)
ticker = st.text_input("Enter Ticker Symbol", value="TSLA").upper()

if st.button("Run Real-Time Analysis") or True:
    data = agent.get_signal(ticker)
    
    if data:
        # Key Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${data['price']}", f"{data['pct']}%")
        c2.metric("RSI (14-period)", data['rsi'])
        c3.metric("Rel. Volume (RVOL)", f"{data['rvol']}x")
        
        # Enhanced Signal Display
        sig = data['overall']
        color = "#2ecc71" if "BUY" in sig else "#e74c3c" if "SELL" in sig else "#f1c40f"
        c4.markdown(f"""
            <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:white; margin:0;">{sig}</h2>
            </div>
        """, unsafe_allow_html=True)

        # Main Chart: Price and Volume
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(data['df']['date'], data['df']['close'], color='#3498db', label="Price", linewidth=2)
        ax1.fill_between(data['df']['date'], data['df']['close'], alpha=0.1, color='#3498db')
        ax1.set_title(f"{ticker} Technical View", fontsize=16)
        ax1.legend()
        ax1.grid(alpha=0.2)

        # Volume bars
        colors = ['#2ecc71' if c >= o else '#e74c3c' for c, o in zip(data['df']['close'], data['df']['open'])]
        ax2.bar(data['df']['date'], data['df']['volume'], color=colors, alpha=0.6)
        ax2.plot(data['df']['date'], data['df']['vol_avg'], color='#f39c12', label="20-MA Vol")
        ax2.set_ylabel("Volume")
        ax2.legend()
        
        st.pyplot(fig)
    
    # Auto-refresh logic
    time.sleep(30)
    st.rerun()