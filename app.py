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
# AI Trade Agent - Multi-Timeframe Pro Version
# =========================
class AITradeAgent:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)

    def fetch_data(self, ticker, timeframe="15min"):
        if timeframe == "15min":
            multiplier, timespan = 15, "minute"
            days_back = 90
        elif timeframe == "day":
            multiplier, timespan = 1, "day"
            days_back = 365 * 2  # 2 years for daily
        elif timeframe == "week":
            multiplier, timespan = 1, "week"
            days_back = 365 * 5  # 5 years for weekly
        else:
            raise ValueError("timeframe must be 15min, day, or week")

        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        from_date = today_et - timedelta(days=days_back)
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
            return df

        except Exception as e:
            st.error(f"Polygon API Error ({timeframe}): {e}")
            return None

    def calculate_indicators(self, df):
        if len(df) < 30:
            return df

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['ema12'] = df["close"].ewm(span=12, adjust=False).mean()
        df['ema26'] = df["close"].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Relative Volume
        df['vol_avg'] = df['volume'].rolling(window=20).mean()
        df['rvol'] = df['volume'] / df['vol_avg'].replace(0, np.nan)

        return df

    def generate_signal(self, df, timeframe_name):
        if df is None or len(df) < 30:
            return "NO DATA", "#95a5a6"

        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        price_up = latest['close'] > prev['close']
        macd_bullish = latest['macd'] > latest['signal_line']
        volume_spike = latest['rvol'] > 1.2 if not pd.isna(latest['rvol']) else False
        rsi_val = latest['rsi']

        if macd_bullish and price_up:
            if volume_spike:
                signal = "STRONG BUY"
                color = "#00ff88"  # Bright green
            elif rsi_val < 65:
                signal = "BUY"
                color = "#2ecc71"  # Green
            else:
                signal = "BULLISH"
                color = "#27ae60"
        elif rsi_val > 75 or (not price_up and not macd_bullish):
            signal = "SELL" if rsi_val > 75 else "BEARISH"
            color = "#e74c3c"  # Red
        else:
            signal = "NEUTRAL"
            color = "#f1c40f"  # Yellow

        # Add timeframe context
        latest_time = latest['date'].strftime('%b %d, %Y') if timeframe_name != "Short" else latest['date'].strftime('%b %d, %I:%M %p')
        caption = f"{timeframe_name}-Term ({latest_time})"

        return f"{signal}<br><small>{caption}</small>", color

    def get_multi_signals(self, ticker):
        # Fetch all timeframes
        df_15min = self.fetch_data(ticker, "15min")
        df_daily = self.fetch_data(ticker, "day")
        df_weekly = self.fetch_data(ticker, "week")

        # Generate signals
        short_signal, short_color = self.generate_signal(df_15min, "Short")
        med_signal, med_color = self.generate_signal(df_daily, "Medium")
        long_signal, long_color = self.generate_signal(df_weekly, "Long")

        # Also return short-term full data for main chart
        if df_15min is not None:
            df_15min = self.calculate_indicators(df_15min)
            latest_15 = df_15min.iloc[-1]
            prev_15 = df_15min.iloc[-2]
            short_data = {
                "price": round(latest_15['close'], 2),
                "change": round(latest_15['close'] - prev_15['close'], 2),
                "pct": round(((latest_15['close'] - prev_15['close']) / prev_15['close']) * 100, 2),
                "rsi": round(latest_15['rsi'], 2) if not pd.isna(latest_15['rsi']) else "N/A",
                "rvol": round(latest_15['rvol'], 2) if not pd.isna(latest_15['rvol']) else "N/A",
                "df": df_15min
            }
        else:
            short_data = None

        return {
            "short": (short_signal, short_color),
            "medium": (med_signal, med_color),
            "long": (long_signal, long_color),
            "short_data": short_data
        }

# =========================
# Streamlit Interface
# =========================
st.set_page_config(page_title="AI Trade Agent Pro - Multi-Timeframe", layout="wide")

now_et = datetime.now(ZoneInfo("America/New_York"))
st.title("🚀 AI Trade Agent Pro - Multi-Timeframe Analysis")
st.markdown(f"**Market Time (ET):** {now_et.strftime('%A, %B %d, %Y | %I:%M:%S %p')}")

api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    st.error("⚠️ Please set POLYGON_API_KEY in Streamlit Secrets.")
    st.stop()

agent = AITradeAgent(api_key)

ticker = st.text_input("Enter Ticker Symbol", value="TSLA", help="e.g., AAPL, NVDA, SPY").upper().strip()

col1, col2 = st.columns([1, 4])
with col1:
    auto_refresh = st.checkbox("Auto Refresh Every 30s", value=True)
with col2:
    if st.button("🔄 Run Analysis Now"):
        st.session_state.force_run = True

if auto_refresh or st.session_state.get("force_run", False):
    with st.spinner(f"Analyzing {ticker} across multiple timeframes..."):
        results = agent.get_multi_signals(ticker)

    if results["short_data"]:
        data = results["short_data"]
        df = data["df"]

        # === Current Price & Metrics (Short-term) ===
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${data['price']}", f"{data['change']} ({data['pct']}%)")
        c2.metric("RSI (14)", data['rsi'])
        c3.metric("Relative Volume", f"{data['rvol']}x")

        # === Multi-Timeframe Signals ===
        st.markdown("### 📊 Trend Signals Across Timeframes")
        s1, s2, s3 = st.columns(3)

        with s1:
            sig_html, color = results["short"]
            s1.markdown(f"""
            <div style="background-color:{color}; color:white; padding:20px; border-radius:12px; text-align:center; font-size:18px; font-weight:bold;">
                Short-Term<br>{sig_html}
            </div>
            """, unsafe_allow_html=True)

        with s2:
            sig_html, color = results["medium"]
            s2.markdown(f"""
            <div style="background-color:{color}; color:white; padding:20px; border-radius:12px; text-align:center; font-size:18px; font-weight:bold;">
                Medium-Term<br>{sig_html}
            </div>
            """, unsafe_allow_html=True)

        with s3:
            sig_html, color = results["long"]
            s3.markdown(f"""
            <div style="background-color:{color}; color:white; padding:20px; border-radius:12px; text-align:center; font-size:18px; font-weight:bold;">
                Long-Term<br>{sig_html}
            </div>
            """, unsafe_allow_html=True)

        # === Main Chart (15min) ===
        st.markdown("### 📈 Short-Term Price & Volume (15min)")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(df['date'], df['close'], color='#3498db', linewidth=2.5, label='Close Price')
        ax1.fill_between(df['date'], df['close'], alpha=0.1, color='#3498db')
        ax1.set_title(f"{ticker} - 15-Minute Chart", fontsize=16)
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

        # Optional recent data
        with st.expander("View Recent 15min Bars"):
            display = df[['date', 'close', 'volume', 'rsi', 'rvol']].tail(15).copy()
            display['date'] = display['date'].dt.strftime('%m/%d %I:%M %p')
            st.dataframe(display.round(2))

    else:
        st.error("Unable to fetch data for the selected ticker. Please check the symbol.")

    # Auto-refresh
    if auto_refresh:
        placeholder = st.empty()
        for i in range(60, 0, -1):
            placeholder.info(f"🔄 Next refresh in {i} seconds...")
            time.sleep(1)
        placeholder.empty()
        st.rerun()

    if st.session_state.get("force_run"):
        del st.session_state.force_run
