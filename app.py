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
# AI Trade Agent - Final Pro Version with Live Price & Rate Limit Safety
# =========================
class AITradeAgent:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)

    def get_live_price(self, ticker):
        """Fetch real-time current price (includes extended hours)"""
        try:
            trade = self.client.get_last_trade(ticker)
            if trade and hasattr(trade, 'price'):
                return round(trade.price, 2)
        except Exception as e:
            st.warning(f"Live price fetch failed: {e}")
        return None

    def get_prev_close(self, ticker):
        """Get previous trading day's close for % change calculation"""
        try:
            # Get yesterday's date (skip weekends if needed, but Polygon handles it)
            yesterday = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).strftime("%Y-%m-%d")
            prev = self.client.get_aggs(ticker, 1, "day", yesterday, yesterday)
            if prev and len(prev) > 0:
                return round(prev[-1].close, 2)
        except:
            pass
        return None

    def fetch_data(self, ticker, timeframe="15min"):
        if timeframe == "15min":
            multiplier, timespan, days_back = 15, "minute", 90
        elif timeframe == "day":
            multiplier, timespan, days_back = 1, "day", 730
        elif timeframe == "week":
            multiplier, timespan, days_back = 1, "week", 1825
        else:
            return None

        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        from_date = today_et - timedelta(days=days_back)
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = today_et.strftime("%Y-%m-%d")

        for attempt in range(3):
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
                if "429" in str(e):
                    wait = (attempt + 1) * 20
                    st.warning(f"Rate limited. Retrying {timeframe} in {wait}s...")
                    time.sleep(wait)
                else:
                    st.error(f"Error fetching {timeframe}: {e}")
                    return None

        return None

    def calculate_indicators(self, df):
        if len(df) < 30:
            return df

        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        df['ema12'] = df["close"].ewm(span=12, adjust=False).mean()
        df['ema26'] = df["close"].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

        df['vol_avg'] = df['volume'].rolling(window=20).mean()
        df['rvol'] = df['volume'] / df['vol_avg'].replace(0, np.nan)

        return df

    def generate_signal(self, df, timeframe_name):
        if df is None or len(df) < 30:
            return "NO DATA<br><small>Error or rate limited</small>", "#95a5a6"

        df = self.calculate_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        price_up = latest['close'] > prev['close']
        macd_bullish = latest['macd'] > latest['signal_line']
        volume_spike = latest['rvol'] > 1.2 if not pd.isna(latest['rvol']) else False
        rsi_val = latest['rsi']

        if macd_bullish and price_up:
            if volume_spike:
                signal, color = "STRONG BUY", "#00ff88"
            elif rsi_val < 65:
                signal, color = "BUY", "#2ecc71"
            else:
                signal, color = "BULLISH", "#27ae60"
        elif rsi_val > 75 or (not price_up and not macd_bullish):
            signal, color = "SELL" if rsi_val > 75 else "BEARISH", "#e74c3c"
        else:
            signal, color = "NEUTRAL", "#f1c40f"

        time_fmt = '%b %d, %I:%M %p' if timeframe_name == "Short" else '%b %d, %Y'
        latest_time = latest['date'].strftime(time_fmt)
        caption = f"{timeframe_name}-Term ({latest_time})"

        return f"{signal}<br><small>{caption}</small>", color

    def get_multi_signals(self, ticker):
        live_price = self.get_live_price(ticker)
        prev_close = self.get_prev_close(ticker)

        df_15min = self.fetch_data(ticker, "15min")
        df_daily = self.fetch_data(ticker, "day")
        df_weekly = self.fetch_data(ticker, "week")

        short_signal, short_color = self.generate_signal(df_15min, "Short")
        med_signal, med_color = self.generate_signal(df_daily, "Medium")
        long_signal, long_color = self.generate_signal(df_weekly, "Long")

        short_data = None
        if df_15min is not None:
            df_15min = self.calculate_indicators(df_15min)
            latest_15 = df_15min.iloc[-1]
            prev_15 = df_15min.iloc[-2]
            short_data = {
                "bar_price": round(latest_15['close'], 2),
                "rsi": round(latest_15['rsi'], 2) if not pd.isna(latest_15['rsi']) else "N/A",
                "rvol": round(latest_15['rvol'], 2) if not pd.isna(latest_15['rvol']) else "N/A",
                "df": df_15min
            }

        return {
            "live_price": live_price,
            "prev_close": prev_close,
            "short": (short_signal, short_color),
            "medium": (med_signal, med_color),
            "long": (long_signal, long_color),
            "short_data": short_data
        }

# =========================
# Streamlit Interface
# =========================
st.set_page_config(page_title="AI Trade Agent Pro", layout="wide")

now_et = datetime.now(ZoneInfo("America/New_York"))
st.title("🚀 AI Trade Agent Pro - Multi-Timeframe + Live Price")
st.markdown(f"**Market Time (ET):** {now_et.strftime('%A, %B %d, %Y | %I:%M:%S %p')}")

api_key = os.getenv("POLYGON_API_KEY")
if not api_key:
    st.error("⚠️ Set POLYGON_API_KEY in Streamlit Secrets.")
    st.stop()

agent = AITradeAgent(api_key)

ticker = st.text_input("Enter Ticker Symbol", value="AMZN", help="e.g., TSLA, AAPL").upper().strip()

col1, col2 = st.columns([1, 4])
with col1:
    auto_refresh = st.checkbox("Auto Refresh (70s - Free Tier Safe)", value=True)
with col2:
    if st.button("🔄 Run Analysis Now"):
        st.session_state.force_run = True

if auto_refresh or st.session_state.get("force_run", False):
    with st.spinner(f"Fetching live data for {ticker}..."):
        results = agent.get_multi_signals(ticker)

    live_price = results["live_price"]
    prev_close = results["prev_close"]
    change = round(live_price - prev_close, 2) if live_price and prev_close else None
    pct = round((change / prev_close) * 100, 2) if change and prev_close else None

    # === Live Price & Metrics ===
    c1, c2, c3, c4 = st.columns(4)
    if live_price:
        c1.metric("**LIVE Current Price**", f"${live_price}",
                  f"{change} ({pct}%)" if change is not None else "Real-time")
    else:
        c1.metric("Current Price", "Unavailable")

    if results["short_data"]:
        data = results["short_data"]
        c2.metric("RSI (14) - 15min", data['rsi'])
        c3.metric("Relative Volume - 15min", f"{data['rvol']}x")
        c4.metric("Latest 15min Bar Close", f"${data['bar_price']}")
    else:
        c2.metric("RSI (14)", "N/A")
        c3.metric("Relative Volume", "N/A")
        c4.metric("Latest Bar", "N/A")

    # === Multi-Timeframe Signals ===
    st.markdown("### 📊 Trend Signals Across Timeframes")
    s1, s2, s3 = st.columns(3)
    with s1:
        sig_html, color = results["short"]
        s1.markdown(f"<div style='background-color:{color}; color:white; padding:20px; border-radius:12px; text-align:center; font-size:18px; font-weight:bold;'>Short-Term<br>{sig_html}</div>", unsafe_allow_html=True)
    with s2:
        sig_html, color = results["medium"]
        s2.markdown(f"<div style='background-color:{color}; color:white; padding:20px; border-radius:12px; text-align:center; font-size:18px; font-weight:bold;'>Medium-Term<br>{sig_html}</div>", unsafe_allow_html=True)
    with s3:
        sig_html, color = results["long"]
        s3.markdown(f"<div style='background-color:{color}; color:white; padding:20px; border-radius:12px; text-align:center; font-size:18px; font-weight:bold;'>Long-Term<br>{sig_html}</div>", unsafe_allow_html=True)

    # === Chart ===
    if results["short_data"]:
        df = results["short_data"]["df"]
        st.markdown("### 📈 Short-Term 15-Minute Chart")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        ax1.plot(df['date'], df['close'], color='#3498db', linewidth=2.5, label='Close Price')
        ax1.fill_between(df['date'], df['close'], alpha=0.1, color='#3498db')
        ax1.set_title(f"{ticker} - 15-Minute Bars", fontsize=16)
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
        st.caption("📊 Chart uses completed 15min bars. LIVE price above includes extended hours & real-time trades.")

        with st.expander("Recent 15min Data"):
            display = df[['date', 'close', 'volume', 'rsi', 'rvol']].tail(15).copy()
            display['date'] = display['date'].dt.strftime('%m/%d %I:%M %p')
            st.dataframe(display.round(2))

    # Auto-refresh (70s for free tier safety)
    if auto_refresh:
        placeholder = st.empty()
        for i in range(70, 0, -1):
            placeholder.info(f"🔄 Refreshing in {i}s... (Free tier safe)")
            time.sleep(1)
        placeholder.empty()
        st.rerun()

    if st.session_state.get("force_run"):
        del st.session_state.force_run
