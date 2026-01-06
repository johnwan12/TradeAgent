import streamlit as st
import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import matplotlib.pyplot as plt
import time
import requests  # For better error handling

# =========================
# AI Trade Agent - Rate-Limit Safe Multi-Timeframe Version
# =========================
class AITradeAgent:
    def __init__(self, api_key):
        self.client = RESTClient(api_key)

    def fetch_data(self, ticker, timeframe="15min"):
        if timeframe == "15min":
            multiplier, timespan, days_back = 15, "minute", 90
        elif timeframe == "day":
            multiplier, timespan, days_back = 1, "day", 730  # 2 years
        elif timeframe == "week":
            multiplier, timespan, days_back = 1, "week", 1825  # 5 years
        else:
            return None

        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        from_date = today_et - timedelta(days=days_back)
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = today_et.strftime("%Y-%m-%d")

        for attempt in range(3):  # Retry up to 3 times on 429
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

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait = (attempt + 1) * 15  # Wait 15s, 30s, 45s
                    st.warning(f"Rate limited by Polygon (429). Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    st.error(f"API HTTP Error ({timeframe}): {e}")
                    return None
            except Exception as e:
                if "429" in str(e):
                    wait = (attempt + 1) * 15
                    st.warning(f"Rate limited. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    st.error(f"Polygon API Error ({timeframe}): {e}")
                    return None

        st.error(f"Failed to fetch {timeframe} data after retries (rate limited).")
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
            return "NO DATA<br><small>Rate limited or error</small>", "#95a5a6"

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

        latest_time = latest['date'].strftime('%b %d, %Y') if "Short" not in timeframe_name else latest['date'].strftime('%b %d, %I:%M %p')
        caption = f"{timeframe_name}-Term ({latest_time})"

        return f"{signal}<br><small>{caption}</small>", color

    def get_multi_signals(self, ticker):
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
                "price": round(latest_15['close'], 2),
                "change": round(latest_15['close'] - prev_15['close'], 2),
                "pct": round(((latest_15['close'] - prev_15['close']) / prev_15['close']) * 100, 2),
                "rsi": round(latest_15['rsi'], 2) if not pd.isna(latest_15['rsi']) else "N/A",
                "rvol": round(latest_15['rvol'], 2) if not pd.isna(latest_15['rvol']) else "N/A",
                "df": df_15min
            }

        return {
            "short": (short_signal, short_color),
            "medium": (med_signal, med_color),
            "long": (long_signal, long_color),
            "short_data": short_data
        }

# =========================
# Streamlit UI (Unchanged except refresh interval)
# =========================
# ... (same as previous version)

# Change auto-refresh to 70 seconds to stay under 5 req/min limit
with col1:
    auto_refresh = st.checkbox("Auto Refresh (70s for free tier)", value=True)

# Auto-refresh countdown (70 seconds)
if auto_refresh:
    placeholder = st.empty()
    for i in range(70, 0, -1):
        placeholder.info(f"🔄 Next refresh in {i} seconds... (safe for free tier)")
        time.sleep(1)
    placeholder.empty()
    st.rerun()
