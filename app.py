import time
import math
import requests
import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="360° AI Crypto Demo Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 0.8 second (800ms) live refresh without full browser reload
st_autorefresh(interval=800, key="live_market_refresh")

st.markdown("""
<style>
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-title { color: #8b949e; font-size: 13px; margin-bottom: 5px; }
    .metric-val { color: #f0f6fc; font-size: 22px; font-weight: 700; }
    .bullish { color: #2ea043; font-weight: bold; }
    .bearish { color: #f85149; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION ($10,000 DEMO)
# ==========================================
COINS = {
    "BTC": "XXBTZUSD",
    "ETH": "XETHZUSD",
    "SOL": "SOLUSD",
    "XRP": "XXRPZUSD",
    "DOGE": "XDGUSD",
    "ADA": "ADAUSD"
}

if "cash_balance" not in st.session_state:
    st.session_state.cash_balance = 10000.0
if "initial_balance" not in st.session_state:
    st.session_state.initial_balance = 10000.0
if "positions" not in st.session_state:
    st.session_state.positions = []  # Active trades
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []  # Closed trades
if "auto_bot" not in st.session_state:
    st.session_state.auto_bot = False
if "daily_loss" not in st.session_state:
    st.session_state.daily_loss = 0.0
if "kill_switch_triggered" not in st.session_state:
    st.session_state.kill_switch_triggered = False

# ==========================================
# 3. DATA FETCHING ENGINES (NO API KEY NEEDED)
# ==========================================
@st.cache_data(ttl=1)
def fetch_kraken_tickers():
    pairs = ",".join(COINS.values())
    url = f"https://api.kraken.com/0/public/Ticker?pair={pairs}"
    try:
        res = requests.get(url, timeout=3).json()
        result = res.get("result", {})
        data = {}
        for sym, k_pair in COINS.items():
            # Kraken keys sometimes vary slightly, match flexibly
            matched_key = next((k for k in result.keys() if k_pair in k or sym in k), None)
            if matched_key:
                info = result[matched_key]
                last_price = float(info["c"][0])
                open_price = float(info["o"])
                vol_24h = float(info["v"][1])
                change_pct = ((last_price - open_price) / open_price) * 100 if open_price else 0.0
                data[sym] = {
                    "price": last_price,
                    "change_24h": change_pct,
                    "volume_24h": vol_24h
                }
        return data
    except Exception:
        return {}

@st.cache_data(ttl=15)
def fetch_kraken_ohlc(coin_sym, interval=15):
    k_pair = COINS[coin_sym]
    url = f"https://api.kraken.com/0/public/OHLC?pair={k_pair}&interval={interval}"
    try:
        res = requests.get(url, timeout=5).json()
        result = res.get("result", {})
        matched_key = next((k for k in result.keys() if k != "last"), None)
        if not matched_key:
            return pd.DataFrame()
        df = pd.DataFrame(
            result[matched_key],
            columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"]
        )
        df["time"] = pd.to_datetime(df["time"], unit="s")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.tail(100).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_background_macro():
    """Background study of S&P 500 (^GSPC), DXY (DX-Y.NYB), VIX (^VIX)"""
    macro = {"sp500_chg": 0.0, "dxy_chg": 0.0, "vix": 18.0, "macro_score": 50}
    try:
        tickers = yf.Tickers("^GSPC DX-Y.NYB ^VIX")
        sp_hist = tickers.tickers["^GSPC"].history(period="2d")
        dxy_hist = tickers.tickers["DX-Y.NYB"].history(period="2d")
        vix_hist = tickers.tickers["^VIX"].history(period="2d")

        if len(sp_hist) >= 2:
            macro["sp500_chg"] = ((sp_hist["Close"].iloc[-1] - sp_hist["Close"].iloc[-2]) / sp_hist["Close"].iloc[-2]) * 100
        if len(dxy_hist) >= 2:
            macro["dxy_chg"] = ((dxy_hist["Close"].iloc[-1] - dxy_hist["Close"].iloc[-2]) / dxy_hist["Close"].iloc[-2]) * 100
        if len(vix_hist) >= 1:
            macro["vix"] = float(vix_hist["Close"].iloc[-1])

        # Macro scoring: SP500 up & DXY down & VIX low = Crypto friendly
        score = 50
        score += 15 if macro["sp500_chg"] > 0 else -15
        score += 15 if macro["dxy_chg"] < 0 else -15
        score += 10 if macro["vix"] < 20 else -15
        macro["macro_score"] = max(0, min(100, score))
    except Exception:
        pass
    return macro

@st.cache_data(ttl=180)
def fetch_news_sentiment():
    """Scans CoinTelegraph & CoinDesk RSS headlines for Bullish/Bearish bias"""
    feeds = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/"
    ]
    bull_words = ["surge", "rally", "bull", "gain", "adoption", "etf", "approval", "high", "growth", "inflow"]
    bear_words = ["crash", "drop", "bear", "hack", "ban", "lawsuit", "sec", "plunge", "fear", "outflow", "fall"]
    
    bull_count, bear_count = 0, 0
    headlines = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:8]:
                title = entry.title
                headlines.append(title)
                t_low = title.lower()
                if any(w in t_low for w in bull_words):
                    bull_count += 1
                if any(w in t_low for w in bear_words):
                    bear_count += 1
        except Exception:
            continue

    total = bull_count + bear_count
    sentiment_score = int((bull_count / total) * 100) if total > 0 else 55
    return sentiment_score, headlines[:5]

# ==========================================
# 4. TECHNICAL, CANDLESTICK & 360° ENGINE
# ==========================================
def analyze_coin_360(df, macro_data, news_score):
    if df.empty or len(df) < 25:
        return {
            "direction": "NEUTRAL", "score_360": 50, "pattern": "Insufficient Data",
            "ema9": 0, "ema21": 0, "rsi": 50, "vol_ratio": 1.0, "atr": 0,
            "reasons": ["পর্যাপ্ত ক্যান্ডেল ডেটা লোড হচ্ছে..."]
        }

    # Indicators: EMA 9, EMA 21, RSI 14, ATR 14, Volume Ratio
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    avg_vol = df["volume"].tail(20).mean()
    vol_ratio = df["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 1.0

    # Candlestick Pattern Detection (Latest closed candle: iloc[-2])
    c = df.iloc[-2]
    p = df.iloc[-3]
    body = abs(c["close"] - c["open"])
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    candle_range = max(c["high"] - c["low"], 1e-8)

    pattern = "Standard Candle"
    candle_bias = 0
    if lower_wick > 2 * body and upper_wick < body:
        pattern = "Bullish Hammer"
        candle_bias = 12
    elif upper_wick > 2 * body and lower_wick < body:
        pattern = "Bearish Shooting Star"
        candle_bias = -12
    elif c["close"] > c["open"] and p["close"] < p["open"] and c["close"] > p["open"] and c["open"] < p["close"]:
        pattern = "Bullish Engulfing"
        candle_bias = 15
    elif c["close"] < c["open"] and p["close"] > p["open"] and c["open"] > p["close"] and c["close"] < p["open"]:
        pattern = "Bearish Engulfing"
        candle_bias = -15

    ema9 = df["ema9"].iloc[-1]
    ema21 = df["ema21"].iloc[-1]
    rsi = df["rsi"].iloc[-1] if not np.isnan(df["rsi"].iloc[-1]) else 50.0
    atr = df["atr"].iloc[-1] if not np.isnan(df["atr"].iloc[-1]) else (df["close"].iloc[-1] * 0.01)

    # Trend Structure (HH/HL vs LH/LL)
    is_uptrend = ema9 > ema21 and df["low"].iloc[-2] > df["low"].iloc[-6]
    is_downtrend = ema9 < ema21 and df["high"].iloc[-2] < df["high"].iloc[-6]

    reasons = []
    bull_points = 45
    bear_points = 45

    if is_uptrend:
        bull_points += 18
        reasons.append("EMA 9 > EMA 21 এবং Higher Low (Uptrend কাঠামো) সক্রিয়।")
    elif is_downtrend:
        bear_points += 18
        reasons.append("EMA 9 < EMA 21 এবং Lower High (Downtrend কাঠামো) সক্রিয়।")
    else:
        reasons.append("মার্কেট বর্তমানে সাইডওয়েজ বা কনসোলিডেশন রেঞ্জে আছে।")

    if candle_bias > 0:
        bull_points += candle_bias
        reasons.append(f"ক্যান্ডেলস্টিক প্যাটার্ন: {pattern} (ক্রেতাদের রিজেকশন চাপ)।")
    elif candle_bias < 0:
        bear_points += abs(candle_bias)
        reasons.append(f"ক্যান্ডেলস্টিক প্যাটার্ন: {pattern} (বিক্রেতাদের চাপ)।")

    if vol_ratio >= 1.15:
        bull_points += 7
        bear_points += 7
        reasons.append(f"ভলিউম রেশিও {vol_ratio:.2f}x — মুভমেন্টের পেছনে ভালো লিকুইডিটি আছে।")
    else:
        reasons.append(f"ভলিউম রেশিও {vol_ratio:.2f}x — ভলিউম তুলনামূলক কম।")

    if rsi < 35:
        bull_points += 8
        reasons.append(f"RSI ({rsi:.1f}) ওভারসোল্ড জোনে — বাউন্স সম্ভাবনা।")
    elif rsi > 68:
        bear_points += 8
        reasons.append(f"RSI ({rsi:.1f}) ওভারবট জোনে — কারেকশন ঝুঁকি।")

    # Macro & News Adjustment
    macro_s = macro_data["macro_score"]
    if macro_s >= 60:
        bull_points += 8
        reasons.append(f"গ্লোবাল ম্যাক্রো অনুকূল (S&P 500: {macro_data['sp500_chg']:+.2f}%, DXY: {macro_data['dxy_chg']:+.2f}%)।")
    elif macro_s <= 40:
        bear_points += 8
        bull_points -= 10
        reasons.append(f"ম্যাক্রো সতর্কতা: ডলার শক্তিশালী বা S&P দুর্বল — অন্ধভাবে Buy এড়ানো হচ্ছে।")

    if news_score >= 60:
        bull_points += 6
        reasons.append(f"নিউজ সেন্টিমেন্ট ইতিবাচক ({news_score}% Bullish)।")
    elif news_score <= 40:
        bear_points += 6
        reasons.append(f"নিউজ সেন্টিমেন্ট নেতিবাচক ({100-news_score}% Bearish)।")

    if bull_points >= bear_points:
        direction = "BUY (Long)" if is_uptrend else "NEUTRAL"
        score_360 = min(92, max(40, int(bull_points)))
    else:
        direction = "SELL (Short)" if is_downtrend else "NEUTRAL"
        score_360 = min(92, max(40, int(bear_points)))

    return {
        "direction": direction,
        "score_360": score_360,
        "pattern": pattern,
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi,
        "vol_ratio": vol_ratio,
        "atr": atr,
        "reasons": reasons
    }

# ==========================================
# 5. ORDER EXECUTION, TRAILING SL & KILL-SWITCH
# ==========================================
def execute_trade(coin, side, amount_usd, entry_price, sl_price, tp_price, mode="MANUAL"):
    # Kill-Switch & Guardrails
    if st.session_state.kill_switch_triggered or st.session_state.daily_loss >= 500.0:
        st.session_state.kill_switch_triggered = True
        st.session_state.auto_bot = False
        return False, "Kill-Switch সক্রিয়: দৈনিক লস $500 (৫%) ছুঁয়েছে!"

    if any(p["coin"] == coin for p in st.session_state.positions):
        return False, f"{coin}-এ ইতিমধ্যে একটি পজিশন চালু আছে (ডাবল অর্ডার ব্লক)।"

    if amount_usd <= 0 or amount_usd > st.session_state.cash_balance:
        return False, "পর্যাপ্ত ভার্চুয়াল ব্যালেন্স নেই।"

    qty = amount_usd / entry_price
    st.session_state.cash_balance -= amount_usd
    st.session_state.positions.append({
        "id": f"ORD-{int(time.time()*1000)%100000}",
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "coin": coin,
        "side": side,
        "entry": entry_price,
        "qty": qty,
        "invested": amount_usd,
        "sl": sl_price,
        "tp": tp_price,
        "mode": mode
    })
    return True, f"{coin} {side} অর্ডার সফলভাবে নেওয়া হয়েছে!"

def manage_active_positions(live_prices):
    remaining = []
    for pos in st.session_state.positions:
        coin = pos["coin"]
        if coin not in live_prices:
            remaining.append(pos)
            continue

        curr_price = live_prices[coin]["price"]
        if pos["side"] == "BUY (Long)":
            pnl = (curr_price - pos["entry"]) * pos["qty"]
            # Smart Trailing Stop-Loss: Lock profit once +1.2% in favor
            if curr_price >= pos["entry"] * 1.012:
                new_sl = max(pos["sl"], pos["entry"] * 1.003)
                pos["sl"] = new_sl

            hit_sl = curr_price <= pos["sl"]
            hit_tp = curr_price >= pos["tp"]
        else:  # SELL (Short)
            pnl = (pos["entry"] - curr_price) * pos["qty"]
            if curr_price <= pos["entry"] * 0.988:
                new_sl = min(pos["sl"], pos["entry"] * 0.997)
                pos["sl"] = new_sl

            hit_sl = curr_price >= pos["sl"]
            hit_tp = curr_price <= pos["tp"]

        if hit_sl or hit_tp:
            reason = "Take-Profit Hit" if hit_tp else "Stop-Loss / Trailing SL Hit"
            returned_cash = pos["invested"] + pnl
            st.session_state.cash_balance += max(0.0, returned_cash)
            if pnl < 0:
                st.session_state.daily_loss += abs(pnl)
                if st.session_state.daily_loss >= 500.0:
                    st.session_state.kill_switch_triggered = True
                    st.session_state.auto_bot = False

            st.session_state.trade_history.append({
                "Time": pos["time"],
                "Coin": coin,
                "Side": pos["side"],
                "Mode": pos["mode"],
                "Entry": round(pos["entry"], 4),
                "Exit": round(curr_price, 4),
                "Invested ($)": round(pos["invested"], 2),
                "PnL ($)": round(pnl, 2),
                "Status": reason
            })
        else:
            remaining.append(pos)

    st.session_state.positions = remaining

# ==========================================
# 6. MAIN TERMINAL EXECUTION
# ==========================================
live_prices = fetch_kraken_tickers()
macro_data = fetch_background_macro()
news_score, top_headlines = fetch_news_sentiment()

# Update open positions & trailing stops every 0.8s
manage_active_positions(live_prices)

# Analyze all coins
analyses = {}
for coin in COINS.keys():
    df_coin = fetch_kraken_ohlc(coin, interval=15)
    analyses[coin] = (df_coin, analyze_coin_360(df_coin, macro_data, news_score))

# Run Auto Bot if ON and Kill-Switch is SAFE
if st.session_state.auto_bot and not st.session_state.kill_switch_triggered:
    for coin, (df_c, info) in analyses.items():
        if info["score_360"] >= 62 and info["direction"] in ["BUY (Long)", "SELL (Short)"]:
            if coin in live_prices and len(st.session_state.positions) < 3:
                price = live_prices[coin]["price"]
                atr = info["atr"] if info["atr"] > 0 else price * 0.015
                trade_amt = min(1000.0, st.session_state.cash_balance * 0.15)  # 15% allocation per trade
                if info["direction"] == "BUY (Long)":
                    sl = price - (1.5 * atr)
                    tp = price + (2.5 * atr)
                else:
                    sl = price + (1.5 * atr)
                    tp = price - (2.5 * atr)
                execute_trade(coin, info["direction"], trade_amt, price, sl, tp, mode="AUTO-BOT")

# Calculate Portfolio Equity
invested_total = sum(p["invested"] for p in st.session_state.positions)
live_pnl = 0.0
for p in st.session_state.positions:
    if p["coin"] in live_prices:
        cp = live_prices[p["coin"]]["price"]
        live_pnl += (cp - p["entry"]) * p["qty"] if p["side"] == "BUY (Long)" else (p["entry"] - cp) * p["qty"]

current_equity = st.session_state.cash_balance + invested_total + live_pnl

# ==========================================
# 7. UI DASHBOARD RENDERING
# ==========================================
st.title("🌐 360° AI Crypto Demo Trading & Market Intelligence Terminal")
st.caption("⚠️ শিক্ষামূলক ডেমো সিমুলেশন | Kraken লাইভ ফিড (0.8s) | পেজ রিলোড করলে ব্যালেন্স $10,000-এ রিসেট হবে।")

# Row 1: Account Summary (4 Cards)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='metric-card'><div class='metric-title'>Available Cash Balance</div><div class='metric-val'>${st.session_state.cash_balance:,.2f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card'><div class='metric-title'>Invested Balance</div><div class='metric-val'>${invested_total:,.2f}</div></div>", unsafe_allow_html=True)
pnl_color = "#2ea043" if live_pnl >= 0 else "#f85149"
c3.markdown(f"<div class='metric-card'><div class='metric-title'>Live Profit / Loss</div><div class='metric-val' style='color:{pnl_color}'>${live_pnl:+,.2f}</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card'><div class='metric-title'>Current Balance (Equity)</div><div class='metric-val'>${current_equity:,.2f}</div></div>", unsafe_allow_html=True)

st.divider()

# Row 2: Auto Action Control & Kill-Switch Status
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 2])
with ctrl_col1:
    bot_toggle = st.toggle("🤖 AUTO ACTION (অটোমেটেড বট — ৩৬০° স্কোর ≥ ৬২%)", value=st.session_state.auto_bot, disabled=st.session_state.kill_switch_triggered)
    st.session_state.auto_bot = bot_toggle
with ctrl_col2:
    if st.session_state.kill_switch_triggered:
        st.error(f"🛑 KILL-SWITCH ACTIVE! Daily Loss: ${st.session_state.daily_loss:.2f}")
    else:
        st.success(f"🛡️ Kill-Switch Safe | Daily Loss: ${st.session_state.daily_loss:.2f} / $500.00")
with ctrl_col3:
    if st.button("🔄 Reset Demo Account ($10,000)"):
        st.session_state.cash_balance = 10000.0
        st.session_state.positions = []
        st.session_state.trade_history = []
        st.session_state.daily_loss = 0.0
        st.session_state.kill_switch_triggered = False
        st.session_state.auto_bot = False
        st.rerun()

# Row 3: Live Market Board + Smart Opportunity Alerts
board_col, alert_col = st.columns([3, 2])

with board_col:
    st.subheader("📊 লাইভ মার্কেট বোর্ড (Kraken Real-Time Feed)")
    board_rows = []
    for coin in COINS.keys():
        p_info = live_prices.get(coin, {"price": 0.0, "change_24h": 0.0, "volume_24h": 0.0})
        a_info = analyses[coin][1]
        bias_label = "🟢 Bullish" if "BUY" in a_info["direction"] else ("🔴 Bearish" if "SELL" in a_info["direction"] else "⚪ Neutral")
        board_rows.append({
            "Coin": coin,
            "Live Price ($)": f"${p_info['price']:,.4f}",
            "24h Change (%)": f"{p_info['change_24h']:+.2f}%",
            "24h Volume": f"{p_info['volume_24h']:,.1f}",
            "360° Score": f"{a_info['score_360']}%",
            "Market Bias": bias_label
        })
    st.dataframe(pd.DataFrame(board_rows), use_container_width=True, hide_index=True)

with alert_col:
    st.subheader("🔔 স্মার্ট অপরচুনিটি অ্যালার্ট")
    for coin in COINS.keys():
        a_info = analyses[coin][1]
        if a_info["score_360"] >= 60:
            st.info(f"**{coin}** → আগামী ১-২ ঘণ্টায় **{a_info['direction']}** সম্ভাবনা **{a_info['score_360']}%** ({a_info['pattern']})")
    st.caption(f"📰 নিউজ সেন্টিমেন্ট স্কোর: {news_score}% Bullish | S&P 500: {macro_data['sp500_chg']:+.2f}% | VIX: {macro_data['vix']:.1f}")

st.divider()

# Row 4: Manual Mode & 360° Search Study
st.subheader("🖐️ MANUAL MODE & ৩৬০° সার্চ স্টাডি")
selected_coin = st.selectbox("বিশ্লেষণ ও অর্ডারের জন্য কয়েন নির্বাচন করুন:", list(COINS.keys()))
df_sel, sel_analysis = analyses[selected_coin]
curr_sel_price = live_prices.get(selected_coin, {}).get("price", 0.0)

chart_col, report_col = st.columns([3, 2])

with chart_col:
    if not df_sel.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_sel["time"], open=df_sel["open"], high=df_sel["high"],
            low=df_sel["low"], close=df_sel["close"], name=selected_coin
        ))
        fig.add_trace(go.Scatter(x=df_sel["time"], y=df_sel["ema9"], line=dict(color="#00b4d8", width=1.5), name="EMA 9"))
        fig.add_trace(go.Scatter(x=df_sel["time"], y=df_sel["ema21"], line=dict(color="#f77f00", width=1.5), name="EMA 21"))
        fig.update_layout(
            template="plotly_dark", height=380, margin=dict(l=10, r=10, t=25, b=10),
            xaxis_rangeslider_visible=False, title=f"{selected_coin}/USD — 15m Live Candlestick & EMA"
        )
        st.plotly_chart(fig, use_container_width=True)

with report_col:
    st.markdown(f"### 🔍 {selected_coin} — ৩৬০° স্টাডি রিপোর্ট (স্কোর: {sel_analysis['score_360']}%)")
    for r in sel_analysis["reasons"]:
        st.write(f"• {r}")

    # Auto-calculated ATR-based order parameters
    atr_val = sel_analysis["atr"] if sel_analysis["atr"] > 0 else curr_sel_price * 0.015
    rec_dir = sel_analysis["direction"] if sel_analysis["direction"] != "NEUTRAL" else "BUY (Long)"
    rec_amt = min(1000.0, round(st.session_state.cash_balance * 0.10, 2))
    rec_sl = round(curr_sel_price - 1.5 * atr_val, 4) if "BUY" in rec_dir else round(curr_sel_price + 1.5 * atr_val, 4)
    rec_tp = round(curr_sel_price + 2.5 * atr_val, 4) if "BUY" in rec_dir else round(curr_sel_price - 2.5 * atr_val, 4)

    with st.form("manual_order_form"):
        o_dir = st.selectbox("Direction", ["BUY (Long)", "SELL (Short)"], index=0 if "BUY" in rec_dir else 1)
        o_amt = st.number_input("Amount ($)", min_value=10.0, max_value=max(10.0, st.session_state.cash_balance), value=max(10.0, rec_amt))
        o_sl = st.number_input("Stop-Loss ($)", value=float(rec_sl), format="%.4f")
        o_tp = st.number_input("Auto-Sell Target ($)", value=float(rec_tp), format="%.4f")
        submitted = st.form_submit_button("✅ Confirm Order")
        if submitted and curr_sel_price > 0:
            ok, msg = execute_trade(selected_coin, o_dir, o_amt, curr_sel_price, o_sl, o_tp, mode="MANUAL")
            if ok:
                st.success(msg)
            else:
                st.error(msg)

st.divider()

# Row 5: Active Positions & Trade Journal (CSV Export)
pos_col, hist_col = st.columns(2)
with pos_col:
    st.subheader("📌 চলমান পজিশন (Trailing SL সক্রিয়)")
    if st.session_state.positions:
        st.dataframe(pd.DataFrame(st.session_state.positions), use_container_width=True, hide_index=True)
    else:
        st.info("বর্তমানে কোনো ওপেন পজিশন নেই।")

with hist_col:
    st.subheader("📒 ট্রেড জার্নাল ও হিস্ট্রি")
    if st.session_state.trade_history:
        hist_df = pd.DataFrame(st.session_state.trade_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        csv_data = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 ট্রেড জার্নাল ডাউনলোড (CSV / Excel)", data=csv_data, file_name="360_crypto_trade_journal.csv", mime="text/csv")
    else:
        st.info("এখনো কোনো ট্রেড ক্লোজ হয়নি।")
