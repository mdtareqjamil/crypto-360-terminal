import json
import time
import threading
import requests
import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timezone
from websocket import WebSocketApp

st.set_page_config(
    page_title="360° AI Crypto Demo Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

COINS = {
    "BTC": "BTC/USD",
    "ETH": "ETH/USD",
    "SOL": "SOL/USD",
    "XRP": "XRP/USD",
    "DOGE": "DOGE/USD",
    "ADA": "ADA/USD"
}

KRAKEN_REST_PAIRS = {
    "BTC": "XXBTZUSD",
    "ETH": "XETHZUSD",
    "SOL": "SOLUSD",
    "XRP": "XXRPZUSD",
    "DOGE": "XDGUSD",
    "ADA": "ADAUSD"
}

# ==========================================
# 1. THREAD-SAFE LIVE TICK STORE (WEBSOCKET + REST FALLBACK)
# ==========================================
@st.cache_resource
def get_live_store():
    store = {
        "lock": threading.Lock(),
        "prices": {
            sym: {"price": 0.0, "change_24h": 0.0, "volume_24h": 0.0, "updated_at": 0.0}
            for sym in COINS
        },
        "ws_started": False
    }

    def seed_initial_rest():
        try:
            pairs = ",".join(KRAKEN_REST_PAIRS.values())
            url = f"https://api.kraken.com/0/public/Ticker?pair={pairs}"
            res = requests.get(url, timeout=4).json().get("result", {})
            with store["lock"]:
                for sym, k_pair in KRAKEN_REST_PAIRS.items():
                    key = next((k for k in res if k_pair in k or sym in k), None)
                    if key:
                        info = res[key]
                        last_p = float(info["c"][0])
                        open_p = float(info["o"])
                        vol_24h = float(info["v"][1])
                        chg = ((last_p - open_p) / open_p * 100) if open_p else 0.0
                        store["prices"][sym] = {
                            "price": last_p,
                            "change_24h": chg,
                            "volume_24h": vol_24h,
                            "updated_at": time.time()
                        }
        except Exception:
            pass

    def on_message(ws, message):
        try:
            msg = json.loads(message)
            if msg.get("channel") == "ticker" and isinstance(msg.get("data"), list):
                with store["lock"]:
                    for item in msg["data"]:
                        sym_pair = item.get("symbol", "")
                        sym = next((s for s, p in COINS.items() if p == sym_pair), None)
                        if sym:
                            last_p = float(item.get("last", store["prices"][sym]["price"]))
                            vol_24h = float(item.get("volume", store["prices"][sym]["volume_24h"]))
                            chg = float(item.get("change_pct", store["prices"][sym]["change_24h"]))
                            store["prices"][sym] = {
                                "price": last_p,
                                "change_24h": chg,
                                "volume_24h": vol_24h,
                                "updated_at": time.time()
                            }
        except Exception:
            pass

    def run_ws():
        seed_initial_rest()
        while True:
            try:
                ws = WebSocketApp(
                    "wss://ws.kraken.com/v2",
                    on_open=lambda w: w.send(json.dumps({
                        "method": "subscribe",
                        "params": {"channel": "ticker", "symbol": list(COINS.values())}
                    })),
                    on_message=on_message
                )
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception:
                pass
            seed_initial_rest()
            time.sleep(2)

    if not store["ws_started"]:
        store["ws_started"] = True
        t = threading.Thread(target=run_ws, daemon=True)
        t.start()

    return store

store = get_live_store()

def get_prices_snapshot():
    with store["lock"]:
        return {k: dict(v) for k, v in store["prices"].items()}

# ==========================================
# 2. SESSION STATE ($10,000 DEMO)
# ==========================================
if "cash_balance" not in st.session_state:
    st.session_state.cash_balance = 10000.0
if "positions" not in st.session_state:
    st.session_state.positions = []
if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "auto_bot" not in st.session_state:
    st.session_state.auto_bot = False
if "daily_loss" not in st.session_state:
    st.session_state.daily_loss = 0.0
if "kill_switch_triggered" not in st.session_state:
    st.session_state.kill_switch_triggered = False

# ==========================================
# 3. OHLC, MACRO & NEWS ENGINES
# ==========================================
@st.cache_data(ttl=20)
def fetch_kraken_ohlc(coin_sym, interval=15):
    k_pair = KRAKEN_REST_PAIRS[coin_sym]
    url = f"https://api.kraken.com/0/public/OHLC?pair={k_pair}&interval={interval}"
    try:
        res = requests.get(url, timeout=5).json().get("result", {})
        matched_key = next((k for k in res if k != "last"), None)
        if not matched_key:
            return pd.DataFrame()
        df = pd.DataFrame(
            res[matched_key],
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
    feeds = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    bull_words = ["surge", "rally", "bull", "gain", "adoption", "etf", "approval", "high", "growth", "inflow"]
    bear_words = ["crash", "drop", "bear", "hack", "ban", "lawsuit", "sec", "plunge", "fear", "outflow", "fall"]
    bull_c, bear_c = 0, 0
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:8]:
                t = entry.title.lower()
                if any(w in t for w in bull_words):
                    bull_c += 1
                if any(w in t for w in bear_words):
                    bear_c += 1
        except Exception:
            continue
    total = bull_c + bear_c
    return (int((bull_c / total) * 100) if total > 0 else 55)

def analyze_coin_360(df, macro_data, news_score):
    if df.empty or len(df) < 25:
        return {"direction": "NEUTRAL", "score_360": 50, "pattern": "Loading", "atr": 0.0, "reasons": ["ডেটা লোড হচ্ছে..."]}

    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    avg_vol = df["volume"].tail(20).mean()
    vol_ratio = df["volume"].iloc[-1] / avg_vol if avg_vol > 0 else 1.0

    c, p = df.iloc[-2], df.iloc[-3]
    body = abs(c["close"] - c["open"])
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]

    pattern, candle_bias = "Standard Candle", 0
    if lower_wick > 2 * body and upper_wick < body:
        pattern, candle_bias = "Bullish Hammer", 12
    elif upper_wick > 2 * body and lower_wick < body:
        pattern, candle_bias = "Bearish Shooting Star", -12
    elif c["close"] > c["open"] and p["close"] < p["open"] and c["close"] > p["open"] and c["open"] < p["close"]:
        pattern, candle_bias = "Bullish Engulfing", 15
    elif c["close"] < c["open"] and p["close"] > p["open"] and c["open"] > p["close"] and c["close"] < p["open"]:
        pattern, candle_bias = "Bearish Engulfing", -15

    ema9, ema21 = df["ema9"].iloc[-1], df["ema21"].iloc[-1]
    atr = df["atr"].iloc[-1] if not np.isnan(df["atr"].iloc[-1]) else (df["close"].iloc[-1] * 0.015)

    is_uptrend = ema9 > ema21 and df["low"].iloc[-2] > df["low"].iloc[-6]
    is_downtrend = ema9 < ema21 and df["high"].iloc[-2] < df["high"].iloc[-6]

    bull_p, bear_p = 45, 45
    reasons = []
    if is_uptrend:
        bull_p += 18
        reasons.append("Uptrend কাঠামো (EMA 9 > EMA 21 ও Higher Low)।")
    elif is_downtrend:
        bear_p += 18
        reasons.append("Downtrend কাঠামো (EMA 9 < EMA 21 ও Lower High)।")
    else:
        reasons.append("বাজার সাইডওয়েজ / কনসোলিডেশনে আছে।")

    if candle_bias != 0:
        if candle_bias > 0:
            bull_p += candle_bias
        else:
            bear_p += abs(candle_bias)
        reasons.append(f"ক্যান্ডেল প্যাটার্ন: {pattern}।")

    if vol_ratio >= 1.15:
        bull_p += 7
        bear_p += 7
        reasons.append(f"ভলিউম রেশিও {vol_ratio:.2f}x (শক্তিশালী অংশগ্রহণ)।")

    if macro_data["macro_score"] >= 60:
        bull_p += 8
        reasons.append("ম্যাক্রো পরিবেশ ক্রিপ্টোর জন্য অনুকূল।")
    elif macro_data["macro_score"] <= 40:
        bull_p -= 10
        bear_p += 8
        reasons.append("ম্যাক্রো সতর্কতা: ঝুঁকি বিবেচনায় Buy স্কোর কমানো হয়েছে।")

    if news_score >= 60:
        bull_p += 6
        reasons.append(f"নিউজ সেন্টিমেন্ট ইতিবাচক ({news_score}%)।")
    elif news_score <= 40:
        bear_p += 6
        reasons.append(f"নিউজ সেন্টিমেন্ট নেতিবাচক ({100-news_score}%)।")

    if bull_p >= bear_p:
        direction = "BUY (Long)" if is_uptrend else "NEUTRAL"
        score_360 = min(92, max(40, int(bull_p)))
    else:
        direction = "SELL (Short)" if is_downtrend else "NEUTRAL"
        score_360 = min(92, max(40, int(bear_p)))

    return {"direction": direction, "score_360": score_360, "pattern": pattern, "atr": atr, "reasons": reasons}

# ==========================================
# 4. TRADE EXECUTION & POSITION ENGINE
# ==========================================
def execute_trade(coin, side, amount_usd, entry_price, sl_price, tp_price, mode="MANUAL"):
    if st.session_state.kill_switch_triggered or st.session_state.daily_loss >= 500.0:
        st.session_state.kill_switch_triggered = True
        st.session_state.auto_bot = False
        return False, "Kill-Switch সক্রিয়: দৈনিক লস $500 (৫%) ছুঁয়েছে!"
    if any(p["coin"] == coin for p in st.session_state.positions):
        return False, f"{coin}-এ ইতিমধ্যে একটি পজিশন চালু আছে।"
    if amount_usd <= 0 or amount_usd > st.session_state.cash_balance:
        return False, "পর্যাপ্ত ক্যাশ ব্যালেন্স নেই।"

    qty = amount_usd / entry_price
    st.session_state.cash_balance -= amount_usd
    st.session_state.positions.append({
        "id": f"ORD-{int(time.time()*1000)%100000}",
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "coin": coin, "side": side, "entry": entry_price, "qty": qty,
        "invested": amount_usd, "sl": sl_price, "tp": tp_price, "mode": mode
    })
    return True, f"{coin} {side} অর্ডার সম্পন্ন হয়েছে!"

def step_positions_and_bot(live_prices, analyses):
    remaining = []
    for pos in st.session_state.positions:
        coin = pos["coin"]
        cp = live_prices.get(coin, {}).get("price", 0.0)
        if cp <= 0:
            remaining.append(pos)
            continue

        if pos["side"] == "BUY (Long)":
            pnl = (cp - pos["entry"]) * pos["qty"]
            if cp >= pos["entry"] * 1.012:
                pos["sl"] = max(pos["sl"], pos["entry"] * 1.003)
            closed = (cp <= pos["sl"]) or (cp >= pos["tp"])
            reason = "Take-Profit Hit" if cp >= pos["tp"] else "Stop-Loss / Trailing SL Hit"
        else:
            pnl = (pos["entry"] - cp) * pos["qty"]
            if cp <= pos["entry"] * 0.988:
                pos["sl"] = min(pos["sl"], pos["entry"] * 0.997)
            closed = (cp >= pos["sl"]) or (cp <= pos["tp"])
            reason = "Take-Profit Hit" if cp <= pos["tp"] else "Stop-Loss / Trailing SL Hit"

        if closed:
            st.session_state.cash_balance += max(0.0, pos["invested"] + pnl)
            if pnl < 0:
                st.session_state.daily_loss += abs(pnl)
                if st.session_state.daily_loss >= 500.0:
                    st.session_state.kill_switch_triggered = True
                    st.session_state.auto_bot = False
            st.session_state.trade_history.append({
                "Time": pos["time"], "Coin": coin, "Side": pos["side"], "Mode": pos["mode"],
                "Entry": round(pos["entry"], 4), "Exit": round(cp, 4),
                "Invested ($)": round(pos["invested"], 2), "PnL ($)": round(pnl, 2), "Status": reason
            })
        else:
            remaining.append(pos)
    st.session_state.positions = remaining

    if st.session_state.auto_bot and not st.session_state.kill_switch_triggered:
        for coin, info in analyses.items():
            if info["score_360"] >= 62 and info["direction"] in ["BUY (Long)", "SELL (Short)"] and len(st.session_state.positions) < 3:
                price = live_prices.get(coin, {}).get("price", 0.0)
                if price > 0:
                    atr = info["atr"] if info["atr"] > 0 else price * 0.015
                    amt = min(1000.0, st.session_state.cash_balance * 0.15)
                    sl = price - 1.5 * atr if "BUY" in info["direction"] else price + 1.5 * atr
                    tp = price + 2.5 * atr if "BUY" in info["direction"] else price - 2.5 * atr
                    execute_trade(coin, info["direction"], amt, price, sl, tp, mode="AUTO-BOT")

# ==========================================
# 5. TOP HEADER & CONTROLS (STATIC SKELETON)
# ==========================================
st.title("⚡ 360° AI Crypto Demo Terminal (Smooth Live Mode)")
st.caption("WebSocket লাইভ ফিড + Fragment আপডেট — পুরো পেজ রিলোড না করেই প্রাইস ও পজিশন আপডেট হয়।")

c_col1, c_col2, c_col3 = st.columns([2, 2, 1.5])
with c_col1:
    st.session_state.auto_bot = st.toggle(
        "🤖 AUTO ACTION (স্কোর ≥ ৬২%)",
        value=st.session_state.auto_bot,
        disabled=st.session_state.kill_switch_triggered
    )
with c_col2:
    if st.session_state.kill_switch_triggered:
        st.error(f"🛑 KILL-SWITCH ACTIVE! Daily Loss: ${st.session_state.daily_loss:.2f}")
    else:
        st.success(f"🛡️ Kill-Switch Safe | Daily Loss: ${st.session_state.daily_loss:.2f} / $500.00")
with c_col3:
    if st.button("🔄 Reset ($10,000)"):
        st.session_state.cash_balance = 10000.0
        st.session_state.positions = []
        st.session_state.trade_history = []
        st.session_state.daily_loss = 0.0
        st.session_state.kill_switch_triggered = False
        st.session_state.auto_bot = False
        st.rerun()

macro_data = fetch_background_macro()
news_score = fetch_news_sentiment()
analyses = {coin: analyze_coin_360(fetch_kraken_ohlc(coin, 15), macro_data, news_score) for coin in COINS}

# ==========================================
# 6. LIVE FRAGMENT (UPDATES EVERY 1s WITHOUT FULL PAGE RELOAD)
# ==========================================
@st.fragment(run_every=1)
def live_ticker_and_positions():
    live_prices = get_prices_snapshot()
    step_positions_and_bot(live_prices, analyses)

    invested_total = sum(p["invested"] for p in st.session_state.positions)
    live_pnl = 0.0
    for p in st.session_state.positions:
        cp = live_prices.get(p["coin"], {}).get("price", 0.0)
        if cp > 0:
            live_pnl += (cp - p["entry"]) * p["qty"] if p["side"] == "BUY (Long)" else (p["entry"] - cp) * p["qty"]
    equity = st.session_state.cash_balance + invested_total + live_pnl

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Available Cash", f"${st.session_state.cash_balance:,.2f}")
    m2.metric("Invested Balance", f"${invested_total:,.2f}")
    m3.metric("Live Profit / Loss", f"${live_pnl:+,.2f}", delta=f"{live_pnl:+.2f}")
    m4.metric("Current Equity", f"${equity:,.2f}")

    left_b, right_a = st.columns([3, 2])
    with left_b:
        st.subheader("📊 লাইভ মার্কেট বোর্ড")
        rows = []
        for coin in COINS:
            info = live_prices.get(coin, {})
            a = analyses[coin]
            bias = "🟢 Bullish" if "BUY" in a["direction"] else ("🔴 Bearish" if "SELL" in a["direction"] else "⚪ Neutral")
            rows.append({
                "Coin": coin,
                "Live Price ($)": f"${info.get('price', 0.0):,.4f}",
                "24h (%)": f"{info.get('change_24h', 0.0):+.2f}%",
                "360° Score": f"{a['score_360']}%",
                "Bias": bias
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with right_a:
        st.subheader("🔔 স্মার্ট অপরচুনিটি অ্যালার্ট")
        alerts_shown = 0
        for coin, a in analyses.items():
            if a["score_360"] >= 60:
                st.info(f"**{coin}** → **{a['direction']}** সম্ভাবনা **{a['score_360']}%** ({a['pattern']})")
                alerts_shown += 1
        if alerts_shown == 0:
            st.write("বর্তমানে ≥ ৬০% হাই-কনফিডেন্স সিগন্যালের অপেক্ষায় আছে।")
        st.caption(f"ম্যাক্রো স্কোর: {macro_data['macro_score']}% (S&P: {macro_data['sp500_chg']:+.2f}%, DXY: {macro_data['dxy_chg']:+.2f}%, VIX: {macro_data['vix']:.1f}) | নিউজ: {news_score}% Bullish")

    st.subheader("📌 চলমান পজিশন")
    if st.session_state.positions:
        st.dataframe(pd.DataFrame(st.session_state.positions), use_container_width=True, hide_index=True)
    else:
        st.caption("কোনো ওপেন পজিশন নেই।")

live_ticker_and_positions()

st.divider()

# ==========================================
# 7. MANUAL MODE & 360° SEARCH STUDY
# ==========================================
st.subheader("🖐️ MANUAL MODE & ৩৬০° সার্চ স্টাডি")
selected_coin = st.selectbox("কয়েন নির্বাচন করুন:", list(COINS.keys()))
df_sel = fetch_kraken_ohlc(selected_coin, 15)
sel_info = analyses[selected_coin]
curr_price = get_prices_snapshot().get(selected_coin, {}).get("price", 0.0)

ch_col, rep_col = st.columns([3, 2])
with ch_col:
    if not df_sel.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_sel["time"], open=df_sel["open"], high=df_sel["high"],
            low=df_sel["low"], close=df_sel["close"], name=selected_coin
        ))
        fig.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=25, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

with rep_col:
    st.markdown(f"### 🔍 {selected_coin} — স্কোর: {sel_info['score_360']}%")
    for r in sel_info["reasons"]:
        st.write(f"• {r}")

    atr_v = sel_info["atr"] if sel_info["atr"] > 0 else max(curr_price * 0.015, 0.0001)
    rec_dir = sel_info["direction"] if sel_info["direction"] != "NEUTRAL" else "BUY (Long)"
    rec_amt = min(1000.0, round(st.session_state.cash_balance * 0.10, 2))
    rec_sl = round(curr_price - 1.5 * atr_v, 4) if "BUY" in rec_dir else round(curr_price + 1.5 * atr_v, 4)
    rec_tp = round(curr_price + 2.5 * atr_v, 4) if "BUY" in rec_dir else round(curr_price - 2.5 * atr_v, 4)

    with st.form("manual_order_form"):
        o_dir = st.selectbox("Direction", ["BUY (Long)", "SELL (Short)"], index=0 if "BUY" in rec_dir else 1)
        o_amt = st.number_input("Amount ($)", min_value=10.0, max_value=max(10.0, st.session_state.cash_balance), value=max(10.0, rec_amt))
        o_sl = st.number_input("Stop-Loss ($)", value=float(rec_sl), format="%.4f")
        o_tp = st.number_input("Auto-Sell Target ($)", value=float(rec_tp), format="%.4f")
        if st.form_submit_button("✅ Confirm Order") and curr_price > 0:
            ok, msg = execute_trade(selected_coin, o_dir, o_amt, curr_price, o_sl, o_tp, mode="MANUAL")
            if ok:
                st.success(msg)
            else:
                st.error(msg)

st.divider()

# ==========================================
# 8. TRADE JOURNAL (CSV EXPORT)
# ==========================================
st.subheader("📒 ট্রেড জার্নাল")
if st.session_state.trade_history:
    hist_df = pd.DataFrame(st.session_state.trade_history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 ডাউনলোড ট্রেড জার্নাল (CSV)",
        data=hist_df.to_csv(index=False).encode("utf-8"),
        file_name="crypto_360_journal.csv",
        mime="text/csv"
    )
else:
    st.info("এখনো কোনো ট্রেড সম্পন্ন হয়নি।")
