#!/usr/bin/env python3
"""
Crypto Futures Signals Bot (Binance USDT-M) — STABLE EDITION

• Market: ONLY USDT-M perpetual futures (no spot/delivery mixing)
• Horizons: 4h view (from 1h bars) and 24h view (from 4h bars)
• Core bias: EMA200 + RSI; Filters: ADX + volume
• Confirmations (scoring): MACD, Bollinger (bandwidth), Stoch RSI, CMF
• Levels: SL via swing & 1.5×ATR; TP1/TP2/TP3 via R-multiples
• Ranking: Avg ADX with funding penalty (prefer cheaper direction)
• Output: Telegram cards (LONG/SHORT/SETUP/WAIT) with explanations

Quick start:
    pip install -U ccxt pandas requests
    python bot6.py
"""

from __future__ import annotations
import os
import time
import traceback
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import requests

try:
    import ccxt
except Exception:
    raise SystemExit("Please install ccxt: pip install ccxt")

# ============================ Config ============================ #
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "1800"))  # 30 min
TOP_LIMIT = int(os.getenv("TOP_LIMIT", "50"))          # universe size
TOP_REPORT = int(os.getenv("TOP_REPORT", "5"))         # report count
VOL_USD_MIN = float(os.getenv("VOL_USD_MIN", "5000000"))  # $5M 24h quote vol

# --- TELEGRAM CONFIG (embedded as requested) ---
BOT_TOKEN = "8282008727:AAF7sNkP0-LrP3l7jNSJFwt-UqezzNbZdMQ"
CHAT_ID = "134296797"

# Risk presets
PRESET = os.getenv("PRESET", "moderate").lower()
if PRESET == "conservative":
    ADX_MIN = float(os.getenv("ADX_MIN", "25"))
    VOL_FILTER = float(os.getenv("VOL_FILTER", "1.5"))
elif PRESET == "aggressive":
    ADX_MIN = float(os.getenv("ADX_MIN", "15"))
    VOL_FILTER = float(os.getenv("VOL_FILTER", "0.7"))
else:  # moderate
    ADX_MIN = float(os.getenv("ADX_MIN", "20"))
    VOL_FILTER = float(os.getenv("VOL_FILTER", "1.0"))

# sanity guard: max distance of SL from price (fraction)
MAX_SL_DIST = float(os.getenv("MAX_SL_DIST", "0.35"))  # 35%

# Funding weight for ranking penalty (prefer cheaper direction)
FUNDING_WEIGHT = float(os.getenv("FUNDING_WEIGHT", "50"))

# Confirmations toggles & weights
USE_MACD = bool(int(os.getenv("USE_MACD", "1")))
USE_BB = bool(int(os.getenv("USE_BB", "1")))
USE_STOCHRSI = bool(int(os.getenv("USE_STOCHRSI", "1")))
USE_CMF = bool(int(os.getenv("USE_CMF", "1")))

W_MACD = float(os.getenv("W_MACD", "1.0"))
W_BB = float(os.getenv("W_BB", "1.0"))
W_STOCH = float(os.getenv("W_STOCH", "1.0"))
W_CMF = float(os.getenv("W_CMF", "1.0"))
SCORE_MIN = float(os.getenv("SCORE_MIN", "2.0"))
VOTE_MIN = int(os.getenv("VOTE_MIN", "2"))

# Timeframes feeding feature sets
TF_4H_SRC = os.getenv("TF_4H_SRC", "1h")
TF_24H_SRC = os.getenv("TF_24H_SRC", "4h")

# newline helper (avoid very long string literals)
NL = chr(10)

# ============================ Exchange ============================ #
exchange = ccxt.binance({
    "enableRateLimit": True,
    "options": {"defaultType": "future"},  # USDT-M futures only
})

# ========= Compatibility helper (snake/camel across ccxt versions) ========= #

def _call(exchange, camel: str, snake: str, *args, **kwargs):
    func = getattr(exchange, snake, None)
    if callable(func):
        return func(*args, **kwargs)
    func = getattr(exchange, camel, None)
    if callable(func):
        return func(*args, **kwargs)
    raise AttributeError("Missing both {} and {} on exchange".format(snake, camel))

# ====================== Telegram helpers ====================== #

def send_telegram(text: str) -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("[WARN] BOT_TOKEN/CHAT_ID not set; printing message instead:")
        print(text)
        return
    try:
        url = "https://api.telegram.org/bot{}/sendMessage".format(BOT_TOKEN)
        data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
        r = requests.post(url, data=data, timeout=15)
        if r.status_code != 200:
            print("[TELEGRAM] non-200: {}".format(r.text))
    except Exception:
        print("[TELEGRAM] error:")
        print(traceback.format_exc())

# ====================== Market helpers ====================== #

def fetch_futures_symbols(limit: int = TOP_LIMIT) -> List[str]:
    """Return top-N ACTIVE USDT-M PERPETUAL linear contracts with decent liquidity."""
    markets = exchange.load_markets()
    tickers = _call(exchange, 'fetchTickers', 'fetch_tickers')
    candidates: List[str] = []
    for m in markets.values():
        if not m.get("active", True):
            continue
        if not (m.get("contract") and m.get("linear") and m.get("quote") == "USDT"):
            continue
        ctype = str(m.get("info", {}).get("contractType") or "").upper()
        if ctype != "PERPETUAL":
            continue
        sym = m["symbol"]  # e.g., BTC/USDT:USDT
        if ":USDT" not in sym:
            continue
        t = tickers.get(sym, {})
        qv = float(t.get("quoteVolume") or (t.get("info", {}) or {}).get("quoteVolume") or 0.0)
        if qv < VOL_USD_MIN:
            continue
        try:
            oi = _call(exchange, 'fetchOpenInterest', 'fetch_open_interest', sym)
            oi_val = float(oi.get("openInterestAmount", 0) or oi.get("openInterest", 0) or 0)
            if oi_val <= 0:
                continue
        except Exception:
            pass
        candidates.append(sym)

    def qv_key(s: str) -> float:
        t = tickers.get(s, {})
        return float(t.get("quoteVolume") or (t.get("info", {}) or {}).get("quoteVolume") or 0.0)

    candidates.sort(key=qv_key, reverse=True)
    return candidates[:limit]

# ====================== Indicators ====================== #

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["h"], df["l"], df["c"]
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr_val)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx_val.fillna(0.0)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["h"], df["l"], df["c"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# Extra confirmations

def macd_line(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def bollinger(close: pd.Series, length: int = 20, k: float = 2.0):
    mid = close.rolling(length).mean()
    std = close.rolling(length).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    bandwidth = (upper - lower) / (mid.replace(0, np.nan))
    return mid, upper, lower, bandwidth

def stochastic_rsi(close: pd.Series, rsi_len: int = 14, stoch_len: int = 14):
    r = rsi(close, rsi_len)
    min_r = r.rolling(stoch_len).min()
    max_r = r.rolling(stoch_len).max()
    srsi = (r - min_r) / (max_r - min_r + 1e-9)
    k = srsi.rolling(3).mean()
    d = k.rolling(3).mean()
    return k, d

def cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
    high, low, close, vol = df["h"], df["l"], df["c"], df["v"]
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
    mfv = mfm * vol
    return mfv.rolling(length).sum() / (vol.rolling(length).sum() + 1e-9)

# Swing levels for SL/TP

def local_extrema(series: pd.Series, window: int = 10):
    lows = series.rolling(window, min_periods=window).min()
    highs = series.rolling(window, min_periods=window).max()
    return lows, highs

# ====================== Funding ====================== #

def fetch_funding_rate_map(symbols: List[str]) -> dict:
    out = {s: 0.0 for s in symbols}
    try:
        data = _call(exchange, 'fetchFundingRates', 'fetch_funding_rates', symbols)
        for s, rec in data.items():
            fr = rec.get("fundingRate")
            if fr is None:
                fr = (rec.get("info", {}) or {}).get("lastFundingRate")
            out[s] = float(fr or 0.0)
    except Exception:
        try:
            data = _call(exchange, 'fetchFundingRates', 'fetch_funding_rates')
            for s in symbols:
                rec = data.get(s, {})
                fr = rec.get("fundingRate") or (rec.get("info", {}) or {}).get("lastFundingRate")
                out[s] = float(fr or 0.0)
        except Exception:
            pass
    return out

# ====================== Signal logic ====================== #

def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    arr = _call(exchange, 'fetchOHLCV', 'fetch_ohlcv', symbol, timeframe, limit)
    df = pd.DataFrame(arr, columns=["ts", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def compute_signal(symbol: str, timeframe: str) -> Tuple[str, float, float, Optional[float], Optional[float], Optional[float], Optional[float], float, str]:
    """Return: (side, adx, price, sl, tp1, tp2, tp3, conf_score, conf_tags)"""
    df = fetch_ohlcv(symbol, timeframe, limit=300)
    if df.empty or len(df) < 50:
        return ("NEUTRAL", 0.0, float('nan'), None, None, None, None, 0.0, "")

    df["ema200"] = ema(df["c"], 200)
    df["rsi"] = rsi(df["c"], 14)
    df["ADX"] = adx(df, 14)
    df["ATR"] = atr(df, 14)

    # confirmations
    macd_val = macd_sig = macd_hist = pd.Series(index=df.index, dtype=float)
    bb_mid = bb_up = bb_low = bb_bw = pd.Series(index=df.index, dtype=float)
    s_k = s_d = pd.Series(index=df.index, dtype=float)
    cmf_val = pd.Series(index=df.index, dtype=float)
    if USE_MACD:
        macd_val, macd_sig, macd_hist = macd_line(df["c"])  # type: ignore
    if USE_BB:
        bb_mid, bb_up, bb_low, bb_bw = bollinger(df["c"])   # type: ignore
    if USE_STOCHRSI:
        s_k, s_d = stochastic_rsi(df["c"])                  # type: ignore
    if USE_CMF:
        cmf_val = cmf(df)

    last = df.iloc[-1]
    price = float(last["c"])

    # --- Sanity check price vs live ticker; fix possible scale mismatch ---
    try:
        t = _call(exchange, 'fetchTicker', 'fetch_ticker', symbol)
        last_t = float(t.get('last') or t.get('close') or (t.get('info', {}) or {}).get('lastPrice') or 0.0)
        if last_t > 0 and abs(price - last_t) / max(1.0, abs(last_t)) > 0.1:
            price = last_t
    except Exception:
        pass

    # volume filter (avoid div by 0)
    vol_mean = float(df["v"].rolling(50).mean().iloc[-1]) if len(df) >= 50 else float(last["v"])
    vol_ok = last["v"] >= VOL_FILTER * (vol_mean if vol_mean > 0 else last["v"])  # avoid div by 0

    if (last["ADX"] < ADX_MIN) or (not vol_ok):
        return ("NEUTRAL", float(last.get("ADX", 0.0)), price, None, None, None, None, 0.0, "")

    # base directional bias
    bias_long = (price > float(last["ema200"])) and (last["rsi"] > 55)
    bias_short = (price < float(last["ema200"])) and (last["rsi"] < 45)

    score = 0.0
    tags: List[str] = []

    if bias_long or bias_short:
        score += 1.0
        tags.append("bias")

    # MACD confirmation
    if USE_MACD and not macd_hist.isna().all():
        mh = float(macd_hist.iloc[-1]) if not np.isnan(macd_hist.iloc[-1]) else 0.0
        ml = float(macd_val.iloc[-1]) if not np.isnan(macd_val.iloc[-1]) else 0.0
        ms = float(macd_sig.iloc[-1]) if not np.isnan(macd_sig.iloc[-1]) else 0.0
        if bias_long and (mh > 0) and (ml > ms):
            score += W_MACD; tags.append("macd+")
        if bias_short and (mh < 0) and (ml < ms):
            score += W_MACD; tags.append("macd-")

    # Bollinger confirmation (bandwidth expansion)
    if USE_BB and not bb_mid.isna().all():
        mid = float(bb_mid.iloc[-1]) if not np.isnan(bb_mid.iloc[-1]) else price
        bw = float(bb_bw.iloc[-1]) if not np.isnan(bb_bw.iloc[-1]) else 0.0
        bw_prev = float(bb_bw.iloc[-10:-1].median()) if not bb_bw.iloc[-10:-1].isna().all() else 0.0
        expanding = bw > bw_prev
        if bias_long and price >= mid and expanding:
            score += W_BB; tags.append("bb+")
        if bias_short and price <= mid and expanding:
            score += W_BB; tags.append("bb-")

    # Stochastic RSI confirmation
    if USE_STOCHRSI and not s_k.isna().all():
        k = float(s_k.iloc[-1]) if not np.isnan(s_k.iloc[-1]) else 0.5
        k_prev = float(s_k.iloc[-2]) if len(s_k) > 1 and not np.isnan(s_k.iloc[-2]) else k
        if bias_long and (k > 0.6) and (k_prev <= k):
            score += W_STOCH; tags.append("stoch+")
        if bias_short and (k < 0.4) and (k_prev >= k):
            score += W_STOCH; tags.append("stoch-")

    # CMF confirmation (volume flow)
    if USE_CMF and not cmf_val.isna().all():
        c = float(cmf_val.iloc[-1])
        if bias_long and c > 0:
            score += W_CMF; tags.append("cmf+")
        if bias_short and c < 0:
            score += W_CMF; tags.append("cmf-")

    side = "NEUTRAL"
    if bias_long and (score >= SCORE_MIN) and (len(tags) >= VOTE_MIN):
        side = "LONG"
    if bias_short and (score >= SCORE_MIN) and (len(tags) >= VOTE_MIN):
        side = "SHORT"

    # If still neutral — return early (no levels)
    if side == "NEUTRAL":
        return ("NEUTRAL", float(last["ADX"]), price, None, None, None, None, score, ",".join(tags))

    # dynamic stops and targets
    lows, highs = local_extrema(df["c"], window=10)
    swing_low = float(lows.iloc[-2]) if pd.notna(lows.iloc[-2]) else float(df["l"].iloc[-10:-1].min())
    swing_high = float(highs.iloc[-2]) if pd.notna(highs.iloc[-2]) else float(df["h"].iloc[-10:-1].max())
    a = float(df["ATR"].iloc[-1]) if float(df["ATR"].iloc[-1]) > 0 else max(1e-8, price * 0.001)

    if side == "LONG":
        sl = min(price - 1.5 * a, swing_low)
        # sanity: SL distance must be reasonable
        if sl <= 0 or (price - sl) / max(price, 1e-9) > MAX_SL_DIST:
            return ("NEUTRAL", float(last["ADX"]), price, None, None, None, None, score, ",".join(tags + ["sl_sanity_fail"]))
        risk = max(price - sl, price * 0.001)
        tp1 = price + 2.0 * risk
        tp2 = price + 1.618 * risk
        tp3 = price + 2.618 * risk
        return (side, float(last["ADX"]), price, sl, tp1, tp2, tp3, score, ",".join(tags))
    elif side == "SHORT":
        sl = max(price + 1.5 * a, swing_high)
        if sl <= 0 or (sl - price) / max(price, 1e-9) > MAX_SL_DIST:
            return ("NEUTRAL", float(last["ADX"]), price, None, None, None, None, score, ",".join(tags + ["sl_sanity_fail"]))
        risk = max(sl - price, price * 0.001)
        tp1 = price - 2.0 * risk
        tp2 = price - 1.618 * risk
        tp3 = price - 2.618 * risk
        return (side, float(last["ADX"]), price, sl, tp1, tp2, tp3, score, ",".join(tags))
    else:
        return ("NEUTRAL", float(last["ADX"]), price, None, None, None, None, score, ",".join(tags))

# ====================== Orchestrator ====================== #

def run_once() -> str:
    syms = fetch_futures_symbols(TOP_LIMIT)
    if not syms:
        return NL.join([
            "ℹ️ *Futures Signals*",
            "",
            "No liquid USDT-M PERPETUAL contracts found (check VOL_USD_MIN).",
        ])

    fr_map = fetch_funding_rate_map(syms)

    rows = []
    for s in syms:
        try:
            sig4 = compute_signal(s, TF_4H_SRC)
            sig24 = compute_signal(s, TF_24H_SRC)
            side4, adx4, price4, sl4, tp14, tp24, tp34, conf4, tags4 = sig4
            side24, adx24, *_ = sig24

            if side4 == "NEUTRAL" and side24 == "NEUTRAL":
                continue

            # strength score = average ADX across horizons, penalize expensive funding
            score = (adx4 + adx24) / 2.0
            fr = float(fr_map.get(s, 0.0))
            if side4 == "LONG" and fr > 0:
                score -= FUNDING_WEIGHT * fr
            elif side4 == "SHORT" and fr < 0:
                score -= FUNDING_WEIGHT * abs(fr)

            rows.append((s, side4, side24, score, price4, sl4, tp14, tp24, tp34, fr, conf4, tags4))
        except ccxt.RateLimitExceeded:
            time.sleep(1)
        except Exception:
            print("[ERR] {}".format(s))
            print(traceback.format_exc())
            continue

    if not rows:
        return NL.join([
            "ℹ️ *Futures Signals*",
            "",
            "No strong signals right now (filtered by ADX/volume).",
        ])

    rows.sort(key=lambda x: x[3], reverse=True)
    rows = rows[:TOP_REPORT]

    def render_card(sym: str, s4: str, s24: str, score: float, p: float,
                     sl, tp1, tp2, tp3, fr: float, conf4: float, tags4: str) -> str:
        action = "⏳ WAIT"
        reason = []
        if s4 == "LONG" and s24 == "LONG":
            action = "✅ LONG"; reason.append("4H and 24H aligned")
        elif s4 == "SHORT" and s24 == "SHORT":
            action = "✅ SHORT"; reason.append("4H and 24H aligned")
        elif s24 == "LONG" and s4 == "NEUTRAL":
            action = "🟡 LONG SETUP"; reason.append("24H bullish; waiting for 4H trigger: close>EMA200 or MACD>0 or StochRSI>0.6")
        elif s24 == "SHORT" and s4 == "NEUTRAL":
            action = "🟠 SHORT SETUP"; reason.append("24H bearish; waiting for 4H trigger: close<EMA200 or MACD<0 or StochRSI<0.4")
        else:
            action = "⏳ WAIT"; reason.append("Timeframes diverge, no consensus")

        fr_txt = "{:.4%}".format(fr) if fr else "0.0000%"
        conf_txt = "{:.1f}".format(conf4) if conf4 > 0 else "0.0"
        tags_txt = tags4 if tags4 else "—"

        if sl is not None and tp1 is not None:
            levels = [
                "• Entry: `{}`".format("{:.6g}".format(p)),
                "• Stop: `{}`".format("{:.6g}".format(sl)),
                "• Targets: `TP1 {}` · `TP2 {}` · `TP3 {}`".format(
                    "{:.6g}".format(tp1), "{:.6g}".format(tp2), "{:.6g}".format(tp3)
                ),
            ]
        else:
            levels = ["• Levels will appear once 4H confirms"]

        body = [
            "*{}*".format(sym),
            "*Action:* {}".format(action),
            "*Context:* 4H={} · 24H={} · Strength≈{} · Funding={}".format(s4, s24, "{:.2f}".format(score), fr_txt),
            "*Confirmations:* {} ({})".format(conf_txt, tags_txt),
            *levels,
            "• Why: {}.".format("; ".join(reason)),
            "• Risk: ≤1% per trade; do not average against stop.",
        ]
        return NL.join(body)

    header = NL.join([
        "🔥 *TOP-{} signals* (Futures, PRESET={}, ADX≥{})".format(TOP_REPORT, PRESET, ADX_MIN),
        "",
        "*Legend:* ‘LONG/SHORT’ — entry; ‘SETUP’ — waiting for 4H trigger; ‘WAIT’ — skip.",
        "4H triggers: close<>EMA200, MACD hist <> 0, StochRSI K <> threshold.",
    ])

    lines = [header]
    for (sym, s4, s24, score, p, sl, tp1, tp2, tp3, fr, conf4, tags4) in rows:
        lines.append(render_card(sym, s4, s24, score, p, sl, tp1, tp2, tp3, fr, conf4, tags4))
        lines.append("")

    return NL.join(lines).strip()

# ====================== Main loop ====================== #

def main_loop():
    print("Futures Signals Bot — Binance USDT-M. Press Ctrl+C to stop.")
    print("Polling every {}s | Top universe={} | Report top={} | PRESET={}".format(POLL_SECONDS, TOP_LIMIT, TOP_REPORT, PRESET))
    while True:
        try:
            msg = run_once()
            send_telegram(msg)
        except Exception:
            print("[FATAL] in main loop:")
            print(traceback.format_exc())
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main_loop()
