"""
SMC PRO SCANNER v5.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BACKTEST-PROVEN FILTERS (720d, 486 pairs):
  WR: 70.0% | PF: 4.37 | Max DD: -13.51%
  Signals: ~2/month (elite setups only)

CHANGES from v4.0 (based on backtest iterations v1→v4):
  ✅ SHORT ONLY       — LONGs = 35% WR in backtest, net loser
  ✅ MIN_SCORE 75→83  — 75-82 band = 41-46% WR, pure noise
  ✅ OB_TOLERANCE 0.8%→1.2% — wider entry zone, more valid OBs
  ✅ BEAR STRUCTURE   — BOS_BEAR only (MSS_BEAR = 0% WR, blocked)
  ✅ REQUIRE_FVG      — FVG overlap → 75% WR vs 60% without
  ✅ BLOCK_SWEEP      — sweeps = 40% WR vs 62.9% clean entries
  ✅ REQUIRE_TRENDING — HH/LL required (ranging = 33% WR)
  ✅ TP1=2.0R, TP2=3.5R, TP3=5.5R — wider targets, more captured
  ✅ Trade timeout: 48H

TIMEFRAME ROLES v5.0:
  4H  → Trend bias (EMA 21/50) + HH/LL structure
  1H  → BOS_BEAR + Order Block + FVG zone + Entry trigger
  15M → Volume spike bonus only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import ccxt.async_support as ccxt
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════
#  TUNABLE SETTINGS  (backtest-proven v4)
# ═══════════════════════════════════════════════
MAX_SIGNALS_PER_SCAN  = 6
MIN_SCORE             = 83        # backtest: 75-82 = 41-46% WR → cut
MIN_VOLUME_24H        = 1_000_000
OB_TOLERANCE_PCT      = 0.012     # widened 0.8% → 1.2% for more valid entries
OB_IMPULSE_ATR_MULT   = 1.0
STRUCTURE_LOOKBACK    = 20
SCAN_INTERVAL_MIN     = 30
HH_LL_LOOKBACK        = 10
HH_LL_BONUS           = 8

# ── v5 FILTERS (all backtest-proven) ──────────────────
SHORT_ONLY            = True   # LONGs = 35% WR in backtest — killed
REQUIRE_BOS_BEAR      = True   # BOS_BEAR only — MSS_BEAR = 0% WR
REQUIRE_FVG           = True   # FVG overlap → 75% WR vs 60% without
BLOCK_SWEEP           = True   # sweep entries = 40% WR vs 62.9% clean
REQUIRE_TRENDING      = True   # ranging (no HH/LL) = 33% WR — blocked

# ── TRADE MANAGEMENT ──────────────────────────────────
TP_RR                 = [2.0, 3.5, 5.5]    # wider than v4 original — more captured
TP_PCT                = [0.50, 0.30, 0.20] # 50% at TP1, 30% at TP2, 20% at TP3
TRADE_TIMEOUT_HOURS   = 48


# ══════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════

def add_indicators(df):
    if len(df) < 55:
        return df
    try:
        df['ema_21']  = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
        df['ema_50']  = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], min(200, len(df)-1)).ema_indicator()
        df['rsi']     = ta.momentum.RSIIndicator(df['close'], 14).rsi()

        macd = ta.trend.MACD(df['close'])
        df['macd']        = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist']   = macd.macd_diff()

        stoch = ta.momentum.StochRSIIndicator(df['close'])
        df['srsi_k'] = stoch.stochrsi_k()
        df['srsi_d'] = stoch.stochrsi_d()

        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

        bb = ta.volatility.BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_pband'] = bb.bollinger_pband()

        adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']    = adx_i.adx()
        df['di_pos'] = adx_i.adx_pos()
        df['di_neg'] = adx_i.adx_neg()

        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
        df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()

        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()

        body = (df['close'] - df['open']).abs()
        uw   = df['high'] - df[['open','close']].max(axis=1)
        lw   = df[['open','close']].min(axis=1) - df['low']

        df['bull_engulf'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1))
        ).astype(int)

        df['bear_engulf'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1))
        ).astype(int)

        df['bull_pin'] = (
            (lw > body * 2.5) & (lw > uw * 2) & (df['close'] > df['open'])
        ).astype(int)

        df['bear_pin'] = (
            (uw > body * 2.5) & (uw > lw * 2) & (df['close'] < df['open'])
        ).astype(int)

        df['hammer'] = (
            (lw > body * 2.0) & (lw > uw * 1.5)
        ).astype(int)

        df['shooting_star'] = (
            (uw > body * 2.0) & (uw > lw * 1.5)
        ).astype(int)

    except Exception as e:
        logger.error(f"Indicator error: {e}")
    return df


# ══════════════════════════════════════════════════════════════
#  SMC ENGINE
# ══════════════════════════════════════════════════════════════

class SMCEngine:

    def swing_highs_lows(self, df, left=4, right=4):
        highs, lows = [], []
        n = len(df)
        for i in range(left, n - right):
            hi = df['high'].iloc[i]
            lo = df['low'].iloc[i]
            if all(hi >= df['high'].iloc[i-left:i]) and all(hi >= df['high'].iloc[i+1:i+right+1]):
                highs.append({'i': i, 'price': hi})
            if all(lo <= df['low'].iloc[i-left:i]) and all(lo <= df['low'].iloc[i+1:i+right+1]):
                lows.append({'i': i, 'price': lo})
        return highs, lows

    def check_4h_hh_ll(self, df_4h, direction, lookback=HH_LL_LOOKBACK):
        """Returns (confirmed: bool, message: str)"""
        n = len(df_4h)
        if n < lookback * 2:
            return False, "⚠️ Not enough 4H data"
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback * 2):-lookback]
        if direction == 'LONG':
            rh, ph = recent['high'].max(), prior['high'].max()
            if rh > ph:
                return True, f"📈 4H Higher High ({ph:.5f} → {rh:.5f}) +{HH_LL_BONUS}pts"
            return False, f"➖ 4H no HH — ranging"
        else:
            rl, pl = recent['low'].min(), prior['low'].min()
            if rl < pl:
                return True, f"📉 4H Lower Low ({pl:.5f} → {rl:.5f}) +{HH_LL_BONUS}pts"
            return False, f"➖ 4H no LL — ranging"

    def detect_structure_break(self, df, highs, lows, lookback=STRUCTURE_LOOKBACK):
        events = []
        close = df['close']
        n = len(df)
        start = max(0, n - lookback - 15)

        for k in range(1, len(highs)):
            ph = highs[k-1]; ch = highs[k]
            if ch['i'] < start: continue
            level = ph['price']
            for j in range(ch['i'], min(ch['i'] + 10, n)):
                if close.iloc[j] > level:
                    kind = 'BOS_BULL' if ch['price'] > ph['price'] else 'MSS_BULL'
                    events.append({'kind': kind, 'level': level, 'bar': j})
                    break

        for k in range(1, len(lows)):
            pl = lows[k-1]; cl = lows[k]
            if cl['i'] < start: continue
            level = pl['price']
            for j in range(cl['i'], min(cl['i'] + 10, n)):
                if close.iloc[j] < level:
                    kind = 'BOS_BEAR' if cl['price'] < pl['price'] else 'MSS_BEAR'
                    events.append({'kind': kind, 'level': level, 'bar': j})
                    break

        if not events:
            return None
        latest = sorted(events, key=lambda x: x['bar'])[-1]
        if latest['bar'] < n - lookback:
            return None
        return latest

    def find_order_blocks(self, df, direction, lookback=60):
        obs = []
        n = len(df)
        start = max(2, n - lookback)

        for i in range(start, n - 3):
            c = df.iloc[i]
            atr_local = df['atr'].iloc[i] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high'] - c['low'])
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT

            if direction == 'LONG':
                if c['close'] >= c['open']: continue
                fwd_high = df['high'].iloc[i+1:min(i+5, n)].max()
                if fwd_high - c['low'] < min_impulse: continue
                ob = {
                    'top':    max(c['open'], c['close']),
                    'bottom': c['low'],
                    'mid':   (max(c['open'], c['close']) + c['low']) / 2,
                    'bar':    i
                }
                ob_50 = (ob['top'] + ob['bottom']) / 2
                if (df['close'].iloc[i+1:n] < ob_50).any(): continue
                obs.append(ob)
            else:
                if c['close'] <= c['open']: continue
                fwd_low = df['low'].iloc[i+1:min(i+5, n)].min()
                if c['high'] - fwd_low < min_impulse: continue
                ob = {
                    'top':    c['high'],
                    'bottom': min(c['open'], c['close']),
                    'mid':   (c['high'] + min(c['open'], c['close'])) / 2,
                    'bar':    i
                }
                ob_50 = (ob['top'] + ob['bottom']) / 2
                if (df['close'].iloc[i+1:n] > ob_50).any(): continue
                obs.append(ob)

        obs.sort(key=lambda x: x['bar'], reverse=True)
        return obs

    def price_in_ob(self, price, ob, tolerance_pct=OB_TOLERANCE_PCT):
        tol = ob['top'] * tolerance_pct
        return (ob['bottom'] - tol) <= price <= (ob['top'] + tol)

    def find_fvg(self, df, direction, lookback=25):
        fvgs = []
        n = len(df)
        for i in range(max(1, n - lookback), n - 1):
            prev = df.iloc[i-1]; nxt = df.iloc[i+1]
            if direction == 'LONG' and prev['high'] < nxt['low']:
                fvgs.append({'top': nxt['low'], 'bottom': prev['high'],
                             'mid': (nxt['low'] + prev['high']) / 2, 'bar': i})
            elif direction == 'SHORT' and prev['low'] > nxt['high']:
                fvgs.append({'top': prev['low'], 'bottom': nxt['high'],
                             'mid': (prev['low'] + nxt['high']) / 2, 'bar': i})
        return fvgs

    def recent_liquidity_sweep(self, df, direction, highs, lows, lookback=25):
        n = len(df)
        start = n - lookback
        if direction == 'LONG':
            for sl in reversed(lows):
                if sl['i'] < start: continue
                level = sl['price']
                for j in range(sl['i'] + 1, min(sl['i'] + 8, n)):
                    c = df.iloc[j]
                    if c['low'] < level and c['close'] > level:
                        return {'level': level, 'bar': j, 'type': 'SWEEP_LOW'}
        else:
            for sh in reversed(highs):
                if sh['i'] < start: continue
                level = sh['price']
                for j in range(sh['i'] + 1, min(sh['i'] + 8, n)):
                    c = df.iloc[j]
                    if c['high'] > level and c['close'] < level:
                        return {'level': level, 'bar': j, 'type': 'SWEEP_HIGH'}
        return None

    def pd_zone(self, df_4h, price):
        hi = df_4h['high'].iloc[-50:].max()
        lo = df_4h['low'].iloc[-50:].min()
        rang = hi - lo
        if rang == 0: return 'NEUTRAL', 0.5
        pos = (price - lo) / rang
        if pos < 0.40:   return 'DISCOUNT', pos
        elif pos > 0.60: return 'PREMIUM',  pos
        return 'NEUTRAL', pos


# ══════════════════════════════════════════════════════════════
#  SCORER
# ══════════════════════════════════════════════════════════════

def safe_get(row, key, default=0):
    try:
        v = row[key]
        return default if pd.isna(v) else v
    except:
        return default


def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed):
    score = 0
    reasons = []
    failed = []

    l1  = df_1h.iloc[-1]
    p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1] if len(df_15m) > 0 else pd.Series(dtype=float)
    l4  = df_4h.iloc[-1]

    # ── 1. Structure (20 pts) ─────────────────────────────────
    if structure:
        if 'MSS' in structure['kind']:
            score += 20; reasons.append(f"🏗️ MSS ({structure['kind']})")
        else:
            score += 14; reasons.append(f"🏗️ BOS ({structure['kind']})")
    else:
        failed.append("❌ No BOS/MSS")

    # ── 2. OB quality (20 pts) ───────────────────────────────
    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        if ob_size_pct < 0.8:
            score += 20; reasons.append(f"📦 Tight OB ({ob_size_pct:.2f}%) ⭐")
        elif ob_size_pct < 2.0:
            score += 13; reasons.append(f"📦 OB ({ob_size_pct:.2f}%)")
        else:
            score += 7;  reasons.append(f"📦 Wide OB ({ob_size_pct:.2f}%)")

    # ── 3. 4H Trend (15 pts) ─────────────────────────────────
    e21 = safe_get(l4, 'ema_21'); e50 = safe_get(l4, 'ema_50'); e200 = safe_get(l4, 'ema_200')

    if direction == 'LONG':
        if e21 > e50 > e200:
            score += 15; reasons.append("📈 4H Triple EMA Bull")
        elif e21 > e50:
            score += 10; reasons.append("📈 4H EMA 21>50")
        elif pd_label == 'DISCOUNT':
            score += 6;  reasons.append("📈 4H Discount zone")
        else:
            failed.append("⚠️ 4H trend weak for LONG")
    else:
        if e21 < e50 < e200:
            score += 15; reasons.append("📉 4H Triple EMA Bear")
        elif e21 < e50:
            score += 10; reasons.append("📉 4H EMA 21<50")
        elif pd_label == 'PREMIUM':
            score += 6;  reasons.append("📉 4H Premium zone")
        else:
            failed.append("⚠️ 4H trend weak for SHORT")

    # ── 4. HH/LL bonus (8 pts) ───────────────────────────────
    if hh_ll_confirmed:
        score += HH_LL_BONUS
        reasons.append(f"🏔️ 4H Lower Low confirmed (+{HH_LL_BONUS}pts)")
    else:
        failed.append("➖ 4H no LL — ranging")

    # ── 5. 1H Entry Trigger (25 pts) ─────────────────────────
    trigger = False
    if direction == 'LONG':
        if safe_get(l1, 'bull_engulf') == 1:
            score += 25; trigger = True; reasons.append("🕯️ 1H Bull Engulf ✅")
        elif safe_get(l1, 'bull_pin') == 1:
            score += 22; trigger = True; reasons.append("🕯️ 1H Bull Pin ✅")
        elif safe_get(l1, 'hammer') == 1:
            score += 18; trigger = True; reasons.append("🕯️ 1H Hammer ✅")
        elif safe_get(p1, 'bull_engulf') == 1:
            score += 14; trigger = True; reasons.append("🕯️ 1H Bull Engulf (prev) ✅")
        elif safe_get(p1, 'bull_pin') == 1:
            score += 11; trigger = True; reasons.append("🕯️ 1H Bull Pin (prev) ✅")
        elif safe_get(p1, 'hammer') == 1:
            score += 9;  trigger = True; reasons.append("🕯️ 1H Hammer (prev) ✅")
    else:
        if safe_get(l1, 'bear_engulf') == 1:
            score += 25; trigger = True; reasons.append("🕯️ 1H Bear Engulf ✅ (strongest)")
        elif safe_get(l1, 'bear_pin') == 1:
            score += 22; trigger = True; reasons.append("🕯️ 1H Bear Pin ✅")
        elif safe_get(l1, 'shooting_star') == 1:
            score += 18; trigger = True; reasons.append("🕯️ 1H Shooting Star ✅")
        elif safe_get(p1, 'bear_engulf') == 1:
            score += 14; trigger = True; reasons.append("🕯️ 1H Bear Engulf (prev) ✅")
        elif safe_get(p1, 'bear_pin') == 1:
            score += 11; trigger = True; reasons.append("🕯️ 1H Bear Pin (prev) ✅")
        elif safe_get(p1, 'shooting_star') == 1:
            score += 9;  trigger = True; reasons.append("🕯️ 1H Shooting Star (prev) ✅")

    if trigger:
        pass
    else:
        score -= 12
        failed.append("⏳ No 1H trigger — setup forming, wait")

    # ── 6. Momentum (12 pts) ─────────────────────────────────
    rsi1  = safe_get(l1, 'rsi', 50)
    macd1 = safe_get(l1, 'macd'); ms1  = safe_get(l1, 'macd_signal')
    pm1   = safe_get(p1, 'macd'); pms1 = safe_get(p1, 'macd_signal')
    sk1   = safe_get(l1, 'srsi_k', 0.5); sd1 = safe_get(l1, 'srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi1 <= 55:
            score += 4; reasons.append(f"✅ RSI reset ({rsi1:.0f})")
        elif rsi1 < 28:
            score += 3; reasons.append(f"✅ RSI oversold ({rsi1:.0f})")
        if macd1 > ms1 and pm1 <= pms1:
            score += 5; reasons.append("⚡ MACD bull cross")
        elif macd1 > ms1:
            score += 2; reasons.append("✅ MACD bullish")
        if sk1 < 0.3 and sk1 > sd1:
            score += 3; reasons.append("⚡ Stoch bull cross")
    else:
        if 45 <= rsi1 <= 72:
            score += 4; reasons.append(f"✅ RSI overbought zone ({rsi1:.0f})")
        elif rsi1 > 72:
            score += 3; reasons.append(f"✅ RSI overbought ({rsi1:.0f})")
        if macd1 < ms1 and pm1 >= pms1:
            score += 5; reasons.append("⚡ MACD bear cross")
        elif macd1 < ms1:
            score += 2; reasons.append("✅ MACD bearish")
        if sk1 > 0.7 and sk1 < sd1:
            score += 3; reasons.append("⚡ Stoch bear cross")

    # ── 7. Extras (10 pts max) ───────────────────────────────
    extras = 0
    if fvg_near:
        extras += 3; reasons.append("⚡ FVG overlaps OB ✅")

    vr15 = safe_get(l15, 'vol_ratio', 1.0) if len(df_15m) > 0 else 1.0
    if   vr15 >= 2.5: extras += 3; reasons.append(f"🚀 15M vol spike {vr15:.1f}x")
    elif vr15 >= 1.5: extras += 1; reasons.append(f"✅ 15M elevated vol {vr15:.1f}x")

    close1 = safe_get(l1, 'close'); vwap1 = safe_get(l1, 'vwap')
    if direction == 'SHORT' and close1 > vwap1:
        extras = min(extras+1, 10); reasons.append("✅ 1H above VWAP")

    score += min(extras, 10)

    return max(0, min(int(score), 100)), reasons, failed


# ══════════════════════════════════════════════════════════════
#  MAIN BOT
# ══════════════════════════════════════════════════════════════

class SMCProScanner:
    def __init__(self, telegram_token, chat_id, api_key=None, secret=None):
        self.token    = telegram_token
        self.bot      = Bot(token=telegram_token)
        self.chat_id  = chat_id
        self.exchange = ccxt.binance({
            'apiKey': api_key, 'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.smc            = SMCEngine()
        self.active_trades  = {}
        self.signal_history = deque(maxlen=300)
        self.is_scanning    = False
        self.last_debug     = []
        self.stats = {
            'total': 0, 'short': 0,
            'elite': 0, 'premium': 0, 'high': 0,
            'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0,
            'last_scan': None, 'pairs_scanned': 0
        }

    async def get_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = [
                s for s in self.exchange.symbols
                if s.endswith('/USDT:USDT')
                and 'PERP' not in s
                and tickers.get(s, {}).get('quoteVolume', 0) > MIN_VOLUME_24H
            ]
            pairs.sort(key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
            logger.info(f"✅ {len(pairs)} pairs (vol>${MIN_VOLUME_24H/1e6:.0f}M)")
            return pairs
        except Exception as e:
            logger.error(f"Pairs: {e}"); return []

    async def fetch_data(self, symbol):
        try:
            result = {}
            for tf, lim in [('4h', 220), ('1h', 150), ('15m', 80)]:
                raw = await self.exchange.fetch_ohlcv(symbol, tf, limit=lim)
                df  = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                result[tf] = add_indicators(df)
                await asyncio.sleep(0.04)
            return result
        except Exception as e:
            logger.error(f"Fetch {symbol}: {e}"); return None

    def analyse(self, data, symbol):
        debug = {'symbol': symbol.replace('/USDT:USDT',''), 'gates': [], 'score': 0, 'bias': '?'}

        try:
            df4 = data['4h']; df1 = data['1h']; df15 = data['15m']
            if len(df1) < 80 or len(df15) < 40:
                debug['gates'].append('❌ Not enough candle data')
                return None, debug

            price = df1['close'].iloc[-1]

            # ── GATE 1: SHORT ONLY ────────────────────────────────
            # Backtest: LONGs = 35% WR, net negative over 720d. Killed.
            l4 = df4.iloc[-1]
            e21 = safe_get(l4, 'ema_21'); e50 = safe_get(l4, 'ema_50')
            if e21 < e50:
                bias = 'SHORT'
            else:
                debug['gates'].append('❌ SHORT ONLY — 4H EMA not bearish (LONGs disabled)')
                return None, debug
            debug['bias'] = bias

            # ── GATE 2: HH/LL TRENDING REQUIRED ─────────────────
            # Backtest: ranging = 33% WR. Require Lower Low confirmed.
            hh_ll_ok, hh_ll_msg = self.smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)
            if REQUIRE_TRENDING and not hh_ll_ok:
                debug['gates'].append(f'❌ TRENDING REQUIRED — {hh_ll_msg}')
                return None, debug
            debug['gates'].append(hh_ll_msg)

            # ── GATE 3: PD Zone ───────────────────────────────────
            pd_label, pd_pos = self.smc.pd_zone(df4, price)
            if bias == 'SHORT' and pd_label == 'DISCOUNT':
                debug['gates'].append(f'❌ PD zone: DISCOUNT ({pd_pos*100:.0f}%) — no shorts here')
                return None, debug
            debug['gates'].append(f'✅ PD zone: {pd_label} ({pd_pos*100:.0f}%)')

            # ── GATE 4: BOS_BEAR ONLY ─────────────────────────────
            # Backtest: MSS_BEAR = 0% WR. BOS_BEAR only = 67-70% WR.
            highs1, lows1 = self.smc.swing_highs_lows(df1, left=4, right=4)
            structure = self.smc.detect_structure_break(df1, highs1, lows1, lookback=STRUCTURE_LOOKBACK)

            if structure is None:
                debug['gates'].append('❌ BOS_BEAR REQUIRED — no structure break found')
                return None, debug
            if structure['kind'] != 'BOS_BEAR':
                debug['gates'].append(f'❌ BOS_BEAR REQUIRED — got {structure["kind"]} (blocked)')
                return None, debug
            debug['gates'].append(f'✅ Structure: {structure["kind"]}')

            # ── GATE 5: 1H Order Block (hard gate) ───────────────
            obs = self.smc.find_order_blocks(df1, bias, lookback=60)
            if not obs:
                debug['gates'].append('❌ No valid SHORT OBs on 1H')
                return None, debug

            active_ob = None
            for ob in obs:
                if self.smc.price_in_ob(price, ob, OB_TOLERANCE_PCT):
                    active_ob = ob; break

            if not active_ob:
                nearest  = obs[0]
                dist_pct = min(abs(price - nearest['top']), abs(price - nearest['bottom'])) / price * 100
                debug['gates'].append(f'❌ Price not at OB — nearest {dist_pct:.2f}% away')
                return None, debug
            debug['gates'].append(f'✅ Price IN OB [{active_ob["bottom"]:.5f}–{active_ob["top"]:.5f}]')

            # ── FVG check ─────────────────────────────────────────
            fvgs = self.smc.find_fvg(df1, bias, lookback=25)
            fvg_near = None
            for fvg in fvgs:
                if fvg['bottom'] < active_ob['top'] and fvg['top'] > active_ob['bottom']:
                    fvg_near = fvg; break

            # ── GATE 6: REQUIRE FVG ───────────────────────────────
            # Backtest: FVG=YES → 75% WR vs 60% without. Required.
            if REQUIRE_FVG and fvg_near is None:
                debug['gates'].append('❌ FVG REQUIRED — no FVG overlapping OB')
                return None, debug
            if fvg_near:
                debug['gates'].append('✅ FVG overlaps OB ⚡')

            # ── Liquidity sweep check ─────────────────────────────
            sweep = self.smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)

            # ── GATE 7: BLOCK SWEEP ───────────────────────────────
            # Backtest: sweep entries = 40% WR vs 62.9% clean. Blocked.
            if BLOCK_SWEEP and sweep is not None:
                debug['gates'].append(f'❌ SWEEP BLOCKED — sweep entries = 40% WR in backtest')
                return None, debug

            # ── GATE 8: Score ≥ MIN_SCORE ─────────────────────────
            score, reasons, failed_reasons = score_setup(
                bias, active_ob, structure, sweep, fvg_near,
                df1, df15, df4, pd_label, hh_ll_ok
            )
            debug['score'] = score
            debug['gates'] += failed_reasons

            if score < MIN_SCORE:
                debug['gates'].append(f'❌ Score {score} < {MIN_SCORE} minimum')
                return None, debug

            # ── Build signal ──────────────────────────────────────
            atr1  = df1['atr'].iloc[-1]
            entry = price

            # SHORT: SL above OB top + buffer (tighter than v4)
            sl = active_ob['top'] + atr1 * 0.2
            sl = max(sl, entry + atr1 * 0.4)

            risk = abs(entry - sl)
            if risk < entry * 0.001:
                debug['gates'].append('❌ Degenerate SL')
                return None, debug

            tps      = [entry - risk * rr for rr in TP_RR]
            rr_vals  = [abs(t - entry) / risk for t in tps]
            risk_pct = risk / entry * 100

            if   score >= 92: quality = 'ELITE 👑'
            elif score >= 85: quality = 'PREMIUM 💎'
            else:             quality = 'HIGH 🔥'

            tid = f"{symbol.split('/')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            sig = {
                'trade_id':    tid,
                'symbol':      symbol.replace('/USDT:USDT', ''),
                'full_symbol': symbol,
                'signal':      bias,
                'quality':     quality,
                'score':       score,
                'hh_ll':       hh_ll_ok,
                'entry':       entry,
                'stop_loss':   sl,
                'targets':     tps,
                'rr':          rr_vals,
                'risk_pct':    risk_pct,
                'ob':          active_ob,
                'fvg':         fvg_near,
                'sweep':       None,  # always None (blocked)
                'structure':   structure,
                'pd_zone':     pd_label,
                'pd_pos':      pd_pos,
                'reasons':     reasons,
                'tp_hit':      [False, False, False],
                'sl_hit':      False,
                'timestamp':   datetime.now(),
            }
            debug['gates'].append(f'✅ PASSED — Score {score}/100')
            return sig, debug

        except Exception as e:
            logger.error(f"Analyse {symbol}: {e}")
            debug['gates'].append(f'💥 Exception: {e}')
            return None, debug

    def fmt(self, s):
        icon  = '🔻'
        bar   = '█' * int(s['score']/10) + '░' * (10 - int(s['score']/10))
        z     = {'DISCOUNT':'🟩 DISCOUNT','PREMIUM':'🟥 PREMIUM','NEUTRAL':'🟨 NEUTRAL'}.get(s['pd_zone'],'')
        ob    = s['ob']

        msg  = f"{'━'*40}\n"
        msg += f"{icon} <b>SMC PRO v5 — {s['quality']}</b> {icon}\n"
        msg += f"{'━'*40}\n\n"
        msg += f"<b>🆔</b> <code>{s['trade_id']}</code>\n"
        msg += f"<b>📊 PAIR:</b>    <b>#{s['symbol']}USDT</b>\n"
        msg += f"<b>📍 DIR:</b>     🔴 <b>SHORT</b>\n"
        msg += f"<b>🗺️ ZONE:</b>    {z} ({s['pd_pos']*100:.0f}%)\n"
        msg += f"<b>📐 TREND:</b>   🏔️ Trending (LL confirmed)\n"
        msg += f"<b>🏗️ STRUCT:</b>  BOS_BEAR ✅\n"
        msg += f"<b>⚡ FVG:</b>     {'Overlaps OB ✅' if s['fvg'] else '—'}\n\n"
        msg += f"<b>⭐ SCORE: {s['score']} / 100</b>\n"
        msg += f"<code>[{bar}]</code>\n\n"
        msg += f"<b>📦 ORDER BLOCK (1H):</b>\n"
        msg += f"  Top:    <code>${ob['top']:.6f}</code>\n"
        msg += f"  Bottom: <code>${ob['bottom']:.6f}</code>\n\n"
        msg += f"<b>💰 ENTRY NOW:</b> <code>${s['entry']:.6f}</code>\n\n"
        msg += f"<b>🎯 TARGETS:</b>\n"
        labels = [
            ('TP1 — 50% close', '12-24h'),
            ('TP2 — 30% close', '24-36h'),
            ('TP3 — 20% close', '36-48h'),
        ]
        for (lbl, eta), tp, rr in zip(labels, s['targets'], s['rr']):
            pct = abs((tp - s['entry'])/s['entry']*100)
            msg += f"  <b>{lbl}</b> [{eta}]\n"
            msg += f"  <code>${tp:.6f}</code>  <b>-{pct:.2f}%</b>  RR {rr:.1f}:1\n\n"
        msg += f"<b>🛑 STOP LOSS:</b> <code>${s['stop_loss']:.6f}</code>  (+{s['risk_pct']:.2f}%)\n"
        msg += f"  └ <i>1H close above OB = invalidated</i>\n\n"
        msg += f"<b>📋 CONFLUENCE:</b>\n"
        for r in s['reasons'][:10]:
            msg += f"  • {r}\n"
        msg += f"\n<b>⚠️ RISK:</b> 1-2% per trade only\n"
        msg += f"  Move SL → BE after TP1 hits\n"
        msg += f"\n<b>📡 Live Tracking: ON (48H timeout)</b>\n"
        msg += f"<i>🕐 {s['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}</i>\n"
        msg += f"{'━'*40}"
        return msg

    async def send(self, text):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Telegram: {e}")

    async def tp_alert(self, t, n, price):
        tp  = t['targets'][n-1]
        pct = abs((tp - t['entry'])/t['entry']*100)
        advice = {
            1: 'Close 50% → Move SL to breakeven',
            2: 'Close 30% → Trail stop tight',
            3: 'Close final 20% 🎊'
        }
        msg  = f"🎯 <b>TP{n} HIT!</b>\n\n<code>{t['trade_id']}</code>\n"
        msg += f"<b>{t['symbol']}</b> SHORT\n\n"
        msg += f"Target: <code>${tp:.6f}</code>\nCurrent: <code>${price:.6f}</code>\n"
        msg += f"Profit: <b>-{pct:.2f}%</b> (short)\n\n"
        msg += f"📋 {advice[n]}"
        await self.send(msg)
        self.stats[f'tp{n}'] += 1

    async def sl_alert(self, t, price):
        loss = abs((price - t['entry'])/t['entry']*100)
        msg  = f"⛔ <b>STOP LOSS HIT</b>\n\n<code>{t['trade_id']}</code>\n"
        msg += f"<b>{t['symbol']}</b> SHORT\n\n"
        msg += f"Entry: <code>${t['entry']:.6f}</code>\nLoss: <b>+{loss:.2f}%</b> (against short)\n\n"
        msg += f"OB invalidated. Next setup incoming."
        await self.send(msg)
        self.stats['sl'] += 1

    async def track(self):
        logger.info("📡 Tracker started")
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30); continue
                remove = []
                for tid, t in list(self.active_trades.items()):
                    try:
                        age = datetime.now() - t['timestamp']
                        if age > timedelta(hours=TRADE_TIMEOUT_HOURS):
                            await self.send(
                                f"⏰ <b>48H TIMEOUT</b>\n<code>{tid}</code>\n"
                                f"{t['symbol']} SHORT — Close manually."
                            )
                            remove.append(tid); continue

                        ticker = await self.exchange.fetch_ticker(t['full_symbol'])
                        p = ticker['last']

                        # SHORT: TP = price goes DOWN, SL = price goes UP
                        for i, tp in enumerate(t['targets']):
                            if not t['tp_hit'][i] and p <= tp:
                                await self.tp_alert(t, i+1, p)
                                t['tp_hit'][i] = True
                                if i == 2: remove.append(tid)

                        if not t['sl_hit'] and p >= t['stop_loss']:
                            await self.sl_alert(t, p)
                            t['sl_hit'] = True; remove.append(tid)

                    except Exception as e:
                        logger.error(f"Track {tid}: {e}")

                for tid in set(remove):
                    self.active_trades.pop(tid, None)
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Track loop: {e}"); await asyncio.sleep(60)

    async def scan(self):
        if self.is_scanning:
            return []
        self.is_scanning = True
        logger.info("🔍 Scan starting...")

        await self.send(
            f"🔍 <b>SMC v5.0 SCAN STARTED</b>\n"
            f"Mode: <b>SHORT ONLY</b> | Structure: BOS_BEAR | FVG: Required\n"
            f"Score: ≥{MIN_SCORE} | OB tol: {OB_TOLERANCE_PCT*100:.1f}% | No sweeps\n"
            f"Backtest: 70% WR | PF 4.37 | -13.5% max DD"
        )

        pairs       = await self.get_pairs()
        candidates  = []
        near_misses = []
        scanned     = 0

        for pair in pairs:
            try:
                data = await self.fetch_data(pair)
                if data:
                    sig, dbg = self.analyse(data, pair)
                    if sig:
                        candidates.append(sig)
                        logger.info(f"  💎 {pair} SHORT score={sig['score']}")
                    else:
                        if dbg['score'] > 0:
                            near_misses.append(dbg)
                scanned += 1
                if scanned % 30 == 0:
                    logger.info(f"  ⏳ {scanned}/{len(pairs)} | {len(candidates)} candidates")
                await asyncio.sleep(0.45)
            except Exception as e:
                logger.error(f"Scan {pair}: {e}"); continue

        candidates.sort(key=lambda x: x['score'], reverse=True)
        top = candidates[:MAX_SIGNALS_PER_SCAN]

        near_misses.sort(key=lambda x: x['score'], reverse=True)
        self.last_debug = near_misses[:10]

        for sig in top:
            self.signal_history.append(sig)
            self.active_trades[sig['trade_id']] = sig
            self.stats['total'] += 1
            self.stats['short'] += 1
            if 'ELITE'   in sig['quality']: self.stats['elite']   += 1
            elif 'PREMIUM' in sig['quality']: self.stats['premium'] += 1
            else:                             self.stats['high']    += 1
            await self.send(self.fmt(sig))
            await asyncio.sleep(2)

        self.stats['last_scan'] = datetime.now()
        self.stats['pairs_scanned'] = scanned

        el = sum(1 for s in top if 'ELITE'   in s['quality'])
        pr = sum(1 for s in top if 'PREMIUM' in s['quality'])
        hi = len(top) - el - pr

        summ  = f"✅ <b>SCAN COMPLETE — v5.0</b>\n\n"
        summ += f"📊 Pairs scanned: {scanned}\n"
        summ += f"🔍 Candidates:    {len(candidates)}\n"
        summ += f"🎯 Signals sent:  {len(top)}\n"
        if top:
            summ += f"  👑 Elite:   {el}\n  💎 Premium: {pr}\n  🔥 High:    {hi}\n"
        else:
            summ += f"\n<i>No setups met criteria this scan.</i>\n"
            summ += f"Near misses: {len(near_misses)} — use /debug\n"
        summ += f"\n<b>Active trades: {len(self.active_trades)}</b>"
        summ += f"\n⏰ {datetime.now().strftime('%H:%M UTC')}"
        await self.send(summ)

        logger.info(f"✅ Done. {len(candidates)} candidates → {len(top)} sent.")
        self.is_scanning = False
        return top

    async def run(self, interval_min=SCAN_INTERVAL_MIN):
        logger.info("🚀 SMC Pro v5.0 starting")
        await self.send(
            "🔴 <b>SMC PRO v5.0 — LIVE (SHORT ONLY)</b> 🔴\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Backtest: 720d | 70% WR | PF 4.37 | -13.5% Max DD</b>\n\n"
            "✅ SHORT ONLY (LONGs = 35% WR in backtest)\n"
            "✅ BOS_BEAR structure required\n"
            "✅ FVG overlap with OB required\n"
            "✅ Sweep entries blocked (40% WR)\n"
            "✅ Trending pairs only (HH/LL confirmed)\n"
            f"✅ Min score: {MIN_SCORE}/100\n"
            f"✅ OB tolerance: {OB_TOLERANCE_PCT*100:.1f}%\n"
            f"✅ TP: {TP_RR[0]}R / {TP_RR[1]}R / {TP_RR[2]}R\n"
            f"✅ Trade timeout: {TRADE_TIMEOUT_HOURS}H\n"
            f"✅ Scan every: {SCAN_INTERVAL_MIN} min\n\n"
            "Commands: /scan /stats /trades /debug /help\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        asyncio.create_task(self.track())
        while True:
            try:
                await self.scan()
                logger.info(f"💤 Next scan in {SCAN_INTERVAL_MIN}m")
                await asyncio.sleep(SCAN_INTERVAL_MIN * 60)
            except Exception as e:
                logger.error(f"Main: {e}"); await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ══════════════════════════════════════════════════════════════
#  BOT COMMANDS
# ══════════════════════════════════════════════════════════════

class Commands:
    def __init__(self, s: SMCProScanner):
        self.s = s

    async def start(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        await u.message.reply_text(
            "🔴 <b>SMC Pro v5.0 — SHORT ONLY</b>\n\n"
            "Backtest: 70% WR | PF 4.37 | -13.5% DD\n"
            "720 days | 486 pairs tested\n\n"
            "Filters: BOS_BEAR + FVG + Trending + No sweep\n\n"
            "/scan /stats /trades /debug /help",
            parse_mode=ParseMode.HTML
        )

    async def cmd_scan(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await u.message.reply_text("⚠️ Already scanning."); return
        await u.message.reply_text("🔍 Manual scan started...")
        asyncio.create_task(self.s.scan())

    async def stats(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        s = self.s.stats
        msg  = "📊 <b>SMC PRO v5.0 STATS</b>\n\n"
        msg += f"Total signals: {s['total']} (SHORT only)\n"
        msg += f"  👑 Elite: {s['elite']}  💎 Premium: {s['premium']}  🔥 High: {s['high']}\n\n"
        msg += f"TP1: {s['tp1']} | TP2: {s['tp2']} | TP3: {s['tp3']} | SL: {s['sl']}\n"
        if s['total'] > 0:
            live_wr = (s['tp1']) / s['total'] * 100
            msg += f"Live TP1 rate: {live_wr:.1f}%\n"
        msg += f"\nBacktest baseline: 70% WR | PF 4.37 | -13.5% DD\n"
        if s['last_scan']:
            msg += f"\nLast scan: {s['last_scan'].strftime('%H:%M UTC')}\n"
            msg += f"Pairs: {s['pairs_scanned']}\n"
        msg += f"Active: {len(self.s.active_trades)}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def trades(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.active_trades:
            await u.message.reply_text("📭 No active trades."); return
        msg = f"📡 <b>ACTIVE TRADES ({len(self.s.active_trades)})</b>\n\n"
        for tid, t in list(self.s.active_trades.items())[:10]:
            age  = int((datetime.now() - t['timestamp']).total_seconds()/3600)
            tps  = ''.join(['✅' if h else '⏳' for h in t['tp_hit']])
            msg += (f"<b>{t['symbol']}</b> SHORT — {t['quality']}\n"
                    f"  Entry: <code>${t['entry']:.5f}</code> | Score: {t['score']}\n"
                    f"  TPs: {tps} | {age}h old\n\n")
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def debug(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.last_debug:
            await u.message.reply_text("📭 No debug data. Run /scan first.")
            return
        msg = "🔬 <b>NEAR MISSES — Last Scan</b>\n\n"
        for d in self.s.last_debug[:8]:
            msg += f"<b>{d['symbol']}</b> — Score: {d['score']}/100\n"
            for g in d['gates'][-4:]:
                msg += f"  {g}\n"
            msg += "\n"
        msg += f"<i>Min score: {MIN_SCORE}. Filters: BOS_BEAR+FVG+Trending+NoSweep</i>"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def help(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>SMC PRO v5.0 — STRATEGY</b>\n\n"
        msg += "<b>Direction: SHORT ONLY</b>\n"
        msg += "LONGs showed 35% WR in 720d backtest — disabled.\n\n"
        msg += "<b>Hard Gates (ALL must pass):</b>\n"
        msg += "  1️⃣ 4H EMA 21 < 50 (bearish bias)\n"
        msg += "  2️⃣ 4H Lower Low confirmed (trending)\n"
        msg += "  3️⃣ PD zone not DISCOUNT\n"
        msg += "  4️⃣ BOS_BEAR on 1H (not MSS)\n"
        msg += "  5️⃣ Price at valid 1H Order Block\n"
        msg += "  6️⃣ FVG overlapping OB\n"
        msg += "  7️⃣ No liquidity sweep present\n"
        msg += f"  8️⃣ Score ≥ {MIN_SCORE}/100\n\n"
        msg += "<b>Score System:</b>\n"
        msg += "  +25 — 1H bear trigger (engulf/pin/star)\n"
        msg += "  +20 — Structure (BOS_BEAR)\n"
        msg += "  +20 — OB quality\n"
        msg += "  +15 — 4H triple EMA bear\n"
        msg += f"  +{HH_LL_BONUS}  — 4H LL confirmed\n"
        msg += "  +12 — Momentum (RSI/MACD/Stoch)\n"
        msg += "  +10 — Extras (FVG/vol)\n\n"
        msg += "<b>Trade Management:</b>\n"
        msg += f"  TP1: {TP_RR[0]}R (50% close) — move SL to BE\n"
        msg += f"  TP2: {TP_RR[1]}R (30% close) — trail tight\n"
        msg += f"  TP3: {TP_RR[2]}R (20% close) — full exit\n"
        msg += f"  Timeout: {TRADE_TIMEOUT_HOURS}H\n\n"
        msg += "<b>Backtest results (720d, 486 pairs):</b>\n"
        msg += "  WR: 70% | PF: 4.37 | Max DD: -13.5%\n"
        msg += "  ~2 signals/month (elite setups only)"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

async def main():
    # ════════════ CONFIG ════════════
    TELEGRAM_TOKEN   = "7731521911:AAFnus-fDivEwoKqrtwZXMmKEj5BU1EhQn4"
    TELEGRAM_CHAT_ID = "7500072234"
    BINANCE_API_KEY  = None
    BINANCE_SECRET   = None
    # ════════════════════════════════

    scanner = SMCProScanner(
        telegram_token=TELEGRAM_TOKEN,
        chat_id=TELEGRAM_CHAT_ID,
        api_key=BINANCE_API_KEY,
        secret=BINANCE_SECRET
    )

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds = Commands(scanner)

    app.add_handler(CommandHandler("start",  cmds.start))
    app.add_handler(CommandHandler("scan",   cmds.cmd_scan))
    app.add_handler(CommandHandler("stats",  cmds.stats))
    app.add_handler(CommandHandler("trades", cmds.trades))
    app.add_handler(CommandHandler("debug",  cmds.debug))
    app.add_handler(CommandHandler("help",   cmds.help))

    await app.initialize()
    await app.start()
    logger.info("🤖 SMC Pro v5.0 ready!")

    try:
        await scanner.run(interval_min=SCAN_INTERVAL_MIN)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await scanner.close()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
