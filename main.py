"""
SMC PRO SCANNER v5.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGES from v4.0:
  [IMP-1] OB Quality: Added displacement strength (body/ATR > 1.5 → bonus)
          OB scored on size + displacement, not size alone
  [IMP-2] ADX Chop Filter: adx_4h < 18 → skip pair entirely (no chop traps)
  [IMP-3] PD Zone Bonus: LONG in discount <30% → +5pts / SHORT in premium >70% → +5pts
  [IMP-4] Structure Strength: break candle body > 1.2 ATR → +4 bonus pts
  [IMP-5] Indicator cleanup: removed CMF, MFI, Bollinger, EMA200, ADX di+/di-
          Kept: EMA21, EMA50, ATR, RSI, MACD, Volume, VWAP, ADX (scalar only)
  [IMP-6] Dynamic TP3: strong trend (HH/LL + triple EMA + ADX>25) → TP3 = 5R
  [IMP-7] Displacement Gate: BOS/MSS candle body > 1.2 ATR AND vol > 1.5x avg
          required — fake BOS penalised -8pts

TIMEFRAME ROLES v5.0:
  4H  → Trend bias (EMA21/50) + HH/LL + ADX chop filter
  1H  → BOS/MSS + OB displacement + Entry trigger candle
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
#  TUNABLE SETTINGS
# ═══════════════════════════════════════════════
MAX_SIGNALS_PER_SCAN  = 6
MIN_SCORE             = 75
MIN_VOLUME_24H        = 5_000_000
OB_TOLERANCE_PCT      = 0.008
OB_IMPULSE_ATR_MULT   = 1.0
STRUCTURE_LOOKBACK    = 20
SCAN_INTERVAL_MIN     = 60
HH_LL_LOOKBACK        = 10
HH_LL_BONUS           = 8

# [IMP-2] ADX chop threshold — 4H ADX below this = skip
ADX_CHOP_THRESHOLD    = 18
# [IMP-6] ADX trend threshold — above this = expand TP3 to 5R
ADX_TREND_THRESHOLD   = 25
# [IMP-7] Displacement gate thresholds
DISP_BODY_ATR_MULT    = 1.2   # break candle body must be > 1.2 ATR
DISP_VOL_MULT         = 1.5   # break candle volume must be > 1.5x average
# [IMP-1] OB displacement bonus threshold
OB_DISP_ATR_MULT      = 1.5   # OB candle body > 1.5 ATR = displacement bonus


# ══════════════════════════════════════════════════════════════
#  INDICATORS  [IMP-5: stripped to essentials]
# ══════════════════════════════════════════════════════════════

def add_indicators(df):
    """
    Lean indicator set — only what SMC scoring actually uses:
    EMA21, EMA50, ATR, RSI, MACD, Volume SMA/ratio, VWAP, ADX (scalar)
    Candle pattern flags (engulf, pin, hammer, shooting_star)
    """
    if len(df) < 55:
        return df
    try:
        # Trend
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], 21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], 50).ema_indicator()

        # Volatility
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()

        # Momentum
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()

        macd = ta.trend.MACD(df['close'])
        df['macd']        = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        stoch = ta.momentum.StochRSIIndicator(df['close'])
        df['srsi_k'] = stoch.stochrsi_k()
        df['srsi_d'] = stoch.stochrsi_d()

        # ADX — scalar strength only (no DI+/DI-)
        adx_i        = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']    = adx_i.adx()

        # Volume
        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

        # VWAP
        tp           = (df['high'] + df['low'] + df['close']) / 3
        df['vwap']   = (tp * df['volume']).cumsum() / df['volume'].cumsum()

        # Candle geometry helpers
        body = (df['close'] - df['open']).abs()
        uw   = df['high'] - df[['open', 'close']].max(axis=1)
        lw   = df[['open', 'close']].min(axis=1) - df['low']
        df['body'] = body          # expose for displacement checks

        # Trigger patterns (1H)
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
        n = len(df_4h)
        if n < lookback * 2:
            return False, "⚠️ Not enough 4H data for HH/LL check"
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback * 2):-lookback]
        if direction == 'LONG':
            rh, ph = recent['high'].max(), prior['high'].max()
            if rh > ph:
                return True,  f"📈 4H Higher High ({ph:.5f} → {rh:.5f}) +{HH_LL_BONUS}pts"
            return False, f"➖ 4H no HH ({rh:.5f} ≤ {ph:.5f}) — ranging"
        else:
            rl, pl = recent['low'].min(), prior['low'].min()
            if rl < pl:
                return True,  f"📉 4H Lower Low ({pl:.5f} → {rl:.5f}) +{HH_LL_BONUS}pts"
            return False, f"➖ 4H no LL ({rl:.5f} ≥ {pl:.5f}) — ranging"

    def detect_structure_break(self, df, highs, lows, lookback=STRUCTURE_LOOKBACK):
        """
        [IMP-7] Attaches displacement quality to each event:
          - disp_body_ok: break candle body > DISP_BODY_ATR_MULT * ATR
          - disp_vol_ok:  break candle volume > DISP_VOL_MULT * vol_sma
        """
        events = []
        close  = df['close']
        n      = len(df)
        start  = max(0, n - lookback - 15)

        for k in range(1, len(highs)):
            ph = highs[k-1]; ch = highs[k]
            if ch['i'] < start: continue
            level = ph['price']
            for j in range(ch['i'], min(ch['i'] + 10, n)):
                if close.iloc[j] > level:
                    kind = 'BOS_BULL' if ch['price'] > ph['price'] else 'MSS_BULL'
                    events.append({
                        'kind': kind, 'level': level, 'bar': j,
                        'disp': self._displacement_ok(df, j)
                    })
                    break

        for k in range(1, len(lows)):
            pl = lows[k-1]; cl = lows[k]
            if cl['i'] < start: continue
            level = pl['price']
            for j in range(cl['i'], min(cl['i'] + 10, n)):
                if close.iloc[j] < level:
                    kind = 'BOS_BEAR' if cl['price'] < pl['price'] else 'MSS_BEAR'
                    events.append({
                        'kind': kind, 'level': level, 'bar': j,
                        'disp': self._displacement_ok(df, j)
                    })
                    break

        if not events:
            return None
        latest = sorted(events, key=lambda x: x['bar'])[-1]
        if latest['bar'] < n - lookback:
            return None
        return latest

    def _displacement_ok(self, df, bar_idx):
        """Returns dict with body_ok and vol_ok flags for a given bar."""
        try:
            row  = df.iloc[bar_idx]
            atr  = row.get('atr', np.nan)
            body = row.get('body', abs(row['close'] - row['open']))
            vsma = row.get('vol_sma', np.nan)
            vol  = row['volume']
            body_ok = (not np.isnan(atr) and atr > 0 and body > DISP_BODY_ATR_MULT * atr)
            vol_ok  = (not np.isnan(vsma) and vsma > 0 and vol > DISP_VOL_MULT * vsma)
            return {'body_ok': body_ok, 'vol_ok': vol_ok}
        except Exception:
            return {'body_ok': False, 'vol_ok': False}

    def find_order_blocks(self, df, direction, lookback=60):
        """
        Classifies each OB as 'swing' or 'internal' — mirroring LuxAlgo zones:
          🔵 swing    = OB anchored at a major swing point (left=10, right=10)
          🟢🔴 internal = OB anchored at a minor internal pivot (left=3, right=3)

        Only OBs with ob_type in ('swing', 'internal') are returned.
        Unclassified OBs (no nearby pivot) are discarded — these are the
        noise OBs that LuxAlgo would not colour.

        Also carries disp_ratio (body/ATR) for displacement scoring [IMP-1].
        """
        obs   = []
        n     = len(df)
        start = max(2, n - lookback)

        # Build swing pivot index sets (LuxAlgo swingsLengthInput=50 → we use left=10/right=10 on 1H)
        swing_pivot_bars    = self._get_pivot_bars(df, left=10, right=10)
        # Build internal pivot index sets (LuxAlgo internal size=5 → left=3/right=3)
        internal_pivot_bars = self._get_pivot_bars(df, left=3,  right=3)

        for i in range(start, n - 3):
            c         = df.iloc[i]
            atr_local = df['atr'].iloc[i] if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i]) else (c['high'] - c['low'])
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT

            body_size  = abs(c['close'] - c['open'])
            disp_ratio = (body_size / atr_local) if atr_local > 0 else 0

            if direction == 'LONG':
                if c['close'] >= c['open']: continue
                fwd_high = df['high'].iloc[i+1:min(i+5, n)].max()
                if fwd_high - c['low'] < min_impulse: continue
                ob = {
                    'top':        max(c['open'], c['close']),
                    'bottom':     c['low'],
                    'mid':       (max(c['open'], c['close']) + c['low']) / 2,
                    'bar':        i,
                    'disp_ratio': disp_ratio,
                    'ob_type':    None
                }
                ob_50 = (ob['top'] + ob['bottom']) / 2
                if (df['close'].iloc[i+1:n] < ob_50).any(): continue

            else:
                if c['close'] <= c['open']: continue
                fwd_low = df['low'].iloc[i+1:min(i+5, n)].min()
                if c['high'] - fwd_low < min_impulse: continue
                ob = {
                    'top':        c['high'],
                    'bottom':     min(c['open'], c['close']),
                    'mid':       (c['high'] + min(c['open'], c['close'])) / 2,
                    'bar':        i,
                    'disp_ratio': disp_ratio,
                    'ob_type':    None
                }
                ob_50 = (ob['top'] + ob['bottom']) / 2
                if (df['close'].iloc[i+1:n] > ob_50).any(): continue

            # ── Classify: swing > internal > discard ─────────────
            # Check if the OB candle (or within ±2 bars) sits at a pivot
            nearby = range(max(0, i - 2), min(n, i + 3))
            if any(b in swing_pivot_bars for b in nearby):
                ob['ob_type'] = 'swing'       # 🔵 Blue — strongest
            elif any(b in internal_pivot_bars for b in nearby):
                ob['ob_type'] = 'internal'    # 🟢🔴 Green/Red
            else:
                continue   # no pivot nearby → not a real LuxAlgo OB, discard

            obs.append(ob)

        obs.sort(key=lambda x: x['bar'], reverse=True)
        return obs

    def _get_pivot_bars(self, df, left=5, right=5):
        """
        Returns a set of bar indices that are pivot highs or lows
        with the given left/right confirmation window.
        Mirrors LuxAlgo's leg() detection logic.
        """
        pivot_bars = set()
        n = len(df)
        for i in range(left, n - right):
            hi = df['high'].iloc[i]
            lo = df['low'].iloc[i]
            if (all(hi >= df['high'].iloc[i-left:i]) and
                    all(hi >= df['high'].iloc[i+1:i+right+1])):
                pivot_bars.add(i)
            if (all(lo <= df['low'].iloc[i-left:i]) and
                    all(lo <= df['low'].iloc[i+1:i+right+1])):
                pivot_bars.add(i)
        return pivot_bars

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
        hi   = df_4h['high'].iloc[-50:].max()
        lo   = df_4h['low'].iloc[-50:].min()
        rang = hi - lo
        if rang == 0: return 'NEUTRAL', 0.5
        pos = (price - lo) / rang
        if pos < 0.40:   return 'DISCOUNT', pos
        elif pos > 0.60: return 'PREMIUM',  pos
        return 'NEUTRAL', pos


# ══════════════════════════════════════════════════════════════
#  SCORER  — v5.0 with all 7 improvements wired in
# ══════════════════════════════════════════════════════════════

def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, pd_pos,
                hh_ll_confirmed, adx_4h):
    score   = 0
    reasons = []
    failed  = []

    l1  = df_1h.iloc[-1]
    p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]
    l4  = df_4h.iloc[-1]

    # ── 1. Structure (20 pts) + displacement bonus [IMP-4] + gate penalty [IMP-7] ─
    if structure:
        disp  = structure.get('disp', {})
        body_ok = disp.get('body_ok', False)
        vol_ok  = disp.get('vol_ok',  False)

        if 'MSS' in structure['kind']:
            score += 20
            reasons.append(f"🏗️ MSS — Early Reversal ({structure['kind']})")
        else:
            score += 14
            reasons.append(f"🏗️ BOS — Pullback Entry ({structure['kind']})")

        # [IMP-4] Strong break candle body → +4
        if body_ok:
            score += 4
            reasons.append("💥 Strong break candle (body > 1.2 ATR) +4pts")
        else:
            reasons.append("⚠️ Weak break candle body")

        # [IMP-7] Both displacement conditions needed; penalise if missing
        if body_ok and vol_ok:
            reasons.append("✅ Displacement confirmed (body + volume)")
        elif not body_ok and not vol_ok:
            score -= 8
            failed.append("❌ Fake BOS? Break candle: weak body & weak volume -8pts")
        elif not vol_ok:
            score -= 3
            failed.append("⚠️ Break candle low volume -3pts")
        elif not body_ok:
            score -= 3
            failed.append("⚠️ Break candle weak body -3pts")
    else:
        failed.append("❌ No BOS/MSS in last 20 candles")

    # ── 2. Order Block quality (20 pts) + type bonus + displacement [IMP-1] ──
    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        disp_ratio  = ob.get('disp_ratio', 0)
        ob_type     = ob.get('ob_type', 'internal')

        # Base quality score
        if ob_size_pct < 0.8:
            score += 20
            reasons.append(f"📦 Tight OB ({ob_size_pct:.2f}%) — high quality")
        elif ob_size_pct < 2.0:
            score += 13
            reasons.append(f"📦 OB ({ob_size_pct:.2f}%)")
        else:
            score += 7
            reasons.append(f"📦 Wide OB ({ob_size_pct:.2f}%) — lower quality")

        # OB type bonus — mirrors LuxAlgo zone hierarchy
        if ob_type == 'swing':
            score += 8
            reasons.append("🔵 Swing OB (strongest zone) +8pts")
        else:
            score += 3
            reasons.append("🟢🔴 Internal OB +3pts")

        # [IMP-1] Displacement strength bonus
        if disp_ratio >= OB_DISP_ATR_MULT:
            score += 5
            reasons.append(f"🔥 OB displacement strong (body {disp_ratio:.1f}x ATR) +5pts")
        elif disp_ratio >= 1.0:
            score += 2
            reasons.append(f"✅ OB moderate displacement ({disp_ratio:.1f}x ATR)")
        else:
            failed.append(f"⚠️ OB weak displacement ({disp_ratio:.1f}x ATR)")
    else:
        failed.append("❌ No valid OB found")

    # ── 3. 4H Trend Alignment (15 pts) ───────────────────────
    e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0)
    triple_ema = False
    if direction == 'LONG':
        # v5 removed ema_200 from indicators [IMP-5] — use only 21/50
        if e21 > e50:
            score += 15; reasons.append("📈 4H EMA 21>50 Bull")
            triple_ema = True
        elif pd_label == 'DISCOUNT':
            score += 6;  reasons.append("📈 4H Discount Zone (counter-trend OK)")
        else:
            failed.append("⚠️ 4H trend weak for LONG")
    else:
        if e21 < e50:
            score += 15; reasons.append("📉 4H EMA 21<50 Bear")
            triple_ema = True
        elif pd_label == 'PREMIUM':
            score += 6;  reasons.append("📉 4H Premium Zone (counter-trend OK)")
        else:
            failed.append("⚠️ 4H trend weak for SHORT")

    # ── 4. 4H HH/LL Bonus (8 pts) ────────────────────────────
    if hh_ll_confirmed:
        score += HH_LL_BONUS
        reasons.append(f"🏔️ 4H HH/LL confirmed (+{HH_LL_BONUS}pts)")
    else:
        failed.append(f"➖ 4H HH/LL not confirmed — ranging")

    # ── 5. 1H Entry Trigger (25 pts) ─────────────────────────
    trigger = False; trigger_label = ""

    if direction == 'LONG':
        if l1.get('bull_engulf', 0) == 1:
            score += 25; trigger = True
            trigger_label = "🕯️ 1H Bullish Engulfing ✅ (strongest)"
        elif l1.get('bull_pin', 0) == 1:
            score += 22; trigger = True
            trigger_label = "🕯️ 1H Bullish Pin Bar ✅"
        elif l1.get('hammer', 0) == 1:
            score += 18; trigger = True
            trigger_label = "🕯️ 1H Hammer ✅"
        elif p1.get('bull_engulf', 0) == 1:
            score += 14; trigger = True
            trigger_label = "🕯️ 1H Bull Engulf (prev) ✅"
        elif p1.get('bull_pin', 0) == 1:
            score += 11; trigger = True
            trigger_label = "🕯️ 1H Bull Pin (prev) ✅"
        elif p1.get('hammer', 0) == 1:
            score += 9;  trigger = True
            trigger_label = "🕯️ 1H Hammer (prev) ✅"
    else:
        if l1.get('bear_engulf', 0) == 1:
            score += 25; trigger = True
            trigger_label = "🕯️ 1H Bearish Engulfing ✅ (strongest)"
        elif l1.get('bear_pin', 0) == 1:
            score += 22; trigger = True
            trigger_label = "🕯️ 1H Bearish Pin Bar ✅"
        elif l1.get('shooting_star', 0) == 1:
            score += 18; trigger = True
            trigger_label = "🕯️ 1H Shooting Star ✅"
        elif p1.get('bear_engulf', 0) == 1:
            score += 14; trigger = True
            trigger_label = "🕯️ 1H Bear Engulf (prev) ✅"
        elif p1.get('bear_pin', 0) == 1:
            score += 11; trigger = True
            trigger_label = "🕯️ 1H Bear Pin (prev) ✅"
        elif p1.get('shooting_star', 0) == 1:
            score += 9;  trigger = True
            trigger_label = "🕯️ 1H Shooting Star (prev) ✅"

    if trigger:
        reasons.append(trigger_label)
    else:
        score -= 12
        failed.append("⏳ No 1H trigger candle yet — setup forming")

    # ── 6. Momentum (12 pts) ─────────────────────────────────
    rsi1  = l1.get('rsi', 50)
    macd1 = l1.get('macd', 0);  ms1  = l1.get('macd_signal', 0)
    pm1   = p1.get('macd', 0);  pms1 = p1.get('macd_signal', 0)
    sk1   = l1.get('srsi_k', 0.5); sd1 = l1.get('srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi1 <= 55:
            score += 4; reasons.append(f"✅ RSI reset zone ({rsi1:.0f})")
        elif rsi1 < 28:
            score += 3; reasons.append(f"✅ RSI oversold ({rsi1:.0f})")
        if macd1 > ms1 and pm1 <= pms1:
            score += 5; reasons.append("⚡ MACD bull cross")
        elif macd1 > ms1:
            score += 2; reasons.append("✅ MACD bullish")
        if sk1 < 0.3 and sk1 > sd1:
            score += 3; reasons.append("⚡ Stoch RSI bull cross")
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
            score += 3; reasons.append("⚡ Stoch RSI bear cross")

    # ── 7. PD Zone Bonus [IMP-3] ─────────────────────────────
    if direction == 'LONG' and pd_pos < 0.30:
        score += 5
        reasons.append(f"🟩 Deep Discount ({pd_pos*100:.0f}%) +5pts")
    elif direction == 'SHORT' and pd_pos > 0.70:
        score += 5
        reasons.append(f"🟥 Deep Premium ({pd_pos*100:.0f}%) +5pts")

    # ── 8. Extras: Sweep / FVG / 15M Vol / VWAP (10 pts cap) ─
    extras = 0
    if sweep:
        extras += 4; reasons.append(f"💧 Liq. sweep @ {sweep['level']:.5f}")
    if fvg_near:
        extras += 3; reasons.append("⚡ FVG overlaps OB")

    vr15 = l15.get('vol_ratio', 1.0)
    if   vr15 >= 2.5:
        extras += 3; reasons.append(f"🚀 15M vol spike {vr15:.1f}x")
    elif vr15 >= 1.5:
        extras += 1; reasons.append(f"✅ 15M elevated vol {vr15:.1f}x")

    close1 = l1.get('close', 0); vwap1 = l1.get('vwap', 0)
    if direction == 'LONG' and close1 < vwap1:
        extras = min(extras + 1, 10); reasons.append("✅ 1H below VWAP")
    elif direction == 'SHORT' and close1 > vwap1:
        extras = min(extras + 1, 10); reasons.append("✅ 1H above VWAP")

    score += min(extras, 10)

    # [IMP-6] Dynamic TP3 flag — scorer just exposes the flag
    strong_trend = hh_ll_confirmed and triple_ema and (adx_4h is not None and adx_4h > ADX_TREND_THRESHOLD)

    return max(0, min(int(score), 100)), reasons, failed, strong_trend


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
            'total': 0, 'long': 0, 'short': 0,
            'elite': 0, 'premium': 0, 'high': 0,
            'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0,
            'last_scan': None, 'pairs_scanned': 0,
            'chop_filtered': 0   # [IMP-2] track how many we blocked
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

            # Gate 1: 4H EMA Bias
            l4  = df4.iloc[-1]
            e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0)
            if   e21 > e50: bias = 'LONG'
            elif e21 < e50: bias = 'SHORT'
            else:
                debug['gates'].append('❌ 4H EMAs flat — no bias')
                return None, debug
            debug['bias'] = bias

            # [IMP-2] Gate 2: ADX Chop Filter
            adx_4h = l4.get('adx', None)
            if adx_4h is not None and not np.isnan(adx_4h):
                if adx_4h < ADX_CHOP_THRESHOLD:
                    self.stats['chop_filtered'] += 1
                    debug['gates'].append(
                        f'❌ [IMP-2] 4H ADX {adx_4h:.1f} < {ADX_CHOP_THRESHOLD} — CHOP, skipping'
                    )
                    return None, debug
                debug['gates'].append(f'✅ 4H ADX {adx_4h:.1f} ≥ {ADX_CHOP_THRESHOLD} — trending')
            else:
                debug['gates'].append('⚠️ ADX unavailable — skipping chop check')

            # Gate 3: HH/LL bonus check (not a hard gate)
            hh_ll_ok, hh_ll_msg = self.smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)
            debug['gates'].append(hh_ll_msg)

            # Gate 4: PD Zone
            pd_label, pd_pos = self.smc.pd_zone(df4, price)
            if bias == 'LONG' and pd_label == 'PREMIUM':
                debug['gates'].append(f'❌ PD zone: PREMIUM ({pd_pos*100:.0f}%) — no longs here')
                return None, debug
            if bias == 'SHORT' and pd_label == 'DISCOUNT':
                debug['gates'].append(f'❌ PD zone: DISCOUNT ({pd_pos*100:.0f}%) — no shorts here')
                return None, debug
            debug['gates'].append(f'✅ PD zone: {pd_label} ({pd_pos*100:.0f}%)')

            # Gate 5: 1H Structure
            highs1, lows1 = self.smc.swing_highs_lows(df1, left=4, right=4)
            structure = self.smc.detect_structure_break(df1, highs1, lows1, lookback=STRUCTURE_LOOKBACK)
            if structure:
                s_bull = 'BULL' in structure['kind']
                s_bear = 'BEAR' in structure['kind']
                if bias == 'LONG' and s_bear:
                    debug['gates'].append(f'❌ Structure ({structure["kind"]}) opposes LONG')
                    return None, debug
                if bias == 'SHORT' and s_bull:
                    debug['gates'].append(f'❌ Structure ({structure["kind"]}) opposes SHORT')
                    return None, debug
                disp = structure.get('disp', {})
                disp_tag = "💥 displaced" if disp.get('body_ok') and disp.get('vol_ok') else "⚠️ weak displacement"
                debug['gates'].append(f'✅ Structure: {structure["kind"]} [{disp_tag}]')
            else:
                debug['gates'].append('⚠️ No recent BOS/MSS (score=0 but continuing)')

            # Gate 6: 1H Order Block (HARD GATE)
            obs = self.smc.find_order_blocks(df1, bias, lookback=60)
            if not obs:
                debug['gates'].append(f'❌ No valid {bias} OBs on 1H')
                return None, debug
            debug['gates'].append(f'✅ {len(obs)} OB(s) found on 1H')

            active_ob = None
            for ob in obs:
                if self.smc.price_in_ob(price, ob, OB_TOLERANCE_PCT):
                    active_ob = ob; break

            if not active_ob:
                nearest  = obs[0]
                dist_pct = min(abs(price - nearest['top']), abs(price - nearest['bottom'])) / price * 100
                debug['gates'].append(
                    f'❌ Price not at OB — nearest {dist_pct:.2f}% away '
                    f'[{nearest["bottom"]:.5f}–{nearest["top"]:.5f}]'
                )
                return None, debug
            dr       = active_ob.get('disp_ratio', 0)
            ob_emoji = '🔵' if active_ob.get('ob_type') == 'swing' else ('🟢' if bias == 'LONG' else '🔴')
            ob_label = active_ob.get('ob_type', '?').upper()
            debug['gates'].append(
                f'✅ Price IN {ob_emoji} {ob_label} OB '
                f'[{active_ob["bottom"]:.5f}–{active_ob["top"]:.5f}] disp={dr:.1f}x ATR'
            )

            # FVG on 1H (bonus)
            fvgs     = self.smc.find_fvg(df1, bias, lookback=25)
            fvg_near = None
            for fvg in fvgs:
                if fvg['bottom'] < active_ob['top'] and fvg['top'] > active_ob['bottom']:
                    fvg_near = fvg; break
            if fvg_near:
                debug['gates'].append('✅ 1H FVG overlaps OB')

            # Liquidity sweep on 1H
            sweep = self.smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)
            if sweep:
                debug['gates'].append(f'✅ 1H liq sweep @ {sweep["level"]:.5f}')

            # Score
            score, reasons, failed, strong_trend = score_setup(
                bias, active_ob, structure, sweep, fvg_near,
                df1, df15, df4, pd_label, pd_pos,
                hh_ll_ok, adx_4h
            )
            debug['score'] = score
            debug['gates'] += failed

            if score < MIN_SCORE:
                debug['gates'].append(f'❌ Score {score} < {MIN_SCORE} minimum')
                return None, debug

            if   score >= 92: quality = 'ELITE 👑'
            elif score >= 85: quality = 'PREMIUM 💎'
            else:             quality = 'HIGH 🔥'

            atr1  = df1['atr'].iloc[-1]
            entry = price

            if bias == 'LONG':
                sl = active_ob['bottom'] - atr1 * 0.2
                sl = min(sl, entry - atr1 * 0.6)
            else:
                sl = active_ob['top'] + atr1 * 0.2
                sl = max(sl, entry + atr1 * 0.6)

            risk = abs(entry - sl)
            if risk < entry * 0.001:
                debug['gates'].append('❌ Degenerate SL')
                return None, debug

            # [IMP-6] Expand TP3 to 5R in strong trends
            tp3_mult = 5.0 if strong_trend else 4.0
            tp3_label = "5R 🏔️ TREND" if strong_trend else "4R"

            if bias == 'LONG':
                tps = [entry + risk*1.5, entry + risk*2.5, entry + risk*tp3_mult]
            else:
                tps = [entry - risk*1.5, entry - risk*2.5, entry - risk*tp3_mult]

            rr       = [abs(t - entry) / risk for t in tps]
            risk_pct = risk / entry * 100
            tid      = f"{symbol.split('/')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            sig = {
                'trade_id':    tid,
                'symbol':      symbol.replace('/USDT:USDT', ''),
                'full_symbol': symbol,
                'signal':      bias,
                'quality':     quality,
                'score':       score,
                'hh_ll':       hh_ll_ok,
                'strong_trend': strong_trend,   # [IMP-6]
                'tp3_label':   tp3_label,        # [IMP-6]
                'adx_4h':      adx_4h,           # [IMP-2]
                'entry':       entry,
                'stop_loss':   sl,
                'targets':     tps,
                'rr':          rr,
                'risk_pct':    risk_pct,
                'ob':          active_ob,
                'fvg':         fvg_near,
                'sweep':       sweep,
                'structure':   structure,
                'pd_zone':     pd_label,
                'pd_pos':      pd_pos,
                'reasons':     reasons,
                'tp_hit':      [False, False, False],
                'sl_hit':      False,
                'timestamp':   datetime.now(),
            }
            debug['gates'].append(f'✅ PASSED — Score {score}')
            return sig, debug

        except Exception as e:
            logger.error(f"Analyse {symbol}: {e}")
            debug['gates'].append(f'💥 Exception: {e}')
            return None, debug

    def fmt(self, s):
        arrow  = '🟢' if s['signal'] == 'LONG' else '🔴'
        icon   = '🚀' if s['signal'] == 'LONG' else '🔻'
        z      = {'DISCOUNT': '🟩 Discount', 'PREMIUM': '🟥 Premium', 'NEUTRAL': '🟨 Neutral'}.get(s['pd_zone'], '')
        ob     = s['ob']
        q_icon = {'ELITE 👑': '👑', 'PREMIUM 💎': '💎', 'HIGH 🔥': '🔥'}.get(s['quality'], '🔥')
        tp3_rr = f"{s['rr'][2]:.0f}R{'  🏔️ Strong Trend' if s.get('strong_trend') else ''}"

        # OB zone type — front and centre
        ob_type    = ob.get('ob_type', 'internal')
        ob_emoji   = '🔵' if ob_type == 'swing' else ('🟢' if s['signal'] == 'LONG' else '🔴')
        ob_label   = 'Swing OB' if ob_type == 'swing' else 'Internal OB'
        ob_display = f"{ob_emoji} <b>{ob_label}</b>"

        msg  = f"{'━'*28}\n"
        msg += f"{icon} <b>#{s['symbol']}USDT  {arrow} {s['signal']}</b>  {ob_emoji}\n"
        msg += f"{q_icon} {s['quality']}   ⭐ {s['score']}/100   {z}\n"
        msg += f"{'━'*28}\n\n"

        msg += f"{ob_display}  <code>${ob['bottom']:.5f} – ${ob['top']:.5f}</code>\n\n"

        msg += f"💰 Entry   <code>${s['entry']:.5f}</code>\n"
        msg += f"🛑 Stop    <code>${s['stop_loss']:.5f}</code>  <i>-{s['risk_pct']:.1f}%</i>\n\n"

        for i, (tp, rr, split) in enumerate(zip(s['targets'], s['rr'], ['50%', '30%', '20%']), 1):
            pct    = abs((tp - s['entry']) / s['entry'] * 100)
            rr_lbl = tp3_rr if i == 3 else f"{rr:.1f}R"
            msg   += f"🎯 TP{i}    <code>${tp:.5f}</code>  <b>+{pct:.1f}%</b>  {rr_lbl}  <i>close {split}</i>\n"

        msg += f"\n<i>⚠️ Risk 1-2% max  ·  Move SL to entry after TP1</i>\n"
        msg += f"<i>🕐 {s['timestamp'].strftime('%d %b  %H:%M UTC')}</i>\n"
        msg += f"{'━'*28}"
        return msg

    async def send(self, text):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Telegram: {e}")

    async def tp_alert(self, t, n, price):
        tp  = t['targets'][n-1]
        pct = abs((tp - t['entry'])/t['entry']*100)
        advice = {1:'Close 50% → Move SL to breakeven', 2:'Close 30% → Trail stop tight', 3:'Close final 20% 🎊'}
        msg  = f"🎯 <b>TP{n} HIT!</b>\n\n<code>{t['trade_id']}</code>\n<b>{t['symbol']}</b> {t['signal']}\n\n"
        msg += f"Target: <code>${tp:.6f}</code>\nCurrent: <code>${price:.6f}</code>\nProfit: <b>+{pct:.2f}%</b>\n\n"
        msg += f"📋 {advice[n]}"
        await self.send(msg)
        self.stats[f'tp{n}'] += 1

    async def sl_alert(self, t, price):
        loss = abs((price - t['entry'])/t['entry']*100)
        msg  = f"⛔ <b>STOP LOSS HIT</b>\n\n<code>{t['trade_id']}</code>\n<b>{t['symbol']}</b> {t['signal']}\n\n"
        msg += f"Entry: <code>${t['entry']:.6f}</code>\nLoss: <b>-{loss:.2f}%</b>\n\nOB invalidated. Next setup incoming."
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
                        if datetime.now() - t['timestamp'] > timedelta(hours=48):
                            await self.send(f"⏰ <b>48H TIMEOUT</b>\n<code>{tid}</code>\n{t['symbol']} — Close manually.")
                            remove.append(tid); continue
                        ticker = await self.exchange.fetch_ticker(t['full_symbol'])
                        p = ticker['last']
                        if t['signal'] == 'LONG':
                            for i, tp in enumerate(t['targets']):
                                if not t['tp_hit'][i] and p >= tp:
                                    await self.tp_alert(t, i+1, p); t['tp_hit'][i] = True
                                    if i == 2: remove.append(tid)
                            if not t['sl_hit'] and p <= t['stop_loss']:
                                await self.sl_alert(t, p); t['sl_hit'] = True; remove.append(tid)
                        else:
                            for i, tp in enumerate(t['targets']):
                                if not t['tp_hit'][i] and p <= tp:
                                    await self.tp_alert(t, i+1, p); t['tp_hit'][i] = True
                                    if i == 2: remove.append(tid)
                            if not t['sl_hit'] and p >= t['stop_loss']:
                                await self.sl_alert(t, p); t['sl_hit'] = True; remove.append(tid)
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
            f"Entry: <b>1H trigger</b> | OB: 1H + displacement | Trend: 4H\n"
            f"Min score: {MIN_SCORE} | ADX chop filter: <{ADX_CHOP_THRESHOLD}\n"
            f"OB disp bonus: >{OB_DISP_ATR_MULT}x ATR | Vol filter: ${MIN_VOLUME_24H/1e6:.0f}M\n"
            f"Strong trend TP3 = 5R (ADX>{ADX_TREND_THRESHOLD} + HH/LL + EMA stack)"
        )

        pairs       = await self.get_pairs()
        candidates  = []
        near_misses = []
        scanned     = 0
        chop_count  = 0

        for pair in pairs:
            try:
                data = await self.fetch_data(pair)
                if data:
                    sig, dbg = self.analyse(data, pair)
                    if any(f'CHOP' in g for g in dbg['gates']):
                        chop_count += 1
                    if sig:
                        candidates.append(sig)
                        logger.info(f"  💎 {pair} {sig['signal']} score={sig['score']}")
                    else:
                        if dbg['score'] > 0 and any('Price IN' in g and 'OB' in g for g in dbg['gates']):
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
            self.stats[sig['signal'].lower()] += 1
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
        lg = sum(1 for s in top if s['signal'] == 'LONG')
        tr = sum(1 for s in top if s.get('hh_ll'))
        st = sum(1 for s in top if s.get('strong_trend'))

        summ  = f"✅ <b>SCAN COMPLETE — v5.0</b>\n\n"
        summ += f"📊 Pairs scanned:    {scanned}\n"
        summ += f"🚫 Chop filtered:    {chop_count} (ADX<{ADX_CHOP_THRESHOLD})\n"
        summ += f"🔍 Candidates:       {len(candidates)}\n"
        summ += f"🎯 Signals sent:     {len(top)}\n"
        if top:
            summ += f"  👑 Elite:    {el}\n  💎 Premium:  {pr}\n  🔥 High:     {hi}\n"
            summ += f"  🟢 Long:     {lg}\n  🔴 Short:    {len(top)-lg}\n"
            summ += f"  🏔️ Trending: {tr}\n  💥 Strong (5R TP3): {st}\n"
        else:
            summ += f"\n<i>No setups met criteria this scan.</i>\n"
            summ += f"Near misses: {len(near_misses)} — use /debug\n"
        summ += f"\n⏰ {datetime.now().strftime('%H:%M UTC')}"
        await self.send(summ)

        logger.info(f"✅ Done. {len(candidates)} candidates → {len(top)} sent. Chop blocked: {chop_count}")
        self.is_scanning = False
        return top

    async def run(self, interval_min=SCAN_INTERVAL_MIN):
        logger.info("🚀 SMC Pro v5.0 starting")
        await self.send(
            "👑 <b>SMC PRO v5.0 — ORDER BLOCK SCANNER</b> 👑\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>4H Trend  →  1H Structure + OB + Entry  →  15M Vol</b>\n\n"
            f"✅ Entry trigger: <b>1H candles</b>\n"
            f"✅ ADX chop filter: ADX < {ADX_CHOP_THRESHOLD} = blocked\n"
            f"✅ OB displacement: body/{OB_DISP_ATR_MULT}x ATR bonus\n"
            f"✅ BOS gate: body > {DISP_BODY_ATR_MULT}x ATR + vol > {DISP_VOL_MULT}x\n"
            f"✅ Deep PD bonus: ±5pts\n"
            f"✅ Strong trend TP3 = 5R (ADX>{ADX_TREND_THRESHOLD} + HH/LL + EMA)\n"
            f"✅ Min score: {MIN_SCORE}/100\n"
            f"✅ OB tolerance: {OB_TOLERANCE_PCT*100:.1f}%\n"
            f"✅ Vol filter: ${MIN_VOLUME_24H/1e6:.0f}M/day\n"
            f"✅ Scan every: {SCAN_INTERVAL_MIN} min\n\n"
            "Commands: /scan /stats /trades /debug /help\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        asyncio.create_task(self.track())
        while True:
            try:
                await self.scan()
                logger.info(f"💤 Next scan in {interval_min}m")
                await asyncio.sleep(interval_min * 60)
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
            "👑 <b>SMC Pro v5.0</b>\n\n"
            "7 improvements: ADX chop filter, OB displacement, "
            "BOS gate, deep PD bonus, structure strength, "
            "dynamic TP3 (5R in trends), lean indicators.\n\n"
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
        msg += f"Total signals: {s['total']}\n"
        msg += f"  👑 Elite: {s['elite']}  💎 Premium: {s['premium']}  🔥 High: {s['high']}\n"
        msg += f"  🟢 Long: {s['long']}  🔴 Short: {s['short']}\n\n"
        msg += f"TP1: {s['tp1']} | TP2: {s['tp2']} | TP3: {s['tp3']} | SL: {s['sl']}\n"
        msg += f"🚫 Chop filtered: {s['chop_filtered']}\n\n"
        if s['last_scan']:
            msg += f"Last scan: {s['last_scan'].strftime('%H:%M UTC')}\n"
            msg += f"Pairs: {s['pairs_scanned']}\n"
        msg += f"Active: {len(self.s.active_trades)}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def trades(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.active_trades:
            await u.message.reply_text("📭 No active trades."); return
        msg = f"📡 <b>ACTIVE TRADES ({len(self.s.active_trades)})</b>\n\n"
        for tid, t in list(self.s.active_trades.items())[:10]:
            age       = int((datetime.now() - t['timestamp']).total_seconds()/3600)
            tps       = ''.join(['✅' if h else '⏳' for h in t['tp_hit']])
            trend_tag = '🏔️' if t.get('hh_ll') else '〰️'
            strong    = ' 💥5R' if t.get('strong_trend') else ''
            adx_tag   = f" ADX={t['adx_4h']:.0f}" if t.get('adx_4h') else ""
            msg += (f"<b>{t['symbol']}</b> {t['signal']} {trend_tag} — {t['quality']}{strong}\n"
                    f"  Entry: <code>${t['entry']:.5f}</code> | Score: {t['score']}{adx_tag}\n"
                    f"  TPs: {tps} | {age}h old\n\n")
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def debug(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.last_debug:
            await u.message.reply_text("📭 No debug data yet. Run /scan first.", parse_mode=ParseMode.HTML)
            return
        msg = "🔬 <b>NEAR MISSES — Last Scan</b>\n"
        msg += "<i>(At OB but below score threshold)</i>\n\n"
        for d in self.s.last_debug[:8]:
            msg += f"<b>{d['symbol']}</b> {d['bias']} — Score: {d['score']}/100\n"
            for g in d['gates'][-4:]:
                msg += f"  {g}\n"
            msg += "\n"
        msg += f"<i>Min score: {MIN_SCORE}. 1H trigger = up to +25pts.</i>"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def help(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>SMC PRO v5.0 — STRATEGY</b>\n\n"
        msg += "<b>Timeframe Stack:</b>\n"
        msg += "  4H  → EMA bias + ADX chop filter + HH/LL\n"
        msg += "  1H  → BOS/MSS (displacement gate) + OB + Trigger\n"
        msg += "  15M → Volume spike bonus only\n\n"
        msg += "<b>Hard Gates (ALL must pass):</b>\n"
        msg += "  1️⃣ 4H EMA 21/50 bias\n"
        msg += f"  2️⃣ 4H ADX ≥ {ADX_CHOP_THRESHOLD} (chop filter) [IMP-2]\n"
        msg += "  3️⃣ PD zone (no longs premium / no shorts discount)\n"
        msg += "  4️⃣ 1H BOS/MSS within 20 candles\n"
        msg += "  5️⃣ Price at valid 1H Order Block\n"
        msg += f"  6️⃣ Score ≥ {MIN_SCORE}/100\n\n"
        msg += "<b>Score System (max 100):</b>\n"
        msg += "  +25 — 1H entry trigger\n"
        msg += "  +20 — MSS structure\n"
        msg += "  +20 — Tight OB\n"
        msg += "  + 5 — OB displacement >1.5x ATR [IMP-1]\n"
        msg += "  + 4 — Break candle body >1.2 ATR [IMP-4]\n"
        msg += "  - 8 — Fake BOS (weak body + weak vol) [IMP-7]\n"
        msg += "  +15 — 4H EMA 21>50\n"
        msg += f"  + 8 — 4H HH/LL\n"
        msg += "  + 5 — Deep PD zone (<30% / >70%) [IMP-3]\n"
        msg += "  +12 — Momentum (RSI/MACD/Stoch)\n"
        msg += "  +10 — Extras (sweep/FVG/vol)\n\n"
        msg += "<b>Dynamic TP3 [IMP-6]:</b>\n"
        msg += f"  Normal:       TP3 = 4R\n"
        msg += f"  Strong trend: TP3 = 5R  (HH/LL + EMA + ADX>{ADX_TREND_THRESHOLD})\n\n"
        msg += "<b>Lean Indicators [IMP-5]:</b>\n"
        msg += "  EMA21/50, ATR, RSI, MACD, Volume, VWAP, ADX\n"
        msg += "  (Removed: CMF, MFI, Bollinger, EMA200, DI+/-)\n\n"
        msg += f"<b>Config:</b>\n"
        msg += f"  MIN_SCORE={MIN_SCORE} | ADX_CHOP={ADX_CHOP_THRESHOLD}\n"
        msg += f"  ADX_TREND={ADX_TREND_THRESHOLD} | HH_LL_BONUS={HH_LL_BONUS}\n"
        msg += f"  OB_DISP_ATR={OB_DISP_ATR_MULT} | OB_TOL={OB_TOLERANCE_PCT}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

async def main():
    # ════════════ CONFIG ════════════
    TELEGRAM_TOKEN   = "8034062612:AAEJYbPA8sMODYvqvt8U-5mM7c3Y3-GOYtM"
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
