"""
SMC PRO SCANNER v4.2 — "BTC AS THE KING"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGES from v4.1:

  BTC FULL SMC ANALYSIS (the big upgrade):
  ─────────────────────────────────────────
  BTC now gets the FULL SMC treatment — same engine used
  on every alt pair, but applied to BTC itself across
  3 timeframes (4H, 1H, 15M). This gives a complete
  BTC structural picture before any alt signal fires.

  What BTC analysis now checks:
    ✅ 4H EMA trend (STRONG_BULL / BULL / RANGING / BEAR / STRONG_BEAR)
    ✅ 4H HH/LL — is BTC in a trending or ranging phase?
    ✅ 4H PD zone — is BTC in discount (potential bounce) or premium (potential dump)?
    ✅ 1H BOS/MSS — did BTC just break structure? Which direction?
    ✅ 1H Order Block — is BTC sitting at its own OB? Top/bottom/mid printed
    ✅ 1H Liquidity Sweep — did BTC just sweep lows/highs before moving?
    ✅ 1H FVG — is there a Fair Value Gap on BTC right now?
    ✅ 1H Trigger candle — engulfing/pin/hammer on BTC itself
    ✅ 1H price momentum (1H chg%, 3H chg%)
    ✅ 15M volume — is BTC seeing a volume spike right now?

  BTC Regime Labels (replaces simple trend label):
    BULL_CONFIRMED   — 4H bull + 1H BOS_BULL + 1H bull trigger
    BULL_STRUCTURE   — 4H bull + 1H BOS_BULL (no trigger yet)
    BULL_RANGING     — 4H bull but no BOS yet
    BEAR_CONFIRMED   — 4H bear + 1H BOS_BEAR + 1H bear trigger
    BEAR_STRUCTURE   — 4H bear + 1H BOS_BEAR (no trigger yet)
    BEAR_RANGING     — 4H bear but no BOS yet
    REVERSAL_BULL    — 4H bear BUT 1H MSS_BULL — potential flip
    REVERSAL_BEAR    — 4H bull BUT 1H MSS_BEAR — potential flip
    RANGING          — No clear bias on either timeframe

  Hard Block Logic (upgraded):
    BEAR_CONFIRMED + price near OB resistance → block ALL LONGS
    BULL_CONFIRMED + price near OB support    → block ALL SHORTS
    BTC 1H sweep DOWN + no recovery           → block LONGS
    BTC 1H dump >2.5%                         → block LONGS
    BTC 1H pump >2.5%                         → block SHORTS
    REVERSAL_BEAR forming                     → block LONGS (early warning)
    REVERSAL_BULL forming                     → block SHORTS (early warning)

  Score modifier matrix (upgraded from v4.1):
    BULL_CONFIRMED   → LONG +15, SHORT -18
    BULL_STRUCTURE   → LONG +10, SHORT -12
    BULL_RANGING     → LONG  +5, SHORT  -6
    BEAR_CONFIRMED   → SHORT +15, LONG -18
    BEAR_STRUCTURE   → SHORT +10, LONG -12
    BEAR_RANGING     → SHORT  +5, LONG  -6
    REVERSAL_BULL    → LONG  +8, SHORT -10  (early flip)
    REVERSAL_BEAR    → SHORT +8, LONG  -10  (early flip)
    RANGING          → neutral (0)
    BTC at own OB    → +5 pts (aligned direction)
    BTC sweep done   → +4 pts (liquidity cleared)
    BTC FVG present  → +3 pts (magnet for price)
    BTC 15M vol spike→ +2 pts

  New /btc command output:
    Now shows full BTC SMC breakdown:
    regime, structure, OB levels, sweep, FVG,
    trigger candle, 4H PD zone, HH/LL status

TIMEFRAME ROLES v4.2:
  BTC 4H  → EMA trend + HH/LL + PD zone
  BTC 1H  → BOS/MSS + OB + sweep + FVG + trigger candle  ← FULL SMC
  BTC 15M → Volume spike on BTC
  ALT 4H  → EMA bias + HH/LL
  ALT 1H  → BOS/MSS + OB + entry trigger
  ALT 15M → Volume bonus
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

# ═══════════════════════════════════════════════════════════════
#  TUNABLE SETTINGS
# ═══════════════════════════════════════════════════════════════
MAX_SIGNALS_PER_SCAN  = 6
MIN_SCORE             = 75
MIN_VOLUME_24H        = 5_000_000
OB_TOLERANCE_PCT      = 0.008
OB_IMPULSE_ATR_MULT   = 1.0
STRUCTURE_LOOKBACK    = 20
SCAN_INTERVAL_MIN     = 60
HH_LL_LOOKBACK        = 10
HH_LL_BONUS           = 8

# ── BTC filter thresholds ───────────────────────────────────────
BTC_HARD_BLOCK_1H_PCT = 2.5    # instant block if BTC 1H candle moves this %
BTC_HARD_BLOCK_3H_PCT = 3.0    # block if BTC strong trend + 3H move this %

# Score modifier table per BTC regime (LONG side, SHORT is mirrored)
BTC_REGIME_SCORES = {
    #  regime              LONG   SHORT
    'BULL_CONFIRMED':    ( +15,   -18),
    'BULL_STRUCTURE':    ( +10,   -12),
    'BULL_RANGING':      (  +5,    -6),
    'BEAR_CONFIRMED':    ( -18,   +15),
    'BEAR_STRUCTURE':    ( -12,   +10),
    'BEAR_RANGING':      (  -6,    +5),
    'REVERSAL_BULL':     (  +8,   -10),
    'REVERSAL_BEAR':     ( -10,    +8),
    'RANGING':           (   0,     0),
}

BTC_OB_BONUS     = 5   # BTC at own OB aligned with signal direction
BTC_SWEEP_BONUS  = 4   # BTC just swept liquidity (cleared stops)
BTC_FVG_BONUS    = 3   # BTC has open FVG in signal direction
BTC_VOL_BONUS    = 2   # BTC 15M volume spike


# ══════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════

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

        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close']
        ).average_true_range()

        bb = ta.volatility.BollingerBands(df['close'], 20, 2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_pband'] = bb.bollinger_pband()

        adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']    = adx_i.adx()
        df['di_pos'] = adx_i.adx_pos()
        df['di_neg'] = adx_i.adx_neg()

        df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).chaikin_money_flow()
        df['mfi'] = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).money_flow_index()

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


# ══════════════════════════════════════════════════════════════════
#  SMC ENGINE  (shared by both ALT and BTC analysis)
# ══════════════════════════════════════════════════════════════════

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
            return False, "⚠️ Not enough 4H data for HH/LL"
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback * 2):-lookback]
        if direction == 'LONG':
            rh, ph = recent['high'].max(), prior['high'].max()
            if rh > ph:
                return True, f"📈 4H HH ({ph:.2f} → {rh:.2f}) +{HH_LL_BONUS}pts"
            return False, f"➖ 4H no HH ({rh:.2f} ≤ {ph:.2f}) ranging"
        else:
            rl, pl = recent['low'].min(), prior['low'].min()
            if rl < pl:
                return True, f"📉 4H LL ({pl:.2f} → {rl:.2f}) +{HH_LL_BONUS}pts"
            return False, f"➖ 4H no LL ({rl:.2f} ≥ {pl:.2f}) ranging"

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
            atr_local = (
                df['atr'].iloc[i]
                if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i])
                else (c['high'] - c['low'])
            )
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
                if (df['close'].iloc[i+1:n] < (ob['top']+ob['bottom'])/2).any(): continue
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
                if (df['close'].iloc[i+1:n] > (ob['top']+ob['bottom'])/2).any(): continue
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
                fvgs.append({
                    'top': nxt['low'], 'bottom': prev['high'],
                    'mid': (nxt['low'] + prev['high']) / 2, 'bar': i
                })
            elif direction == 'SHORT' and prev['low'] > nxt['high']:
                fvgs.append({
                    'top': prev['low'], 'bottom': nxt['high'],
                    'mid': (prev['low'] + nxt['high']) / 2, 'bar': i
                })
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

    def detect_trigger_candle(self, df, direction):
        """Check last 2 candles for trigger patterns. Returns (label, strength)"""
        l1 = df.iloc[-1]
        p1 = df.iloc[-2]
        if direction == 'LONG':
            if l1.get('bull_engulf', 0) == 1: return "Bullish Engulfing", 3
            if l1.get('bull_pin',   0) == 1:  return "Bullish Pin Bar",   2
            if l1.get('hammer',     0) == 1:  return "Hammer",            2
            if p1.get('bull_engulf',0) == 1:  return "Bull Engulf (prev)",1
            if p1.get('bull_pin',   0) == 1:  return "Bull Pin (prev)",   1
            if p1.get('hammer',     0) == 1:  return "Hammer (prev)",     1
        else:
            if l1.get('bear_engulf',  0) == 1: return "Bearish Engulfing", 3
            if l1.get('bear_pin',     0) == 1: return "Bearish Pin Bar",   2
            if l1.get('shooting_star',0) == 1: return "Shooting Star",     2
            if p1.get('bear_engulf',  0) == 1: return "Bear Engulf (prev)",1
            if p1.get('bear_pin',     0) == 1: return "Bear Pin (prev)",   1
            if p1.get('shooting_star',0) == 1: return "Shooting Star (prev)",1
        return None, 0


# ══════════════════════════════════════════════════════════════════
#  BTC FULL SMC ANALYSER  ← THE CORE UPGRADE IN v4.2
# ══════════════════════════════════════════════════════════════════

class BTCAnalyser:
    """
    Runs the full SMC stack on BTC across 4H, 1H, 15M.
    Produces a BTCContext dict consumed by the main scanner.
    """

    def __init__(self, smc: SMCEngine):
        self.smc = smc

    def _ema_trend(self, row) -> str:
        e21, e50, e200 = row.get('ema_21',0), row.get('ema_50',0), row.get('ema_200',0)
        if e21 > e50 > e200: return 'STRONG_BULL'
        if e21 > e50:        return 'BULL'
        if e21 < e50 < e200: return 'STRONG_BEAR'
        if e21 < e50:        return 'BEAR'
        return 'RANGING'

    def _determine_regime(self, ema_trend, structure, trigger_label) -> str:
        """
        Combine 4H EMA trend, 1H structure break, and 1H trigger candle
        into a single BTC regime label.
        """
        has_bull_structure = structure and 'BULL' in structure['kind']
        has_bear_structure = structure and 'BEAR' in structure['kind']
        is_mss_bull        = structure and 'MSS_BULL' in structure['kind']
        is_mss_bear        = structure and 'MSS_BEAR' in structure['kind']
        has_bull_trigger   = trigger_label is not None and 'Bull' in (trigger_label or '') or 'Hammer' in (trigger_label or '') or 'Engulfing' in (trigger_label or '') and 'Bear' not in (trigger_label or '')
        has_bear_trigger   = trigger_label is not None and ('Bear' in (trigger_label or '') or 'Shooting' in (trigger_label or ''))

        bull_4h = ema_trend in ('BULL', 'STRONG_BULL')
        bear_4h = ema_trend in ('BEAR', 'STRONG_BEAR')

        # Full confirmation — trend + structure + trigger all agree
        if bull_4h and has_bull_structure and has_bull_trigger:
            return 'BULL_CONFIRMED'
        if bear_4h and has_bear_structure and has_bear_trigger:
            return 'BEAR_CONFIRMED'

        # Structure matches trend but no trigger candle yet
        if bull_4h and has_bull_structure:
            return 'BULL_STRUCTURE'
        if bear_4h and has_bear_structure:
            return 'BEAR_STRUCTURE'

        # MSS against trend — early reversal signal
        if bull_4h and is_mss_bear:
            return 'REVERSAL_BEAR'
        if bear_4h and is_mss_bull:
            return 'REVERSAL_BULL'

        # Trend exists but no structure break yet
        if bull_4h:
            return 'BULL_RANGING'
        if bear_4h:
            return 'BEAR_RANGING'

        return 'RANGING'

    def analyse(self, df_4h, df_1h, df_15m) -> dict:
        """
        Full BTC SMC analysis. Returns a rich context dictionary.
        """
        ctx = {
            'regime':        'RANGING',
            'ema_trend':     'RANGING',
            'price':          0,
            '1h_chg':         0.0,
            '3h_chg':         0.0,
            'hh_ll':          False,
            'hh_ll_msg':      '',
            'pd_zone':        'NEUTRAL',
            'pd_pos':          0.5,
            'structure':      None,
            'ob_bull':        None,   # nearest bullish OB
            'ob_bear':        None,   # nearest bearish OB
            'at_bull_ob':     False,
            'at_bear_ob':     False,
            'sweep':          None,
            'fvg_bull':       None,
            'fvg_bear':       None,
            'trigger_label':  None,
            'trigger_strength': 0,
            'vol_ratio_15m':  1.0,
            'rsi':            50.0,
            'macd_bull':      False,
        }

        try:
            if len(df_1h) < 80 or len(df_4h) < 60:
                return ctx

            price = df_1h['close'].iloc[-1]
            ctx['price'] = price

            # ── 4H EMA trend ─────────────────────────────────────
            l4 = df_4h.iloc[-1]
            ema_trend = self._ema_trend(l4)
            ctx['ema_trend'] = ema_trend

            # ── 4H HH/LL ─────────────────────────────────────────
            # Check both directions, pick whichever matches trend
            bias_for_hhll = 'LONG' if 'BULL' in ema_trend else 'SHORT'
            hh_ll_ok, hh_ll_msg = self.smc.check_4h_hh_ll(df_4h, bias_for_hhll, HH_LL_LOOKBACK)
            ctx['hh_ll']     = hh_ll_ok
            ctx['hh_ll_msg'] = hh_ll_msg

            # ── 4H PD Zone ────────────────────────────────────────
            pd_label, pd_pos = self.smc.pd_zone(df_4h, price)
            ctx['pd_zone'] = pd_label
            ctx['pd_pos']  = pd_pos

            # ── 1H Structure ──────────────────────────────────────
            highs1, lows1 = self.smc.swing_highs_lows(df_1h, left=4, right=4)
            structure     = self.smc.detect_structure_break(
                df_1h, highs1, lows1, lookback=STRUCTURE_LOOKBACK
            )
            ctx['structure'] = structure

            # ── 1H Order Blocks (both directions) ─────────────────
            obs_bull = self.smc.find_order_blocks(df_1h, 'LONG',  lookback=60)
            obs_bear = self.smc.find_order_blocks(df_1h, 'SHORT', lookback=60)

            ctx['ob_bull'] = obs_bull[0] if obs_bull else None
            ctx['ob_bear'] = obs_bear[0] if obs_bear else None

            ctx['at_bull_ob'] = any(
                self.smc.price_in_ob(price, ob) for ob in obs_bull
            )
            ctx['at_bear_ob'] = any(
                self.smc.price_in_ob(price, ob) for ob in obs_bear
            )

            # ── 1H Liquidity Sweep ────────────────────────────────
            # Check if BTC just swept lows (bullish) or highs (bearish)
            sweep_bull = self.smc.recent_liquidity_sweep(df_1h, 'LONG',  highs1, lows1, lookback=20)
            sweep_bear = self.smc.recent_liquidity_sweep(df_1h, 'SHORT', highs1, lows1, lookback=20)
            ctx['sweep'] = sweep_bull or sweep_bear

            # ── 1H FVG (both directions) ──────────────────────────
            fvg_bull_list = self.smc.find_fvg(df_1h, 'LONG',  lookback=25)
            fvg_bear_list = self.smc.find_fvg(df_1h, 'SHORT', lookback=25)
            ctx['fvg_bull'] = fvg_bull_list[0] if fvg_bull_list else None
            ctx['fvg_bear'] = fvg_bear_list[0] if fvg_bear_list else None

            # ── 1H Trigger Candle ─────────────────────────────────
            # Determine trigger in direction of structure
            if structure and 'BULL' in structure['kind']:
                trig_dir = 'LONG'
            elif structure and 'BEAR' in structure['kind']:
                trig_dir = 'SHORT'
            else:
                trig_dir = 'LONG' if 'BULL' in ema_trend else 'SHORT'

            t_label, t_strength = self.smc.detect_trigger_candle(df_1h, trig_dir)
            ctx['trigger_label']    = t_label
            ctx['trigger_strength'] = t_strength

            # ── 1H Momentum ───────────────────────────────────────
            l1 = df_1h.iloc[-1]
            ctx['rsi']       = l1.get('rsi', 50)
            ctx['macd_bull'] = l1.get('macd', 0) > l1.get('macd_signal', 0)
            ctx['1h_chg']    = (df_1h['close'].iloc[-1] - df_1h['open'].iloc[-1]) / df_1h['open'].iloc[-1] * 100
            ctx['3h_chg']    = (df_1h['close'].iloc[-1] - df_1h['open'].iloc[-3]) / df_1h['open'].iloc[-3] * 100

            # ── 15M Volume ────────────────────────────────────────
            if len(df_15m) > 20:
                ctx['vol_ratio_15m'] = df_15m['vol_ratio'].iloc[-1] if 'vol_ratio' in df_15m.columns else 1.0

            # ── Regime Label ──────────────────────────────────────
            ctx['regime'] = self._determine_regime(ema_trend, structure, t_label)

        except Exception as e:
            logger.error(f"BTC analyser error: {e}")

        return ctx


# ══════════════════════════════════════════════════════════════════
#  BTC ALIGNMENT CHECKER  (uses full BTCContext)
# ══════════════════════════════════════════════════════════════════

def check_btc_alignment(btc_ctx: dict, direction: str):
    """
    Evaluate the BTC regime against the proposed alt trade direction.

    Returns:
        score_delta (int)  — points to add/subtract from signal score
        hard_block (bool)  — True = skip this signal entirely
        label (str)        — human-readable reason
        reasons (list)     — detailed breakdown lines
    """
    if not btc_ctx:
        return 0, False, "⚪ BTC data unavailable — filter skipped", []

    regime   = btc_ctx['regime']
    chg_1h   = btc_ctx['1h_chg']
    chg_3h   = btc_ctx['3h_chg']
    sweep    = btc_ctx['sweep']
    at_bull  = btc_ctx['at_bull_ob']
    at_bear  = btc_ctx['at_bear_ob']
    fvg_bull = btc_ctx['fvg_bull']
    fvg_bear = btc_ctx['fvg_bear']
    vol15    = btc_ctx['vol_ratio_15m']

    reasons = []

    # ── Hard Blocks ───────────────────────────────────────────────

    # Flash move block
    if direction == 'LONG' and chg_1h < -BTC_HARD_BLOCK_1H_PCT:
        return -999, True, f"🚫 BTC flash dump {chg_1h:.1f}% (1H) — LONGs blocked", []

    if direction == 'SHORT' and chg_1h > BTC_HARD_BLOCK_1H_PCT:
        return -999, True, f"🚫 BTC flash pump +{chg_1h:.1f}% (1H) — SHORTs blocked", []

    # Regime-based hard blocks
    if direction == 'LONG' and regime == 'BEAR_CONFIRMED':
        if at_bear:
            return -999, True, "🚫 BTC BEAR_CONFIRMED + price at bear OB — LONGs blocked", []
        if chg_3h < -BTC_HARD_BLOCK_3H_PCT:
            return -999, True, f"🚫 BTC BEAR_CONFIRMED + {chg_3h:.1f}% (3H) — LONGs blocked", []

    if direction == 'SHORT' and regime == 'BULL_CONFIRMED':
        if at_bull:
            return -999, True, "🚫 BTC BULL_CONFIRMED + price at bull OB — SHORTs blocked", []
        if chg_3h > BTC_HARD_BLOCK_3H_PCT:
            return -999, True, f"🚫 BTC BULL_CONFIRMED + +{chg_3h:.1f}% (3H) — SHORTs blocked", []

    # Reversal warning blocks
    if direction == 'LONG' and regime == 'REVERSAL_BEAR':
        return -999, True, "🚫 BTC REVERSAL_BEAR forming (MSS against bull trend) — LONGs blocked", []

    if direction == 'SHORT' and regime == 'REVERSAL_BULL':
        return -999, True, "🚫 BTC REVERSAL_BULL forming (MSS against bear trend) — SHORTs blocked", []

    # ── Score Modifier from Regime ────────────────────────────────
    scores = BTC_REGIME_SCORES.get(regime, (0, 0))
    delta  = scores[0] if direction == 'LONG' else scores[1]

    regime_icons = {
        'BULL_CONFIRMED': '🟢🟢', 'BULL_STRUCTURE': '🟢',  'BULL_RANGING': '🟢',
        'BEAR_CONFIRMED': '🔴🔴', 'BEAR_STRUCTURE': '🔴',  'BEAR_RANGING': '🔴',
        'REVERSAL_BULL':  '🔄🟢', 'REVERSAL_BEAR':  '🔄🔴','RANGING': '🟡',
    }
    icon = regime_icons.get(regime, '⚪')
    sign = f"+{delta}" if delta >= 0 else str(delta)
    reasons.append(f"{icon} BTC regime: {regime} → {sign}pts")

    # ── Bonus: BTC at its own OB ──────────────────────────────────
    if direction == 'LONG' and at_bull:
        delta += BTC_OB_BONUS
        reasons.append(f"🎯 BTC at bullish OB support (+{BTC_OB_BONUS}pts)")
    elif direction == 'SHORT' and at_bear:
        delta += BTC_OB_BONUS
        reasons.append(f"🎯 BTC at bearish OB resistance (+{BTC_OB_BONUS}pts)")

    # ── Bonus: BTC liquidity sweep ────────────────────────────────
    if sweep:
        sweep_type = sweep.get('type','')
        if direction == 'LONG' and sweep_type == 'SWEEP_LOW':
            delta += BTC_SWEEP_BONUS
            reasons.append(f"💧 BTC swept lows @ {sweep['level']:.0f} — stops cleared (+{BTC_SWEEP_BONUS}pts)")
        elif direction == 'SHORT' and sweep_type == 'SWEEP_HIGH':
            delta += BTC_SWEEP_BONUS
            reasons.append(f"💧 BTC swept highs @ {sweep['level']:.0f} — stops cleared (+{BTC_SWEEP_BONUS}pts)")

    # ── Bonus: BTC FVG in signal direction ────────────────────────
    if direction == 'LONG' and fvg_bull:
        delta += BTC_FVG_BONUS
        reasons.append(f"⚡ BTC bull FVG [{fvg_bull['bottom']:.0f}–{fvg_bull['top']:.0f}] (+{BTC_FVG_BONUS}pts)")
    elif direction == 'SHORT' and fvg_bear:
        delta += BTC_FVG_BONUS
        reasons.append(f"⚡ BTC bear FVG [{fvg_bear['bottom']:.0f}–{fvg_bear['top']:.0f}] (+{BTC_FVG_BONUS}pts)")

    # ── Bonus: BTC 15M volume spike ───────────────────────────────
    if vol15 >= 2.0:
        delta += BTC_VOL_BONUS
        reasons.append(f"🚀 BTC 15M vol spike {vol15:.1f}x (+{BTC_VOL_BONUS}pts)")

    summary_label = f"🟠 BTC {regime} | {chg_1h:+.2f}% (1H) | {chg_3h:+.2f}% (3H) → {sign}pts"
    return delta, False, summary_label, reasons


def regime_emoji(regime: str) -> str:
    return {
        'BULL_CONFIRMED': '🟢🟢', 'BULL_STRUCTURE': '🟢',  'BULL_RANGING':  '🟡🟢',
        'BEAR_CONFIRMED': '🔴🔴', 'BEAR_STRUCTURE': '🔴',  'BEAR_RANGING':  '🟡🔴',
        'REVERSAL_BULL':  '🔄🟢', 'REVERSAL_BEAR':  '🔄🔴','RANGING':       '🟡',
    }.get(regime, '⚪')


# ══════════════════════════════════════════════════════════════════
#  SCORER
# ══════════════════════════════════════════════════════════════════

def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed,
                btc_delta=0, btc_reasons=None):
    score   = 0
    reasons = []
    failed  = []

    l1  = df_1h.iloc[-1]
    p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]
    l4  = df_4h.iloc[-1]

    # ── 1. Structure (20 pts) ─────────────────────────────────────
    if structure:
        if 'MSS' in structure['kind']:
            score += 20; reasons.append(f"🏗️ MSS — Early Reversal ({structure['kind']})")
        else:
            score += 14; reasons.append(f"🏗️ BOS — Pullback Entry ({structure['kind']})")
    else:
        failed.append("❌ No BOS/MSS in last 20 candles")

    # ── 2. Order Block quality (20 pts) ──────────────────────────
    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        if ob_size_pct < 0.8:
            score += 20; reasons.append(f"📦 Tight OB ({ob_size_pct:.2f}%) — high quality")
        elif ob_size_pct < 2.0:
            score += 13; reasons.append(f"📦 OB ({ob_size_pct:.2f}%)")
        else:
            score += 7;  reasons.append(f"📦 Wide OB ({ob_size_pct:.2f}%) — lower quality")
    else:
        failed.append("❌ No valid OB found")

    # ── 3. 4H Trend Alignment (15 pts) ───────────────────────────
    e21  = l4.get('ema_21', 0)
    e50  = l4.get('ema_50', 0)
    e200 = l4.get('ema_200', 0)
    if direction == 'LONG':
        if e21 > e50 > e200:
            score += 15; reasons.append("📈 4H Triple EMA Bull Stack")
        elif e21 > e50:
            score += 10; reasons.append("📈 4H EMA 21>50 Bull")
        elif pd_label == 'DISCOUNT':
            score += 6;  reasons.append("📈 4H Discount Zone (counter-trend OK)")
        else:
            failed.append("⚠️ 4H trend weak for LONG")
    else:
        if e21 < e50 < e200:
            score += 15; reasons.append("📉 4H Triple EMA Bear Stack")
        elif e21 < e50:
            score += 10; reasons.append("📉 4H EMA 21<50 Bear")
        elif pd_label == 'PREMIUM':
            score += 6;  reasons.append("📉 4H Premium Zone (counter-trend OK)")
        else:
            failed.append("⚠️ 4H trend weak for SHORT")

    # ── 4. 4H HH/LL Bonus (8 pts) ────────────────────────────────
    if hh_ll_confirmed:
        score += HH_LL_BONUS
        reasons.append(f"🏔️ 4H HH/LL confirmed (+{HH_LL_BONUS}pts)")
    else:
        failed.append("➖ 4H HH/LL not confirmed — ranging")

    # ── 5. 1H Entry Trigger (25 pts) ─────────────────────────────
    trigger = False
    trigger_label = ""
    if direction == 'LONG':
        if l1.get('bull_engulf',  0) == 1: score += 25; trigger=True; trigger_label="🕯️ 1H Bull Engulfing ✅"
        elif l1.get('bull_pin',   0) == 1: score += 22; trigger=True; trigger_label="🕯️ 1H Bull Pin Bar ✅"
        elif l1.get('hammer',     0) == 1: score += 18; trigger=True; trigger_label="🕯️ 1H Hammer ✅"
        elif p1.get('bull_engulf',0) == 1: score += 14; trigger=True; trigger_label="🕯️ 1H Bull Engulf (prev) ✅"
        elif p1.get('bull_pin',   0) == 1: score += 11; trigger=True; trigger_label="🕯️ 1H Bull Pin (prev) ✅"
        elif p1.get('hammer',     0) == 1: score +=  9; trigger=True; trigger_label="🕯️ 1H Hammer (prev) ✅"
    else:
        if l1.get('bear_engulf',   0) == 1: score += 25; trigger=True; trigger_label="🕯️ 1H Bear Engulfing ✅"
        elif l1.get('bear_pin',    0) == 1: score += 22; trigger=True; trigger_label="🕯️ 1H Bear Pin Bar ✅"
        elif l1.get('shooting_star',0)== 1: score += 18; trigger=True; trigger_label="🕯️ 1H Shooting Star ✅"
        elif p1.get('bear_engulf', 0) == 1: score += 14; trigger=True; trigger_label="🕯️ 1H Bear Engulf (prev) ✅"
        elif p1.get('bear_pin',    0) == 1: score += 11; trigger=True; trigger_label="🕯️ 1H Bear Pin (prev) ✅"
        elif p1.get('shooting_star',0)== 1: score +=  9; trigger=True; trigger_label="🕯️ 1H Shooting Star (prev) ✅"

    if trigger:
        reasons.append(trigger_label)
    else:
        score -= 12
        failed.append("⏳ No 1H trigger yet — setup forming")

    # ── 6. Momentum (12 pts) ──────────────────────────────────────
    rsi1  = l1.get('rsi', 50)
    macd1 = l1.get('macd', 0);  ms1  = l1.get('macd_signal', 0)
    pm1   = p1.get('macd', 0);  pms1 = p1.get('macd_signal', 0)
    sk1   = l1.get('srsi_k', 0.5); sd1 = l1.get('srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi1 <= 55:   score += 4; reasons.append(f"✅ RSI reset ({rsi1:.0f})")
        elif rsi1 < 28:        score += 3; reasons.append(f"✅ RSI oversold ({rsi1:.0f})")
        if macd1>ms1 and pm1<=pms1: score+=5; reasons.append("⚡ MACD bull cross")
        elif macd1 > ms1:      score += 2; reasons.append("✅ MACD bullish")
        if sk1 < 0.3 and sk1 > sd1: score+=3; reasons.append("⚡ StochRSI bull cross")
    else:
        if 45 <= rsi1 <= 72:   score += 4; reasons.append(f"✅ RSI OB zone ({rsi1:.0f})")
        elif rsi1 > 72:        score += 3; reasons.append(f"✅ RSI overbought ({rsi1:.0f})")
        if macd1<ms1 and pm1>=pms1: score+=5; reasons.append("⚡ MACD bear cross")
        elif macd1 < ms1:      score += 2; reasons.append("✅ MACD bearish")
        if sk1 > 0.7 and sk1 < sd1: score+=3; reasons.append("⚡ StochRSI bear cross")

    # ── 7. Extras: Sweep / FVG / 15M Vol / VWAP (10 pts) ─────────
    extras = 0
    if sweep:    extras += 4; reasons.append(f"💧 ALT liq sweep @ {sweep['level']:.5f}")
    if fvg_near: extras += 3; reasons.append("⚡ FVG overlaps OB")

    vr15 = l15.get('vol_ratio', 1.0)
    if   vr15 >= 2.5: extras += 3; reasons.append(f"🚀 15M vol spike {vr15:.1f}x")
    elif vr15 >= 1.5: extras += 1; reasons.append(f"✅ 15M elevated vol {vr15:.1f}x")

    close1 = l1.get('close', 0); vwap1 = l1.get('vwap', 0)
    if direction == 'LONG'  and close1 < vwap1: extras = min(extras+1, 10); reasons.append("✅ Below VWAP")
    elif direction == 'SHORT' and close1 > vwap1: extras = min(extras+1, 10); reasons.append("✅ Above VWAP")

    score += min(extras, 10)

    # ── 8. BTC Full SMC Context (up to ±18 pts) ← UPGRADED ───────
    if btc_delta != 0 or btc_reasons:
        score += btc_delta
        if btc_reasons:
            reasons.extend([f"  ↳ {r}" for r in btc_reasons])

    return max(0, min(int(score), 100)), reasons, failed


# ══════════════════════════════════════════════════════════════════
#  MAIN SCANNER BOT
# ══════════════════════════════════════════════════════════════════

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
        self.smc          = SMCEngine()
        self.btc_analyser = BTCAnalyser(self.smc)

        self.active_trades  = {}
        self.signal_history = deque(maxlen=300)
        self.is_scanning    = False
        self.last_debug     = []
        self.last_btc_ctx   = None

        self.stats = {
            'total': 0, 'long': 0, 'short': 0,
            'elite': 0, 'premium': 0, 'high': 0,
            'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0,
            'btc_blocks': 0,
            'last_scan': None, 'pairs_scanned': 0,
        }

    # ── BTC Data Fetcher ─────────────────────────────────────────

    async def fetch_btc_context(self) -> dict:
        """Fetch BTC OHLCV across 3 TFs and run full SMC analysis."""
        try:
            raw_4h  = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '4h',  limit=220)
            raw_1h  = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1h',  limit=150)
            raw_15m = await self.exchange.fetch_ohlcv('BTC/USDT:USDT', '15m', limit=80)

            dfs = {}
            for key, raw in [('4h', raw_4h), ('1h', raw_1h), ('15m', raw_15m)]:
                df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                dfs[key] = add_indicators(df)

            ctx = self.btc_analyser.analyse(dfs['4h'], dfs['1h'], dfs['15m'])
            self.last_btc_ctx = ctx

            logger.info(
                f"🟠 BTC: {ctx['regime']} | ${ctx['price']:,.0f} | "
                f"1H: {ctx['1h_chg']:+.2f}% | 3H: {ctx['3h_chg']:+.2f}% | "
                f"Structure: {ctx['structure']['kind'] if ctx['structure'] else 'None'} | "
                f"At OB: bull={ctx['at_bull_ob']} bear={ctx['at_bear_ob']}"
            )
            return ctx

        except Exception as e:
            logger.error(f"BTC context error: {e}")
            return None

    # ── Alt Data Fetcher ─────────────────────────────────────────

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

    # ── Analysis ──────────────────────────────────────────────────

    def analyse(self, data, symbol, btc_ctx=None):
        debug = {
            'symbol': symbol.replace('/USDT:USDT',''),
            'gates': [], 'score': 0, 'bias': '?'
        }

        try:
            df4 = data['4h']; df1 = data['1h']; df15 = data['15m']
            if len(df1) < 80 or len(df15) < 40:
                debug['gates'].append('❌ Not enough candle data')
                return None, debug

            price = df1['close'].iloc[-1]

            # Gate 1: 4H Bias
            l4 = df4.iloc[-1]
            e21 = l4.get('ema_21',0); e50 = l4.get('ema_50',0)
            if   e21 > e50: bias = 'LONG'
            elif e21 < e50: bias = 'SHORT'
            else:
                debug['gates'].append('❌ 4H EMAs flat — no bias')
                return None, debug
            debug['bias'] = bias

            # ── BTC Full SMC Gate (early) ← UPGRADED ─────────────
            btc_delta, btc_block, btc_summary, btc_reasons = check_btc_alignment(btc_ctx, bias)
            debug['gates'].append(f"🟠 {btc_summary}")
            if btc_block:
                self.stats['btc_blocks'] += 1
                debug['gates'].append(f'🚫 BTC hard block — {bias} skipped')
                return None, debug
            # ─────────────────────────────────────────────────────

            # HH/LL bonus
            hh_ll_ok, hh_ll_msg = self.smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)
            debug['gates'].append(hh_ll_msg)

            # Gate 2: PD Zone
            pd_label, pd_pos = self.smc.pd_zone(df4, price)
            if bias == 'LONG' and pd_label == 'PREMIUM':
                debug['gates'].append(f'❌ PREMIUM zone — no longs'); return None, debug
            if bias == 'SHORT' and pd_label == 'DISCOUNT':
                debug['gates'].append(f'❌ DISCOUNT zone — no shorts'); return None, debug
            debug['gates'].append(f'✅ PD zone: {pd_label} ({pd_pos*100:.0f}%)')

            # Gate 3: 1H Structure
            highs1, lows1 = self.smc.swing_highs_lows(df1, left=4, right=4)
            structure = self.smc.detect_structure_break(df1, highs1, lows1, STRUCTURE_LOOKBACK)
            if structure:
                if bias == 'LONG' and 'BEAR' in structure['kind']:
                    debug['gates'].append(f'❌ Structure {structure["kind"]} opposes LONG')
                    return None, debug
                if bias == 'SHORT' and 'BULL' in structure['kind']:
                    debug['gates'].append(f'❌ Structure {structure["kind"]} opposes SHORT')
                    return None, debug
                debug['gates'].append(f'✅ Structure: {structure["kind"]}')
            else:
                debug['gates'].append('⚠️ No recent BOS/MSS (continuing)')

            # Gate 4: 1H Order Block (HARD GATE)
            obs = self.smc.find_order_blocks(df1, bias, lookback=60)
            if not obs:
                debug['gates'].append(f'❌ No valid {bias} OBs on 1H')
                return None, debug
            debug['gates'].append(f'✅ {len(obs)} OB(s) on 1H')

            active_ob = None
            for ob in obs:
                if self.smc.price_in_ob(price, ob, OB_TOLERANCE_PCT):
                    active_ob = ob; break

            if not active_ob:
                nearest  = obs[0]
                dist_pct = min(abs(price-nearest['top']), abs(price-nearest['bottom'])) / price * 100
                debug['gates'].append(
                    f'❌ Not at OB — nearest {dist_pct:.2f}% away '
                    f'[{nearest["bottom"]:.5f}–{nearest["top"]:.5f}]'
                )
                return None, debug
            debug['gates'].append(f'✅ Price IN OB [{active_ob["bottom"]:.5f}–{active_ob["top"]:.5f}]')

            # FVG
            fvgs     = self.smc.find_fvg(df1, bias, lookback=25)
            fvg_near = None
            for fvg in fvgs:
                if fvg['bottom'] < active_ob['top'] and fvg['top'] > active_ob['bottom']:
                    fvg_near = fvg; break
            if fvg_near:
                debug['gates'].append('✅ 1H FVG overlaps OB')

            # Sweep
            sweep = self.smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)
            if sweep:
                debug['gates'].append(f'✅ 1H liq sweep @ {sweep["level"]:.5f}')

            # Score
            score, reasons, failed = score_setup(
                bias, active_ob, structure, sweep, fvg_near,
                df1, df15, df4, pd_label, hh_ll_ok,
                btc_delta=btc_delta,
                btc_reasons=btc_reasons
            )
            debug['score'] = score
            debug['gates'] += failed

            if score < MIN_SCORE:
                debug['gates'].append(f'❌ Score {score} < {MIN_SCORE}')
                return None, debug

            if   score >= 92: quality = 'ELITE 👑'
            elif score >= 85: quality = 'PREMIUM 💎'
            else:             quality = 'HIGH 🔥'

            atr1  = df1['atr'].iloc[-1]
            entry = price

            if bias == 'LONG':
                sl = min(active_ob['bottom'] - atr1*0.2, entry - atr1*0.6)
            else:
                sl = max(active_ob['top'] + atr1*0.2, entry + atr1*0.6)

            risk = abs(entry - sl)
            if risk < entry * 0.001:
                debug['gates'].append('❌ Degenerate SL'); return None, debug

            tps = (
                [entry+risk*1.5, entry+risk*2.5, entry+risk*4.0]
                if bias == 'LONG' else
                [entry-risk*1.5, entry-risk*2.5, entry-risk*4.0]
            )
            rr       = [abs(t-entry)/risk for t in tps]
            risk_pct = risk/entry*100
            tid      = f"{symbol.split('/')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            sig = {
                'trade_id':    tid,
                'symbol':      symbol.replace('/USDT:USDT',''),
                'full_symbol': symbol,
                'signal':      bias,
                'quality':     quality,
                'score':       score,
                'hh_ll':       hh_ll_ok,
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
                'btc_ctx':     btc_ctx,
                'btc_delta':   btc_delta,
                'btc_reasons': btc_reasons,
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

    # ── Signal Card Formatter ─────────────────────────────────────

    def fmt(self, s):
        arrow  = '🟢' if s['signal'] == 'LONG' else '🔴'
        icon   = '🚀' if s['signal'] == 'LONG' else '🔻'
        bar    = '█' * int(s['score']/10) + '░' * (10 - int(s['score']/10))
        z      = {'DISCOUNT':'🟩 DISCOUNT','PREMIUM':'🟥 PREMIUM','NEUTRAL':'🟨 NEUTRAL'}.get(s['pd_zone'],'')
        ob     = s['ob']
        hh_tag = '🏔️ Trending (HH/LL ✅)' if s.get('hh_ll') else '〰️ Ranging (no HH/LL)'

        # BTC block
        btc = s.get('btc_ctx') or {}
        reg = btc.get('regime', 'UNKNOWN')
        em  = regime_emoji(reg)

        btc_struct = btc.get('structure')
        btc_struct_str = btc_struct['kind'] if btc_struct else 'None'
        btc_ob_str = ''
        if btc.get('at_bull_ob') and s['signal'] == 'LONG':
            ob_b = btc.get('ob_bull')
            btc_ob_str = f"\n  🎯 BTC at bull OB [{ob_b['bottom']:.0f}–{ob_b['top']:.0f}]" if ob_b else "\n  🎯 BTC at bull OB"
        elif btc.get('at_bear_ob') and s['signal'] == 'SHORT':
            ob_b = btc.get('ob_bear')
            btc_ob_str = f"\n  🎯 BTC at bear OB [{ob_b['bottom']:.0f}–{ob_b['top']:.0f}]" if ob_b else "\n  🎯 BTC at bear OB"

        btc_sweep_str = ""
        if btc.get('sweep'):
            sw = btc['sweep']
            btc_sweep_str = f"\n  💧 BTC swept {sw['type']} @ {sw['level']:.0f}"

        btc_trigger_str = ""
        if btc.get('trigger_label'):
            btc_trigger_str = f"\n  🕯️ BTC trigger: {btc['trigger_label']}"

        msg  = f"{'━'*40}\n"
        msg += f"{icon} <b>SMC PRO v4.2 — {s['quality']}</b> {icon}\n"
        msg += f"{'━'*40}\n\n"
        msg += f"<b>🆔</b> <code>{s['trade_id']}</code>\n"
        msg += f"<b>📊 PAIR:</b>   <b>#{s['symbol']}USDT</b>\n"
        msg += f"<b>📍 DIR:</b>    {arrow} <b>{s['signal']}</b>\n"
        msg += f"<b>🗺️ ZONE:</b>   {z} ({s['pd_pos']*100:.0f}%)\n"
        msg += f"<b>📐 4H STR:</b> {hh_tag}\n\n"

        # ── BTC Full SMC Section ───────────────────────────────────
        msg += f"<b>{'─'*38}</b>\n"
        msg += f"<b>🟠 BTC KING ANALYSIS</b>\n"
        msg += f"  {em} Regime:    <b>{reg}</b>\n"
        msg += f"  📊 Trend:     {btc.get('ema_trend','?')}\n"
        msg += f"  🏗️ Structure: {btc_struct_str}\n"
        msg += f"  💰 Price:     ${btc.get('price',0):,.0f}\n"
        msg += f"  📈 1H move:   {btc.get('1h_chg',0):+.2f}%\n"
        msg += f"  📈 3H move:   {btc.get('3h_chg',0):+.2f}%\n"
        msg += f"  🗺️ PD Zone:   {btc.get('pd_zone','?')} ({btc.get('pd_pos',0.5)*100:.0f}%)\n"
        msg += f"  {'🏔️ HH/LL: YES' if btc.get('hh_ll') else '〰️ HH/LL: NO (ranging)'}\n"
        msg += btc_ob_str
        msg += btc_sweep_str
        msg += btc_trigger_str
        if s.get('btc_reasons'):
            for r in s['btc_reasons']:
                msg += f"\n  • {r}"
        msg += f"\n<b>{'─'*38}</b>\n\n"
        # ──────────────────────────────────────────────────────────

        msg += f"<b>⭐ SCORE: {s['score']} / 100</b>\n"
        msg += f"<code>[{bar}]</code>\n\n"
        msg += f"<b>📦 ORDER BLOCK (1H):</b>\n"
        msg += f"  Top:    <code>${ob['top']:.6f}</code>\n"
        msg += f"  Bottom: <code>${ob['bottom']:.6f}</code>\n"
        msg += f"  Mid:    <code>${ob['mid']:.6f}</code>\n\n"
        msg += f"<b>💰 ENTRY NOW:</b> <code>${s['entry']:.6f}</code>\n\n"
        msg += f"<b>🎯 TARGETS:</b>\n"
        for (lbl, eta), tp, rr in zip(
            [('TP1 — 50% exit','6-12h'),('TP2 — 30% exit','12-24h'),('TP3 — 20% exit','24-48h')],
            s['targets'], s['rr']
        ):
            pct = abs((tp - s['entry'])/s['entry']*100)
            msg += f"  <b>{lbl}</b> [{eta}]\n"
            msg += f"  <code>${tp:.6f}</code>  <b>+{pct:.2f}%</b>  RR {rr:.1f}:1\n\n"
        msg += f"<b>🛑 STOP LOSS:</b> <code>${s['stop_loss']:.6f}</code>  (-{s['risk_pct']:.2f}%)\n"
        msg += f"  └ <i>1H close below OB = invalidated</i>\n\n"
        if s['structure']:
            sk  = s['structure']['kind']
            lbl = '🔄 MSS — Early Reversal' if 'MSS' in sk else '💥 BOS — Pullback Entry'
            msg += f"<b>🏗️ STRUCTURE:</b> {lbl}\n\n"
        msg += f"<b>📋 CONFLUENCE:</b>\n"
        for r in s['reasons'][:15]:
            msg += f"  • {r}\n"
        msg += f"\n<b>⚠️ RISK:</b> 1-2% per trade only\n"
        msg += f"  Move SL → BE after TP1 hits\n"
        msg += f"\n<b>📡 Live Tracking: ON</b>\n"
        msg += f"<i>🕐 {s['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}</i>\n"
        msg += f"{'━'*40}"
        return msg

    # ── Telegram Sender ───────────────────────────────────────────

    async def send(self, text):
        try:
            await self.bot.send_message(
                chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    # ── TP/SL Alerts ──────────────────────────────────────────────

    async def tp_alert(self, t, n, price):
        tp  = t['targets'][n-1]
        pct = abs((tp - t['entry'])/t['entry']*100)
        advice = {
            1: 'Close 50% → Move SL to breakeven',
            2: 'Close 30% → Trail stop tight',
            3: 'Close final 20% 🎊'
        }
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

    # ── Trade Tracker ─────────────────────────────────────────────

    async def track(self):
        logger.info("📡 Trade tracker started")
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
                                    await self.tp_alert(t, i+1, p)
                                    t['tp_hit'][i] = True
                                    if i == 2: remove.append(tid)
                            if not t['sl_hit'] and p <= t['stop_loss']:
                                await self.sl_alert(t, p)
                                t['sl_hit'] = True; remove.append(tid)
                        else:
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
                logger.error(f"Track loop error: {e}"); await asyncio.sleep(60)

    # ── Main Scan ─────────────────────────────────────────────────

    async def scan(self):
        if self.is_scanning:
            return []
        self.is_scanning = True
        logger.info("🔍 Scan starting...")

        # Step 1: Full BTC SMC analysis — ONCE for whole scan
        btc_ctx = await self.fetch_btc_context()

        if btc_ctx:
            em  = regime_emoji(btc_ctx['regime'])
            st  = btc_ctx['structure']['kind'] if btc_ctx['structure'] else 'No structure'
            ob_status = []
            if btc_ctx['at_bull_ob']: ob_status.append('🎯 At Bull OB')
            if btc_ctx['at_bear_ob']: ob_status.append('🎯 At Bear OB')
            ob_str = ' | '.join(ob_status) if ob_status else '➖ Not at OB'

            btc_header = (
                f"{em} BTC: <b>{btc_ctx['regime']}</b> | ${btc_ctx['price']:,.0f}\n"
                f"Structure: {st} | {ob_str}\n"
                f"1H: {btc_ctx['1h_chg']:+.2f}% | 3H: {btc_ctx['3h_chg']:+.2f}% | "
                f"RSI: {btc_ctx['rsi']:.0f}"
            )
        else:
            btc_header = "⚪ BTC: data unavailable — filter skipped"

        await self.send(
            f"🔍 <b>SMC v4.2 SCAN STARTED</b>\n\n"
            f"{btc_header}\n\n"
            f"Stack: BTC SMC → 4H trend → 1H OB + structure + trigger → 15M vol\n"
            f"Min score: {MIN_SCORE} | OB tol: {OB_TOLERANCE_PCT*100:.1f}%\n"
            f"Vol filter: ${MIN_VOLUME_24H/1e6:.0f}M"
        )

        pairs       = await self.get_pairs()
        candidates  = []
        near_misses = []
        btc_blocked = 0
        scanned     = 0

        for pair in pairs:
            try:
                data = await self.fetch_data(pair)
                if data:
                    sig, dbg = self.analyse(data, pair, btc_ctx=btc_ctx)
                    if sig:
                        candidates.append(sig)
                        logger.info(f"  💎 {pair} {sig['signal']} score={sig['score']}")
                    else:
                        if any('🚫 BTC' in g or 'BTC hard block' in g for g in dbg['gates']):
                            btc_blocked += 1
                        elif dbg['score'] > 0 and any('✅ Price IN OB' in g for g in dbg['gates']):
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

        self.stats['last_scan']     = datetime.now()
        self.stats['pairs_scanned'] = scanned

        el = sum(1 for s in top if 'ELITE'   in s['quality'])
        pr = sum(1 for s in top if 'PREMIUM' in s['quality'])
        hi = len(top) - el - pr
        lg = sum(1 for s in top if s['signal'] == 'LONG')
        tr = sum(1 for s in top if s.get('hh_ll'))

        summ  = f"✅ <b>SCAN COMPLETE — v4.2</b>\n\n"
        if btc_ctx:
            summ += f"🟠 BTC: {btc_ctx['regime']} | {btc_ctx['1h_chg']:+.2f}% (1H)\n"
        summ += f"📊 Pairs scanned:  {scanned}\n"
        summ += f"🚫 BTC blocked:    {btc_blocked}\n"
        summ += f"🔍 Candidates:     {len(candidates)}\n"
        summ += f"🎯 Signals sent:   {len(top)}\n"
        if top:
            summ += f"  👑 Elite: {el}  💎 Premium: {pr}  🔥 High: {hi}\n"
            summ += f"  🟢 Long: {lg}  🔴 Short: {len(top)-lg}\n"
            summ += f"  🏔️ Trending: {tr}  〰️ Ranging: {len(top)-tr}\n"
        else:
            summ += f"\n<i>No setups met criteria this scan.</i>\n"
            summ += f"Near misses: {len(near_misses)} — use /debug\n"
        summ += f"\n⏰ {datetime.now().strftime('%H:%M UTC')}"
        await self.send(summ)

        logger.info(f"✅ Done. {len(candidates)} candidates → {len(top)} sent. BTC blocked: {btc_blocked}")
        self.is_scanning = False
        return top

    async def run(self, interval_min=SCAN_INTERVAL_MIN):
        logger.info("🚀 SMC Pro v4.2 starting")
        await self.send(
            "👑 <b>SMC PRO v4.2 — FULL BTC SMC ENGINE</b> 👑\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>BTC Full SMC  →  4H Trend  →  1H OB+Structure+Trigger  →  15M Vol</b>\n\n"
            "BTC now gets the full treatment:\n"
            "  ✅ 4H EMA + HH/LL + PD zone\n"
            "  ✅ 1H BOS/MSS structure\n"
            "  ✅ 1H Order Block (both directions)\n"
            "  ✅ 1H Liquidity Sweep\n"
            "  ✅ 1H Fair Value Gap\n"
            "  ✅ 1H Trigger candle\n"
            "  ✅ 15M volume spike\n\n"
            "BTC Regimes:\n"
            "  🟢🟢 BULL_CONFIRMED   🟢 BULL_STRUCTURE   🟡🟢 BULL_RANGING\n"
            "  🔴🔴 BEAR_CONFIRMED   🔴 BEAR_STRUCTURE   🟡🔴 BEAR_RANGING\n"
            "  🔄🟢 REVERSAL_BULL    🔄🔴 REVERSAL_BEAR   🟡 RANGING\n\n"
            f"Min score: {MIN_SCORE} | Scan: every {SCAN_INTERVAL_MIN}min\n\n"
            "Commands: /scan /btc /stats /trades /debug /help\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        asyncio.create_task(self.track())
        while True:
            try:
                await self.scan()
                logger.info(f"💤 Next scan in {interval_min}m")
                await asyncio.sleep(interval_min * 60)
            except Exception as e:
                logger.error(f"Main loop: {e}"); await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ══════════════════════════════════════════════════════════════════
#  BOT COMMANDS
# ══════════════════════════════════════════════════════════════════

class Commands:
    def __init__(self, s: SMCProScanner):
        self.s = s

    async def start(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        await u.message.reply_text(
            "👑 <b>SMC Pro v4.2 — Full BTC SMC Engine</b>\n\n"
            "BTC is now fully analysed with the same SMC stack as every alt:\n"
            "regime, structure, OB, sweep, FVG, trigger candle.\n\n"
            "Stack: BTC Full SMC → 4H trend → 1H OB+structure+trigger → 15M vol\n\n"
            "/scan /btc /stats /trades /debug /help",
            parse_mode=ParseMode.HTML
        )

    async def cmd_scan(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await u.message.reply_text("⚠️ Already scanning."); return
        await u.message.reply_text("🔍 Manual scan started...")
        asyncio.create_task(self.s.scan())

    async def cmd_btc(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        """Full BTC SMC snapshot — the flagship /btc command."""
        await u.message.reply_text("🟠 Running full BTC SMC analysis...")
        ctx = await self.s.fetch_btc_context()
        if not ctx:
            await u.message.reply_text("❌ Failed to fetch BTC data."); return

        em    = regime_emoji(ctx['regime'])
        st    = ctx['structure']['kind'] if ctx['structure'] else 'None detected'
        st_lvl= f" @ {ctx['structure']['level']:.0f}" if ctx['structure'] else ''

        ob_bull = ctx['ob_bull']
        ob_bear = ctx['ob_bear']
        ob_bull_str = f"[{ob_bull['bottom']:.0f}–{ob_bull['top']:.0f}]" if ob_bull else "None"
        ob_bear_str = f"[{ob_bear['bottom']:.0f}–{ob_bear['top']:.0f}]" if ob_bear else "None"

        sweep = ctx['sweep']
        sweep_str = f"{sweep['type']} @ {sweep['level']:.0f}" if sweep else "None"

        fvg_bull = ctx['fvg_bull']
        fvg_bear = ctx['fvg_bear']
        fvg_bull_str = f"[{fvg_bull['bottom']:.0f}–{fvg_bull['top']:.0f}]" if fvg_bull else "None"
        fvg_bear_str = f"[{fvg_bear['bottom']:.0f}–{fvg_bear['top']:.0f}]" if fvg_bear else "None"

        # Impact on alts
        d_long,  blk_long,  lbl_long,  r_long  = check_btc_alignment(ctx, 'LONG')
        d_short, blk_short, lbl_short, r_short = check_btc_alignment(ctx, 'SHORT')

        long_impact  = "🚫 HARD BLOCKED" if blk_long  else f"{'+' if d_long>=0 else ''}{d_long}pts"
        short_impact = "🚫 HARD BLOCKED" if blk_short else f"{'+' if d_short>=0 else ''}{d_short}pts"

        msg  = f"🟠 <b>BTC FULL SMC ANALYSIS v4.2</b>\n"
        msg += f"{'━'*38}\n\n"
        msg += f"<b>💰 Price:</b>  <code>${ctx['price']:,.2f}</code>\n"
        msg += f"<b>🏷️ Regime:</b> {em} <b>{ctx['regime']}</b>\n"
        msg += f"<b>📊 Trend:</b>  {ctx['ema_trend']}\n"
        msg += f"<b>🗺️ PD Zone:</b> {ctx['pd_zone']} ({ctx['pd_pos']*100:.0f}%)\n"
        msg += f"<b>🏔️ HH/LL:</b>  {'✅ YES — ' + ctx['hh_ll_msg'] if ctx['hh_ll'] else '❌ NO — ranging'}\n\n"

        msg += f"<b>🏗️ 1H Structure:</b> {st}{st_lvl}\n\n"

        msg += f"<b>📦 Order Blocks:</b>\n"
        msg += f"  Bull OB: {ob_bull_str} {'🎯 PRICE HERE' if ctx['at_bull_ob'] else ''}\n"
        msg += f"  Bear OB: {ob_bear_str} {'🎯 PRICE HERE' if ctx['at_bear_ob'] else ''}\n\n"

        msg += f"<b>💧 Liq Sweep:</b> {sweep_str}\n\n"

        msg += f"<b>⚡ FVGs:</b>\n"
        msg += f"  Bull FVG: {fvg_bull_str}\n"
        msg += f"  Bear FVG: {fvg_bear_str}\n\n"

        msg += f"<b>🕯️ Trigger Candle:</b> {ctx['trigger_label'] or 'None'}\n\n"

        msg += f"<b>📈 Momentum:</b>\n"
        msg += f"  RSI: {ctx['rsi']:.1f}\n"
        msg += f"  MACD: {'📈 Bull' if ctx['macd_bull'] else '📉 Bear'}\n"
        msg += f"  1H move: {ctx['1h_chg']:+.2f}%\n"
        msg += f"  3H move: {ctx['3h_chg']:+.2f}%\n"
        msg += f"  15M vol: {ctx['vol_ratio_15m']:.1f}x\n\n"

        msg += f"<b>🎯 Alt Signal Impact:</b>\n"
        msg += f"  LONG signals:  {long_impact}\n"
        msg += f"  SHORT signals: {short_impact}\n"
        if r_long:
            for r in r_long: msg += f"    • {r}\n"

        msg += f"\n<i>🕐 {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</i>"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def stats(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        s = self.s.stats
        msg  = "📊 <b>SMC PRO v4.2 STATS</b>\n\n"
        msg += f"Total signals: {s['total']}\n"
        msg += f"  👑 Elite: {s['elite']}  💎 Premium: {s['premium']}  🔥 High: {s['high']}\n"
        msg += f"  🟢 Long: {s['long']}  🔴 Short: {s['short']}\n\n"
        msg += f"🚫 BTC blocked (all-time): {s['btc_blocks']}\n\n"
        msg += f"TP1: {s['tp1']} | TP2: {s['tp2']} | TP3: {s['tp3']} | SL: {s['sl']}\n\n"
        if s['last_scan']:
            msg += f"Last scan: {s['last_scan'].strftime('%H:%M UTC')}\n"
            msg += f"Pairs: {s['pairs_scanned']}\n"
        msg += f"Active trades: {len(self.s.active_trades)}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def trades(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.active_trades:
            await u.message.reply_text("📭 No active trades."); return
        msg = f"📡 <b>ACTIVE TRADES ({len(self.s.active_trades)})</b>\n\n"
        for tid, t in list(self.s.active_trades.items())[:10]:
            age       = int((datetime.now()-t['timestamp']).total_seconds()/3600)
            tps       = ''.join(['✅' if h else '⏳' for h in t['tp_hit']])
            trend_tag = '🏔️' if t.get('hh_ll') else '〰️'
            btc_reg   = t.get('btc_ctx',{}).get('regime','?') if t.get('btc_ctx') else '?'
            msg += (
                f"<b>{t['symbol']}</b> {t['signal']} {trend_tag} — {t['quality']}\n"
                f"  Entry: <code>${t['entry']:.5f}</code> | Score: {t['score']}\n"
                f"  BTC was: {btc_reg} | TPs: {tps} | {age}h old\n\n"
            )
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def debug(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.last_debug:
            await u.message.reply_text("📭 No debug data yet. Run /scan first."); return
        msg = "🔬 <b>NEAR MISSES — Last Scan</b>\n"
        msg += "<i>(At OB but below score threshold)</i>\n\n"
        for d in self.s.last_debug[:8]:
            msg += f"<b>{d['symbol']}</b> {d['bias']} — Score: {d['score']}/100\n"
            for g in d['gates'][-5:]:
                msg += f"  {g}\n"
            msg += "\n"
        msg += f"<i>Min score: {MIN_SCORE}</i>"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def help(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>SMC PRO v4.2 — FULL STRATEGY</b>\n\n"
        msg += "<b>Scan Stack (in order):</b>\n"
        msg += "  0️⃣ BTC Full SMC  ← upgraded gate\n"
        msg += "  1️⃣ ALT 4H EMA bias\n"
        msg += "  2️⃣ PD zone filter\n"
        msg += "  3️⃣ ALT 1H BOS/MSS\n"
        msg += "  4️⃣ ALT 1H Order Block\n"
        msg += f"  5️⃣ Score ≥ {MIN_SCORE}/100\n\n"
        msg += "<b>BTC Regimes:</b>\n"
        msg += "  🟢🟢 BULL_CONFIRMED  = trend+structure+trigger\n"
        msg += "  🟢   BULL_STRUCTURE  = trend+structure (no trigger)\n"
        msg += "  🟡🟢 BULL_RANGING    = trend only\n"
        msg += "  🔴🔴 BEAR_CONFIRMED  = trend+structure+trigger\n"
        msg += "  🔴   BEAR_STRUCTURE  = trend+structure\n"
        msg += "  🟡🔴 BEAR_RANGING    = trend only\n"
        msg += "  🔄🟢 REVERSAL_BULL   = bear trend BUT bull MSS\n"
        msg += "  🔄🔴 REVERSAL_BEAR   = bull trend BUT bear MSS\n"
        msg += "  🟡   RANGING         = no clear bias\n\n"
        msg += "<b>Hard Blocks:</b>\n"
        msg += "  BEAR_CONFIRMED + LONGs → blocked\n"
        msg += "  BULL_CONFIRMED + SHORTs → blocked\n"
        msg += "  REVERSAL_BEAR forming → LONGs blocked\n"
        msg += "  REVERSAL_BULL forming → SHORTs blocked\n"
        msg += f"  Flash move >{BTC_HARD_BLOCK_1H_PCT}% (1H) → opposite blocked\n\n"
        msg += "<b>Score System (max ~100):</b>\n"
        msg += "  +25 — 1H trigger candle\n"
        msg += "  +20 — MSS structure\n"
        msg += "  +20 — Tight OB\n"
        msg += "  +15 — 4H triple EMA\n"
        msg += f"  +15 — BTC BULL/BEAR_CONFIRMED\n"
        msg += f"  +{HH_LL_BONUS}  — 4H HH/LL\n"
        msg += f"  +{BTC_OB_BONUS}  — BTC at own OB\n"
        msg += f"  +{BTC_SWEEP_BONUS}  — BTC liq sweep\n"
        msg += f"  +{BTC_FVG_BONUS}  — BTC FVG\n"
        msg += "  +12 — Momentum\n"
        msg += "  +10 — Extras\n"
        msg += "  -18 — BTC strongly opposing\n\n"
        msg += "<b>Commands:</b>\n"
        msg += "  /btc    — Full BTC SMC snapshot\n"
        msg += "  /scan   — Manual scan\n"
        msg += "  /stats  — Stats\n"
        msg += "  /trades — Active trades\n"
        msg += "  /debug  — Near misses\n"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

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
    app.add_handler(CommandHandler("btc",    cmds.cmd_btc))
    app.add_handler(CommandHandler("stats",  cmds.stats))
    app.add_handler(CommandHandler("trades", cmds.trades))
    app.add_handler(CommandHandler("debug",  cmds.debug))
    app.add_handler(CommandHandler("help",   cmds.help))

    await app.initialize()
    await app.start()
    logger.info("🤖 SMC Pro v4.2 ready!")

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
