"""
SMC PRO SCANNER v4.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGES from v4.0:
  - OB detection now requires BOS confirmation
  - OB detection now requires volume spike on impulse candle
  - FVG inside impulse + sweep before impulse -> ELITE tier
  - Pullback volume filter -> WEAK OB tag
  - OB tiers: ELITE / STANDARD / WEAK
  - REQUIRE_TRIGGER=True makes 1H trigger a hard gate
  - New tunables: OB_VOL_SPIKE_MIN, OB_PULLBACK_VOL_MAX, OB_BOS_SWING_LOOKBACK
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
MAX_SIGNALS_PER_SCAN   = 6
MIN_SCORE              = 75
MIN_VOLUME_24H         = 5_000_000
OB_TOLERANCE_PCT       = 0.008
OB_IMPULSE_ATR_MULT    = 1.0
OB_VOL_SPIKE_MIN       = 1.2     # impulse candle must be >= 1.2x avg vol
OB_PULLBACK_VOL_MAX    = 1.4     # pullback vol above this = WEAK OB
OB_BOS_SWING_LOOKBACK  = 10      # bars back to find swing high/low for BOS check
STRUCTURE_LOOKBACK     = 20
SCAN_INTERVAL_MIN      = 30
HH_LL_LOOKBACK         = 10
HH_LL_BONUS            = 8
REQUIRE_TRIGGER        = True    # True = no 1H trigger = hard fail


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
            return False, "Not enough 4H data for HH/LL check"
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback * 2):-lookback]
        if direction == 'LONG':
            rh, ph = recent['high'].max(), prior['high'].max()
            if rh > ph:
                return True, f"4H Higher High ({ph:.5f} -> {rh:.5f}) +{HH_LL_BONUS}pts"
            return False, f"4H no HH ({rh:.5f} <= {ph:.5f}) -- ranging"
        else:
            rl, pl = recent['low'].min(), prior['low'].min()
            if rl < pl:
                return True, f"4H Lower Low ({pl:.5f} -> {rl:.5f}) +{HH_LL_BONUS}pts"
            return False, f"4H no LL ({rl:.5f} >= {pl:.5f}) -- ranging"

    def detect_structure_break(self, df, highs, lows, lookback=STRUCTURE_LOOKBACK):
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
        """
        v4.1 -- All 6 OB validity checks now enforced.

        Hard gates (all must pass to even be considered):
          1. Candle direction (bearish candle for bullish OB, vice versa)
          2. Strong impulse move after the candle (>= ATR threshold)
          3. BOS confirmation -- impulse close must break recent swing H/L
          4. Volume spike on the OB candle (>= OB_VOL_SPIKE_MIN x average)
          5. OB not yet mitigated past its 50% midpoint

        Quality bonuses (score 2+ = ELITE, 1 = STANDARD):
          + FVG (imbalance) found inside the impulse move
          + Liquidity sweep detected before the impulse formed
          + Very strong vol spike on impulse (>= 2x average)

        Fake OB filter:
          - High average vol on pullback into OB -> tagged WEAK
        """
        obs   = []
        n     = len(df)
        start = max(2, n - lookback)

        for i in range(start, n - 3):
            c         = df.iloc[i]
            atr_local = (
                df['atr'].iloc[i]
                if 'atr' in df.columns and not pd.isna(df['atr'].iloc[i])
                else (c['high'] - c['low'])
            )
            min_impulse = atr_local * OB_IMPULSE_ATR_MULT
            fwd_end     = min(i + 5, n)

            # Gate 1: candle direction
            if direction == 'LONG'  and c['close'] >= c['open']: continue
            if direction == 'SHORT' and c['close'] <= c['open']: continue

            # Gate 2: impulse size
            if direction == 'LONG':
                fwd_high = df['high'].iloc[i+1:fwd_end].max()
                if pd.isna(fwd_high) or (fwd_high - c['low']) < min_impulse: continue
            else:
                fwd_low = df['low'].iloc[i+1:fwd_end].min()
                if pd.isna(fwd_low) or (c['high'] - fwd_low) < min_impulse: continue

            # Gate 3: BOS confirmation -- impulse must close beyond recent swing
            bos_start  = max(0, i - OB_BOS_SWING_LOOKBACK)
            fwd_closes = df['close'].iloc[i+1:fwd_end]
            if direction == 'LONG':
                swing_high = df['high'].iloc[bos_start:i].max()
                if not (fwd_closes > swing_high).any(): continue
            else:
                swing_low = df['low'].iloc[bos_start:i].min()
                if not (fwd_closes < swing_low).any(): continue

            # Gate 4: volume spike on OB candle itself
            vol_ratio_at_ob = (
                df['vol_ratio'].iloc[i] if 'vol_ratio' in df.columns else 1.0
            )
            if pd.isna(vol_ratio_at_ob) or vol_ratio_at_ob < OB_VOL_SPIKE_MIN: continue

            # Build OB dict
            if direction == 'LONG':
                ob = {
                    'top':       max(c['open'], c['close']),
                    'bottom':    c['low'],
                    'mid':      (max(c['open'], c['close']) + c['low']) / 2,
                    'bar':       i,
                    'vol_ratio': vol_ratio_at_ob,
                }
            else:
                ob = {
                    'top':       c['high'],
                    'bottom':    min(c['open'], c['close']),
                    'mid':      (c['high'] + min(c['open'], c['close'])) / 2,
                    'bar':       i,
                    'vol_ratio': vol_ratio_at_ob,
                }

            # Gate 5: not mitigated past 50%
            ob_50 = ob['mid']
            if direction == 'LONG'  and (df['close'].iloc[i+1:n] < ob_50).any(): continue
            if direction == 'SHORT' and (df['close'].iloc[i+1:n] > ob_50).any(): continue

            # Quality bonuses
            quality_pts     = 0
            quality_reasons = []

            # Bonus A: FVG inside impulse
            fvg_found = False
            for fi in range(i + 1, min(i + 4, n - 1)):
                pc = df.iloc[fi - 1]; nc = df.iloc[fi + 1]
                if direction == 'LONG' and pc['high'] < nc['low']:
                    fvg_found = True; quality_pts += 1
                    quality_reasons.append("FVG inside impulse"); break
                elif direction == 'SHORT' and pc['low'] > nc['high']:
                    fvg_found = True; quality_pts += 1
                    quality_reasons.append("FVG inside impulse"); break

            # Bonus B: sweep before impulse
            sweep_found = False
            sw_win = df.iloc[max(0, i - 8):i]
            if len(sw_win) >= 2:
                if direction == 'LONG':
                    r_lo  = sw_win['low'].min()
                    p_lo  = df['low'].iloc[max(0,i-20):max(0,i-8)].min() if i > 8 else r_lo
                    if r_lo <= p_lo * 1.001:
                        sweep_found = True; quality_pts += 1
                        quality_reasons.append("Sweep before impulse")
                else:
                    r_hi  = sw_win['high'].max()
                    p_hi  = df['high'].iloc[max(0,i-20):max(0,i-8)].max() if i > 8 else r_hi
                    if r_hi >= p_hi * 0.999:
                        sweep_found = True; quality_pts += 1
                        quality_reasons.append("Sweep before impulse")

            # Bonus C: very strong vol spike
            if vol_ratio_at_ob >= 2.0:
                quality_pts += 1
                quality_reasons.append(f"Strong vol {vol_ratio_at_ob:.1f}x")

            ob['fvg_inside']      = fvg_found
            ob['sweep_before']    = sweep_found
            ob['quality_reasons'] = quality_reasons

            # Fake OB filter: high pullback volume
            if i + 1 < n:
                pb_bars    = df.iloc[i+1:min(i+4, n)]
                avg_pb_vol = pb_bars['vol_ratio'].mean() if 'vol_ratio' in df.columns else 1.0
                if not pd.isna(avg_pb_vol) and avg_pb_vol > OB_PULLBACK_VOL_MAX:
                    ob['quality'] = 'WEAK'
                    ob['quality_reasons'].append(f"High pullback vol {avg_pb_vol:.1f}x")
                    obs.append(ob)
                    continue

            ob['quality'] = 'ELITE' if quality_pts >= 2 else 'STANDARD'
            obs.append(ob)

        obs.sort(
            key=lambda x: ({'ELITE':2,'STANDARD':1,'WEAK':0}[x['quality']], x['bar']),
            reverse=True
        )
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
        n = len(df); start = n - lookback
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

def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed):
    score     = 0
    reasons   = []
    failed    = []
    hard_fail = False

    l1  = df_1h.iloc[-1];  p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]; l4  = df_4h.iloc[-1]

    # 1. Structure (20 pts)
    if structure:
        if 'MSS' in structure['kind']:
            score += 20; reasons.append(f"MSS Early Reversal ({structure['kind']})")
        else:
            score += 14; reasons.append(f"BOS Pullback Entry ({structure['kind']})")
    else:
        failed.append("No BOS/MSS in last 20 candles")

    # 2. OB quality (20 pts, tiered)
    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        ob_qual     = ob.get('quality', 'STANDARD')
        if ob_qual == 'ELITE':
            score += 20
            reasons.append(f"ELITE OB ({ob_size_pct:.2f}%) BOS+Vol+FVG/Sweep")
        elif ob_qual == 'STANDARD':
            pts = 16 if ob_size_pct < 0.8 else 13
            score += pts
            reasons.append(f"{'Tight ' if ob_size_pct < 0.8 else ''}STANDARD OB ({ob_size_pct:.2f}%)")
        else:
            score += 7
            reasons.append(f"WEAK OB ({ob_size_pct:.2f}%) low conviction")
        for r in ob.get('quality_reasons', []):
            reasons.append(f"  -- {r}")
    else:
        failed.append("No valid OB found")

    # 3. 4H Trend (15 pts)
    e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0); e200 = l4.get('ema_200', 0)
    if direction == 'LONG':
        if e21 > e50 > e200: score += 15; reasons.append("4H Triple EMA Bull Stack")
        elif e21 > e50:      score += 10; reasons.append("4H EMA 21>50 Bull")
        elif pd_label == 'DISCOUNT': score += 6; reasons.append("4H Discount Zone")
        else: failed.append("4H trend weak for LONG")
    else:
        if e21 < e50 < e200: score += 15; reasons.append("4H Triple EMA Bear Stack")
        elif e21 < e50:      score += 10; reasons.append("4H EMA 21<50 Bear")
        elif pd_label == 'PREMIUM': score += 6; reasons.append("4H Premium Zone")
        else: failed.append("4H trend weak for SHORT")

    # 4. HH/LL Bonus (8 pts)
    if hh_ll_confirmed:
        score += HH_LL_BONUS; reasons.append(f"4H HH/LL confirmed (+{HH_LL_BONUS}pts)")
    else:
        failed.append("4H HH/LL not confirmed -- ranging")

    # 5. 1H Entry Trigger (25 pts) -- hard gate if REQUIRE_TRIGGER
    trigger = False; trigger_label = ""
    if direction == 'LONG':
        if   l1.get('bull_engulf',0)==1: score+=25; trigger=True; trigger_label="1H Bull Engulf (strongest)"
        elif l1.get('bull_pin',0)==1:    score+=22; trigger=True; trigger_label="1H Bull Pin Bar"
        elif l1.get('hammer',0)==1:      score+=18; trigger=True; trigger_label="1H Hammer"
        elif p1.get('bull_engulf',0)==1: score+=14; trigger=True; trigger_label="1H Bull Engulf (prev)"
        elif p1.get('bull_pin',0)==1:    score+=11; trigger=True; trigger_label="1H Bull Pin (prev)"
        elif p1.get('hammer',0)==1:      score+=9;  trigger=True; trigger_label="1H Hammer (prev)"
    else:
        if   l1.get('bear_engulf',0)==1:    score+=25; trigger=True; trigger_label="1H Bear Engulf (strongest)"
        elif l1.get('bear_pin',0)==1:       score+=22; trigger=True; trigger_label="1H Bear Pin Bar"
        elif l1.get('shooting_star',0)==1:  score+=18; trigger=True; trigger_label="1H Shooting Star"
        elif p1.get('bear_engulf',0)==1:    score+=14; trigger=True; trigger_label="1H Bear Engulf (prev)"
        elif p1.get('bear_pin',0)==1:       score+=11; trigger=True; trigger_label="1H Bear Pin (prev)"
        elif p1.get('shooting_star',0)==1:  score+=9;  trigger=True; trigger_label="1H Shooting Star (prev)"

    if trigger:
        reasons.append(trigger_label)
    else:
        if REQUIRE_TRIGGER:
            hard_fail = True
            failed.append("No 1H trigger candle -- HARD GATE FAILED")
        else:
            score -= 12
            failed.append("No 1H trigger yet -- wait for candle close")

    # 6. Momentum (12 pts)
    rsi1  = l1.get('rsi', 50)
    macd1 = l1.get('macd', 0);  ms1  = l1.get('macd_signal', 0)
    pm1   = p1.get('macd', 0);  pms1 = p1.get('macd_signal', 0)
    sk1   = l1.get('srsi_k', 0.5); sd1 = l1.get('srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi1 <= 55: score += 4; reasons.append(f"RSI reset zone ({rsi1:.0f})")
        elif rsi1 < 28:      score += 3; reasons.append(f"RSI oversold ({rsi1:.0f})")
        if macd1 > ms1 and pm1 <= pms1: score += 5; reasons.append("MACD bull cross")
        elif macd1 > ms1:               score += 2; reasons.append("MACD bullish")
        if sk1 < 0.3 and sk1 > sd1:    score += 3; reasons.append("StochRSI bull cross")
    else:
        if 45 <= rsi1 <= 72: score += 4; reasons.append(f"RSI overbought zone ({rsi1:.0f})")
        elif rsi1 > 72:      score += 3; reasons.append(f"RSI overbought ({rsi1:.0f})")
        if macd1 < ms1 and pm1 >= pms1: score += 5; reasons.append("MACD bear cross")
        elif macd1 < ms1:               score += 2; reasons.append("MACD bearish")
        if sk1 > 0.7 and sk1 < sd1:    score += 3; reasons.append("StochRSI bear cross")

    # 7. Extras (10 pts)
    extras = 0
    if sweep:    extras += 4; reasons.append(f"Liq sweep @ {sweep['level']:.5f}")
    if fvg_near: extras += 3; reasons.append("FVG overlaps OB")

    vr15 = l15.get('vol_ratio', 1.0) if hasattr(l15, 'get') else 1.0
    if not pd.isna(vr15):
        if   vr15 >= 2.5: extras += 3; reasons.append(f"15M vol spike {vr15:.1f}x")
        elif vr15 >= 1.5: extras += 1; reasons.append(f"15M elevated vol {vr15:.1f}x")

    close1 = l1.get('close', 0); vwap1 = l1.get('vwap', 0)
    if direction == 'LONG'  and close1 < vwap1: extras = min(extras+1,10); reasons.append("1H below VWAP")
    elif direction == 'SHORT' and close1 > vwap1: extras = min(extras+1,10); reasons.append("1H above VWAP")

    score += min(extras, 10)
    return max(0, min(int(score), 100)), reasons, failed, hard_fail


# ══════════════════════════════════════════════════════════════
#  SCANNER BOT
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
            'total':0,'long':0,'short':0,
            'elite':0,'premium':0,'high':0,
            'ob_elite':0,'ob_standard':0,'ob_weak':0,
            'tp1':0,'tp2':0,'tp3':0,'sl':0,
            'last_scan':None,'pairs_scanned':0
        }

    async def get_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = [
                s for s in self.exchange.symbols
                if s.endswith('/USDT:USDT') and 'PERP' not in s
                and tickers.get(s, {}).get('quoteVolume', 0) > MIN_VOLUME_24H
            ]
            pairs.sort(key=lambda x: tickers.get(x,{}).get('quoteVolume',0), reverse=True)
            logger.info(f"Loaded {len(pairs)} pairs")
            return pairs
        except Exception as e:
            logger.error(f"get_pairs: {e}"); return []

    async def fetch_data(self, symbol):
        try:
            result = {}
            for tf, lim in [('4h',220),('1h',150),('15m',80)]:
                raw = await self.exchange.fetch_ohlcv(symbol, tf, limit=lim)
                df  = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                result[tf] = add_indicators(df)
                await asyncio.sleep(0.04)
            return result
        except Exception as e:
            logger.error(f"fetch_data {symbol}: {e}"); return None

    def analyse(self, data, symbol):
        debug = {'symbol': symbol.replace('/USDT:USDT',''), 'gates':[], 'score':0, 'bias':'?'}
        try:
            df4=data['4h']; df1=data['1h']; df15=data['15m']
            if len(df1)<80 or len(df15)<40:
                debug['gates'].append('Not enough data'); return None, debug

            price = df1['close'].iloc[-1]

            l4=df4.iloc[-1]; e21=l4.get('ema_21',0); e50=l4.get('ema_50',0)
            if   e21>e50: bias='LONG'
            elif e21<e50: bias='SHORT'
            else:
                debug['gates'].append('4H EMAs flat'); return None, debug
            debug['bias'] = bias

            hh_ll_ok, hh_ll_msg = self.smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)
            debug['gates'].append(hh_ll_msg)

            pd_label, pd_pos = self.smc.pd_zone(df4, price)
            if bias=='LONG' and pd_label=='PREMIUM':
                debug['gates'].append(f'PD PREMIUM -- no longs'); return None, debug
            if bias=='SHORT' and pd_label=='DISCOUNT':
                debug['gates'].append(f'PD DISCOUNT -- no shorts'); return None, debug
            debug['gates'].append(f'PD zone: {pd_label} ({pd_pos*100:.0f}%)')

            highs1, lows1 = self.smc.swing_highs_lows(df1, left=4, right=4)
            structure = self.smc.detect_structure_break(df1, highs1, lows1, lookback=STRUCTURE_LOOKBACK)
            if structure:
                if bias=='LONG'  and 'BEAR' in structure['kind']:
                    debug['gates'].append(f'Structure {structure["kind"]} opposes LONG'); return None, debug
                if bias=='SHORT' and 'BULL' in structure['kind']:
                    debug['gates'].append(f'Structure {structure["kind"]} opposes SHORT'); return None, debug
                debug['gates'].append(f'Structure: {structure["kind"]}')
            else:
                debug['gates'].append('No recent BOS/MSS')

            obs = self.smc.find_order_blocks(df1, bias, lookback=60)
            if not obs:
                debug['gates'].append(f'No valid {bias} OBs (need BOS+vol+impulse)'); return None, debug

            ob_counts = {'ELITE':0,'STANDARD':0,'WEAK':0}
            for o in obs: ob_counts[o.get('quality','STANDARD')] += 1
            debug['gates'].append(
                f'{len(obs)} OBs -- ELITE:{ob_counts["ELITE"]} '
                f'STANDARD:{ob_counts["STANDARD"]} WEAK:{ob_counts["WEAK"]}'
            )

            active_ob = None
            for qual in ['ELITE','STANDARD','WEAK']:
                for ob in obs:
                    if ob.get('quality')==qual and self.smc.price_in_ob(price, ob, OB_TOLERANCE_PCT):
                        active_ob = ob; break
                if active_ob: break

            if not active_ob:
                nearest  = obs[0]
                dist_pct = min(abs(price-nearest['top']),abs(price-nearest['bottom']))/price*100
                debug['gates'].append(
                    f'Price not at OB -- nearest {dist_pct:.2f}% away '
                    f'[{nearest["bottom"]:.5f}-{nearest["top"]:.5f}] ({nearest.get("quality","?")})'
                ); return None, debug
            debug['gates'].append(
                f'Price IN {active_ob.get("quality","?")} OB '
                f'[{active_ob["bottom"]:.5f}-{active_ob["top"]:.5f}]'
            )

            fvgs     = self.smc.find_fvg(df1, bias, lookback=25)
            fvg_near = next((f for f in fvgs
                             if f['bottom']<active_ob['top'] and f['top']>active_ob['bottom']), None)
            if fvg_near: debug['gates'].append('1H FVG overlaps OB')

            sweep = self.smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)
            if sweep: debug['gates'].append(f'1H liq sweep @ {sweep["level"]:.5f}')

            score, reasons, failed, hard_fail = score_setup(
                bias, active_ob, structure, sweep, fvg_near,
                df1, df15, df4, pd_label, hh_ll_ok
            )
            debug['score'] = score
            debug['gates'] += failed

            if hard_fail:
                debug['gates'].append('HARD GATE FAILED'); return None, debug
            if score < MIN_SCORE:
                debug['gates'].append(f'Score {score} < {MIN_SCORE}'); return None, debug

            quality = 'ELITE' if score>=92 else 'PREMIUM' if score>=85 else 'HIGH'
            atr1    = df1['atr'].iloc[-1]
            entry   = price

            if bias == 'LONG':
                sl  = min(active_ob['bottom'] - atr1*0.2, entry - atr1*0.6)
                tps = [entry+(entry-sl)*r for r in [1.5,2.5,4.0]]
            else:
                sl  = max(active_ob['top'] + atr1*0.2, entry + atr1*0.6)
                tps = [entry-(sl-entry)*r for r in [1.5,2.5,4.0]]

            risk = abs(entry - sl)
            if risk < entry*0.001:
                debug['gates'].append('Degenerate SL'); return None, debug

            tid = f"{symbol.split('/')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            sig = {
                'trade_id':tid, 'symbol':symbol.replace('/USDT:USDT',''),
                'full_symbol':symbol, 'signal':bias, 'quality':quality,
                'score':score, 'hh_ll':hh_ll_ok, 'entry':entry, 'stop_loss':sl,
                'targets':tps, 'rr':[abs(t-entry)/risk for t in tps],
                'risk_pct':risk/entry*100, 'ob':active_ob, 'fvg':fvg_near,
                'sweep':sweep, 'structure':structure, 'pd_zone':pd_label,
                'pd_pos':pd_pos, 'reasons':reasons,
                'tp_hit':[False,False,False], 'sl_hit':False, 'timestamp':datetime.now(),
            }
            debug['gates'].append(f'PASSED -- score {score}')
            return sig, debug

        except Exception as e:
            logger.error(f"analyse {symbol}: {e}")
            debug['gates'].append(f'Exception: {e}'); return None, debug

    def fmt(self, s):
        icon    = '🚀' if s['signal']=='LONG' else '🔻'
        bar     = '█'*int(s['score']/10) + '░'*(10-int(s['score']/10))
        z_map   = {'DISCOUNT':'🟩 DISCOUNT','PREMIUM':'🟥 PREMIUM','NEUTRAL':'🟨 NEUTRAL'}
        ob      = s['ob']
        ob_qual = ob.get('quality','STANDARD')
        ob_icon = {'ELITE':'👑','STANDARD':'📦','WEAK':'⚠️'}.get(ob_qual,'📦')
        q_icon  = {'ELITE':'👑','PREMIUM':'💎','HIGH':'🔥'}.get(s['quality'],'🔥')
        hh_tag  = 'Trending (HH/LL)' if s.get('hh_ll') else 'Ranging (no HH/LL)'
        vol_str = f"{ob.get('vol_ratio',0):.1f}x" if ob.get('vol_ratio') else 'N/A'

        msg  = f"{'━'*40}\n"
        msg += f"{icon} <b>SMC PRO v4.1 -- {s['quality']} {q_icon}</b>\n"
        msg += f"{'━'*40}\n\n"
        msg += f"<b>ID:</b> <code>{s['trade_id']}</code>\n"
        msg += f"<b>PAIR:</b> <b>#{s['symbol']}USDT</b>\n"
        msg += f"<b>DIR:</b>  <b>{s['signal']}</b>  |  <b>ZONE:</b> {z_map.get(s['pd_zone'],'')} ({s['pd_pos']*100:.0f}%)\n"
        msg += f"<b>4H:</b>   {hh_tag}\n\n"
        msg += f"<b>SCORE: {s['score']}/100</b>\n<code>[{bar}]</code>\n\n"
        msg += f"<b>{ob_icon} ORDER BLOCK ({ob_qual})</b>\n"
        msg += f"  Top:    <code>${ob['top']:.6f}</code>\n"
        msg += f"  Bottom: <code>${ob['bottom']:.6f}</code>\n"
        msg += f"  Mid:    <code>${ob['mid']:.6f}</code>\n"
        msg += f"  Impulse vol: {vol_str}  BOS: YES  Vol spike: YES\n"
        if ob.get('fvg_inside'):   msg += f"  FVG inside impulse: YES\n"
        if ob.get('sweep_before'): msg += f"  Sweep before impulse: YES\n"
        msg += f"\n<b>ENTRY:</b> <code>${s['entry']:.6f}</code>\n\n"
        msg += f"<b>TARGETS:</b>\n"
        for (lbl, eta), tp, rr in zip(
            [('TP1 50% exit','6-12h'),('TP2 30% exit','12-24h'),('TP3 20% exit','24-48h')],
            s['targets'], s['rr']
        ):
            pct = abs((tp-s['entry'])/s['entry']*100)
            msg += f"  <b>{lbl}</b> [{eta}]\n  <code>${tp:.6f}</code>  +{pct:.2f}%  RR {rr:.1f}:1\n\n"
        msg += f"<b>STOP LOSS:</b> <code>${s['stop_loss']:.6f}</code>  (-{s['risk_pct']:.2f}%)\n"
        msg += f"  1H close below OB = invalidated\n\n"
        if s['structure']:
            sk = s['structure']['kind']
            msg += f"<b>STRUCTURE:</b> {'MSS Early Reversal' if 'MSS' in sk else 'BOS Pullback Entry'}\n\n"
        msg += f"<b>CONFLUENCE:</b>\n"
        for r in s['reasons'][:14]: msg += f"  . {r}\n"
        msg += f"\n<b>RISK:</b> 1-2% per trade max | Move SL to BE after TP1\n"
        msg += f"<b>Live Tracking: ON</b>\n"
        msg += f"<i>{s['timestamp'].strftime('%Y-%m-%d %H:%M UTC')}</i>\n"
        msg += f"{'━'*40}"
        return msg

    async def send(self, text):
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Telegram: {e}")

    async def tp_alert(self, t, n, price):
        tp = t['targets'][n-1]; pct = abs((tp-t['entry'])/t['entry']*100)
        actions = {1:'Close 50% -- move SL to BE',2:'Close 30% -- trail stop',3:'Close final 20%'}
        msg  = f"<b>TP{n} HIT!</b>\n<code>{t['trade_id']}</code>\n<b>{t['symbol']}</b> {t['signal']}\n\n"
        msg += f"Target:  <code>${tp:.6f}</code>\nPrice:   <code>${price:.6f}</code>\n"
        msg += f"Profit:  <b>+{pct:.2f}%</b>\n\n{actions[n]}"
        await self.send(msg); self.stats[f'tp{n}'] += 1

    async def sl_alert(self, t, price):
        loss = abs((price-t['entry'])/t['entry']*100)
        msg  = f"<b>STOP LOSS HIT</b>\n<code>{t['trade_id']}</code>\n<b>{t['symbol']}</b> {t['signal']}\n\n"
        msg += f"Entry: <code>${t['entry']:.6f}</code>\nLoss:  <b>-{loss:.2f}%</b>\n\nWaiting for next setup."
        await self.send(msg); self.stats['sl'] += 1

    async def track(self):
        logger.info("Tracker started")
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30); continue
                remove = []
                for tid, t in list(self.active_trades.items()):
                    try:
                        if datetime.now() - t['timestamp'] > timedelta(hours=48):
                            await self.send(f"<b>48H TIMEOUT</b>\n<code>{tid}</code>\n{t['symbol']} -- Close manually.")
                            remove.append(tid); continue
                        p = (await self.exchange.fetch_ticker(t['full_symbol']))['last']
                        if t['signal'] == 'LONG':
                            for i, tp in enumerate(t['targets']):
                                if not t['tp_hit'][i] and p >= tp:
                                    await self.tp_alert(t, i+1, p); t['tp_hit'][i]=True
                                    if i==2: remove.append(tid)
                            if not t['sl_hit'] and p <= t['stop_loss']:
                                await self.sl_alert(t, p); t['sl_hit']=True; remove.append(tid)
                        else:
                            for i, tp in enumerate(t['targets']):
                                if not t['tp_hit'][i] and p <= tp:
                                    await self.tp_alert(t, i+1, p); t['tp_hit'][i]=True
                                    if i==2: remove.append(tid)
                            if not t['sl_hit'] and p >= t['stop_loss']:
                                await self.sl_alert(t, p); t['sl_hit']=True; remove.append(tid)
                    except Exception as e:
                        logger.error(f"track {tid}: {e}")
                for tid in set(remove): self.active_trades.pop(tid, None)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"track loop: {e}"); await asyncio.sleep(60)

    async def scan(self):
        if self.is_scanning: return []
        self.is_scanning = True
        await self.send(
            f"<b>SMC v4.1 SCAN</b>\n"
            f"OB: BOS+Vol+Tier | Trigger: {'HARD GATE' if REQUIRE_TRIGGER else 'SOFT'}\n"
            f"Min score: {MIN_SCORE} | OB tol: {OB_TOLERANCE_PCT*100:.1f}%\n"
            f"Vol filter: ${MIN_VOLUME_24H/1e6:.0f}M"
        )
        pairs=await self.get_pairs(); candidates=[]; near_misses=[]; scanned=0
        for pair in pairs:
            try:
                data = await self.fetch_data(pair)
                if data:
                    sig, dbg = self.analyse(data, pair)
                    if sig:
                        candidates.append(sig)
                        logger.info(f"  CANDIDATE: {pair} {sig['signal']} score={sig['score']} ob={sig['ob'].get('quality')}")
                    elif dbg['score']>0 and any('Price IN' in g for g in dbg['gates']):
                        near_misses.append(dbg)
                scanned+=1
                if scanned%30==0: logger.info(f"  {scanned}/{len(pairs)} | {len(candidates)} candidates")
                await asyncio.sleep(0.45)
            except Exception as e:
                logger.error(f"scan {pair}: {e}")

        candidates.sort(key=lambda x: x['score'], reverse=True)
        top = candidates[:MAX_SIGNALS_PER_SCAN]
        self.last_debug = sorted(near_misses, key=lambda x: x['score'], reverse=True)[:10]

        for sig in top:
            self.signal_history.append(sig); self.active_trades[sig['trade_id']]=sig
            self.stats['total']+=1; self.stats[sig['signal'].lower()]+=1
            if sig['quality']=='ELITE': self.stats['elite']+=1
            elif sig['quality']=='PREMIUM': self.stats['premium']+=1
            else: self.stats['high']+=1
            ob_q=sig['ob'].get('quality','STANDARD').lower()
            if f'ob_{ob_q}' in self.stats: self.stats[f'ob_{ob_q}']+=1
            await self.send(self.fmt(sig)); await asyncio.sleep(2)

        self.stats['last_scan']=datetime.now(); self.stats['pairs_scanned']=scanned
        el=sum(1 for s in top if s['quality']=='ELITE')
        pr=sum(1 for s in top if s['quality']=='PREMIUM')
        lg=sum(1 for s in top if s['signal']=='LONG')
        ob_el=sum(1 for s in top if s['ob'].get('quality')=='ELITE')
        ob_st=sum(1 for s in top if s['ob'].get('quality')=='STANDARD')

        summ  = f"<b>SCAN COMPLETE v4.1</b>\n\nPairs: {scanned} | Candidates: {len(candidates)} | Sent: {len(top)}\n"
        if top:
            summ += f"  Elite:{el} Premium:{pr} High:{len(top)-el-pr}\n"
            summ += f"  Long:{lg} Short:{len(top)-lg} | Trending:{sum(1 for s in top if s.get('hh_ll'))}\n"
            summ += f"  OB -- ELITE:{ob_el} STANDARD:{ob_st} WEAK:{len(top)-ob_el-ob_st}\n"
        else:
            summ += f"No setups passed. Near misses: {len(near_misses)} -- /debug\n"
        summ += f"{datetime.now().strftime('%H:%M UTC')}"
        await self.send(summ)
        logger.info(f"Scan done: {len(candidates)} -> {len(top)} sent")
        self.is_scanning=False; return top

    async def run(self, interval_min=SCAN_INTERVAL_MIN):
        await self.send(
            "<b>SMC PRO v4.1 ONLINE</b>\n\n"
            "4H Trend -> 1H OB+Structure+Trigger -> 15M Vol\n\n"
            "v4.1 OB changes:\n"
            "  BOS confirmation required\n  Volume spike required\n"
            "  ELITE/STANDARD/WEAK tiers\n  Fake OB filter (pullback vol)\n"
            f"  Trigger = {'HARD GATE' if REQUIRE_TRIGGER else 'soft'}\n\n"
            f"Min score: {MIN_SCORE} | Scan every: {SCAN_INTERVAL_MIN}min\n"
            "Commands: /scan /stats /trades /debug /help"
        )
        asyncio.create_task(self.track())
        while True:
            try:
                await self.scan()
                await asyncio.sleep(interval_min * 60)
            except Exception as e:
                logger.error(f"run loop: {e}"); await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ══════════════════════════════════════════════════════════════
#  COMMANDS
# ══════════════════════════════════════════════════════════════

class Commands:
    def __init__(self, s: SMCProScanner): self.s = s

    async def start(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        await u.message.reply_text(
            "<b>SMC Pro v4.1</b>\nReal OB detection: BOS+Vol+Tiers\n"
            "Stack: 4H -> 1H OB/Structure/Trigger -> 15M Vol\n\n"
            "/scan /stats /trades /debug /help", parse_mode=ParseMode.HTML)

    async def cmd_scan(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await u.message.reply_text("Already scanning..."); return
        await u.message.reply_text("Manual scan started...")
        asyncio.create_task(self.s.scan())

    async def stats(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        s=self.s.stats
        msg  = "<b>SMC PRO v4.1 STATS</b>\n\n"
        msg += f"Total: {s['total']} | Long:{s['long']} Short:{s['short']}\n"
        msg += f"Elite:{s['elite']} Premium:{s['premium']} High:{s['high']}\n\n"
        msg += f"OB Quality:\n  ELITE:{s['ob_elite']} STANDARD:{s['ob_standard']} WEAK:{s['ob_weak']}\n\n"
        msg += f"TP1:{s['tp1']} TP2:{s['tp2']} TP3:{s['tp3']} SL:{s['sl']}\n"
        if s['last_scan']: msg += f"Last scan: {s['last_scan'].strftime('%H:%M UTC')} | Pairs: {s['pairs_scanned']}\n"
        msg += f"Active trades: {len(self.s.active_trades)}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def trades(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.active_trades:
            await u.message.reply_text("No active trades."); return
        msg = f"<b>ACTIVE TRADES ({len(self.s.active_trades)})</b>\n\n"
        for tid, t in list(self.s.active_trades.items())[:10]:
            age   = int((datetime.now()-t['timestamp']).total_seconds()/3600)
            tps   = ' '.join([f"TP{i+1}:{'HIT' if h else 'wait'}" for i,h in enumerate(t['tp_hit'])])
            ob_q  = t['ob'].get('quality','?')
            trend = 'Trending' if t.get('hh_ll') else 'Ranging'
            msg  += f"<b>{t['symbol']}</b> {t['signal']} | OB:{ob_q} | {trend} | {t['quality']}\n"
            msg  += f"  Entry:<code>${t['entry']:.5f}</code> Score:{t['score']} Age:{age}h\n"
            msg  += f"  {tps}\n\n"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def debug(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.last_debug:
            await u.message.reply_text("No debug data. Run /scan first."); return
        msg = "<b>NEAR MISSES -- Last Scan</b>\n<i>(Reached OB but failed score/trigger)</i>\n\n"
        for d in self.s.last_debug[:8]:
            msg += f"<b>{d['symbol']}</b> {d['bias']} Score:{d['score']}/100\n"
            for g in d['gates'][-5:]: msg += f"  {g}\n"
            msg += "\n"
        msg += f"<i>Min:{MIN_SCORE} | Trigger:{'ON' if REQUIRE_TRIGGER else 'OFF'} | OB needs BOS+vol</i>"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def help(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        msg  = "<b>SMC PRO v4.1 STRATEGY</b>\n\n"
        msg += "<b>Hard Gates:</b>\n"
        msg += "  1. 4H EMA bias\n  2. PD zone filter\n  3. 1H BOS/MSS\n"
        msg += "  4. Price at valid OB (BOS+vol+impulse required)\n"
        if REQUIRE_TRIGGER: msg += "  5. 1H trigger candle (HARD GATE)\n"
        msg += f"  6. Score >= {MIN_SCORE}\n\n"
        msg += "<b>OB Tiers:</b>\n"
        msg += "  ELITE    BOS+Vol+FVG/Sweep  +20pts\n"
        msg += "  STANDARD BOS+Vol            +13pts\n"
        msg += "  WEAK     high pullback vol   +7pts\n\n"
        msg += "<b>Score (max 100):</b>\n"
        msg += "  +25 1H trigger  +20 MSS  +20 ELITE OB\n"
        msg += f"  +15 4H EMA  +{HH_LL_BONUS} HH/LL  +12 Momentum  +10 Extras\n\n"
        msg += "<b>TPs:</b> TP1=1.5R[6-12h] TP2=2.5R[12-24h] TP3=4R[24-48h]\n\n"
        msg += f"<b>Config:</b> MIN_SCORE={MIN_SCORE} OB_VOL_SPIKE={OB_VOL_SPIKE_MIN} "
        msg += f"OB_PB_VOL={OB_PULLBACK_VOL_MAX} BOS_LB={OB_BOS_SWING_LOOKBACK} "
        msg += f"TRIGGER={'ON' if REQUIRE_TRIGGER else 'OFF'}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

async def main():
    TELEGRAM_TOKEN   = "8034062612:AAEJYbPA8sMODYvqvt8U-5mM7c3Y3-GOYtM"
    TELEGRAM_CHAT_ID = "7500072234"
    BINANCE_API_KEY  = None
    BINANCE_SECRET   = None

    scanner = SMCProScanner(
        telegram_token=TELEGRAM_TOKEN,
        chat_id=TELEGRAM_CHAT_ID,
        api_key=BINANCE_API_KEY,
        secret=BINANCE_SECRET
    )
    app  = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds = Commands(scanner)
    app.add_handler(CommandHandler("start",  cmds.start))
    app.add_handler(CommandHandler("scan",   cmds.cmd_scan))
    app.add_handler(CommandHandler("stats",  cmds.stats))
    app.add_handler(CommandHandler("trades", cmds.trades))
    app.add_handler(CommandHandler("debug",  cmds.debug))
    app.add_handler(CommandHandler("help",   cmds.help))
    await app.initialize()
    await app.start()
    logger.info("SMC Pro v4.1 ready")
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
