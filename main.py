"""
SMC PRO SCANNER v4.1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGES from v4.0:
  - LONG signals: DISCOUNT zone ONLY (price in lower 40% of 4H range)
  - SHORT signals: PREMIUM zone ONLY (price in upper 40% of 4H range)
  - NEUTRAL zone signals are fully blocked (not scored, not sent)
  - Duplicate guard: same symbol+direction won't re-fire within 4H
  - Signal card redesigned: clean, minimal, follower-friendly
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
#  SETTINGS
# ═══════════════════════════════════════════════
MAX_SIGNALS_PER_SCAN  = 6
MIN_SCORE             = 75
MIN_VOLUME_24H        = 5_000_000
OB_TOLERANCE_PCT      = 0.008
OB_IMPULSE_ATR_MULT   = 1.0
STRUCTURE_LOOKBACK    = 20
SCAN_INTERVAL_MIN     = 30
HH_LL_LOOKBACK        = 10
HH_LL_BONUS           = 8
DUPLICATE_COOLDOWN_H  = 4   # hours before same symbol+direction can re-fire


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
        n = len(df_4h)
        if n < lookback * 2:
            return False, "⚠️ Not enough 4H data"
        recent = df_4h.iloc[-lookback:]
        prior  = df_4h.iloc[-(lookback * 2):-lookback]
        if direction == 'LONG':
            rh, ph = recent['high'].max(), prior['high'].max()
            if rh > ph:
                return True, f"4H Higher High confirmed"
            return False, f"4H no Higher High — ranging"
        else:
            rl, pl = recent['low'].min(), prior['low'].min()
            if rl < pl:
                return True, f"4H Lower Low confirmed"
            return False, f"4H no Lower Low — ranging"

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

def score_setup(direction, ob, structure, sweep, fvg_near,
                df_1h, df_15m, df_4h, pd_label, hh_ll_confirmed):
    score = 0
    reasons = []
    failed = []

    l1  = df_1h.iloc[-1]
    p1  = df_1h.iloc[-2]
    l15 = df_15m.iloc[-1]
    l4  = df_4h.iloc[-1]

    # 1. Structure (20 pts)
    if structure:
        if 'MSS' in structure['kind']:
            score += 20; reasons.append(f"MSS — Early Reversal")
        else:
            score += 14; reasons.append(f"BOS — Pullback Entry")
    else:
        failed.append("No BOS/MSS in last 20 candles")

    # 2. Order Block quality (20 pts)
    if ob:
        ob_size_pct = (ob['top'] - ob['bottom']) / ob['bottom'] * 100
        if ob_size_pct < 0.8:
            score += 20; reasons.append(f"Tight OB ({ob_size_pct:.2f}%) — high quality")
        elif ob_size_pct < 2.0:
            score += 13; reasons.append(f"OB ({ob_size_pct:.2f}%)")
        else:
            score += 7;  reasons.append(f"Wide OB ({ob_size_pct:.2f}%) — lower quality")
    else:
        failed.append("No valid OB found")

    # 3. 4H Trend Alignment (15 pts)
    e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0); e200 = l4.get('ema_200', 0)
    if direction == 'LONG':
        if e21 > e50 > e200:
            score += 15; reasons.append("4H Triple EMA Bull Stack")
        elif e21 > e50:
            score += 10; reasons.append("4H EMA 21>50 Bullish")
        elif pd_label == 'DISCOUNT':
            score += 6;  reasons.append("4H Discount Zone")
        else:
            failed.append("4H trend weak for LONG")
    else:
        if e21 < e50 < e200:
            score += 15; reasons.append("4H Triple EMA Bear Stack")
        elif e21 < e50:
            score += 10; reasons.append("4H EMA 21<50 Bearish")
        elif pd_label == 'PREMIUM':
            score += 6;  reasons.append("4H Premium Zone")
        else:
            failed.append("4H trend weak for SHORT")

    # 4. 4H HH/LL Bonus (8 pts)
    if hh_ll_confirmed:
        score += HH_LL_BONUS
        reasons.append(f"4H HH/LL confirmed")

    # 5. 1H Entry Trigger (25 pts)
    trigger = False
    trigger_label = ""

    if direction == 'LONG':
        if l1.get('bull_engulf', 0) == 1:
            score += 25; trigger = True; trigger_label = "1H Bullish Engulfing"
        elif l1.get('bull_pin', 0) == 1:
            score += 22; trigger = True; trigger_label = "1H Bullish Pin Bar"
        elif l1.get('hammer', 0) == 1:
            score += 18; trigger = True; trigger_label = "1H Hammer"
        elif p1.get('bull_engulf', 0) == 1:
            score += 14; trigger = True; trigger_label = "1H Bull Engulf (prev)"
        elif p1.get('bull_pin', 0) == 1:
            score += 11; trigger = True; trigger_label = "1H Bull Pin (prev)"
        elif p1.get('hammer', 0) == 1:
            score += 9;  trigger = True; trigger_label = "1H Hammer (prev)"
    else:
        if l1.get('bear_engulf', 0) == 1:
            score += 25; trigger = True; trigger_label = "1H Bearish Engulfing"
        elif l1.get('bear_pin', 0) == 1:
            score += 22; trigger = True; trigger_label = "1H Bearish Pin Bar"
        elif l1.get('shooting_star', 0) == 1:
            score += 18; trigger = True; trigger_label = "1H Shooting Star"
        elif p1.get('bear_engulf', 0) == 1:
            score += 14; trigger = True; trigger_label = "1H Bear Engulf (prev)"
        elif p1.get('bear_pin', 0) == 1:
            score += 11; trigger = True; trigger_label = "1H Bear Pin (prev)"
        elif p1.get('shooting_star', 0) == 1:
            score += 9;  trigger = True; trigger_label = "1H Shooting Star (prev)"

    if trigger:
        reasons.append(trigger_label)
    else:
        score -= 12
        failed.append("No 1H trigger candle yet")

    # 6. Momentum (12 pts)
    rsi1  = l1.get('rsi', 50)
    macd1 = l1.get('macd', 0);  ms1  = l1.get('macd_signal', 0)
    pm1   = p1.get('macd', 0);  pms1 = p1.get('macd_signal', 0)
    sk1   = l1.get('srsi_k', 0.5); sd1 = l1.get('srsi_d', 0.5)

    if direction == 'LONG':
        if 28 <= rsi1 <= 55:
            score += 4; reasons.append(f"RSI reset ({rsi1:.0f})")
        elif rsi1 < 28:
            score += 3; reasons.append(f"RSI oversold ({rsi1:.0f})")
        if macd1 > ms1 and pm1 <= pms1:
            score += 5; reasons.append("MACD bull cross")
        elif macd1 > ms1:
            score += 2; reasons.append("MACD bullish")
        if sk1 < 0.3 and sk1 > sd1:
            score += 3; reasons.append("Stoch RSI bull cross")
    else:
        if 45 <= rsi1 <= 72:
            score += 4; reasons.append(f"RSI overbought zone ({rsi1:.0f})")
        elif rsi1 > 72:
            score += 3; reasons.append(f"RSI overbought ({rsi1:.0f})")
        if macd1 < ms1 and pm1 >= pms1:
            score += 5; reasons.append("MACD bear cross")
        elif macd1 < ms1:
            score += 2; reasons.append("MACD bearish")
        if sk1 > 0.7 and sk1 < sd1:
            score += 3; reasons.append("Stoch RSI bear cross")

    # 7. Extras (10 pts)
    extras = 0
    if sweep:
        extras += 4; reasons.append(f"Liquidity sweep @ {sweep['level']:.5f}")
    if fvg_near:
        extras += 3; reasons.append("FVG overlaps OB")

    vr15 = l15.get('vol_ratio', 1.0)
    if   vr15 >= 2.5:
        extras += 3; reasons.append(f"15M vol spike {vr15:.1f}x")
    elif vr15 >= 1.5:
        extras += 1; reasons.append(f"15M elevated vol {vr15:.1f}x")

    close1 = l1.get('close', 0); vwap1 = l1.get('vwap', 0)
    if direction == 'LONG' and close1 < vwap1:
        extras = min(extras+1, 10); reasons.append("Below VWAP")
    elif direction == 'SHORT' and close1 > vwap1:
        extras = min(extras+1, 10); reasons.append("Above VWAP")

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

        # ── Duplicate guard: key = "SYMBOL_DIRECTION", value = last sent time
        self.sent_signals: dict[str, datetime] = {}

        self.stats = {
            'total': 0, 'long': 0, 'short': 0,
            'elite': 0, 'premium': 0, 'high': 0,
            'tp1': 0, 'tp2': 0, 'tp3': 0, 'sl': 0,
            'skipped_zone': 0, 'skipped_dup': 0,
            'last_scan': None, 'pairs_scanned': 0
        }

    def is_duplicate(self, symbol: str, direction: str) -> bool:
        """Returns True if we already sent this symbol+direction within cooldown window."""
        key = f"{symbol}_{direction}"
        last = self.sent_signals.get(key)
        if last and datetime.now() - last < timedelta(hours=DUPLICATE_COOLDOWN_H):
            return True
        return False

    def mark_sent(self, symbol: str, direction: str):
        key = f"{symbol}_{direction}"
        self.sent_signals[key] = datetime.now()

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
                debug['gates'].append('Not enough candle data')
                return None, debug

            price = df1['close'].iloc[-1]

            # Gate 1: 4H Bias
            l4 = df4.iloc[-1]
            e21 = l4.get('ema_21', 0); e50 = l4.get('ema_50', 0)
            if e21 > e50:       bias = 'LONG'
            elif e21 < e50:     bias = 'SHORT'
            else:
                debug['gates'].append('4H EMAs flat — no bias')
                return None, debug
            debug['bias'] = bias

            # HH/LL bonus check
            hh_ll_ok, hh_ll_msg = self.smc.check_4h_hh_ll(df4, bias, HH_LL_LOOKBACK)
            debug['gates'].append(hh_ll_msg)

            # Gate 2: PD Zone — STRICT v4.1
            # LONG only in DISCOUNT | SHORT only in PREMIUM
            pd_label, pd_pos = self.smc.pd_zone(df4, price)
            if bias == 'LONG' and pd_label != 'DISCOUNT':
                self.stats['skipped_zone'] += 1
                debug['gates'].append(f'BLOCKED — LONG needs DISCOUNT, price is {pd_label} ({pd_pos*100:.0f}%)')
                return None, debug
            if bias == 'SHORT' and pd_label != 'PREMIUM':
                self.stats['skipped_zone'] += 1
                debug['gates'].append(f'BLOCKED — SHORT needs PREMIUM, price is {pd_label} ({pd_pos*100:.0f}%)')
                return None, debug
            debug['gates'].append(f'Zone OK: {pd_label} ({pd_pos*100:.0f}%)')

            # Gate 3: 1H Structure
            highs1, lows1 = self.smc.swing_highs_lows(df1, left=4, right=4)
            structure = self.smc.detect_structure_break(df1, highs1, lows1, lookback=STRUCTURE_LOOKBACK)
            if structure:
                s_bull = 'BULL' in structure['kind']
                s_bear = 'BEAR' in structure['kind']
                if bias == 'LONG' and s_bear:
                    debug['gates'].append(f'Structure ({structure["kind"]}) opposes LONG')
                    return None, debug
                if bias == 'SHORT' and s_bull:
                    debug['gates'].append(f'Structure ({structure["kind"]}) opposes SHORT')
                    return None, debug
                debug['gates'].append(f'Structure: {structure["kind"]}')
            else:
                debug['gates'].append('No recent BOS/MSS (continuing)')

            # Gate 4: 1H Order Block
            obs = self.smc.find_order_blocks(df1, bias, lookback=60)
            if not obs:
                debug['gates'].append(f'No valid {bias} OBs on 1H')
                return None, debug

            active_ob = None
            for ob in obs:
                if self.smc.price_in_ob(price, ob, OB_TOLERANCE_PCT):
                    active_ob = ob; break

            if not active_ob:
                nearest  = obs[0]
                dist_pct = min(abs(price - nearest['top']), abs(price - nearest['bottom'])) / price * 100
                debug['gates'].append(f'Price not at OB — nearest {dist_pct:.2f}% away')
                return None, debug
            debug['gates'].append(f'Price IN OB')

            # FVG bonus
            fvgs = self.smc.find_fvg(df1, bias, lookback=25)
            fvg_near = None
            for fvg in fvgs:
                if fvg['bottom'] < active_ob['top'] and fvg['top'] > active_ob['bottom']:
                    fvg_near = fvg; break

            # Sweep
            sweep = self.smc.recent_liquidity_sweep(df1, bias, highs1, lows1, lookback=20)

            # Score
            score, reasons, failed = score_setup(
                bias, active_ob, structure, sweep, fvg_near,
                df1, df15, df4, pd_label, hh_ll_ok
            )
            debug['score'] = score
            debug['gates'] += failed

            if score < MIN_SCORE:
                debug['gates'].append(f'Score {score} < {MIN_SCORE}')
                return None, debug

            if   score >= 92: quality = 'ELITE'
            elif score >= 85: quality = 'PREMIUM'
            else:             quality = 'HIGH'

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
                debug['gates'].append('Degenerate SL')
                return None, debug

            if bias == 'LONG':
                tps = [entry + risk*1.5, entry + risk*2.5, entry + risk*4.0]
            else:
                tps = [entry - risk*1.5, entry - risk*2.5, entry - risk*4.0]

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
            debug['gates'].append(f'PASSED — Score {score}')
            return sig, debug

        except Exception as e:
            logger.error(f"Analyse {symbol}: {e}")
            debug['gates'].append(f'Exception: {e}')
            return None, debug

    # ══════════════════════════════════════════════════════════
    #  SIGNAL CARD — clean, professional, follower-friendly
    # ══════════════════════════════════════════════════════════

    def fmt(self, s):
        is_long  = s['signal'] == 'LONG'
        dir_icon = '🟢 LONG' if is_long else '🔴 SHORT'
        zone_icon = '🔵 DISCOUNT' if s['pd_zone'] == 'DISCOUNT' else '🟣 PREMIUM'

        quality_map = {
            'ELITE':   '👑 ELITE',
            'PREMIUM': '💎 PREMIUM',
            'HIGH':    '🔥 HIGH',
        }
        quality_label = quality_map.get(s['quality'], s['quality'])

        # Score bar
        filled = int(s['score'] / 10)
        bar = '█' * filled + '░' * (10 - filled)

        # Structure label
        struct_label = '—'
        if s['structure']:
            sk = s['structure']['kind']
            struct_label = 'MSS (Reversal)' if 'MSS' in sk else 'BOS (Pullback)'

        # Trigger candle (first reason that mentions "1H")
        trigger_label = next((r for r in s['reasons'] if '1H' in r), '—')

        # Confluence bullets (clean, no emoji overload)
        conf_items = [r for r in s['reasons'] if r != trigger_label][:5]

        ob = s['ob']
        tps = s['targets']
        rr  = s['rr']

        def pct(a, b):
            return abs((a - b) / b * 100)

        msg  = "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"  {quality_label}  |  {dir_icon}\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        msg += f"<b>#{s['symbol']}USDT  —  Perp Futures</b>\n"
        msg += f"Zone: {zone_icon}  |  Score: <b>{s['score']}/100</b>\n"
        msg += f"<code>[{bar}]</code>\n\n"

        msg += "─────────────────────────\n"
        msg += f"<b>📍 ENTRY</b>\n"
        msg += f"  <code>${s['entry']:.6f}</code>\n\n"

        msg += f"<b>🎯 TAKE PROFITS</b>\n"
        msg += f"  TP1 — <code>${tps[0]:.6f}</code>  (+{pct(tps[0], s['entry']):.2f}%)  RR {rr[0]:.1f}:1\n"
        msg += f"  TP2 — <code>${tps[1]:.6f}</code>  (+{pct(tps[1], s['entry']):.2f}%)  RR {rr[1]:.1f}:1\n"
        msg += f"  TP3 — <code>${tps[2]:.6f}</code>  (+{pct(tps[2], s['entry']):.2f}%)  RR {rr[2]:.1f}:1\n\n"

        msg += f"<b>🛑 STOP LOSS</b>\n"
        msg += f"  <code>${s['stop_loss']:.6f}</code>  (-{s['risk_pct']:.2f}%)\n\n"

        msg += "─────────────────────────\n"
        msg += f"<b>📊 ANALYSIS</b>\n"
        msg += f"  Structure : {struct_label}\n"
        msg += f"  OB Zone   : ${ob['bottom']:.6f} – ${ob['top']:.6f}\n"
        msg += f"  Trigger   : {trigger_label}\n"
        if s.get('sweep'):
            msg += f"  Sweep     : Yes — liquidity taken\n"
        if s.get('fvg'):
            msg += f"  FVG       : Yes — overlaps OB\n"
        msg += "\n"

        if conf_items:
            msg += f"<b>✅ CONFLUENCE</b>\n"
            for item in conf_items:
                msg += f"  · {item}\n"
            msg += "\n"

        msg += "─────────────────────────\n"
        msg += f"<b>📋 TRADE PLAN</b>\n"
        msg += f"  · Risk 1–2% of account only\n"
        msg += f"  · Close 50% at TP1, move SL to entry\n"
        msg += f"  · Close 30% at TP2, trail the rest\n"
        msg += f"  · Close final 20% at TP3\n"
        msg += f"  · 1H close below OB = setup invalidated\n\n"

        msg += f"<i>🕐 {s['timestamp'].strftime('%d %b %Y  %H:%M UTC')}</i>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━━━"
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
            1: 'Close 50% of position\nMove SL to entry (breakeven)',
            2: 'Close 30% of position\nTrail stop tightly',
            3: 'Close final 20% — trade complete 🎊'
        }
        dir_icon = '🟢' if t['signal'] == 'LONG' else '🔴'
        msg  = f"🎯 <b>TP{n} HIT — #{t['symbol']}USDT</b>\n\n"
        msg += f"{dir_icon} {t['signal']}  |  Score {t['score']}\n\n"
        msg += f"Target   : <code>${tp:.6f}</code>\n"
        msg += f"Price    : <code>${price:.6f}</code>\n"
        msg += f"Profit   : <b>+{pct:.2f}%</b>\n\n"
        msg += f"📋 Next step:\n{advice[n]}"
        await self.send(msg)
        self.stats[f'tp{n}'] += 1

    async def sl_alert(self, t, price):
        loss = abs((price - t['entry'])/t['entry']*100)
        dir_icon = '🟢' if t['signal'] == 'LONG' else '🔴'
        msg  = f"⛔ <b>STOP LOSS — #{t['symbol']}USDT</b>\n\n"
        msg += f"{dir_icon} {t['signal']}  |  Entry <code>${t['entry']:.6f}</code>\n\n"
        msg += f"Loss     : <b>-{loss:.2f}%</b>\n"
        msg += f"OB invalidated — wait for next setup."
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
                            await self.send(
                                f"⏰ <b>48H TIMEOUT — #{t['symbol']}USDT</b>\n"
                                f"Close manually if still open."
                            )
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
            f"🔍 <b>SMC v4.1 SCANNING</b>\n"
            f"LONG = Discount zone only  |  SHORT = Premium zone only\n"
            f"Min score: {MIN_SCORE}  |  Dup cooldown: {DUPLICATE_COOLDOWN_H}h"
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
                        sym = sig['symbol']
                        dir = sig['signal']

                        # ── Duplicate guard ──────────────────────────────
                        if self.is_duplicate(sym, dir):
                            logger.info(f"  ⏭ Dup skipped: {sym} {dir}")
                            self.stats['skipped_dup'] += 1
                            continue

                        candidates.append(sig)
                        logger.info(f"  💎 {pair} {dir} score={sig['score']}")
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
            self.mark_sent(sig['symbol'], sig['signal'])   # register duplicate guard
            self.stats['total'] += 1
            self.stats[sig['signal'].lower()] += 1
            if sig['quality'] == 'ELITE':    self.stats['elite']   += 1
            elif sig['quality'] == 'PREMIUM': self.stats['premium'] += 1
            else:                             self.stats['high']    += 1
            await self.send(self.fmt(sig))
            await asyncio.sleep(2)

        self.stats['last_scan']     = datetime.now()
        self.stats['pairs_scanned'] = scanned

        el = sum(1 for s in top if s['quality'] == 'ELITE')
        pr = sum(1 for s in top if s['quality'] == 'PREMIUM')
        hi = len(top) - el - pr
        lg = sum(1 for s in top if s['signal'] == 'LONG')

        summ  = f"✅ <b>SCAN COMPLETE</b>\n\n"
        summ += f"Pairs scanned : {scanned}\n"
        summ += f"Signals sent  : {len(top)}\n"
        if top:
            summ += f"  👑 Elite   : {el}\n"
            summ += f"  💎 Premium : {pr}\n"
            summ += f"  🔥 High    : {hi}\n"
            summ += f"  🟢 Long    : {lg}  |  🔴 Short : {len(top)-lg}\n"
        else:
            summ += f"\n<i>No setups met criteria.</i>\n"
            summ += f"Skipped (zone) : {self.stats['skipped_zone']}\n"
            summ += f"Skipped (dup)  : {self.stats['skipped_dup']}\n"
            summ += f"Near misses    : {len(near_misses)} — use /debug\n"
        summ += f"\n⏰ {datetime.now().strftime('%H:%M UTC')}"
        await self.send(summ)

        logger.info(f"✅ Done. {len(candidates)} candidates → {len(top)} sent.")
        self.is_scanning = False
        return top

    async def run(self, interval_min=SCAN_INTERVAL_MIN):
        logger.info("🚀 SMC Pro v4.1 starting")
        await self.send(
            "👑 <b>SMC PRO v4.1 — LIVE</b>\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>4H Trend  →  1H OB + Structure + Entry  →  15M Vol</b>\n\n"
            "🔵 LONG  : Discount zone only\n"
            "🟣 SHORT : Premium zone only\n"
            f"📵 Dup filter : {DUPLICATE_COOLDOWN_H}h cooldown\n"
            f"⭐ Min score : {MIN_SCORE}/100\n"
            f"⏱ Scan every : {SCAN_INTERVAL_MIN} min\n\n"
            "Commands: /scan /stats /trades /debug /help\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━"
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
            "👑 <b>SMC Pro v4.1</b>\n\n"
            "Only the cleanest OB setups.\n"
            "LONG from Discount · SHORT from Premium\n"
            "No duplicate signals.\n\n"
            "/scan /stats /trades /debug /help",
            parse_mode=ParseMode.HTML
        )

    async def cmd_scan(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if self.s.is_scanning:
            await u.message.reply_text("⚠️ Scan already running."); return
        await u.message.reply_text("🔍 Manual scan started...")
        asyncio.create_task(self.s.scan())

    async def stats(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        s = self.s.stats
        msg  = "📊 <b>SMC PRO v4.1 — STATS</b>\n\n"
        msg += f"Total signals  : {s['total']}\n"
        msg += f"  👑 Elite     : {s['elite']}\n"
        msg += f"  💎 Premium   : {s['premium']}\n"
        msg += f"  🔥 High      : {s['high']}\n\n"
        msg += f"  🟢 Long      : {s['long']}\n"
        msg += f"  🔴 Short     : {s['short']}\n\n"
        msg += f"TP1 : {s['tp1']}  TP2 : {s['tp2']}  TP3 : {s['tp3']}\n"
        msg += f"SL  : {s['sl']}\n\n"
        msg += f"Skipped (zone) : {s['skipped_zone']}\n"
        msg += f"Skipped (dup)  : {s['skipped_dup']}\n\n"
        if s['last_scan']:
            msg += f"Last scan : {s['last_scan'].strftime('%H:%M UTC')}\n"
            msg += f"Pairs     : {s['pairs_scanned']}\n"
        msg += f"Active trades : {len(self.s.active_trades)}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def trades(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.active_trades:
            await u.message.reply_text("📭 No active trades."); return
        msg = f"📡 <b>ACTIVE TRADES ({len(self.s.active_trades)})</b>\n\n"
        for tid, t in list(self.s.active_trades.items())[:10]:
            age  = int((datetime.now() - t['timestamp']).total_seconds() / 3600)
            tps  = ''.join(['✅' if h else '⏳' for h in t['tp_hit']])
            icon = '🟢' if t['signal'] == 'LONG' else '🔴'
            msg += (
                f"<b>#{t['symbol']}USDT</b>  {icon} {t['signal']}  [{t['quality']}]\n"
                f"  Entry : <code>${t['entry']:.5f}</code>  Score : {t['score']}\n"
                f"  TPs   : {tps}  |  Age : {age}h\n\n"
            )
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def debug(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        if not self.s.last_debug:
            await u.message.reply_text("📭 No debug data. Run /scan first.")
            return
        msg = "🔬 <b>NEAR MISSES — Last Scan</b>\n\n"
        for d in self.s.last_debug[:8]:
            msg += f"<b>{d['symbol']}</b>  {d['bias']}  Score: {d['score']}/100\n"
            for g in d['gates'][-3:]:
                msg += f"  · {g}\n"
            msg += "\n"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def help(self, u: Update, c: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>SMC PRO v4.1 — HOW IT WORKS</b>\n\n"
        msg += "<b>Timeframe Stack</b>\n"
        msg += "  4H  → Trend bias (EMA) + HH/LL\n"
        msg += "  1H  → Structure (BOS/MSS) + OB + Trigger\n"
        msg += "  15M → Volume spike bonus\n\n"
        msg += "<b>Hard Rules</b>\n"
        msg += "  · LONG signals: Discount zone only (bottom 40%)\n"
        msg += "  · SHORT signals: Premium zone only (top 40%)\n"
        msg += "  · Neutral zone: always skipped\n"
        msg += f"  · Same signal won't repeat for {DUPLICATE_COOLDOWN_H}h\n"
        msg += f"  · Minimum score: {MIN_SCORE}/100\n\n"
        msg += "<b>Score Breakdown</b>\n"
        msg += "  25 pts — 1H entry candle (engulf/pin/hammer)\n"
        msg += "  20 pts — MSS/BOS structure\n"
        msg += "  20 pts — Order Block quality\n"
        msg += "  15 pts — 4H trend alignment\n"
        msg += f"  {HH_LL_BONUS} pts  — 4H Higher High / Lower Low\n"
        msg += "  12 pts — Momentum (RSI / MACD / Stoch)\n"
        msg += "  10 pts — Extras (sweep / FVG / vol)\n\n"
        msg += "<b>Trade Management</b>\n"
        msg += "  TP1 = 1:1.5 RR  — Close 50%\n"
        msg += "  TP2 = 1:2.5 RR  — Close 30%\n"
        msg += "  TP3 = 1:4.0 RR  — Close 20%\n"
        msg += "  Always risk 1-2% of account per trade\n"
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
    logger.info("🤖 SMC Pro v4.1 ready!")

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
