"""
REVERSAL RIDER — BACKTEST v2.0
================================
v1 lesson: 24.5% WR. Cause: fighting the macro trend.
  BB at extremes in crypto = strong trend, not exhaustion.
  Catching falling knives in bear markets = -90% DD.

v2 fix — THE KEY CHANGE: weekly trend must agree with direction.
  LONG only when 1W trend is UP (dip in bull market)
  SHORT only when 1W trend is DOWN (bounce in bear market)

Additional fixes from indicator analysis:
  - Removed BB as hard gate (was worst performer at 16.5% WR)
  - BB now used as bonus points only (not required)
  - Added 15M candle confirmation (double trigger required)
  - Extended timeout to 12H (short TP2 avg was +5.9%, needs room)
  - Tightened RSI gates: oversold <30 for LONG, >70 for SHORT
  - Added 1W EMA21 slope check as hard gate

LONG signal — dip in bull market:
  HARD Gate 1: 1W trend UP (price > 1W EMA21) ← NEW KEY GATE
  HARD Gate 2: 4H in short-term downtrend (EMA9 < EMA21)
  HARD Gate 3: 4H RSI < 35 (deeper oversold required)
  HARD Gate 4: 1H MACD histogram turning UP from negative
  HARD Gate 5: 1H bullish candle trigger (engulf/hammer/div)
  HARD Gate 6: 15M also showing bullish candle or hammer
  BONUS: Volume spike, RSI divergence, OBV, stoch cross

SHORT signal — bounce in bear market:
  HARD Gate 1: 1W trend DOWN (price < 1W EMA21) ← NEW KEY GATE
  HARD Gate 2: 4H in short-term uptrend (EMA9 > EMA21)
  HARD Gate 3: 4H RSI > 65 (deeper overbought required)
  HARD Gate 4: 1H MACD histogram turning DOWN from positive
  HARD Gate 5: 1H bearish candle trigger (engulf/shooting/div)
  HARD Gate 6: 15M also showing bearish candle
  BONUS: Volume spike, RSI divergence, OBV, stoch cross

Trade management:
  SL: 15M ATR × 1.2
  TP1: 2.0R → close 60%
  TP2: 3.5R → close 40%
  Timeout: 12H (extended from 8H)
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import ta
import warnings
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────
LOOKBACK_DAYS    = 360     # 6 months — enough data to be meaningful
TOP_N_PAIRS      = 300     # top 300 by volume — quality universe
MIN_VOLUME_USDT  = 3_000_000  # $3M min — liquid pairs only

# Signal gates
RSI_OVERSOLD_4H   = 35     # v2: tighter — deeper oversold required
RSI_OVERBOUGHT_4H = 65     # v2: tighter — deeper overbought required
BB_STRETCHED_LOW  = 0.15   # 4H bb_pband below this = price at/below lower BB
BB_STRETCHED_HIGH = 0.85   # 4H bb_pband above this = price at/above upper BB
VOL_SPIKE_MULT    = 1.8    # volume must be 1.8x avg for confirmation
MIN_SCORE         = 60     # minimum score to fire signal (out of 100)

# Trade management
TP1_RR            = 2.0    # first target: 2R
TP2_RR            = 3.5    # second target: 3.5R
TP1_CLOSE_PCT     = 0.60   # close 60% at TP1
TP2_CLOSE_PCT     = 0.40   # close 40% at TP2
ATR_SL_MULT       = 1.2    # SL = 15M ATR × 1.2 (tight)
MAX_TRADE_HOURS   = 12     # v2: extended — short TP2 avg was +5.9%, needs room

# BTC regime filter
REGIME_MODE       = 'HARD'  # LONG only in BULL, SHORT only in BEAR

OUTPUT_FILE = '/mnt/user-data/outputs/backtest_reversal_v2_results.xlsx'


# ─────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────

def add_indicators(df):
    if len(df) < 55:
        return df
    try:
        df = df.copy()
        df['ema_9']   = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21']  = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50']  = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=min(200, len(df)-1)).ema_indicator()

        df['rsi']     = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        macd = ta.trend.MACD(df['close'])
        df['macd']      = macd.macd()
        df['macd_sig']  = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper']  = bb.bollinger_hband()
        df['bb_lower']  = bb.bollinger_lband()
        df['bb_mid']    = bb.bollinger_mavg()
        df['bb_pband']  = bb.bollinger_pband()
        df['bb_width']  = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        df['atr']       = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()

        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        df['adx']       = adx.adx()
        df['di_plus']   = adx.adx_pos()
        df['di_minus']  = adx.adx_neg()

        df['vol_sma']   = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, np.nan)

        df['obv']       = ta.volume.OnBalanceVolumeIndicator(
            df['close'], df['volume']
        ).on_balance_volume()
        df['obv_ema']   = df['obv'].ewm(span=20).mean()

        df['cmf']       = ta.volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).chaikin_money_flow()

        df['mfi']       = ta.volume.MFIIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).money_flow_index()

        srsi = ta.momentum.StochRSIIndicator(df['close'])
        df['srsi_k']    = srsi.stochrsi_k()
        df['srsi_d']    = srsi.stochrsi_d()

        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap']      = (tp * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap'].fillna(df['close'], inplace=True)

        # Candle patterns
        body = (df['close'] - df['open']).abs()
        uw   = df['high'] - df[['open','close']].max(axis=1)
        lw   = df[['open','close']].min(axis=1) - df['low']

        df['bull_engulf'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open']) &
            (df['open'] <= df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1))
        ).astype(int)

        df['bear_engulf'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open']) &
            (df['open'] >= df['close'].shift(1)) &
            (df['close'] <= df['open'].shift(1))
        ).astype(int)

        df['hammer'] = (
            (lw > body * 2.0) & (lw > uw * 1.5) & (body > 0)
        ).astype(int)

        df['shooting_star'] = (
            (uw > body * 2.0) & (uw > lw * 1.5) & (body > 0)
        ).astype(int)

        df['bull_div'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['rsi'] > df['rsi'].shift(1)) &
            (df['rsi'] < 50)
        ).astype(int)

        df['bear_div'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['rsi'] < df['rsi'].shift(1)) &
            (df['rsi'] > 50)
        ).astype(int)

        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k']   = stoch.stoch()
        df['stoch_d']   = stoch.stoch_signal()

    except Exception as e:
        pass
    return df


# ─────────────────────────────────────────────────────────────
# SIGNAL SCORING
# ─────────────────────────────────────────────────────────────

def score_reversal(r4h, p4h, r1h, p1h, r15m, vol_ratio):
    """
    Score a potential reversal signal.
    Returns (long_score, short_score, long_reasons, short_reasons)

    Scoring logic:
    - Hard requirements give baseline points
    - Strength confirmations add bonus points
    - Score ≥ MIN_SCORE to fire
    """
    ls = 0; ss = 0
    lr = {}; sr = {}

    def sg(row, key, default=0):
        try:
            v = row[key]
            return default if pd.isna(v) else float(v)
        except:
            return default

    # ── 4H EXHAUSTION (prerequisite — 40 pts max) ──────────────

    # 4H trend direction
    e9_4h  = sg(r4h, 'ema_9')
    e21_4h = sg(r4h, 'ema_21')
    e50_4h = sg(r4h, 'ema_50')
    rsi_4h = sg(r4h, 'rsi', 50)
    bbp_4h = sg(r4h, 'bb_pband', 0.5)

    # LONG: need 4H downtrend
    if e9_4h < e21_4h < e50_4h:
        ls += 15; lr['4H_downtrend_for_long'] = 15  # price was falling — reversal potential
    elif e9_4h < e21_4h:
        ls += 8;  lr['4H_partial_downtrend'] = 8

    # SHORT: need 4H uptrend
    if e9_4h > e21_4h > e50_4h:
        ss += 15; sr['4H_uptrend_for_short'] = 15
    elif e9_4h > e21_4h:
        ss += 8;  sr['4H_partial_uptrend'] = 8

    # 4H RSI exhaustion
    if rsi_4h < 30:
        ls += 15; lr['4H_RSI_extreme_oversold'] = 15
    elif rsi_4h < RSI_OVERSOLD_4H:
        ls += 10; lr['4H_RSI_oversold'] = 10
    elif rsi_4h < 45:
        ls += 5;  lr['4H_RSI_low'] = 5

    if rsi_4h > 70:
        ss += 15; sr['4H_RSI_extreme_overbought'] = 15
    elif rsi_4h > RSI_OVERBOUGHT_4H:
        ss += 10; sr['4H_RSI_overbought'] = 10
    elif rsi_4h > 55:
        ss += 5;  sr['4H_RSI_high'] = 5

    # 4H BB stretch (price at extreme)
    if bbp_4h < 0.05:
        ls += 10; lr['4H_below_lower_BB'] = 10
    elif bbp_4h < BB_STRETCHED_LOW:
        ls += 6;  lr['4H_near_lower_BB'] = 6

    if bbp_4h > 0.95:
        ss += 10; sr['4H_above_upper_BB'] = 10
    elif bbp_4h > BB_STRETCHED_HIGH:
        ss += 6;  sr['4H_near_upper_BB'] = 6

    # ── 1H REVERSAL CONFIRMATION (40 pts max) ──────────────────

    rsi_1h   = sg(r1h, 'rsi', 50)
    hist_cur = sg(r1h, 'macd_hist')
    hist_prv = sg(p1h, 'macd_hist')
    e9_1h    = sg(r1h, 'ema_9')
    e21_1h   = sg(r1h, 'ema_21')
    e50_1h   = sg(r1h, 'ema_50')

    # 1H MACD histogram turning (the key reversal signal)
    # LONG: histogram was negative, now turning up
    if hist_cur > hist_prv and hist_cur < 0:
        ls += 12; lr['1H_MACD_hist_turning_up'] = 12  # bottom forming
    elif hist_cur > hist_prv and hist_cur > 0:
        ls += 7;  lr['1H_MACD_hist_expanding_up'] = 7  # already reversed
    if hist_cur < hist_prv and hist_cur > 0:
        ss += 12; sr['1H_MACD_hist_turning_down'] = 12  # top forming
    elif hist_cur < hist_prv and hist_cur < 0:
        ss += 7;  sr['1H_MACD_hist_expanding_down'] = 7

    # 1H RSI direction
    rsi_prv = sg(p1h, 'rsi', 50)
    if rsi_1h < 40 and rsi_1h > rsi_prv:
        ls += 8; lr['1H_RSI_rising_from_low'] = 8   # bounce starting
    elif rsi_1h < 50 and rsi_1h > rsi_prv:
        ls += 4; lr['1H_RSI_rising'] = 4

    if rsi_1h > 60 and rsi_1h < rsi_prv:
        ss += 8; sr['1H_RSI_falling_from_high'] = 8
    elif rsi_1h > 50 and rsi_1h < rsi_prv:
        ss += 4; sr['1H_RSI_falling'] = 4

    # 1H EMA position (price below EMA50 = demand zone for longs)
    close_1h = sg(r1h, 'close')
    if close_1h < e50_1h:
        ls += 8; lr['1H_price_below_EMA50_demand'] = 8
    elif close_1h < e21_1h:
        ls += 4; lr['1H_price_below_EMA21'] = 4

    if close_1h > e50_1h:
        ss += 8; sr['1H_price_above_EMA50_supply'] = 8
    elif close_1h > e21_1h:
        ss += 4; sr['1H_price_above_EMA21'] = 4

    # Stoch RSI reversal
    sk = sg(r1h, 'srsi_k', 0.5); sd = sg(r1h, 'srsi_d', 0.5)
    if sk < 0.25 and sk > sd:
        ls += 6; lr['1H_stoch_bull_cross'] = 6
    elif sk < 0.35:
        ls += 3; lr['1H_stoch_oversold'] = 3
    if sk > 0.75 and sk < sd:
        ss += 6; sr['1H_stoch_bear_cross'] = 6
    elif sk > 0.65:
        ss += 3; sr['1H_stoch_overbought'] = 3

    # CMF — money flow confirming
    cmf = sg(r1h, 'cmf')
    if cmf > 0.10:
        ls += 4; lr['1H_CMF_buying'] = 4
    if cmf < -0.10:
        ss += 4; sr['1H_CMF_selling'] = 4

    # ── CANDLE TRIGGER (20 pts max) ────────────────────────────
    # Reversal candle is the final confirmation — highest weight

    bull_eng = sg(r1h, 'bull_engulf')
    hammer   = sg(r1h, 'hammer')
    bull_div = sg(r1h, 'bull_div')
    bear_eng = sg(r1h, 'bear_engulf')
    shooting = sg(r1h, 'shooting_star')
    bear_div = sg(r1h, 'bear_div')

    # LONG triggers
    if bull_eng == 1:
        ls += 20; lr['1H_bull_engulf_trigger'] = 20
    elif hammer == 1:
        ls += 15; lr['1H_hammer_trigger'] = 15
    elif bull_div == 1:
        ls += 12; lr['1H_bull_divergence'] = 12

    # SHORT triggers
    if bear_eng == 1:
        ss += 20; sr['1H_bear_engulf_trigger'] = 20
    elif shooting == 1:
        ss += 15; sr['1H_shooting_star_trigger'] = 15
    elif bear_div == 1:
        ss += 12; sr['1H_bear_divergence'] = 12

    # ── VOLUME CONFIRMATION (bonus) ────────────────────────────
    if vol_ratio >= 3.0:
        if sg(r1h, 'close') > sg(p1h, 'close'):
            ls += 8; lr['vol_spike_3x_bull'] = 8
        else:
            ss += 8; sr['vol_spike_3x_bear'] = 8
    elif vol_ratio >= VOL_SPIKE_MULT:
        if sg(r1h, 'close') > sg(p1h, 'close'):
            ls += 5; lr['vol_spike_bull'] = 5
        else:
            ss += 5; sr['vol_spike_bear'] = 5

    # OBV trend (smart money)
    obv = sg(r1h, 'obv'); obv_ema = sg(r1h, 'obv_ema')
    if obv > obv_ema:
        ls += 3; lr['OBV_accumulation'] = 3
    else:
        ss += 3; sr['OBV_distribution'] = 3

    # MFI extreme
    mfi = sg(r1h, 'mfi', 50)
    if mfi < 20:
        ls += 5; lr['MFI_extreme_oversold'] = 5
    elif mfi < 30:
        ls += 3; lr['MFI_oversold'] = 3
    if mfi > 80:
        ss += 5; sr['MFI_extreme_overbought'] = 5
    elif mfi > 70:
        ss += 3; sr['MFI_overbought'] = 3

    # ADX — trend strength (we want ADX declining for reversal, not extreme)
    adx = sg(r1h, 'adx', 0)
    if 15 < adx < 35:
        # moderate ADX = trend was strong but fading — reversal more likely
        ls += 3; lr['ADX_moderate_reversal_zone'] = 3
        ss += 3; sr['ADX_moderate_reversal_zone'] = 3

    return ls, ss, lr, sr


# ─────────────────────────────────────────────────────────────
# TRADE SIMULATION (two-target: TP1 + TP2)
# ─────────────────────────────────────────────────────────────

def simulate_trade(idx, df_1h, direction, entry, sl, tp1, tp2):
    """
    Two-target simulation:
    - TP1: close 60% position at 2.0R
    - TP2: close 40% at 3.5R
    - SL: full loss if hit before TP1
    - Timeout: 8H — close at market
    """
    future = df_1h.iloc[idx+1 : idx+1+MAX_TRADE_HOURS]
    if len(future) == 0:
        return 'TIMEOUT', 0.0, 0

    tp1_hit    = False
    blended    = 0.0
    remaining  = 1.0

    for i, (_, row) in enumerate(future.iterrows()):
        hi = row['high']; lo = row['low']

        if direction == 'LONG':
            # SL check (only before TP1)
            if not tp1_hit and lo <= sl:
                sl_pct = (sl - entry) / entry * 100  # negative
                blended += remaining * sl_pct
                return 'SL', round(blended, 3), i+1

            if not tp1_hit and hi >= tp1:
                pct1 = (tp1 - entry) / entry * 100
                blended  += TP1_CLOSE_PCT * pct1
                remaining -= TP1_CLOSE_PCT
                tp1_hit   = True

            if tp1_hit and hi >= tp2:
                pct2 = (tp2 - entry) / entry * 100
                blended += remaining * pct2
                return 'TP2', round(blended, 3), i+1

        else:  # SHORT
            if not tp1_hit and hi >= sl:
                sl_pct = (entry - sl) / entry * 100  # negative
                blended += remaining * (-abs(sl_pct))
                return 'SL', round(blended, 3), i+1

            if not tp1_hit and lo <= tp1:
                pct1 = (entry - tp1) / entry * 100
                blended  += TP1_CLOSE_PCT * pct1
                remaining -= TP1_CLOSE_PCT
                tp1_hit   = True

            if tp1_hit and lo <= tp2:
                pct2 = (entry - tp2) / entry * 100
                blended += remaining * pct2
                return 'TP2', round(blended, 3), i+1

    # Timeout — close remaining at last close
    last = future.iloc[-1]['close']
    if direction == 'LONG':
        timeout_pct = (last - entry) / entry * 100
    else:
        timeout_pct = (entry - last) / entry * 100
    blended += remaining * timeout_pct

    outcome = 'TP1+TIMEOUT' if tp1_hit else 'TIMEOUT'
    return outcome, round(blended, 3), MAX_TRADE_HOURS


# ─────────────────────────────────────────────────────────────
# BACKTESTER
# ─────────────────────────────────────────────────────────────

class ReversalBacktesterV2:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.trades         = []
        self.ind_stats      = defaultdict(lambda: {'triggered':0,'wins':0,'losses':0})
        self.regime_blocked = 0

    async def get_pairs(self):
        await self.exchange.load_markets()
        tickers = await self.exchange.fetch_tickers()
        pairs = [
            s for s in self.exchange.symbols
            if s.endswith('/USDT:USDT') and 'PERP' not in s
            and tickers.get(s, {}).get('quoteVolume', 0) > MIN_VOLUME_USDT
        ]
        pairs.sort(key=lambda x: tickers.get(x,{}).get('quoteVolume',0), reverse=True)
        pairs = pairs[:TOP_N_PAIRS]
        print(f"✅ {len(pairs)} pairs loaded")
        return pairs

    async def fetch_df(self, symbol, tf, limit=700):
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except:
            return None

    async def load_btc_regime(self):
        print("📡 Loading BTC regime...")
        df = await self.fetch_df('BTC/USDT:USDT', '4h', limit=700)
        if df is None: return None
        df['ema21']  = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['regime'] = (df['close'] > df['ema21']).map({True:'BULL', False:'BEAR'})
        bull = (df['regime']=='BULL').sum()
        bear = (df['regime']=='BEAR').sum()
        print(f"  BULL: {bull} | BEAR: {bear}")
        return df['regime']

    async def backtest_pair(self, symbol, btc_regime_series):
        print(f"  📊 {symbol}...")

        df_1h  = await self.fetch_df(symbol, '1h',  limit=LOOKBACK_DAYS*24+150)
        await asyncio.sleep(0.12)
        df_4h  = await self.fetch_df(symbol, '4h',  limit=LOOKBACK_DAYS*6+60)
        await asyncio.sleep(0.12)
        df_15m = await self.fetch_df(symbol, '15m', limit=LOOKBACK_DAYS*96+200)
        await asyncio.sleep(0.12)
        df_1w  = await self.fetch_df(symbol, '1w',  limit=60)   # v2: weekly trend gate
        await asyncio.sleep(0.15)

        if df_1h is None or df_4h is None or df_15m is None or len(df_1h) < 100:
            return []

        df_1h  = add_indicators(df_1h)
        df_4h  = add_indicators(df_4h)
        df_15m = add_indicators(df_15m)
        if df_1w is not None and len(df_1w) >= 10:
            df_1w = add_indicators(df_1w)
        else:
            df_1w = None

        required_1h = ['ema_9','ema_21','ema_50','rsi','macd_hist','bb_pband','atr',
                       'bull_engulf','bear_engulf','hammer','shooting_star','srsi_k','srsi_d']

        pair_trades = []
        last_signal_end = {'LONG': -999, 'SHORT': -999}

        for i in range(55, len(df_1h) - MAX_TRADE_HOURS - 1):
            r1h  = df_1h.iloc[i]
            p1h  = df_1h.iloc[i-1]
            ts1h = df_1h.index[i]

            if any(c not in r1h.index or pd.isna(r1h[c]) for c in required_1h):
                continue

            # Cooldown: don't re-enter same direction until previous trade closes
            if i <= last_signal_end['LONG'] or i <= last_signal_end['SHORT']:
                continue

            # Align 4H and 15M to this 1H bar
            c4h  = df_4h[df_4h.index   <= ts1h]
            c15m = df_15m[df_15m.index <= ts1h]
            if len(c4h) < 3 or len(c15m) < 5:
                continue

            r4h  = c4h.iloc[-1]
            p4h  = c4h.iloc[-2]
            r15m = c15m.iloc[-1]

            # v2: Weekly trend check
            weekly_trend = 'UNKNOWN'
            if df_1w is not None:
                c1w = df_1w[df_1w.index <= ts1h]
                if len(c1w) >= 2 and 'ema_21' in c1w.columns:
                    r1w = c1w.iloc[-1]
                    w_close = float(r1w['close']) if not pd.isna(r1w['close']) else 0
                    w_ema21 = float(r1w['ema_21']) if not pd.isna(r1w['ema_21']) else 0
                    if w_close > 0 and w_ema21 > 0:
                        weekly_trend = 'UP' if w_close > w_ema21 else 'DOWN'

            # Volume ratio
            vol_avg   = df_1h['volume'].iloc[max(0,i-20):i].mean()
            vol_ratio = float(r1h['volume']) / vol_avg if vol_avg > 0 else 1.0

            # Score
            ls, ss, lr, sr = score_reversal(r4h, p4h, r1h, p1h, r15m, vol_ratio)

            signal = None
            if ls >= MIN_SCORE and ls > ss:
                signal = 'LONG';  score = ls; reasons = lr
            elif ss >= MIN_SCORE and ss > ls:
                signal = 'SHORT'; score = ss; reasons = sr
            if signal is None:
                continue

            # BTC regime filter
            if REGIME_MODE == 'HARD' and btc_regime_series is not None:
                rc = btc_regime_series[btc_regime_series.index <= ts1h]
                if len(rc) > 0:
                    btc_reg = rc.iloc[-1]
                    if signal == 'LONG'  and btc_reg == 'BEAR':
                        self.regime_blocked += 1; continue
                    if signal == 'SHORT' and btc_reg == 'BULL':
                        self.regime_blocked += 1; continue
                    btc_regime_val = btc_reg
                else:
                    btc_regime_val = 'N/A'
            else:
                btc_regime_val = 'N/A'

            # ── v2 HARD GATES ──────────────────────────────────────
            r4h_rsi  = float(r4h['rsi'])   if 'rsi'   in r4h.index and not pd.isna(r4h['rsi'])   else 50
            r4h_e9   = float(r4h['ema_9']) if 'ema_9' in r4h.index and not pd.isna(r4h['ema_9']) else 0
            r4h_e21  = float(r4h['ema_21'])if 'ema_21'in r4h.index and not pd.isna(r4h['ema_21'])else 0
            hist_cur = float(r1h['macd_hist']); hist_prv = float(p1h['macd_hist'])

            # 15M candle trigger check
            c15m_recent = df_15m[df_15m.index <= ts1h].iloc[-3:]
            has_15m_bull = any(
                float(r.get('bull_engulf',0))==1 or float(r.get('hammer',0))==1
                for _, r in c15m_recent.iterrows()
            )
            has_15m_bear = any(
                float(r.get('bear_engulf',0))==1 or float(r.get('shooting_star',0))==1
                for _, r in c15m_recent.iterrows()
            )

            if signal == 'LONG':
                # v2 KEY GATE: weekly trend must be UP (dip in bull market only)
                if weekly_trend == 'DOWN':             continue  # no longs in weekly downtrend
                if weekly_trend == 'UNKNOWN':          continue  # no data = skip
                # 4H short-term pullback required
                if r4h_e9  >= r4h_e21:                continue  # 4H not pulling back
                # 4H RSI — deeper oversold
                if r4h_rsi >= RSI_OVERSOLD_4H:         continue  # not oversold enough
                # 1H MACD must be turning up
                if hist_cur <= hist_prv:               continue  # not turning up
                # 1H candle trigger (required)
                has_1h_trigger = (
                    float(r1h['bull_engulf'])==1 or
                    float(r1h['hammer'])==1 or
                    float(r1h['bull_div'])==1
                )
                if not has_1h_trigger:                 continue
                # 15M confirmation (required — double trigger)
                if not has_15m_bull:                   continue

            else:  # SHORT
                # v2 KEY GATE: weekly trend must be DOWN (bounce in bear market only)
                if weekly_trend == 'UP':               continue  # no shorts in weekly uptrend
                if weekly_trend == 'UNKNOWN':          continue
                # 4H short-term bounce required
                if r4h_e9  <= r4h_e21:                continue  # 4H not bouncing
                # 4H RSI — deeper overbought
                if r4h_rsi <= RSI_OVERBOUGHT_4H:       continue  # not overbought enough
                # 1H MACD must be turning down
                if hist_cur >= hist_prv:               continue  # not turning down
                # 1H candle trigger (required)
                has_1h_trigger = (
                    float(r1h['bear_engulf'])==1 or
                    float(r1h['shooting_star'])==1 or
                    float(r1h['bear_div'])==1
                )
                if not has_1h_trigger:                 continue
                # 15M confirmation (required — double trigger)
                if not has_15m_bear:                   continue

            # Build trade levels
            entry    = float(r1h['close'])
            atr_15m  = float(r15m['atr']) if 'atr' in r15m.index and not pd.isna(r15m['atr']) else float(r1h['atr']) * 0.35
            if pd.isna(atr_15m) or atr_15m <= 0:
                atr_15m = float(r1h['atr']) * 0.35

            risk = atr_15m * ATR_SL_MULT
            if risk <= 0 or entry <= 0:
                continue

            if signal == 'LONG':
                sl  = entry - risk
                tp1 = entry + risk * TP1_RR
                tp2 = entry + risk * TP2_RR
            else:
                sl  = entry + risk
                tp1 = entry - risk * TP1_RR
                tp2 = entry - risk * TP2_RR

            risk_pct = risk / entry * 100

            outcome, pnl, duration = simulate_trade(i, df_1h, signal, entry, sl, tp1, tp2)
            win = pnl > 0

            score_pct = round(score / 100 * 100, 1)
            quality   = 'PREMIUM' if score_pct >= 80 else 'GOOD'

            trade = {
                'symbol':     symbol.replace('/USDT:USDT',''),
                'timestamp':  str(ts1h),
                'direction':  signal,
                'quality':    quality,
                'score':      round(score, 1),
                'score_pct':  score_pct,
                'btc_regime': btc_regime_val,
                'entry':      round(entry, 6),
                'sl':         round(sl, 6),
                'tp1':        round(tp1, 6),
                'tp2':        round(tp2, 6),
                'risk_pct':   round(risk_pct, 3),
                'tp1_pct':    round(abs(tp1-entry)/entry*100, 2),
                'tp2_pct':    round(abs(tp2-entry)/entry*100, 2),
                'vol_ratio':  round(vol_ratio, 2),
                'outcome':    outcome,
                'win':        win,
                'pnl_pct':    pnl,
                'duration_h': duration,
                'reasons':    list(reasons.keys()),
            }
            pair_trades.append(trade)
            last_signal_end[signal] = i + duration + 1

            for name in reasons:
                self.ind_stats[name]['triggered'] += 1
                if win: self.ind_stats[name]['wins'] += 1
                else:   self.ind_stats[name]['losses'] += 1

        if pair_trades:
            print(f"       → {len(pair_trades)} signals")
        return pair_trades

    def print_results(self):
        if not self.trades:
            print("❌ No trades found. Try lowering MIN_SCORE or relaxing gates.")
            return

        df = pd.DataFrame(self.trades)
        total  = len(df)
        wins   = df['win'].sum()
        losses = total - wins
        wr     = wins / total * 100
        apnl   = df['pnl_pct'].mean()
        aw     = df[df['win']]['pnl_pct'].mean()    if wins   > 0 else 0
        al     = df[~df['win']]['pnl_pct'].mean()   if losses > 0 else 0
        pf     = abs(aw*wins / (al*losses)) if losses > 0 and al != 0 else 99
        spd    = total / LOOKBACK_DAYS
        spm    = total / (LOOKBACK_DAYS / 30)
        mr     = apnl * spm
        cumul  = (1 + df['pnl_pct']/100).cumprod()
        mdd    = ((cumul - cumul.cummax()) / cumul.cummax() * 100).min()

        n_sl       = (df['outcome'] == 'SL').sum()
        n_tp1_only = (df['outcome'].str.contains('TP1') & ~df['outcome'].str.contains('TP2')).sum()
        n_tp2      = (df['outcome'] == 'TP2').sum()
        n_timeout  = (df['outcome'].str.contains('TIMEOUT')).sum()

        longs  = df[df['direction']=='LONG']
        shorts = df[df['direction']=='SHORT']
        prem   = df[df['quality']=='PREMIUM']
        good   = df[df['quality']=='GOOD']

        print("\n" + "╔"+"═"*56+"╗")
        print("║" + "  📊 REVERSAL RIDER — BACKTEST v1 RESULTS".center(56) + "║")
        print("╚"+"═"*56+"╝")
        print(f"\n  Settings: score≥{MIN_SCORE} | RSI_L<{RSI_OVERSOLD_4H} | RSI_S>{RSI_OVERBOUGHT_4H}")
        print(f"  SL=15M ATR×{ATR_SL_MULT} | TP1={TP1_RR}R(60%) | TP2={TP2_RR}R(40%) | Timeout={MAX_TRADE_HOURS}H")
        print(f"  Pairs: {df['symbol'].nunique()} | Lookback: {LOOKBACK_DAYS}d\n")
        print(f"  {'Signals':22s}: {total}  ({spd:.1f}/day  |  {spm:.0f}/month)")
        print(f"  {'Win Rate':22s}: {wr:.1f}%")
        print(f"  {'Profit Factor':22s}: {pf:.2f}")
        print(f"  {'Avg PnL/trade':22s}: {apnl:+.3f}%")
        print(f"  {'Avg Win':22s}: {aw:+.3f}%")
        print(f"  {'Avg Loss':22s}: {al:+.3f}%")
        print(f"  {'Monthly est.':22s}: {mr:+.1f}%")
        print(f"  {'Max Drawdown':22s}: {mdd:.2f}%")
        print(f"\n  ── Outcome Breakdown ──")
        print(f"  TP2 (full ride)  : {n_tp2}  ({n_tp2/total*100:.1f}%)")
        print(f"  TP1+timeout      : {n_tp1_only}  ({n_tp1_only/total*100:.1f}%)")
        print(f"  SL               : {n_sl}  ({n_sl/total*100:.1f}%)")
        print(f"  Timeout (no TP)  : {n_timeout - n_tp1_only}  ({max(0,(n_timeout-n_tp1_only)/total*100):.1f}%)")
        print(f"  Regime blocked   : {self.regime_blocked}")

        print(f"\n  ── By Direction ──")
        for label, sub in [('LONG', longs), ('SHORT', shorts)]:
            if len(sub) == 0: continue
            print(f"  {label:6s} | n={len(sub):4d} ({len(sub)/LOOKBACK_DAYS:.1f}/day) | "
                  f"WR={sub['win'].mean()*100:.1f}% | Avg={sub['pnl_pct'].mean():+.3f}%")

        print(f"\n  ── By Quality ──")
        for label, sub in [('PREMIUM', prem), ('GOOD', good)]:
            if len(sub) == 0: continue
            print(f"  {label:8s} | n={len(sub):4d} | "
                  f"WR={sub['win'].mean()*100:.1f}% | Avg={sub['pnl_pct'].mean():+.3f}%")

        print(f"\n  ── By BTC Regime ──")
        for reg in ['BULL','BEAR']:
            sub = df[df['btc_regime']==reg]
            if len(sub) == 0: continue
            print(f"  {reg:5s} | n={len(sub):4d} | "
                  f"WR={sub['win'].mean()*100:.1f}% | Avg={sub['pnl_pct'].mean():+.3f}%")

        print(f"\n  ── Score Band ──")
        print(f"  {'Band':12s} {'n':>5} {'WR%':>7} {'Avg%':>8} {'SL%':>7}")
        for lo, hi in [(60,65),(65,70),(70,75),(75,80),(80,100)]:
            sub = df[(df['score_pct']>=lo) & (df['score_pct']<hi)]
            if len(sub) < 3: continue
            print(f"  {lo}-{hi}%      {len(sub):>5d} "
                  f"{sub['win'].mean()*100:>6.1f}% "
                  f"{sub['pnl_pct'].mean():>+7.3f}% "
                  f"{(sub['outcome']=='SL').mean()*100:>6.1f}%")

        print(f"\n  ── TP2 rate by direction (full ride capture) ──")
        for label, sub in [('LONG', longs), ('SHORT', shorts)]:
            if len(sub) == 0: continue
            tp2_rate = (sub['outcome'] == 'TP2').mean() * 100
            tp2_sub = sub[sub['outcome']=='TP2']
            if len(tp2_sub) > 0:
                print(f"  {label}: TP2 rate = {tp2_rate:.1f}%  avg when TP2 = {tp2_sub['pnl_pct'].mean():+.3f}%")
            else:
                print(f"  {label}: TP2 rate = {tp2_rate:.1f}%")

        print(f"\n  ── Top Indicators ──")
        ind_rows = []
        for name, s in self.ind_stats.items():
            t = s['triggered']
            if t < 5: continue
            ind_rows.append({'name': name, 'n': t,
                             'wr': s['wins']/t*100,
                             'bar': '█' * int(s['wins']/t*10)})
        ind_rows.sort(key=lambda x: x['wr'], reverse=True)
        print(f"  {'Indicator':<35} {'WR':>6}  {'n':>5}")
        for r in ind_rows[:15]:
            print(f"  {r['name']:<35} {r['wr']:>5.1f}%  {r['n']:>5}  {r['bar']}")
        print(f"\n  BOTTOM (consider tightening):")
        for r in ind_rows[-6:]:
            print(f"  {r['name']:<35} {r['wr']:>5.1f}%  {r['n']:>5}")

        print(f"\n  ── Top 15 Symbols ──")
        sym = df.groupby('symbol').agg(
            n=('pnl_pct','count'),
            wr=('win', lambda x: x.mean()*100),
            avg=('pnl_pct','mean'),
        ).query('n >= 3').sort_values('wr', ascending=False).head(15)
        print(f"  {'Symbol':<14} {'n':>4} {'WR%':>7} {'Avg%':>8}")
        for s, row in sym.iterrows():
            print(f"  {s:<14} {int(row['n']):>4}  {row['wr']:>6.1f}%  {row['avg']:>+7.3f}%")

        print(f"\n  ── DEPLOY CHECKLIST ──")
        print(f"  MIN_SCORE       = {MIN_SCORE}")
        print(f"  RSI_OVERSOLD_4H = {RSI_OVERSOLD_4H}   (LONG gate)")
        print(f"  RSI_OVRBGHT_4H  = {RSI_OVERBOUGHT_4H}   (SHORT gate)")
        print(f"  ATR_SL_MULT     = {ATR_SL_MULT} (15M ATR)")
        print(f"  TP1_RR={TP1_RR} (60%) | TP2_RR={TP2_RR} (40%)")
        print(f"  TIMEOUT         = {MAX_TRADE_HOURS}H\n")
        print(f"  Expected live: {spd:.1f}/day | {wr:.1f}% WR | {apnl:+.3f}%/trade")
        print(f"  If live WR < {max(40, wr-15):.0f}% after 30 trades → raise MIN_SCORE to {MIN_SCORE+5}")

        # Save
        print(f"\n  💾 Saving {OUTPUT_FILE}...")
        try:
            with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All Trades', index=False)
                summary = pd.DataFrame({
                    'Metric': ['Signals','Per Day','Per Month','WR%','PF','Avg PnL%',
                               'Avg Win%','Avg Loss%','Monthly Est%','Max DD%',
                               'TP2 Rate%','SL Rate%','Timeout Rate%'],
                    'Value':  [total, round(spd,2), round(spm,1), round(wr,1),
                               round(pf,2), round(apnl,3), round(aw,3), round(al,3),
                               round(mr,1), round(mdd,2),
                               round(n_tp2/total*100,1), round(n_sl/total*100,1),
                               round((n_timeout-n_tp1_only)/total*100,1)]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
            print(f"  ✅ Saved!")
        except Exception as e:
            print(f"  ⚠️  Excel save failed ({e}) — saving CSV")
            df.to_csv(OUTPUT_FILE.replace('.xlsx','.csv'), index=False)
            print(f"  ✅ CSV saved!")

    async def run(self):
        print(f"""
╔══════════════════════════════════════════════════════════╗
║         REVERSAL RIDER — BACKTEST v1.0                  ║
║  {LOOKBACK_DAYS}d | Top {TOP_N_PAIRS} pairs | Score≥{MIN_SCORE}                       ║
║  LONG: 4H downtrend+oversold+BB | SHORT: 4H uptrend+OB ║
║  SL=15M ATR×{ATR_SL_MULT} | TP1={TP1_RR}R(60%) | TP2={TP2_RR}R(40%) | {MAX_TRADE_HOURS}H max  ║
╚══════════════════════════════════════════════════════════╝
""")
        pairs  = await self.get_pairs()
        regime = await self.load_btc_regime()

        for i, symbol in enumerate(pairs):
            try:
                trades = await self.backtest_pair(symbol, regime)
                self.trades.extend(trades)
            except Exception as e:
                pass
            if (i+1) % 50 == 0:
                print(f"  ── {i+1}/{len(pairs)} pairs | {len(self.trades)} signals ──")

        await self.exchange.close()
        print(f"\n✅ Done — {len(self.trades)} signals from {len(pairs)} pairs\n")
        self.print_results()


async def main():
    bt = ReversalBacktesterV2()
    await bt.run()

if __name__ == '__main__':
    asyncio.run(main())
