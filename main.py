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


# ─────────────────────────────────────────────
#  ORDER BLOCK UTILITIES
# ─────────────────────────────────────────────

def detect_order_blocks(df, lookback=50, swing_strength=3):
    """
    Detect Bullish and Bearish Order Blocks using SMC logic.

    Bullish OB  = last BEARISH candle before a strong bullish impulse that
                  broke a recent swing high.
    Bearish OB  = last BULLISH candle before a strong bearish impulse that
                  broke a recent swing low.

    Returns a list of dicts:
        {'type': 'bullish'|'bearish',
         'top': float, 'bottom': float,
         'index': int,           # bar index in df
         'mitigated': bool}
    """
    obs = []

    highs = df['high'].values
    lows  = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    n = len(df)

    if n < lookback + swing_strength * 2:
        return obs

    start = max(swing_strength, n - lookback)

    for i in range(start, n - swing_strength):
        # ── BULLISH OB ──────────────────────────────────────────────
        # Look for a bearish candle followed by a bullish impulse that
        # clears the swing high from before the bearish candle.
        if closes[i] < opens[i]:                          # bearish candle
            # Impulse: next `swing_strength` candles are mostly bullish
            impulse_high = max(highs[i+1 : i+1+swing_strength])
            prior_swing_high = max(highs[max(0, i-swing_strength) : i])

            if impulse_high > prior_swing_high:            # broke swing high → valid OB
                ob = {
                    'type': 'bullish',
                    'top': max(opens[i], closes[i]),
                    'bottom': min(opens[i], closes[i]),
                    'index': i,
                    'mitigated': False
                }
                obs.append(ob)

        # ── BEARISH OB ──────────────────────────────────────────────
        elif closes[i] > opens[i]:                        # bullish candle
            impulse_low  = min(lows[i+1 : i+1+swing_strength])
            prior_swing_low = min(lows[max(0, i-swing_strength) : i])

            if impulse_low < prior_swing_low:              # broke swing low → valid OB
                ob = {
                    'type': 'bearish',
                    'top': max(opens[i], closes[i]),
                    'bottom': min(opens[i], closes[i]),
                    'index': i,
                    'mitigated': False
                }
                obs.append(ob)

    # ── Mark mitigated OBs (price already traded through them) ──────
    current_price = closes[-1]
    for ob in obs:
        if ob['type'] == 'bullish' and current_price < ob['bottom']:
            ob['mitigated'] = True   # price went below, OB invalidated
        elif ob['type'] == 'bearish' and current_price > ob['top']:
            ob['mitigated'] = True

    return obs


def price_at_order_block(current_price, obs, tolerance=0.003):
    """
    Check if current price is inside (or very near) an unmitigated OB.
    tolerance = 0.3% cushion above/below the OB zone.

    Returns:
        ('bullish', ob_dict)  — price is at a bullish OB  → long bias
        ('bearish', ob_dict)  — price is at a bearish OB  → short bias
        (None, None)
    """
    best = None
    best_type = None

    for ob in obs:
        if ob['mitigated']:
            continue

        cushion = current_price * tolerance
        in_zone = (current_price >= ob['bottom'] - cushion and
                   current_price <= ob['top']    + cushion)

        if in_zone:
            # Prefer the most recent (highest index) OB
            if best is None or ob['index'] > best['index']:
                best = ob
                best_type = ob['type']

    return best_type, best


# ─────────────────────────────────────────────
#  SMART MONEY CONCEPTS MODULE (LuxAlgo port)
# ─────────────────────────────────────────────

def detect_swing_points(df, swing_length=5):
    """
    Detect swing highs and lows using pivot logic (same as LuxAlgo leg() function).
    Returns arrays of (index, price) for swing highs and lows.
    """
    highs  = df['high'].values
    lows   = df['low'].values
    n      = len(df)
    s      = swing_length

    swing_highs = []   # list of (bar_index, price)
    swing_lows  = []

    for i in range(s, n - s):
        # Swing high: highest in window centered on i
        if highs[i] == max(highs[i-s : i+s+1]):
            swing_highs.append((i, highs[i]))
        # Swing low: lowest in window centered on i
        if lows[i] == min(lows[i-s : i+s+1]):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def detect_bos_choch(df, swing_length=5):
    """
    Detect Break of Structure (BOS) and Change of Character (CHoCH).

    BOS   = price breaks a swing high/low in the SAME direction as the trend
            → trend continuation confirmation
    CHoCH = price breaks a swing high/low AGAINST the current trend
            → potential trend reversal

    Returns dict:
        last_bos_bull   : last bullish BOS bar index (or None)
        last_bos_bear   : last bearish BOS bar index
        last_choch_bull : last bullish CHoCH bar index
        last_choch_bear : last bearish CHoCH bar index
        swing_trend     : 'bullish' | 'bearish' | 'neutral'
        recent_bull_structure : bool — bullish BOS or CHoCH in last 10 bars
        recent_bear_structure : bool — bearish BOS or CHoCH in last 10 bars
    """
    closes = df['close'].values
    n      = len(df)

    swing_highs, swing_lows = detect_swing_points(df, swing_length)

    if not swing_highs or not swing_lows:
        return {
            'last_bos_bull': None, 'last_bos_bear': None,
            'last_choch_bull': None, 'last_choch_bear': None,
            'swing_trend': 'neutral',
            'recent_bull_structure': False,
            'recent_bear_structure': False,
        }

    # Track trend using HH/HL or LH/LL logic
    swing_trend = 'neutral'
    last_bos_bull = last_bos_bear = None
    last_choch_bull = last_choch_bear = None

    # Determine trend from last two swing highs and lows
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
            swing_trend = 'bullish'   # HH + HL
        elif swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
            swing_trend = 'bearish'   # LH + LL

    # Check last swing high/low for BOS or CHoCH
    # Bullish break = close crosses ABOVE a swing high
    last_sh_idx, last_sh_price = swing_highs[-1]
    last_sl_idx, last_sl_price = swing_lows[-1]

    # Scan recent bars for crossovers
    lookback = min(20, n - 1)
    for i in range(n - lookback, n):
        # Bullish structure break (close crosses above last swing high)
        if closes[i] > last_sh_price and (i == 0 or closes[i-1] <= last_sh_price):
            if swing_trend == 'bearish':
                last_choch_bull = i    # reversal
            else:
                last_bos_bull   = i    # continuation

        # Bearish structure break (close crosses below last swing low)
        if closes[i] < last_sl_price and (i == 0 or closes[i-1] >= last_sl_price):
            if swing_trend == 'bullish':
                last_choch_bear = i    # reversal
            else:
                last_bos_bear   = i    # continuation

    recent_cutoff = n - 10
    recent_bull_structure = (
        (last_bos_bull   is not None and last_bos_bull   >= recent_cutoff) or
        (last_choch_bull is not None and last_choch_bull >= recent_cutoff)
    )
    recent_bear_structure = (
        (last_bos_bear   is not None and last_bos_bear   >= recent_cutoff) or
        (last_choch_bear is not None and last_choch_bear >= recent_cutoff)
    )

    return {
        'last_bos_bull':        last_bos_bull,
        'last_bos_bear':        last_bos_bear,
        'last_choch_bull':      last_choch_bull,
        'last_choch_bear':      last_choch_bear,
        'swing_trend':          swing_trend,
        'recent_bull_structure': recent_bull_structure,
        'recent_bear_structure': recent_bear_structure,
    }


def detect_fair_value_gaps(df, min_gap_pct=0.001):
    """
    Detect Fair Value Gaps (FVG) — 3-candle imbalance pattern from LuxAlgo SMC.

    Bullish FVG : candle[2].low > candle[0].high  (gap up — price left a hole)
    Bearish FVG : candle[2].high < candle[0].low  (gap down)

    Only unmitigated FVGs (price hasn't filled them yet) are returned.

    Returns:
        bull_fvgs : list of {'top': float, 'bottom': float, 'index': int}
        bear_fvgs : list of {'top': float, 'bottom': float, 'index': int}
        nearest_bull_fvg : closest unmitigated bullish FVG to current price
        nearest_bear_fvg : closest unmitigated bearish FVG to current price
    """
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    n      = len(df)

    bull_fvgs = []
    bear_fvgs = []

    for i in range(2, n):
        # Bullish FVG: gap between candle[i-2] high and candle[i] low
        if lows[i] > highs[i-2]:
            gap_size = (lows[i] - highs[i-2]) / highs[i-2]
            if gap_size >= min_gap_pct:
                bull_fvgs.append({
                    'top':    lows[i],
                    'bottom': highs[i-2],
                    'mid':    (lows[i] + highs[i-2]) / 2,
                    'index':  i,
                })

        # Bearish FVG: gap between candle[i] high and candle[i-2] low
        if highs[i] < lows[i-2]:
            gap_size = (lows[i-2] - highs[i]) / lows[i-2]
            if gap_size >= min_gap_pct:
                bear_fvgs.append({
                    'top':    lows[i-2],
                    'bottom': highs[i],
                    'mid':    (lows[i-2] + highs[i]) / 2,
                    'index':  i,
                })

    # Filter mitigated FVGs (price has traded into them)
    current_price = closes[-1]

    active_bull = [f for f in bull_fvgs if current_price > f['bottom']]  # not fully filled
    active_bear = [f for f in bear_fvgs if current_price < f['top']]

    # Nearest to current price
    nearest_bull = min(active_bull, key=lambda f: abs(current_price - f['mid']), default=None) if active_bull else None
    nearest_bear = min(active_bear, key=lambda f: abs(current_price - f['mid']), default=None) if active_bear else None

    return {
        'bull_fvgs':      active_bull[-5:],   # keep last 5
        'bear_fvgs':      active_bear[-5:],
        'nearest_bull':   nearest_bull,
        'nearest_bear':   nearest_bear,
    }


def detect_premium_discount(df):
    """
    Premium / Discount / Equilibrium zones from LuxAlgo SMC.

    Uses trailing swing high and low (same as LuxAlgo drawPremiumDiscountZones):
      Premium    = top 5% of range  → SELL zone (short bias)
      Equilibrium= middle 5% band   → neutral / reversal watch
      Discount   = bottom 5%        → BUY zone (long bias)

    Returns:
        zone           : 'premium' | 'discount' | 'equilibrium' | 'neutral'
        range_pct      : how far price is through the range (0=bottom, 100=top)
        swing_high     : trailing swing high
        swing_low      : trailing swing low
    """
    highs  = df['high'].values
    lows   = df['low'].values

    swing_high = highs.max()
    swing_low  = lows.min()
    rng        = swing_high - swing_low

    if rng == 0:
        return {'zone': 'neutral', 'range_pct': 50, 'swing_high': swing_high, 'swing_low': swing_low}

    current = df['close'].iloc[-1]
    range_pct = (current - swing_low) / rng * 100

    if range_pct >= 95:
        zone = 'premium'        # top 5% — short zone
    elif range_pct <= 5:
        zone = 'discount'       # bottom 5% — long zone
    elif 47.5 <= range_pct <= 52.5:
        zone = 'equilibrium'    # mid 5% — watch for reversal
    else:
        zone = 'neutral'

    return {
        'zone':       zone,
        'range_pct':  range_pct,
        'swing_high': swing_high,
        'swing_low':  swing_low,
    }


def detect_equal_highs_lows(df, length=3, threshold_atr_mult=0.1):
    """
    Equal Highs (EQH) and Equal Lows (EQL) — liquidity pools / stop hunt zones.

    Two swing points are 'equal' if their prices are within threshold ATR of each other.
    These mark where stop losses cluster → smart money hunts them.

    Returns:
        eqh : list of price levels with equal highs
        eql : list of price levels with equal lows
        nearest_eqh : closest EQH above current price (liquidity above)
        nearest_eql : closest EQL below current price (liquidity below)
    """
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    n      = len(df)

    # Simple ATR estimate
    atr = np.mean([abs(highs[i] - lows[i]) for i in range(max(0, n-14), n)])
    threshold = threshold_atr_mult * atr

    swing_highs, swing_lows = detect_swing_points(df, length)

    eqh = []
    for i in range(len(swing_highs) - 1):
        for j in range(i+1, len(swing_highs)):
            if abs(swing_highs[i][1] - swing_highs[j][1]) < threshold:
                eqh.append(round((swing_highs[i][1] + swing_highs[j][1]) / 2, 8))

    eql = []
    for i in range(len(swing_lows) - 1):
        for j in range(i+1, len(swing_lows)):
            if abs(swing_lows[i][1] - swing_lows[j][1]) < threshold:
                eql.append(round((swing_lows[i][1] + swing_lows[j][1]) / 2, 8))

    current = closes[-1]
    nearest_eqh = min([h for h in eqh if h > current], default=None) if eqh else None
    nearest_eql = max([l for l in eql if l < current], default=None) if eql else None

    return {
        'eqh':         sorted(set(eqh)),
        'eql':         sorted(set(eql)),
        'nearest_eqh': nearest_eqh,
        'nearest_eql': nearest_eql,
    }


# ─────────────────────────────────────────────
#  MAIN SCANNER
# ─────────────────────────────────────────────

class AdvancedDayTradingScanner:
    def __init__(self, telegram_token, telegram_chat_id, binance_api_key=None, binance_secret=None):
        self.telegram_token = telegram_token
        self.telegram_bot = Bot(token=telegram_token)
        self.chat_id = telegram_chat_id
        self.exchange = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.signal_history = deque(maxlen=200)
        self.active_trades = {}
        self.stats = {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'premium_signals': 0,
            'tp1_hits': 0,
            'tp2_hits': 0,
            'tp3_hits': 0,
            'ob_signals': 0,
            'last_scan_time': None,
            'pairs_scanned': 0,
            # ── Daily report counters (reset every 24h) ──
            'daily_signals': 0,
            'daily_wins': 0,       # any TP hit = win
            'daily_losses': 0,     # SL hit = loss
            'daily_be': 0,         # 24h timeout = breakeven
            'report_start': datetime.now(),
        }
        self.is_scanning = False
        self.is_tracking = False

    async def get_all_usdt_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = []
            for symbol in self.exchange.symbols:
                if symbol.endswith('/USDT:USDT') and 'PERP' not in symbol:
                    ticker = tickers.get(symbol)
                    if ticker and ticker.get('quoteVolume', 0) > 1000000:
                        pairs.append(symbol)
            sorted_pairs = sorted(pairs, key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
            logger.info(f"✅ Found {len(sorted_pairs)} high-quality pairs")
            return sorted_pairs
        except Exception as e:
            logger.error(f"Error fetching pairs: {e}")
            return []

    async def fetch_day_trading_data(self, symbol):
        timeframes = {'1h': 100, '4h': 100, '15m': 50}
        data = {}
        try:
            for tf, limit in timeframes.items():
                ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                data[tf] = df
                await asyncio.sleep(0.05)
            return data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def calculate_supertrend(self, df, period=10, multiplier=3):
        try:
            hl2 = (df['high'] + df['low']) / 2
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            supertrend = [0] * len(df)
            for i in range(1, len(df)):
                if df['close'].iloc[i] > upper_band.iloc[i-1]:
                    supertrend[i] = lower_band.iloc[i]
                elif df['close'].iloc[i] < lower_band.iloc[i-1]:
                    supertrend[i] = upper_band.iloc[i]
                else:
                    supertrend[i] = supertrend[i-1]
            return pd.Series(supertrend, index=df.index)
        except:
            return pd.Series([0] * len(df), index=df.index)

    def calculate_advanced_indicators(self, df):
        try:
            if len(df) < 30:
                return df

            df['ema_9']  = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=min(50, len(df)-1)).ema_indicator()
            df['supertrend'] = self.calculate_supertrend(df)

            psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
            df['psar'] = psar.psar()

            df['rsi']   = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_6'] = ta.momentum.RSIIndicator(df['close'], window=6).rsi()

            stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
            df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

            macd = ta.trend.MACD(df['close'])
            df['macd']        = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist']   = macd.macd_diff()

            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
            df['uo']  = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()

            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper']  = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower']  = bb.bollinger_lband()
            df['bb_width']  = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pband']  = bb.bollinger_pband()

            kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            df['kc_upper'] = kc.keltner_channel_hband()
            df['kc_lower'] = kc.keltner_channel_lband()

            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

            dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
            df['dc_upper'] = dc.donchian_channel_hband()
            df['dc_lower'] = dc.donchian_channel_lband()

            df['volume_sma']   = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            df['obv']     = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            df['mfi']     = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            df['ad']      = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
            df['cmf']     = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()

            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx']      = adx.adx()
            df['di_plus']  = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()

            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

            aroon = ta.trend.AroonIndicator(df['high'], df['low'])
            df['aroon_up']   = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            df['aroon_ind']  = df['aroon_up'] - df['aroon_down']

            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'] = df['vwap'].fillna(df['close'])

            df['bullish_candle'] = (df['close'] > df['open']).astype(int)
            df['bullish_engulfing'] = (
                (df['close'].shift(1) < df['open'].shift(1)) &
                (df['close'] > df['open']) &
                (df['open'] <= df['close'].shift(1)) &
                (df['close'] >= df['open'].shift(1))
            ).astype(int)
            df['bearish_engulfing'] = (
                (df['close'].shift(1) > df['open'].shift(1)) &
                (df['close'] < df['open']) &
                (df['open'] >= df['close'].shift(1)) &
                (df['close'] <= df['open'].shift(1))
            ).astype(int)

            df['bullish_divergence'] = (
                (df['low'] < df['low'].shift(1)) &
                (df['rsi'] > df['rsi'].shift(1))
            ).astype(int)
            df['bearish_divergence'] = (
                (df['high'] > df['high'].shift(1)) &
                (df['rsi'] < df['rsi'].shift(1))
            ).astype(int)

            return df
        except Exception as e:
            logger.error(f"Indicator error: {e}")
            return df

    def calculate_volume_profile(self, df, n_rows=25):
        """
        Volume Profile ported from LuxAlgo Money Flow Profile.

        For each price row (bucket) across the lookback window we calculate:
          - total_vol  : total volume traded at that price level
          - bull_vol   : buying volume (bar polarity: close > open)
          - bear_vol   : selling volume
          - sentiment  : bull_vol - bear_vol (positive = buyer dominated)

        From those we derive:
          poc_price    : Point of Control — price row with most volume (magnet)
          htn_levels   : High Traded Nodes (>53% of max) — consolidation / S&R
          ltn_levels   : Low Traded Nodes (<37% of max)  — supply/demand gaps
          poc_sentiment: is the POC buyer or seller dominated?
          current_node : which node type is current price sitting in?
          current_sent : sentiment at current price level (bull or bear)

        Returns a dict with all of the above.
        """
        try:
            if len(df) < 10:
                return None

            highs   = df['high'].values
            lows    = df['low'].values
            closes  = df['close'].values
            opens   = df['open'].values
            volumes = df['volume'].values

            p_low  = lows.min()
            p_high = highs.max()
            if p_high <= p_low:
                return None

            step = (p_high - p_low) / n_rows

            total_vol = np.zeros(n_rows)
            bull_vol  = np.zeros(n_rows)   # close > open (buying bar)
            bear_vol  = np.zeros(n_rows)   # close <= open (selling bar)

            for i in range(len(df)):
                v    = volumes[i]
                is_bull = closes[i] > opens[i]

                for row in range(n_rows):
                    row_low  = p_low + row * step
                    row_high = row_low + step

                    # Skip if bar doesn't touch this row
                    if highs[i] < row_low or lows[i] >= row_high:
                        continue

                    # Proportion of bar that overlaps with this row (LuxAlgo vPOR logic)
                    bar_range = highs[i] - lows[i]
                    if bar_range == 0:
                        por = 1.0
                    elif lows[i] >= row_low and highs[i] > row_high:
                        por = (row_high - lows[i]) / bar_range
                    elif highs[i] <= row_high and lows[i] < row_low:
                        por = (highs[i] - row_low) / bar_range
                    elif lows[i] >= row_low and highs[i] <= row_high:
                        por = 1.0
                    else:
                        por = step / bar_range

                    allocated = v * por
                    total_vol[row] += allocated
                    if is_bull:
                        bull_vol[row] += allocated
                    else:
                        bear_vol[row] += allocated

            max_vol = total_vol.max()
            if max_vol == 0:
                return None

            poc_row   = int(np.argmax(total_vol))
            poc_price = p_low + (poc_row + 0.5) * step

            # Classify rows exactly like LuxAlgo thresholds (53% / 37%)
            htn_levels = []   # High Traded Nodes
            ltn_levels = []   # Low Traded Nodes
            for row in range(n_rows):
                ratio = total_vol[row] / max_vol
                mid   = p_low + (row + 0.5) * step
                if ratio >= 0.53:
                    htn_levels.append(mid)
                elif ratio <= 0.37:
                    ltn_levels.append(mid)

            # POC sentiment
            poc_sentiment = 'bullish' if bull_vol[poc_row] >= bear_vol[poc_row] else 'bearish'

            # Current price context
            current_price = closes[-1]
            current_row   = min(int((current_price - p_low) / step), n_rows - 1)
            current_row   = max(current_row, 0)

            cur_ratio = total_vol[current_row] / max_vol
            if cur_ratio >= 0.53:
                current_node = 'high'       # consolidation — expect mean reversion
            elif cur_ratio <= 0.37:
                current_node = 'low'        # thin air — expect fast move
            else:
                current_node = 'average'

            current_sent = 'bullish' if bull_vol[current_row] >= bear_vol[current_row] else 'bearish'

            # Nearest HTN above and below (act as dynamic S/R)
            htn_above = [h for h in htn_levels if h > current_price]
            htn_below = [h for h in htn_levels if h < current_price]
            nearest_resistance = min(htn_above) if htn_above else None
            nearest_support    = max(htn_below) if htn_below else None

            return {
                'poc_price':          poc_price,
                'poc_sentiment':      poc_sentiment,
                'htn_levels':         htn_levels,
                'ltn_levels':         ltn_levels,
                'current_node':       current_node,
                'current_sentiment':  current_sent,
                'nearest_resistance': nearest_resistance,
                'nearest_support':    nearest_support,
                'total_vol':          total_vol,
                'bull_vol':           bull_vol,
                'bear_vol':           bear_vol,
                'p_low':              p_low,
                'step':               step,
                'n_rows':             n_rows,
            }
        except Exception as e:
            logger.error(f"Volume profile error: {e}")
            return None

    def detect_volume_spike(self, df):
        """Legacy spike check — kept for backward compat, used internally."""
        if len(df) < 20:
            return False, 1.0
        recent = df['volume'].iloc[-1]
        avg = df['volume'].iloc[-20:].mean()
        if avg == 0 or pd.isna(avg):
            return False, 1.0
        ratio = recent / avg
        return recent > avg * 2.5, ratio

    def analyze_volume_direction(self, df, lookback=6):
        """
        Determine if volume is genuinely backing a LONG or SHORT move.

        Checks 4 things:
          1. Buying vs selling volume split (candle body direction x volume)
          2. Whether recent volume is above average (active market)
          3. Volume trend — is it rising or fading into the move?
          4. Large candle bodies with high volume (conviction candles)

        Returns:
            long_vol_ok  : bool  — volume confirms a long
            short_vol_ok : bool  — volume confirms a short
            vol_ratio    : float — recent vol vs 20-bar avg
            buy_pct      : float — pct of recent volume that is buying
        """
        if len(df) < max(lookback, 20):
            return False, False, 1.0, 50.0

        closes  = df['close'].values
        opens   = df['open'].values
        volumes = df['volume'].values
        highs   = df['high'].values
        lows    = df['low'].values

        avg_vol = volumes[-20:].mean()
        if avg_vol == 0 or np.isnan(avg_vol):
            return False, False, 1.0, 50.0

        recent_vol = volumes[-lookback:]
        vol_ratio  = recent_vol.mean() / avg_vol   # >1 = above-average activity

        # 1. Buy / Sell volume split using candle wick analysis
        buy_vol  = 0.0
        sell_vol = 0.0
        for i in range(-lookback, 0):
            candle_range = highs[i] - lows[i]
            if candle_range == 0:
                continue
            buy_frac  = (closes[i] - lows[i])  / candle_range
            sell_frac = (highs[i]  - closes[i]) / candle_range
            buy_vol  += volumes[i] * buy_frac
            sell_vol += volumes[i] * sell_frac

        total_vol = buy_vol + sell_vol
        buy_pct   = (buy_vol / total_vol * 100) if total_vol > 0 else 50.0

        # 2. Volume trend — rising into the move or fading?
        mid        = lookback // 2
        early      = volumes[-lookback : -mid].mean()
        late       = volumes[-mid:].mean()
        vol_rising = late > early * 1.1    # 10% increase = rising
        vol_fading = late < early * 0.85   # 15% drop = fading

        # 3. Conviction candles: big body + above-avg volume
        long_conviction  = 0
        short_conviction = 0
        for i in range(-lookback, 0):
            body         = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i] if highs[i] != lows[i] else 1
            body_pct     = body / candle_range
            if volumes[i] > avg_vol and body_pct > 0.5:
                if closes[i] > opens[i]:
                    long_conviction += 1
                else:
                    short_conviction += 1

        # Decision gates
        long_vol_ok = (
            buy_pct > 55 and
            vol_ratio > 0.8 and
            (vol_rising or long_conviction >= 2) and
            not (buy_pct < 45 and vol_fading)
        )

        short_vol_ok = (
            buy_pct < 45 and
            vol_ratio > 0.8 and
            (vol_rising or short_conviction >= 2) and
            not (buy_pct > 55 and vol_fading)
        )

        return long_vol_ok, short_vol_ok, vol_ratio, buy_pct

    def detect_signal(self, data, symbol):
        try:
            if not data or '1h' not in data:
                return None

            for tf in data:
                data[tf] = self.calculate_advanced_indicators(data[tf])

            df_1h  = data['1h']
            df_4h  = data['4h']
            df_15m = data['15m']

            if len(df_1h) < 50:
                return None

            latest_1h  = df_1h.iloc[-1]
            prev_1h    = df_1h.iloc[-2]
            latest_4h  = df_4h.iloc[-1]
            latest_15m = df_15m.iloc[-1]

            required_cols = ['ema_9', 'ema_21', 'rsi', 'macd', 'vwap', 'bb_pband']
            for col in required_cols:
                if col not in latest_1h.index or pd.isna(latest_1h[col]):
                    return None

            volume_spike, vol_ratio = self.detect_volume_spike(df_1h)
            long_vol_ok, short_vol_ok, vol_ratio, buy_pct = self.analyze_volume_direction(df_1h, lookback=6)

            # ── VOLUME PROFILE (LuxAlgo logic) ───────────────────────
            vp = self.calculate_volume_profile(df_1h, n_rows=25)

            # ── SMC: BOS / CHoCH / FVG / Premium-Discount / EQH-EQL ─
            smc_struct = detect_bos_choch(df_1h,  swing_length=5)
            smc_fvg    = detect_fair_value_gaps(df_1h, min_gap_pct=0.001)
            smc_pd     = detect_premium_discount(df_1h)
            smc_eq     = detect_equal_highs_lows(df_1h, length=3)

            # ── ORDER BLOCK DETECTION ────────────────────────────────
            # Detect on 1H (primary) and 4H (higher timeframe confirmation)
            obs_1h = detect_order_blocks(df_1h,  lookback=60, swing_strength=3)
            obs_4h = detect_order_blocks(df_4h,  lookback=60, swing_strength=3)

            current_price = latest_15m['close']

            ob_type_1h, ob_1h = price_at_order_block(current_price, obs_1h, tolerance=0.003)
            ob_type_4h, ob_4h = price_at_order_block(current_price, obs_4h, tolerance=0.004)

            # ─────────────────────────────────────────────────────────

            long_score  = 0
            short_score = 0
            max_score   = 56          # 46 base + 10 from SMC
            long_reasons  = []
            short_reasons = []

            # ── ORDER BLOCK SCORING (5 pts new) ─────────────────────
            ob_tag_long  = None
            ob_tag_short = None

            if ob_type_1h == 'bullish':
                long_score += 3
                ob_tag_long = f"🧱 1H Bullish OB [{ob_1h['bottom']:.4f}–{ob_1h['top']:.4f}]"
                long_reasons.append(ob_tag_long)

            elif ob_type_1h == 'bearish':
                short_score += 3
                ob_tag_short = f"🧱 1H Bearish OB [{ob_1h['bottom']:.4f}–{ob_1h['top']:.4f}]"
                short_reasons.append(ob_tag_short)

            if ob_type_4h == 'bullish':
                long_score += 2
                long_reasons.append(f"🏗️ 4H Bullish OB Confirmed")

            elif ob_type_4h == 'bearish':
                short_score += 2
                short_reasons.append(f"🏗️ 4H Bearish OB Confirmed")

            # ── VOLUME PROFILE SCORING (6 pts new) ───────────────────
            # Ported from LuxAlgo Money Flow Profile logic
            if vp:
                cp = latest_15m['close']

                # 1. Current price in a Low Traded Node (thin air) → fast move likely
                if vp['current_node'] == 'low':
                    if vp['current_sentiment'] == 'bullish':
                        long_score += 2
                        long_reasons.append('🔵 VP: Low Node (bullish) — fast up move likely')
                    else:
                        short_score += 2
                        short_reasons.append('🔵 VP: Low Node (bearish) — fast down move likely')

                # 2. POC sentiment aligns with signal direction (+2)
                if vp['poc_sentiment'] == 'bullish':
                    long_score += 2
                    long_reasons.append(f"📍 VP POC Bullish ({vp['poc_price']:.4f})")
                else:
                    short_score += 2
                    short_reasons.append(f"📍 VP POC Bearish ({vp['poc_price']:.4f})")

                # 3. Price near support (HTN below) for longs / resistance (HTN above) for shorts
                if vp['nearest_support'] and abs(cp - vp['nearest_support']) / cp < 0.015:
                    long_score += 1.5
                    long_reasons.append(f"🟨 VP: Near HTN Support ({vp['nearest_support']:.4f})")
                if vp['nearest_resistance'] and abs(cp - vp['nearest_resistance']) / cp < 0.015:
                    short_score += 1.5
                    short_reasons.append(f"🟨 VP: Near HTN Resistance ({vp['nearest_resistance']:.4f})")

                # 4. POC acts as magnet — if price is moving toward POC, add points
                if cp < vp['poc_price'] * 0.995:    # price below POC → likely to rally to it
                    long_score += 0.5
                    long_reasons.append(f"🧲 VP: POC Magnet Above")
                elif cp > vp['poc_price'] * 1.005:  # price above POC → likely to drop to it
                    short_score += 0.5
                    short_reasons.append(f"🧲 VP: POC Magnet Below")

            # ── SMC SCORING (max ~10 pts) ─────────────────────────────

            # 1. BOS / CHoCH structural confirmation (+3 pts)
            if smc_struct['recent_bull_structure']:
                long_score += 3
                tag = 'BOS' if smc_struct['last_bos_bull'] else 'CHoCH'
                long_reasons.append(f"⚡ Bullish {tag} — Structure Break")
            if smc_struct['recent_bear_structure']:
                short_score += 3
                tag = 'BOS' if smc_struct['last_bos_bear'] else 'CHoCH'
                short_reasons.append(f"⚡ Bearish {tag} — Structure Break")

            # Swing trend alignment bonus (+1)
            if smc_struct['swing_trend'] == 'bullish':
                long_score += 1
                long_reasons.append('📈 SMC: Bullish Swing Trend (HH+HL)')
            elif smc_struct['swing_trend'] == 'bearish':
                short_score += 1
                short_reasons.append('📉 SMC: Bearish Swing Trend (LH+LL)')

            # 2. Fair Value Gap — price near unmitigated FVG (+2 pts)
            cp = latest_15m['close']
            if smc_fvg['nearest_bull'] and abs(cp - smc_fvg['nearest_bull']['mid']) / cp < 0.008:
                long_score += 2
                long_reasons.append(f"🟩 Bullish FVG Retest ({smc_fvg['nearest_bull']['bottom']:.4f}–{smc_fvg['nearest_bull']['top']:.4f})")
            if smc_fvg['nearest_bear'] and abs(cp - smc_fvg['nearest_bear']['mid']) / cp < 0.008:
                short_score += 2
                short_reasons.append(f"🟥 Bearish FVG Retest ({smc_fvg['nearest_bear']['bottom']:.4f}–{smc_fvg['nearest_bear']['top']:.4f})")

            # 3. Premium / Discount zone filter (+2 pts reward, hard block for wrong zone)
            if smc_pd['zone'] == 'discount':
                long_score  += 2
                long_reasons.append(f"💚 Discount Zone ({smc_pd['range_pct']:.0f}% of range) — Smart Buy Area")
            elif smc_pd['zone'] == 'premium':
                short_score += 2
                short_reasons.append(f"🔴 Premium Zone ({smc_pd['range_pct']:.0f}% of range) — Smart Sell Area")
            elif smc_pd['zone'] == 'equilibrium':
                long_score  += 0.5
                short_score += 0.5   # neutral — slight nudge both ways

            # HARD BLOCK: Don't long in premium, don't short in discount
            if smc_pd['zone'] == 'premium' and long_score > short_score:
                logger.info(f"⛔ {symbol} LONG blocked — price in PREMIUM zone ({smc_pd['range_pct']:.0f}%)")
                return None
            if smc_pd['zone'] == 'discount' and short_score > long_score:
                logger.info(f"⛔ {symbol} SHORT blocked — price in DISCOUNT zone ({smc_pd['range_pct']:.0f}%)")
                return None

            # 4. EQH/EQL — liquidity nearby (+1 pt, acts as TP magnet or stop hunt warning)
            if smc_eq['nearest_eqh'] and cp < smc_eq['nearest_eqh']:
                dist_pct = (smc_eq['nearest_eqh'] - cp) / cp * 100
                if dist_pct < 2.0:   # within 2% — price likely to hunt it
                    long_score += 1
                    long_reasons.append(f"💧 EQH Liquidity Target ({smc_eq['nearest_eqh']:.4f})")
            if smc_eq['nearest_eql'] and cp > smc_eq['nearest_eql']:
                dist_pct = (cp - smc_eq['nearest_eql']) / cp * 100
                if dist_pct < 2.0:
                    short_score += 1
                    short_reasons.append(f"💧 EQL Liquidity Target ({smc_eq['nearest_eql']:.4f})")

            # ── TREND (6 pts) ────────────────────────────────────────
            if latest_4h['ema_9'] > latest_4h['ema_21'] > latest_4h['ema_50']:
                long_score += 3
                long_reasons.append('🔥 4H Uptrend')
            elif latest_4h['ema_9'] < latest_4h['ema_21'] < latest_4h['ema_50']:
                short_score += 3
                short_reasons.append('🔥 4H Downtrend')

            if latest_1h['ema_9'] > latest_1h['ema_21']:
                long_score += 2
                long_reasons.append('1H Bullish')
            elif latest_1h['ema_9'] < latest_1h['ema_21']:
                short_score += 2
                short_reasons.append('1H Bearish')

            if latest_1h['close'] > latest_1h['supertrend']:
                long_score += 1
                long_reasons.append('SuperTrend Bull')
            elif latest_1h['close'] < latest_1h['supertrend']:
                short_score += 1
                short_reasons.append('SuperTrend Bear')

            # ── MOMENTUM (9 pts) ─────────────────────────────────────
            if latest_1h['rsi'] < 30:
                long_score += 3.5
                long_reasons.append(f'💎 RSI Deep Oversold ({latest_1h["rsi"]:.0f})')
            elif latest_1h['rsi'] < 40:
                long_score += 2
                long_reasons.append(f'RSI Oversold ({latest_1h["rsi"]:.0f})')
            elif 40 <= latest_1h['rsi'] <= 50:
                long_score += 1
                long_reasons.append('RSI Buy Zone')

            if latest_1h['rsi'] > 70:
                short_score += 3.5
                short_reasons.append(f'💎 RSI Deep Overbought ({latest_1h["rsi"]:.0f})')
            elif latest_1h['rsi'] > 60:
                short_score += 2
                short_reasons.append(f'RSI Overbought ({latest_1h["rsi"]:.0f})')
            elif 50 <= latest_1h['rsi'] <= 60:
                short_score += 1
                short_reasons.append('RSI Sell Zone')

            if latest_1h['stoch_rsi_k'] < 0.2 and latest_1h['stoch_rsi_k'] > latest_1h['stoch_rsi_d']:
                long_score += 2
                long_reasons.append('⚡ Stoch RSI Cross')
            elif latest_1h['stoch_rsi_k'] > 0.8 and latest_1h['stoch_rsi_k'] < latest_1h['stoch_rsi_d']:
                short_score += 2
                short_reasons.append('⚡ Stoch RSI Cross')

            if latest_1h['macd'] > latest_1h['macd_signal'] and prev_1h['macd'] <= prev_1h['macd_signal']:
                long_score += 2.5
                long_reasons.append('🎯 MACD Cross')
            elif latest_1h['macd'] < latest_1h['macd_signal'] and prev_1h['macd'] >= prev_1h['macd_signal']:
                short_score += 2.5
                short_reasons.append('🎯 MACD Cross')

            if latest_1h['uo'] < 30:
                long_score += 1.5
                long_reasons.append('UO Oversold')
            elif latest_1h['uo'] > 70:
                short_score += 1.5
                short_reasons.append('UO Overbought')

            # ── VOLUME (5 pts) ───────────────────────────────────────
            # Directional volume — only score in the right direction
            if long_vol_ok:
                long_score += 3
                long_reasons.append(f'📈 Buy Vol Confirmed ({buy_pct:.0f}% buying, {vol_ratio:.1f}x avg)')
            if short_vol_ok:
                short_score += 3
                short_reasons.append(f'📉 Sell Vol Confirmed ({100-buy_pct:.0f}% selling, {vol_ratio:.1f}x avg)')

            # Extra spike bonus — still valid but only when direction matches
            if volume_spike:
                if latest_1h['close'] > prev_1h['close'] and long_vol_ok:
                    long_score += 1
                    long_reasons.append(f'🚀 Vol Spike ({vol_ratio:.1f}x)')
                elif latest_1h['close'] < prev_1h['close'] and short_vol_ok:
                    short_score += 1
                    short_reasons.append(f'💥 Vol Dump ({vol_ratio:.1f}x)')

            if latest_1h['mfi'] < 20:
                long_score += 1.5
                long_reasons.append(f'MFI Oversold ({latest_1h["mfi"]:.0f})')
            elif latest_1h['mfi'] > 80:
                short_score += 1.5
                short_reasons.append(f'MFI Overbought ({latest_1h["mfi"]:.0f})')

            if latest_1h['cmf'] > 0.15:
                long_score += 1
                long_reasons.append('Strong Buying (CMF)')
            elif latest_1h['cmf'] < -0.15:
                short_score += 1
                short_reasons.append('Strong Selling (CMF)')

            obv_trend = df_1h['obv'].iloc[-5:].diff().mean()
            if obv_trend > 0 and latest_1h['obv'] > latest_1h['obv_ema']:
                long_score += 0.5
                long_reasons.append('OBV Accumulation')
            elif obv_trend < 0 and latest_1h['obv'] < latest_1h['obv_ema']:
                short_score += 0.5
                short_reasons.append('OBV Distribution')

            # ── VOLATILITY (6 pts) ───────────────────────────────────
            if latest_1h['bb_pband'] < 0.1:
                long_score += 2.5
                long_reasons.append('💎 Lower BB')
            elif latest_1h['bb_pband'] > 0.9:
                short_score += 2.5
                short_reasons.append('💎 Upper BB')

            if latest_1h['cci'] < -150:
                long_score += 1.5
                long_reasons.append('CCI Deep Oversold')
            elif latest_1h['cci'] > 150:
                short_score += 1.5
                short_reasons.append('CCI Deep Overbought')

            if latest_1h['williams_r'] < -85:
                long_score += 1
                long_reasons.append('Williams Oversold')
            elif latest_1h['williams_r'] > -15:
                short_score += 1
                short_reasons.append('Williams Overbought')

            if latest_1h['close'] < latest_1h['vwap'] * 0.98:
                long_score += 1
                long_reasons.append('Below VWAP')
            elif latest_1h['close'] > latest_1h['vwap'] * 1.02:
                short_score += 1
                short_reasons.append('Above VWAP')

            # ── TREND STRENGTH (4 pts) ───────────────────────────────
            if latest_1h['adx'] > 30:
                if latest_1h['di_plus'] > latest_1h['di_minus']:
                    long_score += 2
                    long_reasons.append(f'🔥 Strong Up (ADX:{latest_1h["adx"]:.0f})')
                else:
                    short_score += 2
                    short_reasons.append(f'🔥 Strong Down (ADX:{latest_1h["adx"]:.0f})')
            elif latest_1h['adx'] > 25:
                if latest_1h['di_plus'] > latest_1h['di_minus']:
                    long_score += 1
                else:
                    short_score += 1

            if latest_1h['aroon_ind'] > 50:
                long_score += 1
                long_reasons.append('Aroon Up')
            elif latest_1h['aroon_ind'] < -50:
                short_score += 1
                short_reasons.append('Aroon Down')

            if latest_1h['roc'] > 3:
                long_score += 1
                long_reasons.append('Strong Momentum')
            elif latest_1h['roc'] < -3:
                short_score += 1
                short_reasons.append('Strong Momentum')

            # ── DIVERGENCE & PATTERNS (3 pts) ────────────────────────
            if latest_1h['bullish_divergence'] == 1:
                long_score += 2
                long_reasons.append('🎯 Bullish Divergence')
            elif latest_1h['bearish_divergence'] == 1:
                short_score += 2
                short_reasons.append('🎯 Bearish Divergence')

            if latest_15m['bullish_engulfing'] == 1:
                long_score += 1.5
                long_reasons.append('📊 Bullish Engulfing')
            elif latest_15m['bearish_engulfing'] == 1:
                short_score += 1.5
                short_reasons.append('📊 Bearish Engulfing')

            # ── HTF CONFIRMATION (2 pts) ─────────────────────────────
            if latest_4h['close'] > latest_4h['vwap']:
                long_score += 1
            else:
                short_score += 1

            if latest_4h['rsi'] < 50:
                long_score += 1
            elif latest_4h['rsi'] > 50:
                short_score += 1

            # ── DETERMINE SIGNAL ─────────────────────────────────────
            min_threshold = max_score * 0.48

            signal = None
            ob_active = None

            if long_score > short_score and long_score >= min_threshold:
                # HARD GATE: volume must confirm direction
                if not long_vol_ok:
                    logger.info(f"⛔ {symbol} LONG blocked — no buy volume confirmation (buy%={buy_pct:.0f})")
                    return None
                signal  = 'LONG'
                score   = long_score
                reasons = long_reasons
                ob_active = ob_1h if ob_type_1h == 'bullish' else None

                if long_score >= max_score * 0.70:
                    quality = 'PREMIUM 💎'
                elif long_score >= max_score * 0.58:
                    quality = 'HIGH 🔥'
                else:
                    quality = 'GOOD ✅'

            elif short_score > long_score and short_score >= min_threshold:
                # HARD GATE: volume must confirm direction
                if not short_vol_ok:
                    logger.info(f"⛔ {symbol} SHORT blocked — no sell volume confirmation (buy%={buy_pct:.0f})")
                    return None
                signal  = 'SHORT'
                score   = short_score
                reasons = short_reasons
                ob_active = ob_1h if ob_type_1h == 'bearish' else None

                if short_score >= max_score * 0.70:
                    quality = 'PREMIUM 💎'
                elif short_score >= max_score * 0.58:
                    quality = 'HIGH 🔥'
                else:
                    quality = 'GOOD ✅'

            if signal:
                entry = latest_15m['close']
                atr   = latest_1h['atr']

                # If we're at an OB, use the OB boundary as tighter SL
                if ob_active:
                    if signal == 'LONG':
                        # SL just below the OB bottom
                        ob_sl = ob_active['bottom'] * 0.998
                        sl = min(entry - (atr * 1.5), ob_sl)
                    else:
                        # SL just above the OB top
                        ob_sl = ob_active['top'] * 1.002
                        sl = max(entry + (atr * 1.5), ob_sl)
                else:
                    if signal == 'LONG':
                        sl = entry - (atr * 1.5)
                    else:
                        sl = entry + (atr * 1.5)

                if signal == 'LONG':
                    tp1 = entry + (atr * 1)
                    # If POC is above entry and between ATR*1.5 and ATR*3, use it as TP2
                    tp2 = entry + (atr * 2)
                    tp3 = entry + (atr * 3.5)
                    if vp and vp['poc_price'] > entry * 1.005:
                        poc_dist = vp['poc_price'] - entry
                        if atr * 1.2 < poc_dist < atr * 3.5:
                            tp2 = vp['poc_price']
                    targets = [tp1, tp2, tp3]
                else:
                    tp1 = entry - (atr * 1)
                    tp2 = entry - (atr * 2)
                    tp3 = entry - (atr * 3.5)
                    if vp and vp['poc_price'] < entry * 0.995:
                        poc_dist = entry - vp['poc_price']
                        if atr * 1.2 < poc_dist < atr * 3.5:
                            tp2 = vp['poc_price']
                    targets = [tp1, tp2, tp3]

                risk_pct = abs((sl - entry) / entry * 100)
                rr = [(abs(tp - entry) / abs(sl - entry)) for tp in targets]

                trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

                if ob_active:
                    self.stats['ob_signals'] += 1

                return {
                    'trade_id': trade_id,
                    'symbol': symbol.replace('/USDT:USDT', ''),
                    'full_symbol': symbol,
                    'signal': signal,
                    'quality': quality,
                    'score': score,
                    'max_score': max_score,
                    'score_percent': (score / max_score) * 100,
                    'entry': entry,
                    'stop_loss': sl,
                    'targets': targets,
                    'reward_ratios': rr,
                    'risk_percent': risk_pct,
                    'reasons': reasons[:12],
                    'ob_zone': ob_active,
                    'ob_type': ob_type_1h,
                    'buy_pct': buy_pct,
                    'vol_ratio': vol_ratio,
                    'vp_poc': vp['poc_price'] if vp else None,
                    'vp_node': vp['current_node'] if vp else None,
                    'vp_support': vp['nearest_support'] if vp else None,
                    'vp_resistance': vp['nearest_resistance'] if vp else None,
                    'smc_trend':   smc_struct['swing_trend'],
                    'smc_zone':    smc_pd['zone'],
                    'smc_zone_pct': smc_pd['range_pct'],
                    'smc_fvg_bull': smc_fvg['nearest_bull'],
                    'smc_fvg_bear': smc_fvg['nearest_bear'],
                    'smc_eqh':     smc_eq['nearest_eqh'],
                    'smc_eql':     smc_eq['nearest_eql'],
                    'smc_bos_choch': (
                        'Bullish BOS'   if smc_struct['last_bos_bull']   else
                        'Bullish CHoCH' if smc_struct['last_choch_bull'] else
                        'Bearish BOS'   if smc_struct['last_bos_bear']   else
                        'Bearish CHoCH' if smc_struct['last_choch_bear'] else None
                    ),
                    'tp_hit': [False, False, False],
                    'sl_hit': False,
                    'timestamp': datetime.now(),
                    'status': 'ACTIVE'
                }

        except Exception as e:
            logger.error(f"Signal detection error for {symbol}: {e}")
            return None

        return None

    def format_signal(self, sig):
        is_long   = sig['signal'] == 'LONG'
        dir_emoji = "🟢" if is_long else "🔴"
        dir_label = "LONG  📈" if is_long else "SHORT 📉"

        # Header badge
        quality_line = {
            'PREMIUM 💎': "💎 PREMIUM SIGNAL",
            'HIGH 🔥':    "🔥 HIGH QUALITY SIGNAL",
            'GOOD ✅':    "✅ SIGNAL",
        }.get(sig['quality'], "✅ SIGNAL")

        ob_badge  = "  •  🧱 OB"  if sig.get('ob_zone')  else ""
        vp_badge  = "  •  📊 VP"  if sig.get('vp_poc')   else ""
        smc_badge = "  •  🧠 SMC" if sig.get('smc_bos_choch') or sig.get('smc_zone') != 'neutral' else ""

        # Price formatting — strip trailing zeros nicely
        def fmt(p):
            if p >= 100:   return f"{p:.2f}"
            if p >= 1:     return f"{p:.3f}"
            if p >= 0.01:  return f"{p:.4f}"
            return f"{p:.6f}"

        entry = sig['entry']
        sl    = sig['stop_loss']
        tp1, tp2, tp3 = sig['targets']
        rr1, rr2, rr3 = sig['reward_ratios']

        pct = lambda p: abs((p - entry) / entry * 100)

        msg  = f"<b>{quality_line}{ob_badge}{vp_badge}{smc_badge}</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"  {dir_emoji} <b>#{sig['symbol']}USDT  •  {dir_label}</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n\n"

        # OB zone — one clean line
        if sig.get('ob_zone'):
            ob = sig['ob_zone']
            msg += f"🧱 <b>OB Zone:</b>  {fmt(ob['bottom'])} – {fmt(ob['top'])}\n"

        # Volume Profile context
        if sig.get('vp_poc'):
            node_label = {
                'high':    '🟡 Consolidation',
                'low':     '⚡ Thin Air — fast move',
                'average': '⚪ Average'
            }.get(sig.get('vp_node'), '')
            msg += f"📍 <b>VP POC:</b>  {fmt(sig['vp_poc'])}  <i>({node_label})</i>\n"
            if sig.get('vp_support') and is_long:
                msg += f"🟩 <b>VP Support:</b>  {fmt(sig['vp_support'])}\n"
            if sig.get('vp_resistance') and not is_long:
                msg += f"🟥 <b>VP Resist:</b>   {fmt(sig['vp_resistance'])}\n"

        # SMC context line
        smc_parts = []
        if sig.get('smc_bos_choch'):
            smc_parts.append(f"⚡ {sig['smc_bos_choch']}")
        zone = sig.get('smc_zone', 'neutral')
        zone_pct = sig.get('smc_zone_pct', 50)
        if zone == 'discount':
            smc_parts.append(f"💚 Discount Zone ({zone_pct:.0f}%)")
        elif zone == 'premium':
            smc_parts.append(f"🔴 Premium Zone ({zone_pct:.0f}%)")
        elif zone == 'equilibrium':
            smc_parts.append(f"⚖️ Equilibrium ({zone_pct:.0f}%)")
        if is_long and sig.get('smc_fvg_bull'):
            f = sig['smc_fvg_bull']
            smc_parts.append(f"🟩 FVG {fmt(f['bottom'])}–{fmt(f['top'])}")
        if not is_long and sig.get('smc_fvg_bear'):
            f = sig['smc_fvg_bear']
            smc_parts.append(f"🟥 FVG {fmt(f['bottom'])}–{fmt(f['top'])}")
        if is_long and sig.get('smc_eqh'):
            smc_parts.append(f"💧 EQH {fmt(sig['smc_eqh'])}")
        if not is_long and sig.get('smc_eql'):
            smc_parts.append(f"💧 EQL {fmt(sig['smc_eql'])}")
        if smc_parts:
            msg += f"🧠 <b>SMC:</b>  {' · '.join(smc_parts)}\n"
        msg += "\n"

        msg += f"💰 <b>Entry</b>       {fmt(entry)}\n"

        # Volume confidence bar
        buy_pct   = sig.get('buy_pct', 50)
        vol_ratio = sig.get('vol_ratio', 1.0)
        vol_filled = int(buy_pct / 10) if is_long else int((100 - buy_pct) / 10)
        vol_bar    = "🟦" * vol_filled + "⬜" * (10 - vol_filled)
        vol_label  = f"{buy_pct:.0f}% buy pressure" if is_long else f"{100-buy_pct:.0f}% sell pressure"
        msg += f"📊 <b>Volume</b>      {vol_bar}  <i>{vol_label}  ({vol_ratio:.1f}x avg)</i>\n\n"

        msg += f"🎯 <b>TP 1</b>  →  <code>{fmt(tp1)}</code>  <i>(+{pct(tp1):.1f}%  •  RR {rr1:.1f}x)</i>\n"
        msg += f"🎯 <b>TP 2</b>  →  <code>{fmt(tp2)}</code>  <i>(+{pct(tp2):.1f}%  •  RR {rr2:.1f}x)</i>\n"
        msg += f"🎯 <b>TP 3</b>  →  <code>{fmt(tp3)}</code>  <i>(+{pct(tp3):.1f}%  •  RR {rr3:.1f}x)</i>\n\n"

        msg += f"🛑 <b>Stop Loss</b>  <code>{fmt(sl)}</code>  <i>(-{sig['risk_percent']:.1f}%)</i>\n\n"

        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"⭐ Score  {sig['score']:.0f}/{sig['max_score']}   "
        msg += f"{'▰' * int(sig['score_percent']/10)}{'▱' * (10 - int(sig['score_percent']/10))}\n"
        msg += f"🔍 <i>{' · '.join(r.lstrip('🔥💎🎯⚡🚀💥📊 ') for r in sig['reasons'][:5])}</i>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"<i>⏰ {sig['timestamp'].strftime('%H:%M')}  •  📡 Live tracking on</i>"

        return msg

    async def send_msg(self, msg):
        try:
            await self.telegram_bot.send_message(chat_id=self.chat_id, text=msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Send error: {e}")

    async def send_tp_alert(self, trade, tp_num, price):
        emoji = "🎉" if trade['signal'] == 'LONG' else "💰"
        tp  = trade['targets'][tp_num - 1]
        pct = abs((tp - trade['entry']) / trade['entry'] * 100)

        msg  = f"{emoji} <b>TARGET HIT!</b> {emoji}\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"<b>{trade['symbol']}</b> {trade['signal']}"
        if trade.get('ob_zone'):
            msg += " 🧱 OB Setup"
        msg += f"\n\n<b>✅ TP{tp_num} HIT!</b>\n"
        msg += f"Target: ${tp:.6f}\n"
        msg += f"Current: ${price:.6f}\n"
        msg += f"Profit: +{pct:.2f}%\n\n"

        if tp_num == 1:
            msg += f"📋 Take 50% profit NOW\nMove SL to breakeven"
        elif tp_num == 2:
            msg += f"📋 Take 30% profit NOW"
        else:
            msg += f"📋 Take remaining 20%\n🎊 TRADE COMPLETE!"

        await self.send_msg(msg)

        if tp_num == 1:   self.stats['tp1_hits'] += 1
        elif tp_num == 2: self.stats['tp2_hits'] += 1
        else:
            self.stats['tp3_hits'] += 1
            self.stats['daily_wins'] += 1   # TP3 = full win

    async def send_sl_alert(self, trade, price):
        loss = abs((price - trade['entry']) / trade['entry'] * 100)
        msg  = f"⚠️ <b>STOP LOSS HIT!</b> ⚠️\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"{trade['symbol']} {trade['signal']}\n\n"
        msg += f"Entry: ${trade['entry']:.6f}\n"
        msg += f"SL: ${trade['stop_loss']:.6f}\n"
        msg += f"Current: ${price:.6f}\n"
        msg += f"Loss: -{loss:.2f}%"
        await self.send_msg(msg)
        self.stats['daily_losses'] += 1

    async def track_trades(self):
        self.is_tracking = True
        logger.info("📡 Tracking started")

        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30)
                    continue

                to_remove = []

                for tid, trade in list(self.active_trades.items()):
                    try:
                        if datetime.now() - trade['timestamp'] > timedelta(hours=24):
                            msg = f"⏰ 24H LIMIT\n<code>{tid}</code>\n{trade['symbol']}\nClose position!"
                            await self.send_msg(msg)
                            # Count as win if at least TP1 was hit, otherwise breakeven
                            if any(trade['tp_hit']):
                                self.stats['daily_wins'] += 1
                            else:
                                self.stats['daily_be'] += 1
                            to_remove.append(tid)
                            continue

                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price  = ticker['last']

                        if trade['signal'] == 'LONG':
                            if not trade['tp_hit'][0] and price >= trade['targets'][0]:
                                await self.send_tp_alert(trade, 1, price)
                                trade['tp_hit'][0] = True
                                self.stats['daily_wins'] += 1
                            if not trade['tp_hit'][1] and price >= trade['targets'][1]:
                                await self.send_tp_alert(trade, 2, price)
                                trade['tp_hit'][1] = True
                            if not trade['tp_hit'][2] and price >= trade['targets'][2]:
                                await self.send_tp_alert(trade, 3, price)
                                trade['tp_hit'][2] = True
                                to_remove.append(tid)
                            if not trade['sl_hit'] and price <= trade['stop_loss']:
                                await self.send_sl_alert(trade, price)
                                trade['sl_hit'] = True
                                to_remove.append(tid)
                        else:
                            if not trade['tp_hit'][0] and price <= trade['targets'][0]:
                                await self.send_tp_alert(trade, 1, price)
                                trade['tp_hit'][0] = True
                                self.stats['daily_wins'] += 1
                            if not trade['tp_hit'][1] and price <= trade['targets'][1]:
                                await self.send_tp_alert(trade, 2, price)
                                trade['tp_hit'][1] = True
                            if not trade['tp_hit'][2] and price <= trade['targets'][2]:
                                await self.send_tp_alert(trade, 3, price)
                                trade['tp_hit'][2] = True
                                to_remove.append(tid)
                            if not trade['sl_hit'] and price >= trade['stop_loss']:
                                await self.send_sl_alert(trade, price)
                                trade['sl_hit'] = True
                                to_remove.append(tid)

                    except Exception as e:
                        logger.error(f"Track error {tid}: {e}")
                        continue

                for tid in to_remove:
                    del self.active_trades[tid]
                    logger.info(f"✅ Trade done: {tid}")

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Tracking error: {e}")
                await asyncio.sleep(60)

    async def scan_all(self):
        if self.is_scanning:
            logger.info("⚠️ Already scanning...")
            return []

        self.is_scanning = True
        logger.info("🔍 Starting scan...")

        pairs   = await self.get_all_usdt_pairs()
        signals = []
        scanned = 0

        for pair in pairs:
            try:
                logger.info(f"📊 {pair}...")
                data = await self.fetch_day_trading_data(pair)

                if data:
                    sig = self.detect_signal(data, pair)
                    if sig:
                        signals.append(sig)
                        self.signal_history.append(sig)
                        self.stats['total_signals'] += 1
                        self.stats['daily_signals'] += 1

                        if sig['signal'] == 'LONG':
                            self.stats['long_signals'] += 1
                        else:
                            self.stats['short_signals'] += 1

                        if sig['quality'] == 'PREMIUM 💎':
                            self.stats['premium_signals'] += 1

                        self.active_trades[sig['trade_id']] = sig

                        msg = self.format_signal(sig)
                        await self.send_msg(msg)
                        await asyncio.sleep(1.5)

                scanned += 1
                if scanned % 25 == 0:
                    logger.info(f"📈 {scanned}/{len(pairs)}")

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ {pair}: {e}")
                continue

        self.stats['last_scan_time'] = datetime.now()
        self.stats['pairs_scanned']  = scanned

        ob_count = sum(1 for s in signals if s.get('ob_zone'))

        summary  = f"✅ <b>SCAN COMPLETE</b>\n\n"
        summary += f"📊 Scanned: {scanned}\n"
        summary += f"🎯 Signals: {len(signals)}\n"

        if signals:
            longs   = sum(1 for s in signals if s['signal'] == 'LONG')
            shorts  = len(signals) - longs
            premium = sum(1 for s in signals if s['quality'] == 'PREMIUM 💎')

            summary += f"  🟢 Long: {longs}\n"
            summary += f"  🔴 Short: {shorts}\n"
            summary += f"  💎 Premium: {premium}\n"
            summary += f"  🧱 OB Setups: {ob_count}\n"

        summary += f"  📡 Tracking: {len(self.active_trades)}\n"
        summary += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"

        await self.send_msg(summary)
        logger.info(f"🎉 Done! {len(signals)} signals, {ob_count} OB setups")

        self.is_scanning = False
        return signals

    async def _daily_report_loop(self):
        """Wait 24 hours then send report, repeat forever"""
        while True:
            await asyncio.sleep(24 * 60 * 60)   # 24 hours
            try:
                await self.send_daily_report()
            except Exception as e:
                logger.error(f"Daily report error: {e}")

    async def send_daily_report(self):
        """Send 24h performance report and reset daily counters"""
        s = self.stats
        total    = s['daily_signals']
        wins     = s['daily_wins']
        losses   = s['daily_losses']
        be       = s['daily_be']
        closed   = wins + losses + be
        winrate  = (wins / closed * 100) if closed > 0 else 0

        # Bar visualisation  (10 blocks)
        filled = int(winrate / 10)
        bar    = "🟩" * filled + "⬜" * (10 - filled)

        # Streak label
        if winrate >= 70:
            perf = "🔥 ON FIRE"
        elif winrate >= 55:
            perf = "💪 SOLID"
        elif winrate >= 40:
            perf = "😐 AVERAGE"
        else:
            perf = "⚠️ ROUGH DAY"

        period_start = s['report_start'].strftime('%d %b  %H:%M')
        period_end   = datetime.now().strftime('%d %b  %H:%M')

        msg  = f"━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"📊 <b>24H PERFORMANCE REPORT</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
        msg += f"🗓 <i>{period_start}  →  {period_end}</i>\n\n"
        msg += f"📡 <b>Signal Calls:</b>  {total} calls\n"
        msg += f"📊 <b>Win Rate:</b>  {winrate:.2f}%\n"
        msg += f"{bar}\n\n"
        msg += f"🟢 <b>Profit Trades:</b>  {wins}\n"
        msg += f"🚫 <b>Loss Trades:</b>  {losses}\n"
        if be:
            msg += f"⚪ <b>Breakeven:</b>  {be}\n"
        msg += f"\n<b>TP Breakdown:</b>\n"
        msg += f"  🎯 TP1 hits:  {s['tp1_hits']}\n"
        msg += f"  🎯 TP2 hits:  {s['tp2_hits']}\n"
        msg += f"  🎯 TP3 hits:  {s['tp3_hits']}\n\n"
        msg += f"<b>{perf}</b>\n"
        msg += f"━━━━━━━━━━━━━━━━━━━━━━"

        await self.send_msg(msg)
        logger.info(f"📊 Daily report sent — {winrate:.1f}% winrate")

        # Reset daily counters
        self.stats['daily_signals'] = 0
        self.stats['daily_wins']    = 0
        self.stats['daily_losses']  = 0
        self.stats['daily_be']      = 0
        self.stats['tp1_hits']      = 0
        self.stats['tp2_hits']      = 0
        self.stats['tp3_hits']      = 0
        self.stats['report_start']  = datetime.now()

    async def run(self, interval=60):
        logger.info("🚀 ADVANCED DAY TRADING SCANNER")

        welcome  = "🔥 <b>ADVANCED 24H DAY TRADING SCANNER</b> 🔥\n\n"
        welcome += "✅ ALL USDT pairs\n"
        welcome += "✅ 25+ indicators\n"
        welcome += "✅ 🧱 Order Block detection (1H + 4H)\n"
        welcome += "✅ 📊 Volume Profile (LuxAlgo MFP)\n"
        welcome += "✅ 🧠 Smart Money Concepts (BOS/CHoCH/FVG/Zones)\n"
        welcome += "✅ Divergence + Pattern recognition\n"
        welcome += "✅ Live TP/SL tracking + 24H report\n"
        welcome += "✅ 56-point scoring\n\n"
        welcome += f"Scans every {interval} min\n\n"
        welcome += "<b>Commands:</b>\n"
        welcome += "/scan /stats /trades /help\n\n"
        welcome += "🎯 Full SMC + Volume Profile signals!"

        await self.send_msg(welcome)

        asyncio.create_task(self.track_trades())
        asyncio.create_task(self._daily_report_loop())

        while True:
            try:
                await self.scan_all()
                logger.info(f"💤 Next scan in {interval} min")
                await asyncio.sleep(interval * 60)
            except Exception as e:
                logger.error(f"❌ {e}")
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ─────────────────────────────────────────────
#  BOT COMMANDS
# ─────────────────────────────────────────────

class BotCommands:
    def __init__(self, scanner):
        self.scanner = scanner

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg  = "🚀 <b>Advanced Day Trading Scanner + Live Tracking!</b>\n\n"
        msg += "Scans ALL USDT pairs with 25+ indicators + 🧱 Order Blocks.\n\n"
        msg += "<b>Commands:</b>\n"
        msg += "/scan - Force scan\n"
        msg += "/stats - Statistics\n"
        msg += "/trades - Active trades\n"
        msg += "/help - Help\n\n"
        msg += "💰 Get alerts when TPs hit!"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.scanner.is_scanning:
            await update.message.reply_text("⚠️ Scan running!")
            return
        await update.message.reply_text("🔍 Starting scan...")
        await self.scanner.scan_all()

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        s = self.scanner.stats
        msg  = f"📊 <b>STATISTICS</b>\n\n"
        msg += f"Total: {s['total_signals']}\n"
        msg += f"Long: {s['long_signals']} 🟢\n"
        msg += f"Short: {s['short_signals']} 🔴\n"
        msg += f"Premium: {s['premium_signals']} 💎\n"
        msg += f"OB Setups: {s['ob_signals']} 🧱\n\n"
        msg += f"<b>TP Hits:</b>\n"
        msg += f"  TP1: {s['tp1_hits']} 🎯\n"
        msg += f"  TP2: {s['tp2_hits']} 🎯\n"
        msg += f"  TP3: {s['tp3_hits']} 🎯\n\n"
        if s['last_scan_time']:
            msg += f"Last: {s['last_scan_time'].strftime('%H:%M:%S')}\n"
            msg += f"Pairs: {s['pairs_scanned']}"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = self.scanner.active_trades
        if not trades:
            await update.message.reply_text("📭 No active trades")
            return

        msg = f"📡 <b>ACTIVE TRADES ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:10]:
            age = datetime.now() - t['timestamp']
            hrs = int(age.total_seconds() / 3600)
            tp_status = ""
            for hit in t['tp_hit']:
                tp_status += "✅" if hit else "⏳"
            ob_tag = " 🧱" if t.get('ob_zone') else ""
            msg += f"<b>{t['symbol']}</b> {t['signal']}{ob_tag}\n"
            msg += f"  {tp_status} | {hrs}h old\n\n"

        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg  = "📚 <b>ADVANCED DAY TRADING SCANNER</b>\n\n"
        msg += "<b>Features:</b>\n"
        msg += "• 25+ advanced indicators\n"
        msg += "• 🧱 Order Block detection (1H + 4H)\n"
        msg += "• Divergence detection\n"
        msg += "• Pattern recognition\n"
        msg += "• Volume analysis\n"
        msg += "• Live TP/SL tracking\n"
        msg += "• 40-point scoring\n\n"
        msg += "<b>Order Block Logic:</b>\n"
        msg += "• Bullish OB = last bearish candle before a big up impulse\n"
        msg += "• Bearish OB = last bullish candle before a big down impulse\n"
        msg += "• +3 pts for 1H OB, +2 pts for 4H OB confirmation\n"
        msg += "• SL placed beyond the OB zone for tighter risk\n\n"
        msg += "<b>Quality:</b>\n"
        msg += "💎 PREMIUM (70%+)\n"
        msg += "🔥 HIGH (58%+)\n"
        msg += "✅ GOOD (48%+)\n\n"
        msg += "<b>Commands:</b>\n"
        msg += "/scan /stats /trades /help"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

async def main():
    TELEGRAM_TOKEN  = "8034062612:AAEJYbPA8sMODYvqvt8U-5mM7c3Y3-GOYtM"
    TELEGRAM_CHAT_ID = "7500072234"
    BINANCE_API_KEY  = None
    BINANCE_SECRET   = None

    scanner = AdvancedDayTradingScanner(
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        binance_api_key=BINANCE_API_KEY,
        binance_secret=BINANCE_SECRET
    )

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    commands = BotCommands(scanner)

    app.add_handler(CommandHandler("start",  commands.cmd_start))
    app.add_handler(CommandHandler("scan",   commands.cmd_scan))
    app.add_handler(CommandHandler("stats",  commands.cmd_stats))
    app.add_handler(CommandHandler("trades", commands.cmd_trades))
    app.add_handler(CommandHandler("help",   commands.cmd_help))

    await app.initialize()
    await app.start()

    logger.info("🤖 Bot ready!")

    try:
        await scanner.run(interval=15)
    except KeyboardInterrupt:
        logger.info("⚠️ Shutting down...")
    finally:
        await scanner.close()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
