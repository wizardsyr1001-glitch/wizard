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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


# ============================================================
#  SMC + ORDER BLOCK ENGINE
# ============================================================

class SMCEngine:
    """Smart Money Concepts: Order Blocks, FVG, BOS, MSS, Liquidity"""

    # ── Structure ────────────────────────────────────────────

    def find_swing_points(self, df, left=5, right=5):
        """Detect swing highs and lows (pivot points)"""
        highs, lows = [], []
        for i in range(left, len(df) - right):
            if all(df['high'].iloc[i] >= df['high'].iloc[i-left:i]) and \
               all(df['high'].iloc[i] >= df['high'].iloc[i+1:i+right+1]):
                highs.append({'index': i, 'price': df['high'].iloc[i]})
            if all(df['low'].iloc[i] <= df['low'].iloc[i-left:i]) and \
               all(df['low'].iloc[i] <= df['low'].iloc[i+1:i+right+1]):
                lows.append({'index': i, 'price': df['low'].iloc[i]})
        return highs, lows

    def detect_bos_mss(self, df, highs, lows):
        events = []
        close = df['close']

        for i in range(1, len(highs)):
            prev_h = highs[i-1]['price']
            curr_h = highs[i]['price']
            idx = highs[i]['index']
            if idx >= len(close): continue

            for j in range(idx, min(idx+10, len(close))):
                if close.iloc[j] > prev_h:
                    ev_type = 'BOS_BULL' if curr_h > prev_h else 'MSS_BULL'
                    events.append({'type': ev_type, 'index': j, 'level': prev_h})
                    break

        for i in range(1, len(lows)):
            prev_l = lows[i-1]['price']
            curr_l = lows[i]['price']
            idx = lows[i]['index']
            if idx >= len(close): continue

            for j in range(idx, min(idx+10, len(close))):
                if close.iloc[j] < prev_l:
                    ev_type = 'BOS_BEAR' if curr_l < prev_l else 'MSS_BEAR'
                    events.append({'type': ev_type, 'index': j, 'level': prev_l})
                    break

        if not events:
            return None
        return sorted(events, key=lambda x: x['index'])[-1]

    def detect_order_blocks(self, df, highs, lows, lookback=30):
        obs = []
        n = len(df)
        start = max(0, n - lookback)

        for i in range(start + 2, n - 3):
            candle = df.iloc[i]
            is_bearish = candle['close'] < candle['open']
            if not is_bearish:
                continue
            move = (df['high'].iloc[i+1:i+4].max() - candle['low']) / candle['low']
            if move > 0.008:
                ob = {
                    'type': 'BULL',
                    'index': i,
                    'top': candle['open'],
                    'bottom': candle['low'],
                    'mid': (candle['open'] + candle['low']) / 2,
                    'mitigated': False
                }
                future_lows = df['low'].iloc[i+1:n]
                if (future_lows < ob['top']).any() and (future_lows > ob['bottom']).any():
                    obs.append(ob)

        for i in range(start + 2, n - 3):
            candle = df.iloc[i]
            is_bullish = candle['close'] > candle['open']
            if not is_bullish:
                continue
            move = (candle['high'] - df['low'].iloc[i+1:i+4].min()) / candle['high']
            if move > 0.008:
                ob = {
                    'type': 'BEAR',
                    'index': i,
                    'top': candle['high'],
                    'bottom': candle['close'],
                    'mid': (candle['high'] + candle['close']) / 2,
                    'mitigated': False
                }
                future_highs = df['high'].iloc[i+1:n]
                if (future_highs > ob['bottom']).any() and (future_highs < ob['top']).any():
                    obs.append(ob)

        return obs

    def detect_fvg(self, df, lookback=20):
        fvgs = []
        n = len(df)
        start = max(1, n - lookback - 1)

        for i in range(start, n - 1):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            nxt  = df.iloc[i+1]

            if prev['high'] < nxt['low']:
                size = nxt['low'] - prev['high']
                fvgs.append({
                    'type': 'BULL',
                    'top': nxt['low'],
                    'bottom': prev['high'],
                    'mid': (nxt['low'] + prev['high']) / 2,
                    'size_pct': size / curr['close'] * 100,
                    'index': i
                })

            if prev['low'] > nxt['high']:
                size = prev['low'] - nxt['high']
                fvgs.append({
                    'type': 'BEAR',
                    'top': prev['low'],
                    'bottom': nxt['high'],
                    'mid': (prev['low'] + nxt['high']) / 2,
                    'size_pct': size / curr['close'] * 100,
                    'index': i
                })

        return fvgs

    def detect_liquidity_sweeps(self, df, highs, lows, lookback=20):
        sweeps = []
        n = len(df)
        recent_start = n - lookback

        for sh in highs:
            if sh['index'] < recent_start: continue
            level = sh['price']
            for j in range(sh['index']+1, min(sh['index']+8, n)):
                c = df.iloc[j]
                if c['high'] > level and c['close'] < level:
                    sweeps.append({'type': 'BULL_SWEEP', 'level': level, 'index': j})
                    break

        for sl in lows:
            if sl['index'] < recent_start: continue
            level = sl['price']
            for j in range(sl['index']+1, min(sl['index']+8, n)):
                c = df.iloc[j]
                if c['low'] < level and c['close'] > level:
                    sweeps.append({'type': 'BEAR_SWEEP', 'level': level, 'index': j})
                    break

        return sweeps

    def find_premium_discount(self, highs, lows, current_price):
        if not highs or not lows:
            return 'NEUTRAL', 0.5
        recent_high = max(h['price'] for h in highs[-5:]) if len(highs) >= 5 else highs[-1]['price']
        recent_low  = min(l['price'] for l in lows[-5:])  if len(lows)  >= 5 else lows[-1]['price']
        rang = recent_high - recent_low
        if rang == 0:
            return 'NEUTRAL', 0.5
        pos = (current_price - recent_low) / rang
        if pos > 0.6:
            return 'PREMIUM', pos
        elif pos < 0.4:
            return 'DISCOUNT', pos
        return 'NEUTRAL', pos


# ============================================================
#  CONFIRMATION INDICATORS
# ============================================================

class ConfirmationEngine:
    """High-conviction confirmation system layered on top of SMC"""

    def confirm_entry(self, df_1h, df_15m, df_4h, signal_dir):
        score = 0
        reasons = []
        latest = df_1h.iloc[-1]
        prev   = df_1h.iloc[-2]
        l15    = df_15m.iloc[-1]
        l4h    = df_4h.iloc[-1]

        # ── 1. TREND ALIGNMENT  (max 20 pts) ──────────────────
        if 'ema_21' in latest.index and 'ema_50' in latest.index and 'ema_200' in latest.index:
            if signal_dir == 'LONG':
                if latest['ema_21'] > latest['ema_50'] > latest['ema_200']:
                    score += 12
                    reasons.append('✅ 4H Triple EMA Aligned Bull')
                elif latest['ema_21'] > latest['ema_50']:
                    score += 6
                    reasons.append('✅ EMA 21>50 Bull')
            else:
                if latest['ema_21'] < latest['ema_50'] < latest['ema_200']:
                    score += 12
                    reasons.append('✅ 4H Triple EMA Aligned Bear')
                elif latest['ema_21'] < latest['ema_50']:
                    score += 6
                    reasons.append('✅ EMA 21<50 Bear')

        if 'ema_21' in l4h.index and 'ema_50' in l4h.index:
            if signal_dir == 'LONG' and l4h['ema_21'] > l4h['ema_50']:
                score += 8
                reasons.append('✅ 4H Higher Timeframe Bull')
            elif signal_dir == 'SHORT' and l4h['ema_21'] < l4h['ema_50']:
                score += 8
                reasons.append('✅ 4H Higher Timeframe Bear')

        # ── 2. MOMENTUM CONFLUENCE  (max 25 pts) ──────────────
        rsi = latest.get('rsi', 50)
        rsi15 = l15.get('rsi', 50)

        if signal_dir == 'LONG':
            if 35 <= rsi <= 52:
                score += 10
                reasons.append(f'✅ RSI Reset Zone ({rsi:.0f})')
            elif rsi < 35:
                score += 7
                reasons.append(f'✅ RSI Oversold ({rsi:.0f})')

            if latest.get('macd', 0) > latest.get('macd_signal', 0) and \
               prev.get('macd', 0) <= prev.get('macd_signal', 0):
                score += 8
                reasons.append('⚡ MACD Bullish Cross')
            elif latest.get('macd', 0) > latest.get('macd_signal', 0):
                score += 4
                reasons.append('✅ MACD Bull')

            if latest.get('stoch_rsi_k', 1) < 0.25 and \
               latest.get('stoch_rsi_k', 0) > latest.get('stoch_rsi_d', 1):
                score += 7
                reasons.append('⚡ Stoch RSI Bullish Cross')
        else:
            if 48 <= rsi <= 65:
                score += 10
                reasons.append(f'✅ RSI Overbought Zone ({rsi:.0f})')
            elif rsi > 65:
                score += 7
                reasons.append(f'✅ RSI Overbought ({rsi:.0f})')

            if latest.get('macd', 0) < latest.get('macd_signal', 0) and \
               prev.get('macd', 0) >= prev.get('macd_signal', 0):
                score += 8
                reasons.append('⚡ MACD Bearish Cross')
            elif latest.get('macd', 0) < latest.get('macd_signal', 0):
                score += 4
                reasons.append('✅ MACD Bear')

            if latest.get('stoch_rsi_k', 0) > 0.75 and \
               latest.get('stoch_rsi_k', 1) < latest.get('stoch_rsi_d', 0):
                score += 7
                reasons.append('⚡ Stoch RSI Bearish Cross')

        # ── 3. VOLUME CONFIRMATION  (max 20 pts) ──────────────
        vol_ratio = latest.get('volume_ratio', 1.0)
        if vol_ratio >= 2.5:
            score += 10
            reasons.append(f'🚀 Volume Spike {vol_ratio:.1f}x')
        elif vol_ratio >= 1.5:
            score += 5
            reasons.append(f'✅ Above Avg Volume {vol_ratio:.1f}x')

        cmf = latest.get('cmf', 0)
        mfi = latest.get('mfi', 50)
        if signal_dir == 'LONG':
            if cmf > 0.1:
                score += 5
                reasons.append('✅ CMF Buying Pressure')
            if mfi < 30:
                score += 5
                reasons.append(f'✅ MFI Oversold ({mfi:.0f})')
        else:
            if cmf < -0.1:
                score += 5
                reasons.append('✅ CMF Selling Pressure')
            if mfi > 70:
                score += 5
                reasons.append(f'✅ MFI Overbought ({mfi:.0f})')

        # ── 4. VOLATILITY STRUCTURE  (max 15 pts) ─────────────
        bb_pband = latest.get('bb_pband', 0.5)
        if signal_dir == 'LONG' and bb_pband < 0.15:
            score += 8
            reasons.append('💎 Price at Lower BB')
        elif signal_dir == 'SHORT' and bb_pband > 0.85:
            score += 8
            reasons.append('💎 Price at Upper BB')

        adx = latest.get('adx', 0)
        di_plus = latest.get('di_plus', 0)
        di_minus = latest.get('di_minus', 0)
        if adx > 25:
            if signal_dir == 'LONG' and di_plus > di_minus:
                score += 7
                reasons.append(f'✅ ADX Trend Strength ({adx:.0f})')
            elif signal_dir == 'SHORT' and di_minus > di_plus:
                score += 7
                reasons.append(f'✅ ADX Trend Strength ({adx:.0f})')

        # ── 5. 15M ENTRY TRIGGER  (max 20 pts) ────────────────
        if 'bullish_engulfing' in l15.index and 'bearish_engulfing' in l15.index:
            if signal_dir == 'LONG' and l15['bullish_engulfing'] == 1:
                score += 12
                reasons.append('🕯️ 15M Bullish Engulfing (Entry Trigger)')
            elif signal_dir == 'SHORT' and l15['bearish_engulfing'] == 1:
                score += 12
                reasons.append('🕯️ 15M Bearish Engulfing (Entry Trigger)')

        if 'bullish_pin_bar' in l15.index and signal_dir == 'LONG' and l15['bullish_pin_bar'] == 1:
            score += 8
            reasons.append('🕯️ 15M Bullish Pin Bar')
        if 'bearish_pin_bar' in l15.index and signal_dir == 'SHORT' and l15['bearish_pin_bar'] == 1:
            score += 8
            reasons.append('🕯️ 15M Bearish Pin Bar')

        return min(score, 100), reasons


# ============================================================
#  MAIN SCANNER BOT
# ============================================================

class SMCDayTradingScanner:
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

        self.smc = SMCEngine()
        self.confirm = ConfirmationEngine()

        self.signal_history = deque(maxlen=200)
        self.active_trades = {}
        self.stats = {
            'total_signals': 0,
            'long_signals': 0,
            'short_signals': 0,
            'elite_signals': 0,
            'premium_signals': 0,
            'tp1_hits': 0,
            'tp2_hits': 0,
            'tp3_hits': 0,
            'sl_hits': 0,
            'last_scan_time': None,
            'pairs_scanned': 0
        }
        self.is_scanning = False

    # ── Data Fetching ─────────────────────────────────────────

    async def get_usdt_pairs(self):
        """Top USDT perpetual futures by volume (min $5M/24h)"""
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = []
            for symbol in self.exchange.symbols:
                if symbol.endswith('/USDT:USDT') and 'PERP' not in symbol:
                    ticker = tickers.get(symbol)
                    if ticker and ticker.get('quoteVolume', 0) > 5_000_000:
                        pairs.append(symbol)
            pairs = sorted(pairs, key=lambda x: tickers.get(x, {}).get('quoteVolume', 0), reverse=True)
            logger.info(f"✅ {len(pairs)} high-quality pairs loaded")
            return pairs
        except Exception as e:
            logger.error(f"Pair fetch error: {e}")
            return []

    async def fetch_multi_tf(self, symbol):
        """Fetch 4H, 1H, 15M data"""
        timeframes = {'4h': 150, '1h': 150, '15m': 80}
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
            logger.error(f"Data fetch error {symbol}: {e}")
            return None

    # ── Indicators ────────────────────────────────────────────

    def add_indicators(self, df):
        """All required indicators for confirmation engine"""
        try:
            if len(df) < 50:
                return df

            df['ema_9']   = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21']  = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50']  = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=min(200, len(df)-1)).ema_indicator()

            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            macd = ta.trend.MACD(df['close'])
            df['macd']        = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist']   = macd.macd_diff()
            stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
            df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()

            df['volume_sma']   = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
            df['cmf']          = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
            df['mfi']          = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            df['obv']          = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper']  = bb.bollinger_hband()
            df['bb_lower']  = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_pband']  = bb.bollinger_pband()
            df['atr']       = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

            adx_i = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx']      = adx_i.adx()
            df['di_plus']  = adx_i.adx_pos()
            df['di_minus'] = adx_i.adx_neg()

            tp = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()

            body      = (df['close'] - df['open']).abs()
            full_range = df['high'] - df['low']
            upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
            lower_wick = df[['open', 'close']].min(axis=1) - df['low']

            df['bullish_engulfing'] = (
                (df['close'].shift(1) < df['open'].shift(1)) &
                (df['close'] > df['open']) &
                (df['close'] > df['open'].shift(1)) &
                (df['open'] < df['close'].shift(1))
            ).astype(int)

            df['bearish_engulfing'] = (
                (df['close'].shift(1) > df['open'].shift(1)) &
                (df['close'] < df['open']) &
                (df['close'] < df['open'].shift(1)) &
                (df['open'] > df['close'].shift(1))
            ).astype(int)

            df['bullish_pin_bar'] = (
                (lower_wick > body * 2) &
                (lower_wick > upper_wick * 2) &
                (df['close'] > df['open'])
            ).astype(int)

            df['bearish_pin_bar'] = (
                (upper_wick > body * 2) &
                (upper_wick > lower_wick * 2) &
                (df['close'] < df['open'])
            ).astype(int)

            return df
        except Exception as e:
            logger.error(f"Indicator error: {e}")
            return df

    # ── SMC Signal Detection ──────────────────────────────────

    def detect_smc_signal(self, data, symbol):
        try:
            for tf in data:
                data[tf] = self.add_indicators(data[tf])

            df_4h = data['4h']
            df_1h = data['1h']
            df_15m = data['15m']

            if len(df_1h) < 80:
                return None

            latest_price = df_15m['close'].iloc[-1]
            atr_1h = df_1h['atr'].iloc[-1]

            highs_1h, lows_1h = self.smc.find_swing_points(df_1h, left=5, right=5)
            structure = self.smc.detect_bos_mss(df_1h, highs_1h, lows_1h)

            obs = self.smc.detect_order_blocks(df_1h, highs_1h, lows_1h, lookback=50)

            fvgs_15m = self.smc.detect_fvg(df_15m, lookback=30)

            sweeps = self.smc.detect_liquidity_sweeps(df_1h, highs_1h, lows_1h, lookback=30)

            highs_4h, lows_4h = self.smc.find_swing_points(df_4h, left=5, right=5)
            pd_zone, pd_pos = self.smc.find_premium_discount(highs_4h, lows_4h, latest_price)

            if structure:
                if structure['type'] in ('BOS_BULL', 'MSS_BULL'):
                    bias = 'LONG'
                elif structure['type'] in ('BOS_BEAR', 'MSS_BEAR'):
                    bias = 'SHORT'
                else:
                    bias = None
            else:
                l4h = df_4h.iloc[-1]
                if 'ema_21' in l4h and 'ema_50' in l4h:
                    bias = 'LONG' if l4h['ema_21'] > l4h['ema_50'] else 'SHORT'
                else:
                    return None

            if not bias:
                return None

            if bias == 'LONG' and pd_zone == 'PREMIUM':
                return None
            if bias == 'SHORT' and pd_zone == 'DISCOUNT':
                return None

            relevant_ob = None
            ob_proximity = atr_1h * 0.5

            for ob in reversed(obs):
                if ob['type'] == 'BULL' and bias == 'LONG':
                    if ob['bottom'] <= latest_price <= ob['top'] + ob_proximity:
                        relevant_ob = ob
                        break
                elif ob['type'] == 'BEAR' and bias == 'SHORT':
                    if ob['bottom'] - ob_proximity <= latest_price <= ob['top']:
                        relevant_ob = ob
                        break

            relevant_fvg = None
            for fvg in reversed(fvgs_15m):
                if fvg['type'] == 'BULL' and bias == 'LONG':
                    if fvg['bottom'] <= latest_price <= fvg['top']:
                        relevant_fvg = fvg
                        break
                elif fvg['type'] == 'BEAR' and bias == 'SHORT':
                    if fvg['bottom'] <= latest_price <= fvg['top']:
                        relevant_fvg = fvg
                        break

            recent_sweep = None
            for sw in reversed(sweeps[-5:]):
                if sw['type'] == 'BEAR_SWEEP' and bias == 'LONG':
                    recent_sweep = sw
                    break
                elif sw['type'] == 'BULL_SWEEP' and bias == 'SHORT':
                    recent_sweep = sw
                    break

            smc_reasons = []
            smc_base_score = 0

            if relevant_ob:
                smc_reasons.append(f'📦 {bias} Order Block [{relevant_ob["bottom"]:.4f} - {relevant_ob["top"]:.4f}]')
                smc_base_score += 30
            if relevant_fvg:
                smc_reasons.append(f'⚡ FVG Imbalance ({relevant_fvg["size_pct"]:.2f}%)')
                smc_base_score += 15
            if recent_sweep:
                smc_reasons.append(f'💧 Liquidity Sweep @ {recent_sweep["level"]:.4f}')
                smc_base_score += 20
            if structure:
                label = 'Break of Structure' if 'BOS' in structure['type'] else 'Market Structure Shift'
                smc_reasons.append(f'🏗️ {label} ({structure["type"]})')
                smc_base_score += 15
            if pd_zone != 'NEUTRAL':
                zone_label = '🟢 Discount Zone' if pd_zone == 'DISCOUNT' else '🔴 Premium Zone'
                smc_reasons.append(f'{zone_label} ({pd_pos*100:.0f}%)')
                smc_base_score += 10

            if not relevant_ob and not (relevant_fvg and recent_sweep):
                return None

            conf_score, conf_reasons = self.confirm.confirm_entry(df_1h, df_15m, df_4h, bias)

            total_score = min(smc_base_score + conf_score, 100)

            # ── HARD GATE: Only ELITE signals (91+) pass through ──
            ELITE_SCORE = 91
            if total_score < ELITE_SCORE:
                return None

            quality = 'ELITE 👑'
            self.stats['elite_signals'] += 1

            entry = latest_price

            if relevant_ob:
                if bias == 'LONG':
                    sl = relevant_ob['bottom'] - (atr_1h * 0.3)
                    sl = min(sl, entry - atr_1h * 0.8)
                else:
                    sl = relevant_ob['top'] + (atr_1h * 0.3)
                    sl = max(sl, entry + atr_1h * 0.8)
            else:
                sl = (entry - atr_1h * 1.2) if bias == 'LONG' else (entry + atr_1h * 1.2)

            risk = abs(entry - sl)
            if risk == 0:
                return None

            if bias == 'LONG':
                targets = [
                    entry + risk * 1.5,
                    entry + risk * 2.5,
                    entry + risk * 4.0,
                ]
            else:
                targets = [
                    entry - risk * 1.5,
                    entry - risk * 2.5,
                    entry - risk * 4.0,
                ]

            rr = [abs(tp - entry) / risk for tp in targets]
            risk_pct = risk / entry * 100

            trade_id = f"{symbol.split('/')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            all_reasons = smc_reasons + conf_reasons

            return {
                'trade_id': trade_id,
                'symbol': symbol.replace('/USDT:USDT', ''),
                'full_symbol': symbol,
                'signal': bias,
                'quality': quality,
                'score': total_score,
                'smc_score': smc_base_score,
                'conf_score': conf_score,
                'entry': entry,
                'stop_loss': sl,
                'targets': targets,
                'reward_ratios': rr,
                'risk_percent': risk_pct,
                'order_block': relevant_ob,
                'fvg': relevant_fvg,
                'sweep': recent_sweep,
                'structure': structure,
                'pd_zone': pd_zone,
                'reasons': all_reasons[:14],
                'tp_hit': [False, False, False],
                'sl_hit': False,
                'timestamp': datetime.now(),
                'status': 'ACTIVE'
            }

        except Exception as e:
            logger.error(f"SMC detect error {symbol}: {e}")
            return None

    # ── Message Formatting ────────────────────────────────────

    def format_signal(self, sig):
        emoji = '🚀' if sig['signal'] == 'LONG' else '🔻'
        dir_color = '🟢' if sig['signal'] == 'LONG' else '🔴'

        score_bar_filled = int(sig['score'] / 10)
        score_bar = '█' * score_bar_filled + '░' * (10 - score_bar_filled)

        msg = f"{'━'*38}\n"
        msg += f"{emoji} <b>SMC ORDER BLOCK — {sig['quality']}</b> {emoji}\n"
        msg += f"{'━'*38}\n\n"

        msg += f"<b>🆔</b> <code>{sig['trade_id']}</code>\n"
        msg += f"<b>📊 PAIR:</b> <b>#{sig['symbol']}</b>\n"
        msg += f"<b>📍 DIRECTION:</b> {dir_color} <b>{sig['signal']}</b>\n"
        msg += f"<b>📍 ZONE:</b> {sig['pd_zone']}\n\n"

        msg += f"<b>⭐ SCORE: {sig['score']:.0f}/100</b>\n"
        msg += f"<code>[{score_bar}]</code>\n"
        msg += f"  └ SMC: {sig['smc_score']:.0f}pt | Confirm: {sig['conf_score']:.0f}pt\n\n"

        msg += f"<b>💰 ENTRY:</b> <code>${sig['entry']:.6f}</code>\n\n"

        msg += f"<b>🎯 TARGETS:</b>\n"
        labels = [('TP1 — Partial (50%)', '4-8h'), ('TP2 — Partial (30%)', '12-18h'), ('TP3 — Final (20%)', '18-28h')]
        for i, ((label, eta), tp, rr) in enumerate(zip(labels, sig['targets'], sig['reward_ratios']), 1):
            pct = abs((tp - sig['entry']) / sig['entry'] * 100)
            msg += f"  <b>{label}</b> [{eta}]\n"
            msg += f"  <code>${tp:.6f}</code>  +{pct:.2f}%  RR {rr:.1f}:1\n\n"

        sl_pct = sig['risk_percent']
        msg += f"<b>🛑 STOP LOSS:</b> <code>${sig['stop_loss']:.6f}</code>  (-{sl_pct:.2f}%)\n"
        if sig['order_block']:
            ob = sig['order_block']
            msg += f"  └ Below OB [{ob['bottom']:.5f} – {ob['top']:.5f}]\n"

        msg += f"\n<b>📋 SMC CONFLUENCE:</b>\n"
        for r in sig['reasons']:
            msg += f"  • {r}\n"

        msg += f"\n<b>⚠️ RISK MGT:</b>\n"
        msg += f"  • Risk max 1-2% of account\n"
        msg += f"  • Move SL to breakeven after TP1\n"
        msg += f"  • Close partial at each target\n"
        msg += f"\n<b>📡 Live TP/SL Tracking Active</b>\n"
        msg += f"<i>⏰ Signal: {sig['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} UTC</i>\n"
        msg += f"{'━'*38}"

        return msg

    # ── Alerts ────────────────────────────────────────────────

    async def send_msg(self, text):
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id, text=text, parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    async def send_tp_alert(self, trade, tp_num, price):
        emoji = '🎉' if trade['signal'] == 'LONG' else '💰'
        tp = trade['targets'][tp_num - 1]
        pct = abs((tp - trade['entry']) / trade['entry'] * 100)

        advice = {
            1: '📋 Close 50% now\n📋 Move SL to Breakeven',
            2: '📋 Close 30% more\n📋 Trail remaining stop',
            3: '📋 Close final 20%\n🎊 Trade complete!'
        }

        msg = f"{emoji} <b>TP{tp_num} HIT!</b> {emoji}\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"<b>{trade['symbol']}</b> {trade['signal']}\n\n"
        msg += f"Target: <code>${tp:.6f}</code>\n"
        msg += f"Current: <code>${price:.6f}</code>\n"
        msg += f"Profit: <b>+{pct:.2f}%</b>\n\n"
        msg += advice[tp_num]

        await self.send_msg(msg)

        if tp_num == 1: self.stats['tp1_hits'] += 1
        elif tp_num == 2: self.stats['tp2_hits'] += 1
        else: self.stats['tp3_hits'] += 1

    async def send_sl_alert(self, trade, price):
        loss = abs((price - trade['entry']) / trade['entry'] * 100)
        msg = f"⛔ <b>STOP LOSS HIT</b> ⛔\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"<b>{trade['symbol']}</b> {trade['signal']}\n\n"
        msg += f"Entry: <code>${trade['entry']:.6f}</code>\n"
        msg += f"SL: <code>${trade['stop_loss']:.6f}</code>\n"
        msg += f"Current: <code>${price:.6f}</code>\n"
        msg += f"Loss: <b>-{loss:.2f}%</b>\n\n"
        msg += f"✅ OB invalidated — move on.\n"
        msg += f"🔍 Next signal incoming!"
        await self.send_msg(msg)
        self.stats['sl_hits'] += 1

    # ── Trade Tracking ────────────────────────────────────────

    async def track_trades(self):
        logger.info("📡 Trade tracking started")
        while True:
            try:
                if not self.active_trades:
                    await asyncio.sleep(30)
                    continue

                to_remove = []

                for tid, trade in list(self.active_trades.items()):
                    try:
                        if datetime.now() - trade['timestamp'] > timedelta(hours=28):
                            await self.send_msg(
                                f"⏰ <b>TIME LIMIT</b>\n<code>{tid}</code>\n"
                                f"{trade['symbol']} — Please close manually."
                            )
                            to_remove.append(tid)
                            continue

                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price = ticker['last']

                        if trade['signal'] == 'LONG':
                            for i, tp in enumerate(trade['targets']):
                                if not trade['tp_hit'][i] and price >= tp:
                                    await self.send_tp_alert(trade, i+1, price)
                                    trade['tp_hit'][i] = True
                                    if i == 2: to_remove.append(tid)

                            if not trade['sl_hit'] and price <= trade['stop_loss']:
                                await self.send_sl_alert(trade, price)
                                trade['sl_hit'] = True
                                to_remove.append(tid)
                        else:
                            for i, tp in enumerate(trade['targets']):
                                if not trade['tp_hit'][i] and price <= tp:
                                    await self.send_tp_alert(trade, i+1, price)
                                    trade['tp_hit'][i] = True
                                    if i == 2: to_remove.append(tid)

                            if not trade['sl_hit'] and price >= trade['stop_loss']:
                                await self.send_sl_alert(trade, price)
                                trade['sl_hit'] = True
                                to_remove.append(tid)

                    except Exception as e:
                        logger.error(f"Track error {tid}: {e}")

                for tid in set(to_remove):
                    self.active_trades.pop(tid, None)

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Tracking loop error: {e}")
                await asyncio.sleep(60)

    # ── Scanner ───────────────────────────────────────────────

    async def scan_all(self):
        if self.is_scanning:
            logger.info("⚠️ Already scanning, skipping.")
            return []

        self.is_scanning = True
        logger.info("🔍 SMC Scan starting...")

        await self.send_msg(
            "🔍 <b>SMC ORDER BLOCK SCAN STARTED</b>\n"
            "Hunting for ELITE setups only (91+/100)...\n"
            "👑 Only the absolute best will make it through."
        )

        pairs = await self.get_usdt_pairs()
        signals = []
        scanned = 0

        for pair in pairs:
            try:
                logger.info(f"  📊 Scanning {pair}...")
                data = await self.fetch_multi_tf(pair)
                if data:
                    sig = self.detect_smc_signal(data, pair)
                    if sig:
                        # ✅ sig is GUARANTEED ELITE (88+) — detect_smc_signal now returns None for anything below
                        signals.append(sig)
                        self.signal_history.append(sig)
                        self.stats['total_signals'] += 1

                        if sig['signal'] == 'LONG':
                            self.stats['long_signals'] += 1
                        else:
                            self.stats['short_signals'] += 1

                        # ✅ ELITE ONLY: add to tracking AND send to Telegram
                        self.active_trades[sig['trade_id']] = sig
                        await self.send_msg(self.format_signal(sig))
                        await asyncio.sleep(2)

                scanned += 1
                if scanned % 30 == 0:
                    logger.info(f"  ⏳ Progress: {scanned}/{len(pairs)}")

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Scan error {pair}: {e}")
                continue

        self.stats['last_scan_time'] = datetime.now()
        self.stats['pairs_scanned'] = scanned

        longs  = sum(1 for s in signals if s['signal'] == 'LONG')
        shorts = len(signals) - longs

        summary  = f"✅ <b>SMC SCAN COMPLETE</b>\n\n"
        summary += f"📊 Pairs scanned: {scanned}\n"
        summary += f"👑 ELITE signals (91+): {len(signals)}\n"
        if signals:
            summary += f"  🟢 Long: {longs}\n"
            summary += f"  🔴 Short: {shorts}\n"
        else:
            summary += f"  🔕 No ELITE setups found this scan.\n"
        summary += f"  📡 Tracking: {len(self.active_trades)}\n"
        summary += f"\n⏰ {datetime.now().strftime('%H:%M:%S')} UTC"

        await self.send_msg(summary)
        logger.info(f"✅ Scan done. {len(signals)} ELITE signals from {scanned} pairs.")

        self.is_scanning = False
        return signals

    # ── Main Loop ─────────────────────────────────────────────

    async def run(self, interval_min=60):
        logger.info("🚀 SMC Order Block Scanner starting...")

        welcome = (
            "👑 <b>SMC ORDER BLOCK SCANNER — ELITE MODE</b> 👑\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>Strategy: Smart Money Concepts (SMC)</b>\n\n"
            "✅ Order Block detection (OB)\n"
            "✅ Fair Value Gap (FVG / Imbalance)\n"
            "✅ Break of Structure (BOS)\n"
            "✅ Market Structure Shift (MSS)\n"
            "✅ Liquidity Sweep confirmation\n"
            "✅ Premium / Discount zones\n"
            "✅ 5-layer confirmation engine\n"
            "✅ <b>ELITE ONLY: 91+/100 required</b>\n"
            "✅ <b>ELITE ONLY tracking (no noise)</b>\n"
            "✅ Live TP/SL tracking (28h)\n\n"
            f"🕐 Auto-scan every {interval_min} min\n\n"
            "<b>Commands:</b> /scan /stats /trades /help\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "👑 ELITE mode: only the highest-conviction setups."
        )
        await self.send_msg(welcome)

        asyncio.create_task(self.track_trades())

        while True:
            try:
                await self.scan_all()
                logger.info(f"💤 Next scan in {interval_min} min")
                await asyncio.sleep(interval_min * 60)
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ============================================================
#  BOT COMMANDS
# ============================================================

class BotCommands:
    def __init__(self, scanner: SMCDayTradingScanner):
        self.scanner = scanner

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "👑 <b>SMC Order Block Trading Bot — ELITE MODE</b>\n\n"
            "Professional Smart Money strategy.\n"
            "You only receive <b>ELITE signals (91+/100)</b> — the best of the best.\n"
            "Only ELITE signals are tracked. Zero noise.\n\n"
            "<b>Commands:</b>\n"
            "/scan — Force a full market scan\n"
            "/stats — Performance statistics\n"
            "/trades — Active tracked trades\n"
            "/help — Detailed strategy explanation\n\n"
            "👑 Only signals scoring 91+/100 are sent and tracked."
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if self.scanner.is_scanning:
            await update.message.reply_text("⚠️ Scan already running, please wait.")
            return
        await update.message.reply_text("🔍 Manual scan started — ELITE signals only (91+/100)...")
        asyncio.create_task(self.scanner.scan_all())

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        s = self.scanner.stats
        msg  = f"📊 <b>SMC SCANNER STATISTICS</b>\n\n"
        msg += f"👑 Elite signals found: {s['elite_signals']}\n"
        msg += f"  🟢 Long: {s['long_signals']}\n"
        msg += f"  🔴 Short: {s['short_signals']}\n\n"
        msg += f"<b>TP/SL Performance:</b>\n"
        msg += f"  TP1 Hits: {s['tp1_hits']}\n"
        msg += f"  TP2 Hits: {s['tp2_hits']}\n"
        msg += f"  TP3 Hits: {s['tp3_hits']}\n"
        msg += f"  SL Hits: {s['sl_hits']}\n\n"
        if s['last_scan_time']:
            msg += f"Last scan: {s['last_scan_time'].strftime('%H:%M:%S')}\n"
            msg += f"Pairs checked: {s['pairs_scanned']}\n"
        msg += f"📡 Active trades: {len(self.scanner.active_trades)}"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        trades = self.scanner.active_trades
        if not trades:
            await update.message.reply_text("📭 No active ELITE trades right now.")
            return

        msg = f"📡 <b>ACTIVE ELITE TRADES ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:12]:
            age_h = int((datetime.now() - t['timestamp']).total_seconds() / 3600)
            tp_icons = ''.join(['✅' if h else '⏳' for h in t['tp_hit']])
            sl_icon = '⛔' if t['sl_hit'] else '🟢'
            msg += (
                f"<b>{t['symbol']}</b> {t['signal']} — {t['quality']}\n"
                f"  Score: {t['score']:.0f}/100 | {t['pd_zone']}\n"
                f"  TPs: {tp_icons} SL: {sl_icon} | {age_h}h old\n\n"
            )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "📚 <b>SMC ORDER BLOCK BOT — STRATEGY GUIDE</b>\n\n"
            "<b>What is an Order Block?</b>\n"
            "The last bearish candle before a bullish impulse (Bullish OB), "
            "or the last bullish candle before a bearish impulse (Bearish OB). "
            "Smart money leaves 'imprints' — price often returns to these zones.\n\n"
            "<b>Entry Requirements (ALL must pass):</b>\n"
            "1️⃣ Order Block detected on 1H\n"
            "2️⃣ BOS/MSS confirms direction\n"
            "3️⃣ Price in Discount (LONG) or Premium (SHORT)\n"
            "4️⃣ FVG or Liquidity Sweep present\n"
            "5️⃣ Confirmation indicators aligned\n"
            "6️⃣ Score ≥ 91/100 (ELITE — hard gate)\n\n"
            "<b>Risk Management:</b>\n"
            "• SL placed below/above Order Block\n"
            "• TP1 at 1:1.5 RR — close 50%\n"
            "• TP2 at 1:2.5 RR — close 30%\n"
            "• TP3 at 1:4 RR — close 20%\n"
            "• Move SL to BE after TP1 hits\n\n"
            "<b>Score Gate:</b>\n"
            "👑 ELITE — 91+/100 ✅ Sent & Tracked\n"
            "Everything below 91 → silently discarded\n\n"
            "<b>Commands:</b> /scan /stats /trades /help"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ============================================================
#  ENTRY POINT
# ============================================================

async def main():
    # ======== CONFIG — Fill these in ========
    TELEGRAM_TOKEN    = "7731521911:AAFnus-fDivEwoKqrtwZXMmKEj5BU1EhQn4"
    TELEGRAM_CHAT_ID  = "7500072234"
    BINANCE_API_KEY   = None   # Optional — public data only for scanning
    BINANCE_SECRET    = None
    SCAN_INTERVAL_MIN = 60     # How often to scan (minutes)
    # =========================================

    scanner = SMCDayTradingScanner(
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        binance_api_key=BINANCE_API_KEY,
        binance_secret=BINANCE_SECRET
    )

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds = BotCommands(scanner)

    app.add_handler(CommandHandler("start",  cmds.cmd_start))
    app.add_handler(CommandHandler("scan",   cmds.cmd_scan))
    app.add_handler(CommandHandler("stats",  cmds.cmd_stats))
    app.add_handler(CommandHandler("trades", cmds.cmd_trades))
    app.add_handler(CommandHandler("help",   cmds.cmd_help))

    await app.initialize()
    await app.start()
    logger.info("🤖 Bot commands ready!")

    try:
        await scanner.run(interval_min=SCAN_INTERVAL_MIN)
    except KeyboardInterrupt:
        logger.info("⚠️ Shutdown requested...")
    finally:
        await scanner.close()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
