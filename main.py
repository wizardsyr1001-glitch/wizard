import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# PASTE YOUR ORIGINAL SCANNER CLASS HERE (or import it)
# OR just run this file standalone — it re-implements all logic
# ============================================================

class Backtester:
    """
    Backtests the AdvancedDayTradingScanner strategy on real historical data.

    How it works:
    1. Fetches the maximum available 1h/4h/15m candles for each symbol
    2. Walks forward candle-by-candle (no lookahead bias)
    3. Fires the SAME detect_signal() logic you use live
    4. Simulates TP1/TP2/TP3/SL hits with realistic price action
    5. Sends a full HTML-style Telegram report

    Config at the bottom of this file — set your symbols, date range, etc.
    """

    def __init__(self, binance_api_key=None, binance_secret=None, telegram_token=None, telegram_chat_id=None):
        self.exchange = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.telegram_token = telegram_token
        self.chat_id = telegram_chat_id

        # ── Backtest params ──────────────────────────────────
        self.LOOKBACK_DAYS   = 60          # How many days of history to test
        self.TOP_N_PAIRS     = 30          # Test top N pairs by volume
        self.MIN_VOLUME_USDT = 5_000_000   # Minimum 24h volume filter
        self.ATR_SL_MULT     = 1.5         # Must match your live bot
        self.ATR_TP_MULTS    = [1.0, 2.0, 3.5]
        self.MAX_HOLD_HOURS  = 24          # Auto-close after N hours
        self.COMMISSION_PCT  = 0.04        # 0.04% taker fee each side
        # ────────────────────────────────────────────────────

        self.results = []
        self.equity_curve = [10_000]       # Start with $10k notional

    # ──────────────────────────────────────────────
    # HELPERS (duplicated from your scanner)
    # ──────────────────────────────────────────────

    def _calculate_supertrend(self, df, period=10, multiplier=3):
        try:
            hl2 = (df['high'] + df['low']) / 2
            atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            supertrend = [0.0] * len(df)
            for i in range(1, len(df)):
                if df['close'].iloc[i] > upper_band.iloc[i - 1]:
                    supertrend[i] = lower_band.iloc[i]
                elif df['close'].iloc[i] < lower_band.iloc[i - 1]:
                    supertrend[i] = upper_band.iloc[i]
                else:
                    supertrend[i] = supertrend[i - 1]
            return pd.Series(supertrend, index=df.index)
        except:
            return pd.Series([0.0] * len(df), index=df.index)

    def _add_indicators(self, df):
        if len(df) < 30:
            return df
        try:
            df['ema_9']  = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=min(50, len(df)-1)).ema_indicator()
            df['supertrend'] = self._calculate_supertrend(df)

            psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
            df['psar'] = psar.psar()

            df['rsi']   = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_6'] = ta.momentum.RSIIndicator(df['close'], window=6).rsi()

            srsi = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi_k'] = srsi.stochrsi_k()
            df['stoch_rsi_d'] = srsi.stochrsi_d()

            macd = ta.trend.MACD(df['close'])
            df['macd']        = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist']   = macd.macd_diff()

            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            df['roc']        = ta.momentum.ROCIndicator(df['close'], window=12).roc()
            df['uo']         = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()

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
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)

            df['obv']     = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            df['mfi']     = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            df['ad']      = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
            df['cmf']     = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()

            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx']      = adx.adx()
            df['di_plus']  = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()
            df['cci']      = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()

            aroon = ta.trend.AroonIndicator(df['high'], df['low'])
            df['aroon_up']   = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            df['aroon_ind']  = df['aroon_up'] - df['aroon_down']

            tp = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'] = df['vwap'].fillna(df['close'])

            df['bullish_candle']    = (df['close'] > df['open']).astype(int)
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
                (df['low'] < df['low'].shift(1)) & (df['rsi'] > df['rsi'].shift(1))
            ).astype(int)
            df['bearish_divergence'] = (
                (df['high'] > df['high'].shift(1)) & (df['rsi'] < df['rsi'].shift(1))
            ).astype(int)
        except Exception as e:
            logger.error(f"Indicator error: {e}")
        return df

    def _detect_signal_at(self, df_1h_slice, df_4h_slice, df_15m_slice):
        """
        Same scoring as your live bot — returns signal dict or None.
        Runs on the LAST row of each slice (walk-forward).
        """
        try:
            df_1h  = self._add_indicators(df_1h_slice.copy())
            df_4h  = self._add_indicators(df_4h_slice.copy())
            df_15m = self._add_indicators(df_15m_slice.copy())

            if len(df_1h) < 50:
                return None

            lat1  = df_1h.iloc[-1]
            prev1 = df_1h.iloc[-2]
            lat4  = df_4h.iloc[-1]
            lat15 = df_15m.iloc[-1]

            required = ['ema_9','ema_21','ema_50','rsi','macd','macd_signal',
                        'vwap','bb_pband','atr','stoch_rsi_k','stoch_rsi_d']
            for c in required:
                if c not in lat1.index or pd.isna(lat1[c]):
                    return None

            # ── Volume spike ──────────────────────────────────
            vol_avg = df_1h['volume'].iloc[-20:].mean()
            vol_ratio = lat1['volume'] / vol_avg if vol_avg > 0 else 1.0
            volume_spike = vol_ratio > 2.5

            long_score  = 0.0
            short_score = 0.0
            max_score   = 35

            # TREND (6)
            if lat4['ema_9'] > lat4['ema_21'] > lat4['ema_50']:
                long_score += 3
            elif lat4['ema_9'] < lat4['ema_21'] < lat4['ema_50']:
                short_score += 3

            if lat1['ema_9'] > lat1['ema_21']:
                long_score += 2
            elif lat1['ema_9'] < lat1['ema_21']:
                short_score += 2

            if lat1['close'] > lat1['supertrend']:
                long_score += 1
            elif lat1['close'] < lat1['supertrend']:
                short_score += 1

            # MOMENTUM (9)
            rsi = lat1['rsi']
            if rsi < 30:   long_score += 3.5
            elif rsi < 40: long_score += 2
            elif rsi <= 50: long_score += 1

            if rsi > 70:   short_score += 3.5
            elif rsi > 60: short_score += 2
            elif rsi >= 50: short_score += 1

            if lat1['stoch_rsi_k'] < 0.2 and lat1['stoch_rsi_k'] > lat1['stoch_rsi_d']:
                long_score += 2
            elif lat1['stoch_rsi_k'] > 0.8 and lat1['stoch_rsi_k'] < lat1['stoch_rsi_d']:
                short_score += 2

            if lat1['macd'] > lat1['macd_signal'] and prev1['macd'] <= prev1['macd_signal']:
                long_score += 2.5
            elif lat1['macd'] < lat1['macd_signal'] and prev1['macd'] >= prev1['macd_signal']:
                short_score += 2.5

            if lat1['uo'] < 30:  long_score += 1.5
            elif lat1['uo'] > 70: short_score += 1.5

            # VOLUME (5)
            if volume_spike:
                if lat1['close'] > prev1['close']: long_score += 3
                else:                               short_score += 3

            if lat1['mfi'] < 20:  long_score += 1.5
            elif lat1['mfi'] > 80: short_score += 1.5

            if lat1['cmf'] > 0.15:   long_score += 1
            elif lat1['cmf'] < -0.15: short_score += 1

            obv_trend = df_1h['obv'].iloc[-5:].diff().mean()
            if obv_trend > 0 and lat1['obv'] > lat1['obv_ema']:   long_score += 0.5
            elif obv_trend < 0 and lat1['obv'] < lat1['obv_ema']: short_score += 0.5

            # VOLATILITY (6)
            if lat1['bb_pband'] < 0.1:   long_score += 2.5
            elif lat1['bb_pband'] > 0.9:  short_score += 2.5

            if lat1['cci'] < -150:  long_score += 1.5
            elif lat1['cci'] > 150:  short_score += 1.5

            if lat1['williams_r'] < -85:  long_score += 1
            elif lat1['williams_r'] > -15: short_score += 1

            if lat1['close'] < lat1['vwap'] * 0.98:   long_score += 1
            elif lat1['close'] > lat1['vwap'] * 1.02:  short_score += 1

            # TREND STRENGTH (4)
            if lat1['adx'] > 30:
                if lat1['di_plus'] > lat1['di_minus']: long_score += 2
                else:                                   short_score += 2
            elif lat1['adx'] > 25:
                if lat1['di_plus'] > lat1['di_minus']: long_score += 1
                else:                                   short_score += 1

            if lat1['aroon_ind'] > 50:   long_score += 1
            elif lat1['aroon_ind'] < -50: short_score += 1

            if lat1['roc'] > 3:  long_score += 1
            elif lat1['roc'] < -3: short_score += 1

            # DIVERGENCE & PATTERNS (3)
            if lat1['bullish_divergence'] == 1:  long_score += 2
            elif lat1['bearish_divergence'] == 1: short_score += 2

            if lat15['bullish_engulfing'] == 1:  long_score += 1.5
            elif lat15['bearish_engulfing'] == 1: short_score += 1.5

            # HTF CONFIRMATION (2)
            if lat4['close'] > lat4['vwap']: long_score += 1
            else:                             short_score += 1

            if lat4['rsi'] < 50:  long_score += 1
            elif lat4['rsi'] > 50: short_score += 1

            # ── Determine signal ─────────────────────────────
            threshold = max_score * 0.51
            signal = quality = None

            if long_score > short_score and long_score >= threshold:
                signal = 'LONG';  score = long_score
                if long_score >= max_score * 0.71:  quality = 'PREMIUM'
                elif long_score >= max_score * 0.60: quality = 'HIGH'
                else:                                quality = 'GOOD'
            elif short_score > long_score and short_score >= threshold:
                signal = 'SHORT'; score = short_score
                if short_score >= max_score * 0.71:  quality = 'PREMIUM'
                elif short_score >= max_score * 0.60: quality = 'HIGH'
                else:                                quality = 'GOOD'

            if signal:
                entry = lat15['close']
                atr   = lat1['atr']
                if signal == 'LONG':
                    sl  = entry - atr * self.ATR_SL_MULT
                    tps = [entry + atr * m for m in self.ATR_TP_MULTS]
                else:
                    sl  = entry + atr * self.ATR_SL_MULT
                    tps = [entry - atr * m for m in self.ATR_TP_MULTS]

                return {
                    'signal': signal, 'quality': quality,
                    'score': score, 'max_score': max_score,
                    'score_pct': score / max_score * 100,
                    'entry': entry, 'sl': sl, 'tps': tps,
                    'atr': atr,
                    'risk_pct': abs(sl - entry) / entry * 100,
                    'signal_time': df_1h.iloc[-1]['timestamp']
                }
        except Exception as e:
            logger.error(f"Signal detect error: {e}")
        return None

    # ──────────────────────────────────────────────
    # DATA FETCHING
    # ──────────────────────────────────────────────

    async def _fetch_full_history(self, symbol, timeframe, days):
        """Fetch full OHLCV history for the backtest window"""
        try:
            since = int((datetime.utcnow() - timedelta(days=days + 10)).timestamp() * 1000)
            all_ohlcv = []
            while True:
                batch = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not batch:
                    break
                all_ohlcv.extend(batch)
                since = batch[-1][0] + 1
                if len(batch) < 1000:
                    break
                await asyncio.sleep(0.15)

            df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
            cutoff = datetime.utcnow() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff].reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Fetch history {symbol} {timeframe}: {e}")
            return None

    async def _get_top_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = []
            for sym in self.exchange.symbols:
                if sym.endswith('/USDT:USDT') and 'PERP' not in sym:
                    t = tickers.get(sym, {})
                    if t.get('quoteVolume', 0) > self.MIN_VOLUME_USDT:
                        pairs.append((sym, t['quoteVolume']))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [p[0] for p in pairs[:self.TOP_N_PAIRS]]
        except Exception as e:
            logger.error(f"Get pairs: {e}")
            return []

    # ──────────────────────────────────────────────
    # WALK-FORWARD SIMULATION
    # ──────────────────────────────────────────────

    def _simulate_trade(self, signal, future_candles_1h):
        """
        Given a signal, walk forward through future 1h candles
        and check if TP1/TP2/TP3 or SL is hit first.
        Returns outcome dict.
        """
        entry   = signal['entry']
        sl      = signal['sl']
        tps     = signal['tps']
        direction = signal['signal']

        tp_hit = [False, False, False]
        sl_hit = False
        exit_price = None
        exit_reason = None
        hours_held  = 0

        for i, row in future_candles_1h.iterrows():
            hours_held += 1
            hi = row['high']
            lo = row['low']

            if direction == 'LONG':
                # Check SL first (conservative)
                if lo <= sl:
                    sl_hit = True
                    exit_price  = sl
                    exit_reason = 'SL'
                    break
                for j, tp in enumerate(tps):
                    if not tp_hit[j] and hi >= tp:
                        tp_hit[j] = True
                if tp_hit[2]:
                    exit_price  = tps[2]
                    exit_reason = 'TP3'
                    break
            else:  # SHORT
                if hi >= sl:
                    sl_hit = True
                    exit_price  = sl
                    exit_reason = 'SL'
                    break
                for j, tp in enumerate(tps):
                    if not tp_hit[j] and lo <= tp:
                        tp_hit[j] = True
                if tp_hit[2]:
                    exit_price  = tps[2]
                    exit_reason = 'TP3'
                    break

            if hours_held >= self.MAX_HOLD_HOURS:
                exit_price  = row['close']
                exit_reason = 'TIMEOUT'
                break

        if exit_price is None:
            # Ran out of candles
            exit_price  = future_candles_1h.iloc[-1]['close'] if len(future_candles_1h) else entry
            exit_reason = 'INCOMPLETE'

        # ── P&L calculation ─────────────────────────────────
        # Weighted partial exit: TP1=50%, TP2=30%, TP3=20%
        weights    = [0.50, 0.30, 0.20]
        pnl_pct    = 0.0
        tp_reached = sum(tp_hit)

        if sl_hit:
            pnl_pct = -signal['risk_pct'] - self.COMMISSION_PCT * 2
        elif exit_reason in ('TIMEOUT', 'INCOMPLETE'):
            raw = (exit_price - entry) / entry * 100
            if direction == 'SHORT': raw = -raw
            pnl_pct = raw - self.COMMISSION_PCT * 2
        else:
            # Partial takes
            for j in range(3):
                if tp_hit[j]:
                    raw = abs(tps[j] - entry) / entry * 100
                    pnl_pct += raw * weights[j]
            # Any remaining position exits at last exit_price
            filled_weight = sum(weights[:tp_reached]) if tp_reached else 0
            remaining = 1.0 - filled_weight
            if remaining > 0:
                last_raw = (exit_price - entry) / entry * 100
                if direction == 'SHORT': last_raw = -last_raw
                pnl_pct += last_raw * remaining
            pnl_pct -= self.COMMISSION_PCT * 2

        win = pnl_pct > 0

        return {
            'exit_reason': exit_reason,
            'tp_hit': tp_hit,
            'tp_reached': tp_reached,
            'sl_hit': sl_hit,
            'exit_price': exit_price,
            'hours_held': hours_held,
            'pnl_pct': round(pnl_pct, 4),
            'win': win
        }

    def _walk_forward(self, symbol, df_1h, df_4h, df_15m):
        """Walk forward through all 1h candles and fire signals"""
        trades = []
        open_trade_until = None  # Prevent overlapping trades

        # Need at least 100 candles warm-up
        warm = 100
        if len(df_1h) < warm + 20:
            return trades

        for i in range(warm, len(df_1h) - 1):
            ts = df_1h.iloc[i]['timestamp']

            # Skip if we're still inside a previous trade
            if open_trade_until and ts < open_trade_until:
                continue

            # Slice historical context (no future data!)
            slice_1h  = df_1h.iloc[:i+1].reset_index(drop=True)

            # Align 4h and 15m to same timestamp
            slice_4h  = df_4h[df_4h['timestamp'] <= ts].tail(100).reset_index(drop=True)
            slice_15m = df_15m[df_15m['timestamp'] <= ts].tail(50).reset_index(drop=True)

            if len(slice_4h) < 50 or len(slice_15m) < 20:
                continue

            sig = self._detect_signal_at(slice_1h, slice_4h, slice_15m)
            if not sig:
                continue

            # Simulate on future candles
            future = df_1h.iloc[i+1:i+1+self.MAX_HOLD_HOURS].reset_index(drop=True)
            if len(future) == 0:
                break

            outcome = self._simulate_trade(sig, future)

            trade = {
                'symbol': symbol.replace('/USDT:USDT', ''),
                'signal': sig['signal'],
                'quality': sig['quality'],
                'score_pct': round(sig['score_pct'], 1),
                'entry': sig['entry'],
                'sl': sig['sl'],
                'tp1': sig['tps'][0],
                'tp2': sig['tps'][1],
                'tp3': sig['tps'][2],
                'signal_time': ts.strftime('%Y-%m-%d %H:%M'),
                **outcome
            }
            trades.append(trade)

            # Block new signals during trade hold
            open_trade_until = ts + timedelta(hours=outcome['hours_held'])

            logger.info(f"  📊 {trade['symbol']} {trade['signal']} {trade['signal_time']} → "
                        f"{trade['exit_reason']} {trade['pnl_pct']:+.2f}%")

        return trades

    # ──────────────────────────────────────────────
    # STATS + REPORT
    # ──────────────────────────────────────────────

    def _compute_stats(self, all_trades):
        if not all_trades:
            return {}

        df = pd.DataFrame(all_trades)

        total  = len(df)
        wins   = df['win'].sum()
        losses = total - wins
        wr     = wins / total * 100

        avg_win  = df[df['win']]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = df[~df['win']]['pnl_pct'].mean() if losses > 0 else 0
        rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        total_pnl = df['pnl_pct'].sum()
        best      = df['pnl_pct'].max()
        worst     = df['pnl_pct'].min()
        avg_hold  = df['hours_held'].mean()

        # Expectancy per trade
        expectancy = (wr/100 * avg_win) + ((1 - wr/100) * avg_loss)

        # TP breakdown
        tp1_rate = df['tp_hit'].apply(lambda x: x[0]).mean() * 100
        tp2_rate = df['tp_hit'].apply(lambda x: x[1]).mean() * 100
        tp3_rate = df['tp_hit'].apply(lambda x: x[2]).mean() * 100
        sl_rate  = df['sl_hit'].mean() * 100

        # By quality
        quality_stats = {}
        for q in ['PREMIUM', 'HIGH', 'GOOD']:
            sub = df[df['quality'] == q]
            if len(sub) > 0:
                quality_stats[q] = {
                    'count': len(sub),
                    'wr': round(sub['win'].mean() * 100, 1),
                    'avg_pnl': round(sub['pnl_pct'].mean(), 2)
                }

        # By direction
        dir_stats = {}
        for d in ['LONG', 'SHORT']:
            sub = df[df['signal'] == d]
            if len(sub) > 0:
                dir_stats[d] = {
                    'count': len(sub),
                    'wr': round(sub['win'].mean() * 100, 1),
                    'avg_pnl': round(sub['pnl_pct'].mean(), 2)
                }

        # Consecutive losses (max drawdown proxy)
        max_consec_loss = 0
        cur_loss = 0
        for w in df['win']:
            if not w:
                cur_loss += 1
                max_consec_loss = max(max_consec_loss, cur_loss)
            else:
                cur_loss = 0

        # Exit reason breakdown
        exit_counts = df['exit_reason'].value_counts().to_dict()

        # Equity curve (cumulative sum assuming equal % risk per trade)
        equity = 10_000
        equity_curve = [equity]
        for pnl in df['pnl_pct']:
            equity *= (1 + pnl / 100)
            equity_curve.append(round(equity, 2))

        return {
            'total': total, 'wins': int(wins), 'losses': int(losses),
            'win_rate': round(wr, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'rr': round(rr, 2),
            'total_pnl': round(total_pnl, 2),
            'best_trade': round(best, 2), 'worst_trade': round(worst, 2),
            'avg_hold_hours': round(avg_hold, 1),
            'expectancy': round(expectancy, 3),
            'tp1_rate': round(tp1_rate, 1),
            'tp2_rate': round(tp2_rate, 1),
            'tp3_rate': round(tp3_rate, 1),
            'sl_rate': round(sl_rate, 1),
            'quality_stats': quality_stats,
            'dir_stats': dir_stats,
            'max_consec_losses': max_consec_loss,
            'exit_counts': exit_counts,
            'final_equity': round(equity_curve[-1], 2),
            'equity_curve': equity_curve,
            'df': df
        }

    def _format_telegram_report(self, stats, pairs_tested, days):
        """Format a beautiful Telegram report"""
        s = stats
        wr_emoji = "🟢" if s['win_rate'] >= 55 else ("🟡" if s['win_rate'] >= 45 else "🔴")

        msg  = "═" * 38 + "\n"
        msg += f"📊 <b>BACKTEST REPORT</b> 📊\n"
        msg += "═" * 38 + "\n\n"
        msg += f"🗓 <b>Period:</b> Last {days} days\n"
        msg += f"📦 <b>Pairs Tested:</b> {pairs_tested}\n"
        msg += f"🔢 <b>Total Trades:</b> {s['total']}\n\n"

        msg += "━" * 38 + "\n"
        msg += f"<b>📈 PERFORMANCE</b>\n"
        msg += "━" * 38 + "\n"
        msg += f"{wr_emoji} <b>Win Rate:</b> {s['win_rate']}%  ({s['wins']}W / {s['losses']}L)\n"
        msg += f"💰 <b>Avg Win:</b>  +{s['avg_win']}%\n"
        msg += f"💸 <b>Avg Loss:</b> {s['avg_loss']}%\n"
        msg += f"⚖️ <b>Risk/Reward:</b> {s['rr']}:1\n"
        msg += f"🎯 <b>Expectancy:</b> {s['expectancy']}% per trade\n"
        msg += f"📊 <b>Total PnL:</b> {s['total_pnl']:+.1f}% cumulative\n"
        msg += f"💵 <b>$10k → ${s['final_equity']:,.0f}</b>\n\n"

        msg += "━" * 38 + "\n"
        msg += "<b>🎯 TARGET / SL HIT RATES</b>\n"
        msg += "━" * 38 + "\n"
        msg += f"  TP1 Hit: {s['tp1_rate']}%\n"
        msg += f"  TP2 Hit: {s['tp2_rate']}%\n"
        msg += f"  TP3 Hit: {s['tp3_rate']}%\n"
        msg += f"  SL Hit:  {s['sl_rate']}%\n"
        msg += f"  Max Consec Losses: {s['max_consec_losses']}\n"
        msg += f"  Avg Hold: {s['avg_hold_hours']}h\n\n"

        if s['quality_stats']:
            msg += "━" * 38 + "\n"
            msg += "<b>💎 BY SIGNAL QUALITY</b>\n"
            msg += "━" * 38 + "\n"
            for q, qs in s['quality_stats'].items():
                emoji = "💎" if q == 'PREMIUM' else ("🔥" if q == 'HIGH' else "✅")
                msg += f"  {emoji} {q}: {qs['count']} trades | WR {qs['wr']}% | Avg {qs['avg_pnl']:+.2f}%\n"
            msg += "\n"

        if s['dir_stats']:
            msg += "━" * 38 + "\n"
            msg += "<b>📍 LONG vs SHORT</b>\n"
            msg += "━" * 38 + "\n"
            for d, ds in s['dir_stats'].items():
                emoji = "🟢" if d == 'LONG' else "🔴"
                msg += f"  {emoji} {d}: {ds['count']} trades | WR {ds['wr']}% | Avg {ds['avg_pnl']:+.2f}%\n"
            msg += "\n"

        msg += "━" * 38 + "\n"
        msg += "<b>🚪 EXIT REASONS</b>\n"
        msg += "━" * 38 + "\n"
        for reason, count in s['exit_counts'].items():
            pct = count / s['total'] * 100
            msg += f"  {reason}: {count} ({pct:.0f}%)\n"
        msg += "\n"

        msg += f"<b>🏆 Best Trade:</b>  +{s['best_trade']}%\n"
        msg += f"<b>💀 Worst Trade:</b> {s['worst_trade']}%\n\n"

        # Verdict
        msg += "═" * 38 + "\n"
        msg += "<b>🔬 VERDICT</b>\n"
        msg += "═" * 38 + "\n"
        if s['win_rate'] >= 60 and s['rr'] >= 1.5:
            msg += "✅ <b>STRATEGY LOOKS SOLID</b>\nGood win rate + favorable RR.\nConsider running live with small size.\n"
        elif s['win_rate'] >= 50 and s['rr'] >= 2.0:
            msg += "🟡 <b>DECENT — RR SAVES IT</b>\nWin rate modest but high RR.\nConsider filtering to PREMIUM only.\n"
        elif s['win_rate'] < 45:
            msg += "🔴 <b>NEEDS IMPROVEMENT</b>\nWin rate too low.\nSuggested fixes: raise threshold, PREMIUM only, add trend filter.\n"
        else:
            msg += "🟡 <b>MARGINAL EDGE</b>\nNeeds optimization.\nTest PREMIUM-only signals.\n"

        msg += f"\n⏰ Backtest run: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        return msg

    async def _send_telegram(self, msg):
        if not self.telegram_token or not self.chat_id:
            print(msg)
            return
        try:
            from telegram import Bot
            from telegram.constants import ParseMode
            bot = Bot(token=self.telegram_token)
            # Split if > 4000 chars
            for chunk in [msg[i:i+4000] for i in range(0, len(msg), 4000)]:
                await bot.send_message(chat_id=self.chat_id, text=chunk, parse_mode=ParseMode.HTML)
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Telegram send: {e}")
            print(msg)

    # ──────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ──────────────────────────────────────────────

    async def run(self):
        logger.info("🚀 Starting backtest...")
        await self._send_telegram(f"⏳ <b>Backtest started</b>\n{self.TOP_N_PAIRS} pairs × {self.LOOKBACK_DAYS} days\nThis takes a few minutes...")

        pairs = await self._get_top_pairs()
        logger.info(f"📋 Testing {len(pairs)} pairs")

        all_trades = []
        pair_summaries = []

        for i, symbol in enumerate(pairs):
            logger.info(f"[{i+1}/{len(pairs)}] Fetching {symbol}...")
            try:
                df_1h  = await self._fetch_full_history(symbol, '1h',  self.LOOKBACK_DAYS)
                df_4h  = await self._fetch_full_history(symbol, '4h',  self.LOOKBACK_DAYS + 30)
                df_15m = await self._fetch_full_history(symbol, '15m', self.LOOKBACK_DAYS)
                await asyncio.sleep(0.3)

                if df_1h is None or df_4h is None or df_15m is None:
                    continue
                if len(df_1h) < 120:
                    logger.info(f"  ⚠️ Not enough 1h data for {symbol}")
                    continue

                trades = self._walk_forward(symbol, df_1h, df_4h, df_15m)
                logger.info(f"  ✅ {symbol}: {len(trades)} signals")

                if trades:
                    all_trades.extend(trades)
                    sub = pd.DataFrame(trades)
                    pair_summaries.append({
                        'symbol': symbol.replace('/USDT:USDT', ''),
                        'trades': len(trades),
                        'wr': round(sub['win'].mean() * 100, 1),
                        'pnl': round(sub['pnl_pct'].sum(), 2)
                    })

            except Exception as e:
                logger.error(f"Error {symbol}: {e}")
                continue

        logger.info(f"🏁 Done. {len(all_trades)} total trades across {len(pair_summaries)} pairs.")

        if not all_trades:
            await self._send_telegram("❌ No trades found. Check your config / symbols.")
            await self.exchange.close()
            return

        # ── Save raw results ─────────────────────────────────
        df_all = pd.DataFrame(all_trades)
        df_all.to_csv('/tmp/backtest_results.csv', index=False)
        logger.info("💾 Saved to /tmp/backtest_results.csv")

        # ── Compute + send stats ─────────────────────────────
        stats = self._compute_stats(all_trades)
        report = self._format_telegram_report(stats, len(pairs), self.LOOKBACK_DAYS)
        await self._send_telegram(report)

        # ── Top / worst pairs ────────────────────────────────
        if pair_summaries:
            psum = sorted(pair_summaries, key=lambda x: x['pnl'], reverse=True)
            top5    = psum[:5]
            worst5  = psum[-5:]

            extra  = "\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
            extra += "<b>🏆 TOP 5 PAIRS</b>\n"
            for p in top5:
                extra += f"  {p['symbol']}: {p['trades']}t | WR {p['wr']}% | {p['pnl']:+.1f}%\n"
            extra += "\n<b>💀 WORST 5 PAIRS</b>\n"
            for p in worst5:
                extra += f"  {p['symbol']}: {p['trades']}t | WR {p['wr']}% | {p['pnl']:+.1f}%\n"
            await self._send_telegram(extra)

        await self.exchange.close()
        logger.info("✅ Backtest complete!")
        return stats


# ════════════════════════════════════════════════════
# ENTRY POINT — configure below and run:
#   python backtester.py
# ════════════════════════════════════════════════════

async def main():
    # ── Your credentials ────────────────────────────
    TELEGRAM_TOKEN  = "8034062612:AAEJYbPA8sMODYvqvt8U-5mM7c3Y3-GOYtM"
    TELEGRAM_CHAT_ID = "7500072234"
    BINANCE_API_KEY  = None   # Not needed for public data
    BINANCE_SECRET   = None
    # ────────────────────────────────────────────────

    bt = Backtester(
        binance_api_key=BINANCE_API_KEY,
        binance_secret=BINANCE_SECRET,
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )

    # ── Tweak these ──────────────────────────────────
    bt.LOOKBACK_DAYS   = 60   # 60 days of history
    bt.TOP_N_PAIRS     = 30   # Top 30 pairs by volume
    bt.MIN_VOLUME_USDT = 5_000_000
    # ────────────────────────────────────────────────

    await bt.run()


if __name__ == "__main__":
    asyncio.run(main())
