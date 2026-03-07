import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  BACKTESTER V3 — Balanced settings (not too strict, not loose)
#
#  V1 → V2 was too aggressive, filtered everything to zero.
#  V3 finds the sweet spot:
#
#  Score threshold : 51% → 55%   (was 60% in V2 — too strict)
#  ATR cap         : none → 3%   (was 2% in V2 — too strict)
#  Volume filter   : $1M → $5M   (was $10M in V2 — too strict)
#  BTC regime      : hard block → SOFT BIAS
#                    BULL  → longs get +2 bonus pts, shorts get -2
#                    BEAR  → shorts get +2 bonus pts, longs get -2
#                    NEUTRAL → no change
#                    (replaces full block — still filters but doesn't zero out)
#  SL multiplier   : 1.5x → 1.2x (was 1.0x in V2 — slightly too tight)
#  TP multipliers  : [1.0,2.0,3.5] → [1.0,2.0,3.2]
# ═══════════════════════════════════════════════════════════════

class BacktesterV3:

    def __init__(self, binance_api_key=None, binance_secret=None,
                 telegram_token=None, telegram_chat_id=None):
        self.exchange = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.telegram_token = telegram_token
        self.chat_id        = telegram_chat_id

        # ── V3 Balanced params ────────────────────────────────
        self.LOOKBACK_DAYS    = 60
        self.TOP_N_PAIRS      = 30
        self.MIN_VOLUME_USDT  = 5_000_000   # V3: $5M (V1=$1M, V2=$10M)
        self.ATR_SL_MULT      = 1.2         # V3: 1.2x (V1=1.5x, V2=1.0x)
        self.ATR_TP_MULTS     = [1.0, 2.0, 3.2]  # V3 (V1=[1.0,2.0,3.5])
        self.ATR_PCT_CAP      = 0.03        # V3: 3% (V2=2%)
        self.SCORE_THRESHOLD  = 0.55        # V3: 55% (V1=51%, V2=60%)
        self.REGIME_BIAS_PTS  = 2.0         # bonus/penalty points for regime
        self.MAX_HOLD_HOURS   = 24
        self.COMMISSION_PCT   = 0.04
        # ─────────────────────────────────────────────────────

        self.btc_df_4h = None

    # ─────────────────────────────────────────────────────────
    #  BTC regime — SOFT BIAS (not hard block)
    # ─────────────────────────────────────────────────────────
    def _get_btc_regime_at(self, ts):
        if self.btc_df_4h is None or len(self.btc_df_4h) == 0:
            return 'NEUTRAL'
        past = self.btc_df_4h[self.btc_df_4h['timestamp'] <= ts]
        if len(past) < 50:
            return 'NEUTRAL'
        price = past['close'].iloc[-1]
        ema21 = past['close'].ewm(span=21).mean().iloc[-1]
        ema50 = past['close'].ewm(span=50).mean().iloc[-1]
        if price > ema21 > ema50:
            return 'BULL'
        elif price < ema21 < ema50:
            return 'BEAR'
        return 'NEUTRAL'

    # ─────────────────────────────────────────────────────────
    #  Indicators
    # ─────────────────────────────────────────────────────────
    def _calculate_supertrend(self, df, period=10, multiplier=3):
        try:
            hl2 = (df['high'] + df['low']) / 2
            atr = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=period
            ).average_true_range()
            upper = hl2 + (multiplier * atr)
            lower = hl2 - (multiplier * atr)
            st = [0.0] * len(df)
            for i in range(1, len(df)):
                if df['close'].iloc[i] > upper.iloc[i-1]:   st[i] = lower.iloc[i]
                elif df['close'].iloc[i] < lower.iloc[i-1]: st[i] = upper.iloc[i]
                else:                                         st[i] = st[i-1]
            return pd.Series(st, index=df.index)
        except:
            return pd.Series([0.0] * len(df), index=df.index)

    def _add_indicators(self, df):
        if len(df) < 30:
            return df
        try:
            df['ema_9']  = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=min(50,len(df)-1)).ema_indicator()
            df['supertrend'] = self._calculate_supertrend(df)
            psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
            df['psar'] = psar.psar()
            df['rsi']   = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            srsi = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi_k'] = srsi.stochrsi_k()
            df['stoch_rsi_d'] = srsi.stochrsi_d()
            macd = ta.trend.MACD(df['close'])
            df['macd']        = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist']   = macd.macd_diff()
            df['williams_r']  = ta.momentum.WilliamsRIndicator(
                df['high'], df['low'], df['close']
            ).williams_r()
            df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
            df['uo']  = ta.momentum.UltimateOscillator(
                df['high'], df['low'], df['close']
            ).ultimate_oscillator()
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper']  = bb.bollinger_hband()
            df['bb_lower']  = bb.bollinger_lband()
            df['bb_width']  = (df['bb_upper'] - df['bb_lower']) / bb.bollinger_mavg()
            df['bb_pband']  = bb.bollinger_pband()
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close']
            ).average_true_range()
            df['volume_sma']   = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
            df['obv']     = ta.volume.OnBalanceVolumeIndicator(
                df['close'], df['volume']
            ).on_balance_volume()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            df['mfi']     = ta.volume.MFIIndicator(
                df['high'], df['low'], df['close'], df['volume']
            ).money_flow_index()
            df['cmf']     = ta.volume.ChaikinMoneyFlowIndicator(
                df['high'], df['low'], df['close'], df['volume']
            ).chaikin_money_flow()
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx']      = adx.adx()
            df['di_plus']  = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()
            df['cci']      = ta.trend.CCIIndicator(
                df['high'], df['low'], df['close']
            ).cci()
            aroon = ta.trend.AroonIndicator(df['high'], df['low'])
            df['aroon_up']   = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            df['aroon_ind']  = df['aroon_up'] - df['aroon_down']
            tp_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'] = df['vwap'].fillna(df['close'])
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

    # ─────────────────────────────────────────────────────────
    #  Signal detection — V3 with soft regime bias
    # ─────────────────────────────────────────────────────────
    def _detect_signal_at(self, df_1h_slice, df_4h_slice, df_15m_slice, ts):
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

            # ATR cap — V3 = 3% (was 2% in V2)
            atr_pct = lat1['atr'] / lat1['close']
            if atr_pct > self.ATR_PCT_CAP:
                return None

            vol_avg   = df_1h['volume'].iloc[-20:].mean()
            vol_ratio = lat1['volume'] / vol_avg if vol_avg > 0 else 1.0
            vol_spike = vol_ratio > 2.5

            ls = ss = 0.0
            max_score = 35

            # ── TREND (6) ────────────────────────────────────
            if lat4['ema_9'] > lat4['ema_21'] > lat4['ema_50']:   ls += 3
            elif lat4['ema_9'] < lat4['ema_21'] < lat4['ema_50']: ss += 3
            if lat1['ema_9'] > lat1['ema_21']:   ls += 2
            elif lat1['ema_9'] < lat1['ema_21']: ss += 2
            if lat1['close'] > lat1['supertrend']:   ls += 1
            elif lat1['close'] < lat1['supertrend']: ss += 1

            # ── MOMENTUM (9) ─────────────────────────────────
            rsi = lat1['rsi']
            if rsi < 30:    ls += 3.5
            elif rsi < 40:  ls += 2
            elif rsi <= 50: ls += 1
            if rsi > 70:    ss += 3.5
            elif rsi > 60:  ss += 2
            elif rsi >= 50: ss += 1
            if lat1['stoch_rsi_k'] < 0.2 and lat1['stoch_rsi_k'] > lat1['stoch_rsi_d']: ls += 2
            elif lat1['stoch_rsi_k'] > 0.8 and lat1['stoch_rsi_k'] < lat1['stoch_rsi_d']: ss += 2
            if lat1['macd'] > lat1['macd_signal'] and prev1['macd'] <= prev1['macd_signal']: ls += 2.5
            elif lat1['macd'] < lat1['macd_signal'] and prev1['macd'] >= prev1['macd_signal']: ss += 2.5
            if lat1['uo'] < 30:  ls += 1.5
            elif lat1['uo'] > 70: ss += 1.5

            # ── VOLUME (5) ───────────────────────────────────
            if vol_spike:
                if lat1['close'] > prev1['close']: ls += 3
                else:                               ss += 3
            if lat1['mfi'] < 20:   ls += 1.5
            elif lat1['mfi'] > 80:  ss += 1.5
            if lat1['cmf'] > 0.15:   ls += 1
            elif lat1['cmf'] < -0.15: ss += 1
            obv_trend = df_1h['obv'].iloc[-5:].diff().mean()
            if obv_trend > 0 and lat1['obv'] > lat1['obv_ema']:   ls += 0.5
            elif obv_trend < 0 and lat1['obv'] < lat1['obv_ema']: ss += 0.5

            # ── VOLATILITY (6) ───────────────────────────────
            if lat1['bb_pband'] < 0.1:   ls += 2.5
            elif lat1['bb_pband'] > 0.9:  ss += 2.5
            if lat1['cci'] < -150:  ls += 1.5
            elif lat1['cci'] > 150:  ss += 1.5
            if lat1['williams_r'] < -85:  ls += 1
            elif lat1['williams_r'] > -15: ss += 1
            if lat1['close'] < lat1['vwap'] * 0.98:   ls += 1
            elif lat1['close'] > lat1['vwap'] * 1.02:  ss += 1

            # ── TREND STRENGTH (4) ───────────────────────────
            if lat1['adx'] > 30:
                if lat1['di_plus'] > lat1['di_minus']: ls += 2
                else:                                   ss += 2
            elif lat1['adx'] > 25:
                if lat1['di_plus'] > lat1['di_minus']: ls += 1
                else:                                   ss += 1
            if lat1['aroon_ind'] > 50:   ls += 1
            elif lat1['aroon_ind'] < -50: ss += 1
            if lat1['roc'] > 3:  ls += 1
            elif lat1['roc'] < -3: ss += 1

            # ── PATTERNS & DIVERGENCE (3) ────────────────────
            if lat1['bullish_divergence'] == 1:  ls += 2
            elif lat1['bearish_divergence'] == 1: ss += 2
            if lat15['bullish_engulfing'] == 1:  ls += 1.5
            elif lat15['bearish_engulfing'] == 1: ss += 1.5

            # ── HTF (2) ──────────────────────────────────────
            if lat4['close'] > lat4['vwap']: ls += 1
            else:                             ss += 1
            if lat4['rsi'] < 50:  ls += 1
            elif lat4['rsi'] > 50: ss += 1

            # ── SOFT REGIME BIAS (V3 key change) ─────────────
            # Instead of hard blocking, we nudge scores by 2 pts
            # This makes counter-trend signals need to be STRONGER
            # to pass the threshold, but doesn't zero them out
            regime = self._get_btc_regime_at(ts)
            if regime == 'BULL':
                ls += self.REGIME_BIAS_PTS    # reward longs in bull
                ss -= self.REGIME_BIAS_PTS    # penalise shorts in bull
            elif regime == 'BEAR':
                ss += self.REGIME_BIAS_PTS    # reward shorts in bear
                ls -= self.REGIME_BIAS_PTS    # penalise longs in bear
            # NEUTRAL → no adjustment

            # Clamp to zero (can't go negative)
            ls = max(ls, 0)
            ss = max(ss, 0)

            # ── Threshold — V3 = 55% ─────────────────────────
            threshold = max_score * self.SCORE_THRESHOLD
            signal = quality = None

            if ls > ss and ls >= threshold:
                signal = 'LONG';  score = ls
                if ls >= max_score * 0.71:    quality = 'PREMIUM'
                elif ls >= max_score * 0.60:  quality = 'HIGH'
                else:                         quality = 'GOOD'
            elif ss > ls and ss >= threshold:
                signal = 'SHORT'; score = ss
                if ss >= max_score * 0.71:    quality = 'PREMIUM'
                elif ss >= max_score * 0.60:  quality = 'HIGH'
                else:                         quality = 'GOOD'

            if not signal:
                return None

            entry = lat15['close']
            atr   = lat1['atr']

            # SL = 1.2x ATR, TPs = 1.0/2.0/3.2x ATR
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
                'atr': atr, 'atr_pct': atr_pct * 100,
                'risk_pct': abs(sl - entry) / entry * 100,
                'regime': regime, 'signal_time': ts
            }
        except Exception as e:
            logger.error(f"Signal error: {e}")
        return None

    # ─────────────────────────────────────────────────────────
    #  Trade simulation
    # ─────────────────────────────────────────────────────────
    def _simulate_trade(self, signal, future_candles):
        entry = signal['entry']
        sl    = signal['sl']
        tps   = signal['tps']
        direction = signal['signal']

        tp_hit = [False, False, False]
        sl_hit = False
        exit_price = exit_reason = None
        hours_held = 0

        for _, row in future_candles.iterrows():
            hours_held += 1
            hi, lo = row['high'], row['low']

            if direction == 'LONG':
                if lo <= sl:
                    sl_hit = True; exit_price = sl; exit_reason = 'SL'; break
                for j, tp in enumerate(tps):
                    if not tp_hit[j] and hi >= tp: tp_hit[j] = True
                if tp_hit[2]:
                    exit_price = tps[2]; exit_reason = 'TP3'; break
            else:
                if hi >= sl:
                    sl_hit = True; exit_price = sl; exit_reason = 'SL'; break
                for j, tp in enumerate(tps):
                    if not tp_hit[j] and lo <= tp: tp_hit[j] = True
                if tp_hit[2]:
                    exit_price = tps[2]; exit_reason = 'TP3'; break

            if hours_held >= self.MAX_HOLD_HOURS:
                exit_price = row['close']; exit_reason = 'TIMEOUT'; break

        if exit_price is None:
            exit_price  = future_candles.iloc[-1]['close'] if len(future_candles) > 0 else entry
            exit_reason = 'INCOMPLETE'

        weights    = [0.50, 0.30, 0.20]
        tp_reached = sum(tp_hit)
        pnl_pct    = 0.0

        if sl_hit:
            pnl_pct = -signal['risk_pct'] - self.COMMISSION_PCT * 2
        elif exit_reason in ('TIMEOUT', 'INCOMPLETE'):
            raw = (exit_price - entry) / entry * 100
            if direction == 'SHORT': raw = -raw
            pnl_pct = raw - self.COMMISSION_PCT * 2
        else:
            for j in range(3):
                if tp_hit[j]:
                    pnl_pct += abs(tps[j] - entry) / entry * 100 * weights[j]
            filled = sum(weights[:tp_reached]) if tp_reached else 0
            remaining = 1.0 - filled
            if remaining > 0:
                last_raw = (exit_price - entry) / entry * 100
                if direction == 'SHORT': last_raw = -last_raw
                pnl_pct += last_raw * remaining
            pnl_pct -= self.COMMISSION_PCT * 2

        return {
            'exit_reason': exit_reason,
            'tp_hit': tp_hit, 'tp_reached': tp_reached,
            'sl_hit': sl_hit, 'exit_price': exit_price,
            'hours_held': hours_held,
            'pnl_pct': round(pnl_pct, 4),
            'win': pnl_pct > 0
        }

    # ─────────────────────────────────────────────────────────
    #  Walk-forward
    # ─────────────────────────────────────────────────────────
    def _walk_forward(self, symbol, df_1h, df_4h, df_15m):
        trades = []
        open_trade_until = None
        warm = 100
        if len(df_1h) < warm + 20:
            return trades

        for i in range(warm, len(df_1h) - 1):
            ts = df_1h.iloc[i]['timestamp']
            if open_trade_until and ts < open_trade_until:
                continue

            slice_1h  = df_1h.iloc[:i+1].reset_index(drop=True)
            slice_4h  = df_4h[df_4h['timestamp'] <= ts].tail(100).reset_index(drop=True)
            slice_15m = df_15m[df_15m['timestamp'] <= ts].tail(50).reset_index(drop=True)

            if len(slice_4h) < 50 or len(slice_15m) < 20:
                continue

            sig = self._detect_signal_at(slice_1h, slice_4h, slice_15m, ts)
            if not sig:
                continue

            future = df_1h.iloc[i+1:i+1+self.MAX_HOLD_HOURS].reset_index(drop=True)
            if len(future) == 0:
                break

            outcome = self._simulate_trade(sig, future)
            trade = {
                'symbol': symbol.replace('/USDT:USDT', ''),
                'signal': sig['signal'], 'quality': sig['quality'],
                'score_pct': round(sig['score_pct'], 1),
                'atr_pct': round(sig['atr_pct'], 2),
                'regime': sig['regime'],
                'entry': sig['entry'], 'sl': sig['sl'],
                'tp1': sig['tps'][0], 'tp2': sig['tps'][1], 'tp3': sig['tps'][2],
                'signal_time': ts.strftime('%Y-%m-%d %H:%M'),
                **outcome
            }
            trades.append(trade)
            open_trade_until = ts + timedelta(hours=outcome['hours_held'])

            logger.info(
                f"  📊 {trade['symbol']:12s} {trade['signal']:5s} "
                f"[{trade['regime']:7s}] [ATR {trade['atr_pct']:.1f}%] "
                f"[Score {trade['score_pct']:.0f}%] "
                f"{trade['signal_time']} → {trade['exit_reason']:10s} {trade['pnl_pct']:+.2f}%"
            )
        return trades

    # ─────────────────────────────────────────────────────────
    #  Data fetching
    # ─────────────────────────────────────────────────────────
    async def _fetch_history(self, symbol, timeframe, days):
        try:
            since = int((datetime.utcnow() - timedelta(days=days+10)).timestamp() * 1000)
            all_ohlcv = []
            while True:
                batch = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not batch: break
                all_ohlcv.extend(batch)
                since = batch[-1][0] + 1
                if len(batch) < 1000: break
                await asyncio.sleep(0.15)
            df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
            cutoff = datetime.utcnow() - timedelta(days=days)
            return df[df['timestamp'] >= cutoff].reset_index(drop=True)
        except Exception as e:
            logger.error(f"Fetch {symbol} {timeframe}: {e}")
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
            selected = [p[0] for p in pairs[:self.TOP_N_PAIRS]]
            logger.info(f"✅ {len(selected)} pairs ≥ ${self.MIN_VOLUME_USDT/1e6:.0f}M volume")
            return selected
        except Exception as e:
            logger.error(f"Get pairs: {e}")
            return []

    # ─────────────────────────────────────────────────────────
    #  Stats
    # ─────────────────────────────────────────────────────────
    def _compute_stats(self, all_trades):
        if not all_trades:
            return {}
        df = pd.DataFrame(all_trades)
        total  = len(df)
        wins   = int(df['win'].sum())
        losses = total - wins
        wr     = wins / total * 100
        avg_win  = df[df['win']]['pnl_pct'].mean() if wins > 0 else 0
        avg_loss = df[~df['win']]['pnl_pct'].mean() if losses > 0 else 0
        rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        expectancy = (wr/100 * avg_win) + ((1 - wr/100) * avg_loss)

        tp1_rate = df['tp_hit'].apply(lambda x: x[0]).mean() * 100
        tp2_rate = df['tp_hit'].apply(lambda x: x[1]).mean() * 100
        tp3_rate = df['tp_hit'].apply(lambda x: x[2]).mean() * 100
        sl_rate  = df['sl_hit'].mean() * 100

        quality_stats = {}
        for q in ['PREMIUM', 'HIGH', 'GOOD']:
            sub = df[df['quality'] == q]
            if len(sub):
                quality_stats[q] = {
                    'count': len(sub),
                    'wr': round(sub['win'].mean() * 100, 1),
                    'avg_pnl': round(sub['pnl_pct'].mean(), 2)
                }

        dir_stats = {}
        for d in ['LONG', 'SHORT']:
            sub = df[df['signal'] == d]
            if len(sub):
                dir_stats[d] = {
                    'count': len(sub),
                    'wr': round(sub['win'].mean() * 100, 1),
                    'avg_pnl': round(sub['pnl_pct'].mean(), 2)
                }

        regime_stats = {}
        for r in ['BULL', 'BEAR', 'NEUTRAL']:
            sub = df[df['regime'] == r]
            if len(sub):
                regime_stats[r] = {
                    'count': len(sub),
                    'wr': round(sub['win'].mean() * 100, 1),
                    'avg_pnl': round(sub['pnl_pct'].mean(), 2)
                }

        max_cl = cur = 0
        for w in df['win']:
            cur = 0 if w else cur + 1
            max_cl = max(max_cl, cur)

        equity = 10_000.0
        for pnl in df['pnl_pct']:
            equity *= (1 + pnl / 100)

        return {
            'total': total, 'wins': wins, 'losses': losses,
            'win_rate': round(wr, 1),
            'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
            'rr': round(rr, 2),
            'total_pnl': round(df['pnl_pct'].sum(), 2),
            'best_trade': round(df['pnl_pct'].max(), 2),
            'worst_trade': round(df['pnl_pct'].min(), 2),
            'avg_hold': round(df['hours_held'].mean(), 1),
            'expectancy': round(expectancy, 3),
            'tp1_rate': round(tp1_rate, 1),
            'tp2_rate': round(tp2_rate, 1),
            'tp3_rate': round(tp3_rate, 1),
            'sl_rate': round(sl_rate, 1),
            'quality_stats': quality_stats,
            'dir_stats': dir_stats,
            'regime_stats': regime_stats,
            'max_consec_losses': max_cl,
            'exit_counts': df['exit_reason'].value_counts().to_dict(),
            'final_equity': round(equity, 2),
            'df': df
        }

    # ─────────────────────────────────────────────────────────
    #  Report
    # ─────────────────────────────────────────────────────────
    def _format_report(self, s, pairs_tested, days):
        wr_emoji = "🟢" if s['win_rate'] >= 55 else ("🟡" if s['win_rate'] >= 45 else "🔴")

        msg  = "═" * 38 + "\n"
        msg += "📊 <b>BACKTEST V3 REPORT</b> 📊\n"
        msg += "═" * 38 + "\n\n"

        msg += "<b>⚙️ V3 SETTINGS</b>\n"
        msg += f"  Threshold : {self.SCORE_THRESHOLD*100:.0f}%\n"
        msg += f"  ATR cap   : {self.ATR_PCT_CAP*100:.0f}%\n"
        msg += f"  Volume    : ${self.MIN_VOLUME_USDT/1e6:.0f}M\n"
        msg += f"  SL mult   : {self.ATR_SL_MULT}x ATR\n"
        msg += f"  Regime    : Soft bias (±{self.REGIME_BIAS_PTS}pts)\n\n"

        msg += f"🗓 Period: {days} days | Pairs: {pairs_tested} | Trades: {s['total']}\n\n"

        msg += "━" * 38 + "\n"
        msg += "<b>📈 V1 → V3 COMPARISON</b>\n"
        msg += "━" * 38 + "\n"
        wr_delta = s['win_rate'] - 48.5
        msg += f"Win Rate:    48.5% → {s['win_rate']}%  ({wr_delta:+.1f}%)\n"
        msg += f"Worst trade: -21.4% → {s['worst_trade']}%\n"
        msg += f"Expectancy:  ??? → {s['expectancy']:+.3f}% per trade\n"
        msg += f"$10k equity: ??? → ${s['final_equity']:,.0f}\n\n"

        msg += "━" * 38 + "\n"
        msg += f"<b>📊 PERFORMANCE</b>\n"
        msg += "━" * 38 + "\n"
        msg += f"{wr_emoji} Win Rate:  {s['win_rate']}%  ({s['wins']}W / {s['losses']}L)\n"
        msg += f"💰 Avg Win:  +{s['avg_win']}%\n"
        msg += f"💸 Avg Loss: {s['avg_loss']}%\n"
        msg += f"⚖️  RR:       {s['rr']}:1\n"
        msg += f"🎯 Expect:   {s['expectancy']:+.3f}%\n"
        msg += f"🏆 Best:     +{s['best_trade']}%\n"
        msg += f"💀 Worst:    {s['worst_trade']}%\n"
        msg += f"⏱ Avg hold: {s['avg_hold']}h\n\n"

        msg += "━" * 38 + "\n"
        msg += "<b>🎯 TP / SL RATES</b>\n"
        msg += "━" * 38 + "\n"
        msg += f"  TP1 {s['tp1_rate']}% | TP2 {s['tp2_rate']}% | TP3 {s['tp3_rate']}%\n"
        msg += f"  SL  {s['sl_rate']}% | Max consec loss: {s['max_consec_losses']}\n\n"

        if s.get('regime_stats'):
            msg += "━" * 38 + "\n"
            msg += "<b>📡 BY BTC REGIME</b>\n"
            msg += "━" * 38 + "\n"
            for r, rs in s['regime_stats'].items():
                e = "🟢" if r=='BULL' else ("🔴" if r=='BEAR' else "⚪")
                msg += f"  {e} {r}: {rs['count']}t | WR {rs['wr']}% | Avg {rs['avg_pnl']:+.2f}%\n"
            msg += "\n"

        if s.get('quality_stats'):
            msg += "━" * 38 + "\n"
            msg += "<b>💎 BY QUALITY</b>\n"
            msg += "━" * 38 + "\n"
            for q, qs in s['quality_stats'].items():
                e = "💎" if q=='PREMIUM' else ("🔥" if q=='HIGH' else "✅")
                msg += f"  {e} {q}: {qs['count']}t | WR {qs['wr']}% | Avg {qs['avg_pnl']:+.2f}%\n"
            msg += "\n"

        if s.get('dir_stats'):
            msg += "━" * 38 + "\n"
            msg += "<b>📍 LONG vs SHORT</b>\n"
            msg += "━" * 38 + "\n"
            for d, ds in s['dir_stats'].items():
                e = "🟢" if d=='LONG' else "🔴"
                msg += f"  {e} {d}: {ds['count']}t | WR {ds['wr']}% | Avg {ds['avg_pnl']:+.2f}%\n"
            msg += "\n"

        msg += "━" * 38 + "\n"
        msg += "<b>🚪 EXIT REASONS</b>\n"
        msg += "━" * 38 + "\n"
        for reason, count in s['exit_counts'].items():
            msg += f"  {reason}: {count} ({count/s['total']*100:.0f}%)\n"
        msg += "\n"

        msg += "═" * 38 + "\n"
        msg += "<b>🔬 VERDICT</b>\n"
        msg += "═" * 38 + "\n"
        if s['win_rate'] >= 60 and s['rr'] >= 1.5:
            msg += "🚀 <b>STRONG EDGE — GO LIVE SMALL SIZE</b>\n"
        elif s['win_rate'] >= 55 and s['rr'] >= 1.3:
            msg += "✅ <b>SOLID IMPROVEMENT — KEEP OPTIMIZING</b>\n"
        elif s['win_rate'] >= 50 and s['rr'] >= 1.5:
            msg += "🟡 <b>MARGINAL — HIGH RR SAVES IT</b>\n"
        elif s['win_rate'] >= 50:
            msg += "🟡 <b>COIN FLIP — NEEDS MORE WORK</b>\n"
        else:
            msg += "🔴 <b>STILL LOSING — SHARE LOGS, WILL RETWEAK</b>\n"

        msg += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        return msg

    async def _send_telegram(self, msg):
        if not self.telegram_token or not self.chat_id:
            print(msg); return
        try:
            from telegram import Bot
            from telegram.constants import ParseMode
            bot = Bot(token=self.telegram_token)
            for chunk in [msg[i:i+4000] for i in range(0, len(msg), 4000)]:
                await bot.send_message(chat_id=self.chat_id, text=chunk, parse_mode=ParseMode.HTML)
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"Telegram: {e}"); print(msg)

    # ─────────────────────────────────────────────────────────
    #  MAIN
    # ─────────────────────────────────────────────────────────
    async def run(self):
        logger.info("🚀 Backtester V3 starting...")
        await self._send_telegram(
            f"⏳ <b>Backtest V3 started</b>\n\n"
            f"<b>Balanced settings:</b>\n"
            f"  📊 Threshold : {self.SCORE_THRESHOLD*100:.0f}%\n"
            f"  📉 ATR cap   : {self.ATR_PCT_CAP*100:.0f}%\n"
            f"  💧 Volume    : ${self.MIN_VOLUME_USDT/1e6:.0f}M\n"
            f"  🛑 SL mult   : {self.ATR_SL_MULT}x ATR\n"
            f"  📡 Regime    : Soft bias\n\n"
            f"~75 min runtime..."
        )

        # Fetch BTC 4H for regime
        logger.info("Fetching BTC 4H for regime filter...")
        self.btc_df_4h = await self._fetch_history('BTC/USDT:USDT', '4h', self.LOOKBACK_DAYS + 30)
        if self.btc_df_4h is not None:
            logger.info(f"✅ BTC 4H: {len(self.btc_df_4h)} candles")
        else:
            logger.warning("⚠️ BTC 4H fetch failed — regime disabled")

        pairs = await self._get_top_pairs()
        logger.info(f"📋 Testing {len(pairs)} pairs")

        all_trades     = []
        pair_summaries = []

        for i, symbol in enumerate(pairs):
            logger.info(f"[{i+1}/{len(pairs)}] {symbol}")
            try:
                df_1h  = await self._fetch_history(symbol, '1h',  self.LOOKBACK_DAYS)
                df_4h  = await self._fetch_history(symbol, '4h',  self.LOOKBACK_DAYS + 30)
                df_15m = await self._fetch_history(symbol, '15m', self.LOOKBACK_DAYS)
                await asyncio.sleep(0.3)

                if any(x is None for x in [df_1h, df_4h, df_15m]):
                    continue
                if len(df_1h) < 120:
                    logger.info(f"  ⚠️ Not enough 1H data")
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

        logger.info(f"🏁 Done. {len(all_trades)} trades across {len(pair_summaries)} pairs.")

        if not all_trades:
            await self._send_telegram(
                "❌ Still 0 trades. Something structural is wrong.\n"
                "Share terminal logs — I'll diagnose it directly."
            )
            await self.exchange.close()
            return

        pd.DataFrame(all_trades).to_csv('/tmp/backtest_v3_results.csv', index=False)
        logger.info("💾 Saved /tmp/backtest_v3_results.csv")

        stats  = self._compute_stats(all_trades)
        report = self._format_report(stats, len(pairs), self.LOOKBACK_DAYS)
        await self._send_telegram(report)

        if pair_summaries:
            psum  = sorted(pair_summaries, key=lambda x: x['pnl'], reverse=True)
            extra = "\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
            extra += "<b>🏆 TOP 5 PAIRS</b>\n"
            for p in psum[:5]:
                extra += f"  {p['symbol']}: {p['trades']}t | WR {p['wr']}% | {p['pnl']:+.1f}%\n"
            extra += "\n<b>💀 WORST 5 PAIRS</b>\n"
            for p in psum[-5:]:
                extra += f"  {p['symbol']}: {p['trades']}t | WR {p['wr']}% | {p['pnl']:+.1f}%\n"
            await self._send_telegram(extra)

        await self.exchange.close()
        logger.info("✅ Backtester V3 complete!")
        return stats


# ════════════════════════════════════════════════════
#  RUN:  python backtester_v3.py
# ════════════════════════════════════════════════════
async def main():
    TELEGRAM_TOKEN   = "8034062612:AAEJYbPA8sMODYvqvt8U-5mM7c3Y3-GOYtM"
    TELEGRAM_CHAT_ID = "7500072234"

    bt = BacktesterV3(
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )

    await bt.run()


if __name__ == "__main__":
    asyncio.run(main())
