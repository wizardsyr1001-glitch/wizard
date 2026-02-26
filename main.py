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

# Suppress pandas warnings
import warnings
warnings.filterwarnings('ignore')

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
            'last_scan_time': None,
            'pairs_scanned': 0
        }
        self.is_scanning = False
        self.is_tracking = False
        
    async def get_all_usdt_pairs(self):
        """Get ALL USDT perpetual pairs with quality volume filter"""
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
        """Fetch multi-timeframe data"""
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
        """Calculate SuperTrend"""
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
        """Calculate 25+ advanced indicators"""
        try:
            if len(df) < 30:
                return df
            
            # TREND
            df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=min(50, len(df)-1)).ema_indicator()
            df['supertrend'] = self.calculate_supertrend(df)
            
            psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
            df['psar'] = psar.psar()
            
            # MOMENTUM
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['rsi_6'] = ta.momentum.RSIIndicator(df['close'], window=6).rsi()
            
            stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
            df['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
            df['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
            
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close']).williams_r()
            df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
            df['uo'] = ta.momentum.UltimateOscillator(df['high'], df['low'], df['close']).ultimate_oscillator()
            
            # VOLATILITY
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pband'] = bb.bollinger_pband()
            
            kc = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
            df['kc_upper'] = kc.keltner_channel_hband()
            df['kc_lower'] = kc.keltner_channel_lband()
            
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            dc = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'])
            df['dc_upper'] = dc.donchian_channel_hband()
            df['dc_lower'] = dc.donchian_channel_lband()
            
            # VOLUME
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_ema'] = df['obv'].ewm(span=20).mean()
            
            df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            df['ad'] = ta.volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index()
            df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume']).chaikin_money_flow()
            
            # TREND STRENGTH
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            df['di_plus'] = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()
            
            df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close']).cci()
            
            aroon = ta.trend.AroonIndicator(df['high'], df['low'])
            df['aroon_up'] = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
            df['aroon_ind'] = df['aroon_up'] - df['aroon_down']
            
            # VWAP (fixed calculation)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap'] = df['vwap'].fillna(df['close'])  # Fill any NaN values
            
            # PATTERNS
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
            
            # DIVERGENCE
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
    
    def detect_volume_spike(self, df):
        """Detect volume spikes"""
        if len(df) < 20:
            return False, 1.0
        recent = df['volume'].iloc[-1]
        avg = df['volume'].iloc[-20:].mean()
        if avg == 0 or pd.isna(avg):  # Prevent division by zero
            return False, 1.0
        ratio = recent / avg
        return recent > avg * 2.5, ratio
    
    def detect_signal(self, data, symbol):
        """Advanced signal detection with 35-point scoring"""
        try:
            if not data or '1h' not in data:
                return None
            
            for tf in data:
                data[tf] = self.calculate_advanced_indicators(data[tf])
            
            df_1h = data['1h']
            df_4h = data['4h']
            df_15m = data['15m']
            
            if len(df_1h) < 50:
                return None
            
            latest_1h = df_1h.iloc[-1]
            prev_1h = df_1h.iloc[-2]
            latest_4h = df_4h.iloc[-1]
            latest_15m = df_15m.iloc[-1]
            
            # Check for required columns
            required_cols = ['ema_9', 'ema_21', 'rsi', 'macd', 'vwap', 'bb_pband']
            for col in required_cols:
                if col not in latest_1h.index or pd.isna(latest_1h[col]):
                    logger.error(f"Missing or NaN column: {col}")
                    return None
            
            volume_spike, vol_ratio = self.detect_volume_spike(df_1h)
            
            long_score = 0
            short_score = 0
            max_score = 35
            long_reasons = []
            short_reasons = []
            
            # TREND (6 points)
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
            
            # MOMENTUM (9 points)
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
            
            # VOLUME (5 points)
            if volume_spike:
                if latest_1h['close'] > prev_1h['close']:
                    long_score += 3
                    long_reasons.append(f'🚀 VOL SPIKE ({vol_ratio:.1f}x)')
                else:
                    short_score += 3
                    short_reasons.append(f'💥 VOL DUMP ({vol_ratio:.1f}x)')
            
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
            
            # VOLATILITY (6 points)
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
            
            # TREND STRENGTH (4 points)
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
            
            # DIVERGENCE & PATTERNS (3 points)
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
            
            # HTF CONFIRMATION (2 points)
            if latest_4h['close'] > latest_4h['vwap']:
                long_score += 1
            else:
                short_score += 1
            
            if latest_4h['rsi'] < 50:
                long_score += 1
            elif latest_4h['rsi'] > 50:
                short_score += 1
            
            # DETERMINE SIGNAL
            min_threshold = max_score * 0.51
            signal = None
            
            if long_score > short_score and long_score >= min_threshold:
                signal = 'LONG'
                score = long_score
                reasons = long_reasons
                if long_score >= max_score * 0.71:
                    quality = 'PREMIUM 💎'
                elif long_score >= max_score * 0.60:
                    quality = 'HIGH 🔥'
                else:
                    quality = 'GOOD ✅'
            elif short_score > long_score and short_score >= min_threshold:
                signal = 'SHORT'
                score = short_score
                reasons = short_reasons
                if short_score >= max_score * 0.71:
                    quality = 'PREMIUM 💎'
                elif short_score >= max_score * 0.60:
                    quality = 'HIGH 🔥'
                else:
                    quality = 'GOOD ✅'
            
            if signal:
                entry = latest_15m['close']
                atr = latest_1h['atr']
                
                if signal == 'LONG':
                    sl = entry - (atr * 1.5)
                    targets = [entry + (atr * 1), entry + (atr * 2), entry + (atr * 3.5)]
                else:
                    sl = entry + (atr * 1.5)
                    targets = [entry - (atr * 1), entry - (atr * 2), entry - (atr * 3.5)]
                
                risk_pct = abs((sl - entry) / entry * 100)
                rr = [(abs(tp - entry) / abs(sl - entry)) for tp in targets]
                
                trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
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
                    'reasons': reasons[:10],
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
        """Format signal message"""
        emoji = "🚀" if sig['signal'] == 'LONG' else "🔻"
        
        msg = f"{'='*40}\n"
        msg += f"{emoji} <b>24H DAY TRADE - {sig['quality']}</b> {emoji}\n"
        msg += f"{'='*40}\n\n"
        msg += f"<b>🆔:</b> <code>{sig['trade_id']}</code>\n"
        msg += f"<b>📊 PAIR:</b> #{sig['symbol']}\n"
        msg += f"<b>📍 DIR:</b> <b>{sig['signal']}</b>\n"
        msg += f"<b>⭐ SCORE:</b> {sig['score']:.1f}/{sig['max_score']} ({sig['score_percent']:.0f}%)\n"
        msg += f"{'▰' * int(sig['score_percent']/10)}{'▱' * (10-int(sig['score_percent']/10))}\n\n"
        
        msg += f"<b>💰 ENTRY:</b> ${sig['entry']:.6f}\n\n"
        msg += f"<b>🎯 TARGETS:</b>\n"
        times = ['2-4h', '8-12h', '16-24h']
        for i, (tp, rr, t) in enumerate(zip(sig['targets'], sig['reward_ratios'], times), 1):
            pct = abs((tp - sig['entry']) / sig['entry'] * 100)
            msg += f"  TP{i} ({t}): ${tp:.6f} (+{pct:.2f}%) [RR {rr:.1f}:1]\n"
        
        msg += f"\n<b>🛑 SL:</b> ${sig['stop_loss']:.6f} (-{sig['risk_percent']:.2f}%)\n\n"
        
        msg += f"<b>✅ REASONS:</b>\n"
        for r in sig['reasons']:
            msg += f"  • {r}\n"
        
        msg += f"\n<b>📡 LIVE TRACKING ACTIVE!</b>\n"
        msg += f"<i>⏰ {sig['timestamp'].strftime('%H:%M:%S')}</i>\n"
        msg += f"{'='*40}"
        
        return msg
    
    async def send_msg(self, msg):
        """Send telegram message"""
        try:
            await self.telegram_bot.send_message(chat_id=self.chat_id, text=msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Send error: {e}")
    
    async def send_tp_alert(self, trade, tp_num, price):
        """TP hit alert"""
        emoji = "🎉" if trade['signal'] == 'LONG' else "💰"
        tp = trade['targets'][tp_num - 1]
        pct = abs((tp - trade['entry']) / trade['entry'] * 100)
        
        msg = f"{emoji} <b>TARGET HIT!</b> {emoji}\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"<b>{trade['symbol']}</b> {trade['signal']}\n\n"
        msg += f"<b>✅ TP{tp_num} HIT!</b>\n"
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
        
        if tp_num == 1:
            self.stats['tp1_hits'] += 1
        elif tp_num == 2:
            self.stats['tp2_hits'] += 1
        else:
            self.stats['tp3_hits'] += 1
    
    async def send_sl_alert(self, trade, price):
        """SL hit alert"""
        loss = abs((price - trade['entry']) / trade['entry'] * 100)
        
        msg = f"⚠️ <b>STOP LOSS HIT!</b> ⚠️\n\n"
        msg += f"<code>{trade['trade_id']}</code>\n"
        msg += f"{trade['symbol']} {trade['signal']}\n\n"
        msg += f"Entry: ${trade['entry']:.6f}\n"
        msg += f"SL: ${trade['stop_loss']:.6f}\n"
        msg += f"Current: ${price:.6f}\n"
        msg += f"Loss: -{loss:.2f}%"
        
        await self.send_msg(msg)
    
    async def track_trades(self):
        """Live trade tracking"""
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
                        # 24h timeout
                        if datetime.now() - trade['timestamp'] > timedelta(hours=24):
                            msg = f"⏰ 24H LIMIT\n<code>{tid}</code>\n{trade['symbol']}\nClose position!"
                            await self.send_msg(msg)
                            to_remove.append(tid)
                            continue
                        
                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price = ticker['last']
                        
                        if trade['signal'] == 'LONG':
                            if not trade['tp_hit'][0] and price >= trade['targets'][0]:
                                await self.send_tp_alert(trade, 1, price)
                                trade['tp_hit'][0] = True
                            
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
                        
                        else:  # SHORT
                            if not trade['tp_hit'][0] and price <= trade['targets'][0]:
                                await self.send_tp_alert(trade, 1, price)
                                trade['tp_hit'][0] = True
                            
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
        """Scan all pairs"""
        if self.is_scanning:
            logger.info("⚠️ Already scanning...")
            return []
        
        self.is_scanning = True
        logger.info("🔍 Starting scan...")
        
        pairs = await self.get_all_usdt_pairs()
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
        self.stats['pairs_scanned'] = scanned
        
        summary = f"✅ <b>SCAN COMPLETE</b>\n\n"
        summary += f"📊 Scanned: {scanned}\n"
        summary += f"🎯 Signals: {len(signals)}\n"
        
        if signals:
            longs = sum(1 for s in signals if s['signal'] == 'LONG')
            shorts = len(signals) - longs
            premium = sum(1 for s in signals if s['quality'] == 'PREMIUM 💎')
            
            summary += f"  🟢 Long: {longs}\n"
            summary += f"  🔴 Short: {shorts}\n"
            summary += f"  💎 Premium: {premium}\n"
        
        summary += f"  📡 Tracking: {len(self.active_trades)}\n"
        summary += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        await self.send_msg(summary)
        logger.info(f"🎉 Done! {len(signals)} signals")
        
        self.is_scanning = False
        return signals
    
    async def run(self, interval=60):
        """Run continuously"""
        logger.info("🚀 ADVANCED DAY TRADING SCANNER")
        
        welcome = "🔥 <b>ADVANCED 24H DAY TRADING SCANNER</b> 🔥\n\n"
        welcome += "✅ ALL USDT pairs\n"
        welcome += "✅ 25+ indicators\n"
        welcome += "✅ Divergence detection\n"
        welcome += "✅ Pattern recognition\n"
        welcome += "✅ Live TP/SL tracking\n"
        welcome += "✅ 35-point scoring\n\n"
        welcome += f"Scans every {interval} min\n\n"
        welcome += "<b>Commands:</b>\n"
        welcome += "/scan /stats /trades /help\n\n"
        welcome += "🎯 Advanced signals + Live alerts!"
        
        await self.send_msg(welcome)
        
        asyncio.create_task(self.track_trades())
        
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

class BotCommands:
    def __init__(self, scanner):
        self.scanner = scanner
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = "🚀 <b>Advanced Day Trading Scanner + Live Tracking!</b>\n\n"
        msg += "Scans ALL USDT pairs with 25+ indicators.\n\n"
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
        msg = f"📊 <b>STATISTICS</b>\n\n"
        msg += f"Total: {s['total_signals']}\n"
        msg += f"Long: {s['long_signals']} 🟢\n"
        msg += f"Short: {s['short_signals']} 🔴\n"
        msg += f"Premium: {s['premium_signals']} 💎\n\n"
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
            for i, hit in enumerate(t['tp_hit'], 1):
                tp_status += "✅" if hit else "⏳"
            
            msg += f"<b>{t['symbol']}</b> {t['signal']}\n"
            msg += f"  {tp_status} | {hrs}h old\n\n"
        
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = "📚 <b>ADVANCED DAY TRADING SCANNER</b>\n\n"
        msg += "<b>Features:</b>\n"
        msg += "• 25+ advanced indicators\n"
        msg += "• Divergence detection\n"
        msg += "• Pattern recognition\n"
        msg += "• Volume analysis\n"
        msg += "• Live TP/SL tracking\n"
        msg += "• 35-point scoring\n\n"
        msg += "<b>Indicators:</b>\n"
        msg += "• Trend: EMA, SuperTrend, PSAR, Aroon\n"
        msg += "• Momentum: RSI, MACD, Stoch RSI, UO\n"
        msg += "• Volume: OBV, MFI, CMF, A/D\n"
        msg += "• Volatility: BB, Keltner, Donchian\n"
        msg += "• Patterns: Engulfing, Divergence\n\n"
        msg += "<b>Quality:</b>\n"
        msg += "💎 PREMIUM (71%+) - 25+ points\n"
        msg += "🔥 HIGH (60%+) - 21+ points\n"
        msg += "✅ GOOD (51%+) - 18+ points\n\n"
        msg += "<b>Commands:</b>\n"
        msg += "/scan /stats /trades /help"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def main():
    # ========== CONFIG ==========
    TELEGRAM_TOKEN = "7957028587:AAE7aSYtE4hCxxTIPkAs_1ULJ9e8alkY6Ic"
    TELEGRAM_CHAT_ID = "7500072234"
    BINANCE_API_KEY = None
    BINANCE_SECRET = None
    
    scanner = AdvancedDayTradingScanner(
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        binance_api_key=BINANCE_API_KEY,
        binance_secret=BINANCE_SECRET
    )
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    commands = BotCommands(scanner)
    
    app.add_handler(CommandHandler("start", commands.cmd_start))
    app.add_handler(CommandHandler("scan", commands.cmd_scan))
    app.add_handler(CommandHandler("stats", commands.cmd_stats))
    app.add_handler(CommandHandler("trades", commands.cmd_trades))
    app.add_handler(CommandHandler("help", commands.cmd_help))
    
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
