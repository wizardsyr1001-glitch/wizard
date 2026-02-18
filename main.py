import asyncio
import ccxt.async_support as ccxt
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging
from collections import deque
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

class SwingTradingScanner:
    """
    🎯 SWING TRADING BEAST MODE
    
    Finds HIGH QUALITY swing setups (3-10 day holds)
    - Daily + 4H timeframes ONLY
    - Major S/R levels with multiple touches
    - Strong trend reversals & breakouts
    - 5-20% target moves
    - 1:3 to 1:6 reward/risk minimum
    """
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
        
        # SWING TRADING CONFIGURATION
        self.scan_interval = 14400  # 4 hours (swing trades don't need hourly scans)
        self.min_score_threshold = 65  # Lower to catch more gems (horizontal bounces)
        self.max_alerts_per_scan = 5  # More alerts for gems
        self.price_check_interval = 300  # 5 minutes (not 30 seconds - we're swing trading)
        
        # Swing trading parameters
        self.min_reward_ratio = 3.0  # Minimum 1:3 RR
        self.min_target_percent = 5.0  # Minimum 5% move target
        self.max_risk_percent = 5.0  # Max 5% risk per trade
        
        # State tracking
        self.signal_history = deque(maxlen=200)
        self.alerted_pairs = {}
        self.active_trades = {}
        self.last_scan_time = None
        self.is_scanning = False
        self.is_tracking = False
        
        # Stats
        self.stats = {
            'total_scans': 0,
            'total_pairs_scanned': 0,
            'signals_found': 0,
            'high_conviction_signals': 0,
            'avg_scan_time': 0,
            'last_scan_date': None,
            'trades_tracked': 0,
            'tp1_hits': 0,
            'tp2_hits': 0,
            'tp3_hits': 0,
            'sl_hits': 0,
            'active_trades_count': 0,
            'avg_hold_time_hours': 0
        }
        
        self.pairs_to_scan = []
        self.all_symbols = []
    
    async def get_symbol_format(self, symbol_input):
        """Convert user input to proper exchange format"""
        try:
            await self.exchange.load_markets()
            symbol_input = symbol_input.upper().strip()
            
            possible_formats = [
                f"{symbol_input}/USDT:USDT",
                f"{symbol_input}USDT/USDT:USDT",
                symbol_input
            ]
            
            if symbol_input.endswith('USDT'):
                base = symbol_input[:-4]
                possible_formats.insert(0, f"{base}/USDT:USDT")
            
            for fmt in possible_formats:
                if fmt in self.exchange.symbols:
                    return fmt
            
            return None
        except Exception as e:
            logger.error(f"Symbol format error: {e}")
            return None
    
    async def load_all_usdt_pairs(self):
        """Load top 150 most liquid USDT perpetual pairs (quality over quantity for swing)"""
        try:
            logger.info("🔄 Loading top liquid USDT pairs for swing trading...")
            
            await self.exchange.load_markets()
            
            usdt_perpetuals = []
            for symbol, market in self.exchange.markets.items():
                if (market.get('quote') == 'USDT' and 
                    market.get('type') == 'swap' and 
                    market.get('settle') == 'USDT' and
                    market.get('active', True)):
                    
                    base = market['base']
                    usdt_perpetuals.append({
                        'base': base,
                        'symbol': symbol,
                        'volume': market.get('info', {}).get('volume', 0)
                    })
            
            # Sort by volume and take top 150 (most liquid = best for swing)
            usdt_perpetuals.sort(key=lambda x: float(x['volume']) if x['volume'] else 0, reverse=True)
            usdt_perpetuals = usdt_perpetuals[:150]  # Top 150 only
            
            self.pairs_to_scan = [pair['base'] for pair in usdt_perpetuals]
            self.all_symbols = [pair['symbol'] for pair in usdt_perpetuals]
            
            logger.info(f"✅ Loaded {len(self.pairs_to_scan)} liquid pairs for swing trading")
            logger.info(f"📊 Top 10: {', '.join(self.pairs_to_scan[:10])}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading pairs: {e}")
            self.pairs_to_scan = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'DOT', 'MATIC']
            logger.warning(f"⚠️ Using fallback pairs")
            return False
    
    async def fetch_data(self, symbol):
        """Fetch SWING TRADING timeframes - Daily + 4H + 1D"""
        timeframes = {
            '1d': 200,   # Daily - PRIMARY for swing
            '4h': 200,   # 4H - confirmation
            '1h': 100,   # 1H - fine entry timing only
        }
        data = {}
        try:
            for tf, limit in timeframes.items():
                ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                data[tf] = df
                await asyncio.sleep(0.02)
            return data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def find_major_support_resistance(self, df, lookback=100):
        """
        Find HORIZONTAL S/R zones - like the screenshot gems!
        Focus on flat support/resistance with multiple touches
        """
        levels = []
        
        # Find swing highs (resistance) with wider window for daily
        for i in range(10, len(df) - 10):
            if df['high'].iloc[i] == df['high'].iloc[i-10:i+11].max():
                levels.append({
                    'price': df['high'].iloc[i],
                    'type': 'RESISTANCE',
                    'touches': 1,
                    'index': i,
                    'strength': 0
                })
        
        # Find swing lows (support) with wider window
        for i in range(10, len(df) - 10):
            if df['low'].iloc[i] == df['low'].iloc[i-10:i+11].min():
                levels.append({
                    'price': df['low'].iloc[i],
                    'type': 'SUPPORT',
                    'touches': 1,
                    'index': i,
                    'strength': 0
                })
        
        # Cluster similar levels - TIGHT clustering for horizontal zones
        clustered_levels = []
        for level in levels:
            found_cluster = False
            for cluster in clustered_levels:
                # 0.8% tolerance for TIGHT horizontal zones
                if abs(level['price'] - cluster['price']) / cluster['price'] < 0.008:
                    cluster['touches'] += 1
                    cluster['price'] = (cluster['price'] + level['price']) / 2
                    cluster['last_touch_idx'] = max(cluster.get('last_touch_idx', cluster['index']), level['index'])
                    found_cluster = True
                    break
            
            if not found_cluster:
                level['last_touch_idx'] = level['index']
                clustered_levels.append(level)
        
        # Calculate strength - HEAVILY favor multiple touches + recency
        current_idx = len(df) - 1
        for level in clustered_levels:
            recency_factor = 1 - ((current_idx - level['last_touch_idx']) / len(df))
            # MAJOR bonus for 3+ touches (those screenshot gems!)
            touch_score = level['touches'] * 30 if level['touches'] >= 3 else level['touches'] * 15
            level['strength'] = touch_score + recency_factor * 40
        
        # Sort by strength
        clustered_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return clustered_levels[:8]  # Top 8 strongest levels
    
    def detect_horizontal_consolidation(self, df, lookback=30):
        """
        Detect HORIZONTAL consolidation zones - those perfect box patterns!
        Returns info about tight ranges that precede breakouts
        """
        analysis = {
            'is_consolidating': False,
            'consolidation_strength': 0,
            'support_zone': None,
            'resistance_zone': None,
            'duration_candles': 0,
            'breakout_ready': False
        }
        
        # Look at recent price action
        recent_df = df.tail(lookback)
        
        high_price = recent_df['high'].max()
        low_price = recent_df['low'].min()
        range_pct = ((high_price - low_price) / low_price) * 100
        
        # Tight consolidation: <8% range over period
        if range_pct < 8:
            analysis['is_consolidating'] = True
            analysis['support_zone'] = low_price
            analysis['resistance_zone'] = high_price
            analysis['duration_candles'] = lookback
            
            # Check how tight the consolidation is
            if range_pct < 5:
                analysis['consolidation_strength'] = 40  # SUPER tight
            elif range_pct < 6.5:
                analysis['consolidation_strength'] = 30  # Very tight
            else:
                analysis['consolidation_strength'] = 20  # Decent
            
            # Check if we're at support edge (ready to bounce)
            current_price = df['close'].iloc[-1]
            distance_from_bottom = ((current_price - low_price) / (high_price - low_price))
            
            if distance_from_bottom < 0.25:  # In bottom 25% of range
                analysis['breakout_ready'] = True
                analysis['consolidation_strength'] += 15  # Bonus!
        
        return analysis
    
    def detect_major_trend_break(self, df):
        """Detect MAJOR trend line breaks on daily chart"""
        analysis = {
            'bullish_break': False,
            'bearish_break': False,
            'break_strength': 0,
            'consolidation_break': False
        }
        
        # Look for consolidation breakouts (horizontal range) - GEMS!
        last_50_high = df['high'].iloc[-50:].max()
        last_50_low = df['low'].iloc[-50:].min()
        range_size = (last_50_high - last_50_low) / last_50_low
        
        current_price = df['close'].iloc[-1]
        
        # Consolidation break (powerful for swing) - like your screenshots!
        if range_size < 0.12:  # Less than 12% range = tight consolidation
            if current_price > last_50_high * 1.015:  # Broke above with 1.5% buffer
                analysis['consolidation_break'] = True
                analysis['bullish_break'] = True
                analysis['break_strength'] = 45  # HUGE score for this!
            elif current_price < last_50_low * 0.985:  # Broke below
                analysis['consolidation_break'] = True
                analysis['bearish_break'] = True
                analysis['break_strength'] = 45
        
        # Major trendline breaks
        swing_lows = []
        for i in range(20, len(df) - 5):
            if df['low'].iloc[i] == df['low'].iloc[i-10:i+11].min():
                swing_lows.append({'index': i, 'price': df['low'].iloc[i]})
        
        if len(swing_lows) >= 3:
            # Use last 3 swing lows for trendline
            recent_lows = swing_lows[-3:]
            
            # Calculate trendline
            x = [low['index'] for low in recent_lows]
            y = [low['price'] for low in recent_lows]
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            
            current_idx = len(df) - 1
            projected_price = y[-1] + slope * (current_idx - x[-1])
            
            # Strong bullish break - price well above uptrend line
            if current_price > projected_price * 1.03 and slope > 0:
                analysis['bullish_break'] = True
                analysis['break_strength'] = min(50, abs((current_price - projected_price) / projected_price) * 200)
        
        # Downtrend breaks
        swing_highs = []
        for i in range(20, len(df) - 5):
            if df['high'].iloc[i] == df['high'].iloc[i-10:i+11].max():
                swing_highs.append({'index': i, 'price': df['high'].iloc[i]})
        
        if len(swing_highs) >= 3:
            recent_highs = swing_highs[-3:]
            
            x = [high['index'] for high in recent_highs]
            y = [high['price'] for high in recent_highs]
            slope = (y[-1] - y[0]) / (x[-1] - x[0])
            
            current_idx = len(df) - 1
            projected_price = y[-1] + slope * (current_idx - x[-1])
            
            # Strong bearish break
            if current_price < projected_price * 0.97 and slope < 0:
                analysis['bearish_break'] = True
                analysis['break_strength'] = min(50, abs((current_price - projected_price) / projected_price) * 200)
        
        return analysis
    
    def calculate_swing_indicators(self, df):
        """Calculate indicators for SWING TRADING"""
        if len(df) < 100:
            return df
        
        # Longer EMAs for swing
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_100'] = ta.trend.EMAIndicator(df['close'], window=min(100, len(df)-1)).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=min(200, len(df)-1)).ema_indicator()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD with swing settings
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Volume
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, 1)
        
        # ATR for stop placement
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # ADX - trend strength (crucial for swing)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        return df
    
    def analyze_swing_setup(self, data, symbol):
        """
        🎯 SWING TRADING ANALYSIS
        
        Focuses on:
        - Daily chart primary
        - Major S/R with 3+ touches
        - Strong trend confirmations
        - Big targets (5-20%)
        - Clean breakouts
        """
        try:
            if not data or '1d' not in data:
                return None
            
            # Calculate indicators
            for tf in data:
                data[tf] = self.calculate_swing_indicators(data[tf])
            
            df_daily = data['1d']
            df_4h = data['4h']
            df_1h = data['1h']
            
            current_price = df_daily['close'].iloc[-1]
            
            # Get latest data
            l_d = df_daily.iloc[-1]
            p_d = df_daily.iloc[-2]
            pp_d = df_daily.iloc[-3]
            l_4h = df_4h.iloc[-1]
            l_1h = df_1h.iloc[-1]
            
            # Find MAJOR S/R levels on daily
            sr_levels = self.find_major_support_resistance(df_daily, lookback=100)
            
            # Detect horizontal consolidation - THE GEMS!
            consolidation_info = self.detect_horizontal_consolidation(df_daily, lookback=30)
            
            # Find nearest major support and resistance
            nearest_support = None
            nearest_resistance = None
            
            for level in sr_levels:
                if level['type'] == 'SUPPORT' and level['price'] < current_price:
                    if nearest_support is None or level['price'] > nearest_support['price']:
                        nearest_support = level
                elif level['type'] == 'RESISTANCE' and level['price'] > current_price:
                    if nearest_resistance is None or level['price'] < nearest_resistance['price']:
                        nearest_resistance = level
            
            # Detect major trend breaks
            trend_analysis = self.detect_major_trend_break(df_daily)
            
            # 🎯 GEM-FOCUSED SCORING SYSTEM (0-100)
            long_score = 0
            short_score = 0
            reasons = []
            warnings = []
            
            # [1] HORIZONTAL CONSOLIDATION BOUNCE (40 points) - SCREENSHOT PATTERN!
            if consolidation_info['is_consolidating'] and consolidation_info['breakout_ready']:
                long_score += consolidation_info['consolidation_strength']
                reasons.append(f"📦 HORIZONTAL CONSOLIDATION ZONE ({consolidation_info['duration_candles']} candles)")
                reasons.append(f"💎 AT SUPPORT - READY TO BOUNCE!")
            
            # [2] MAJOR SUPPORT LEVEL (35 points) - Must have 3+ touches for gems
            if nearest_support and nearest_support['touches'] >= 3:
                distance = (current_price - nearest_support['price']) / current_price * 100
                
                if distance < 1.0:  # VERY close to major support
                    long_score += 35
                    reasons.append(f"💎 TOUCHING MAJOR SUPPORT ${nearest_support['price']:.4f} ({nearest_support['touches']}x TESTED)")
                elif distance < 2.0:
                    long_score += 28
                    reasons.append(f"🎯 Near Major Support ${nearest_support['price']:.4f} ({nearest_support['touches']}x)")
                elif distance < 3.5:
                    long_score += 18
                    reasons.append(f"📍 Approaching Support ${nearest_support['price']:.4f} ({nearest_support['touches']}x)")
            elif nearest_support and nearest_support['touches'] == 2:
                distance = (current_price - nearest_support['price']) / current_price * 100
                if distance < 1.5:
                    long_score += 20
                    reasons.append(f"🎯 At Support ${nearest_support['price']:.4f} (2x tested)")
            
            if nearest_resistance and nearest_resistance['touches'] >= 3:
                distance = (nearest_resistance['price'] - current_price) / current_price * 100
                
                if distance < 1.0:
                    short_score += 35
                    reasons.append(f"💎 TOUCHING MAJOR RESISTANCE ${nearest_resistance['price']:.4f} ({nearest_resistance['touches']}x TESTED)")
                elif distance < 2.0:
                    short_score += 28
                    reasons.append(f"🎯 Near Major Resistance ${nearest_resistance['price']:.4f} ({nearest_resistance['touches']}x)")
                elif distance < 3.5:
                    short_score += 18
            
            # [2] TREND BREAK / CONSOLIDATION BREAK (30 points)
            if trend_analysis['bullish_break']:
                long_score += trend_analysis['break_strength']
                if trend_analysis['consolidation_break']:
                    reasons.append(f"🚀 CONSOLIDATION BREAKOUT (Strength: {trend_analysis['break_strength']:.0f})")
                else:
                    reasons.append(f"📈 MAJOR UPTREND BREAK")
            
            if trend_analysis['bearish_break']:
                short_score += trend_analysis['break_strength']
                if trend_analysis['consolidation_break']:
                    reasons.append(f"💥 CONSOLIDATION BREAKDOWN")
                else:
                    reasons.append(f"📉 MAJOR DOWNTREND BREAK")
            
            # [3] DAILY TREND ALIGNMENT (20 points) - crucial for swing
            # Strong uptrend on daily
            if (l_d['ema_20'] > l_d['ema_50'] > l_d['ema_100'] and 
                l_d['close'] > l_d['ema_20']):
                long_score += 20
                reasons.append("✅ DAILY STRONG UPTREND")
            elif (l_d['ema_20'] > l_d['ema_50'] and l_d['close'] > l_d['ema_20']):
                long_score += 12
                reasons.append("✅ Daily Uptrend")
            
            # Strong downtrend on daily
            if (l_d['ema_20'] < l_d['ema_50'] < l_d['ema_100'] and 
                l_d['close'] < l_d['ema_20']):
                short_score += 20
                reasons.append("✅ DAILY STRONG DOWNTREND")
            elif (l_d['ema_20'] < l_d['ema_50'] and l_d['close'] < l_d['ema_20']):
                short_score += 12
                reasons.append("✅ Daily Downtrend")
            
            # [4] ADX TREND STRENGTH (15 points) - swing needs strong trends
            if l_d['adx'] > 30:  # Strong trend
                if l_d['ema_20'] > l_d['ema_50']:
                    long_score += 15
                    reasons.append(f"💪 STRONG TREND (ADX: {l_d['adx']:.0f})")
                else:
                    short_score += 15
                    reasons.append(f"💪 STRONG TREND (ADX: {l_d['adx']:.0f})")
            elif l_d['adx'] > 25:
                if l_d['ema_20'] > l_d['ema_50']:
                    long_score += 10
                else:
                    short_score += 10
            
            # [5] DAILY MACD (15 points)
            if l_d['macd'] > l_d['macd_signal'] and l_d['macd_hist'] > 0:
                long_score += 12
                if p_d['macd'] <= p_d['macd_signal']:  # Fresh cross
                    long_score += 3
                    reasons.append("⚡ DAILY MACD BULLISH CROSS")
            
            if l_d['macd'] < l_d['macd_signal'] and l_d['macd_hist'] < 0:
                short_score += 12
                if p_d['macd'] >= p_d['macd_signal']:
                    short_score += 3
                    reasons.append("⚡ DAILY MACD BEARISH CROSS")
            
            # [6] RSI POSITIONING (10 points) - not extreme
            if 40 <= l_d['rsi'] <= 60:  # Healthy zone
                if l_d['ema_20'] > l_d['ema_50']:
                    long_score += 8
                else:
                    short_score += 8
            elif 30 <= l_d['rsi'] < 40:  # Oversold but recovering
                long_score += 10
                reasons.append(f"💎 RSI RECOVERY ZONE ({l_d['rsi']:.0f})")
            elif 60 < l_d['rsi'] <= 70:  # Overbought but strong
                short_score += 10
                reasons.append(f"🔥 RSI HOT ZONE ({l_d['rsi']:.0f})")
            
            # [7] VOLUME CONFIRMATION (10 points)
            if l_d['volume_ratio'] > 1.5:  # Strong volume on daily
                if l_d['close'] > l_d['open']:
                    long_score += 10
                    reasons.append(f"📊 HEAVY VOLUME BUY ({l_d['volume_ratio']:.1f}x)")
                else:
                    short_score += 10
                    reasons.append(f"📊 HEAVY VOLUME SELL ({l_d['volume_ratio']:.1f}x)")
            elif l_d['volume_ratio'] > 1.2:
                if l_d['close'] > l_d['open']:
                    long_score += 6
                else:
                    short_score += 6
            
            # [8] PATTERN CONFIRMATION (bonus points)
            # Higher highs and higher lows (bullish)
            if (l_d['high'] > p_d['high'] and l_d['low'] > p_d['low'] and
                p_d['high'] > pp_d['high'] and p_d['low'] > pp_d['low']):
                long_score += 8
                reasons.append("📈 HIGHER HIGHS & LOWS")
            
            # Lower highs and lower lows (bearish)
            if (l_d['high'] < p_d['high'] and l_d['low'] < p_d['low'] and
                p_d['high'] < pp_d['high'] and p_d['low'] < pp_d['low']):
                short_score += 8
                reasons.append("📉 LOWER HIGHS & LOWS")
            
            # ⚠️ WARNINGS
            if l_d['rsi'] > 75:
                warnings.append("⚠️ DAILY RSI EXTREMELY HIGH - Wait for pullback")
                long_score -= 15
            elif l_d['rsi'] < 25:
                warnings.append("⚠️ DAILY RSI EXTREMELY LOW - Possible further drop")
                short_score -= 15
            
            if l_d['adx'] < 20:
                warnings.append("⚠️ WEAK TREND - Low ADX, choppy market")
                long_score -= 10
                short_score -= 10
            
            # 🎯 DECISION - HIGHER threshold for swing
            if long_score >= 70 and long_score > short_score + 20:
                signal = 'LONG'
                score = long_score
                confidence = 'EXTREME 🔥🔥🔥' if long_score >= 85 else ('HIGH 💎' if long_score >= 75 else 'GOOD ✅')
                
                # Swing SL: Use ATR or major support
                atr = l_d['atr']
                sl_atr = current_price - (2.5 * atr)  # 2.5 ATR stop
                sl_support = nearest_support['price'] * 0.99 if nearest_support else current_price * 0.95
                sl = max(sl_atr, sl_support)  # Use tighter of the two
                
                risk = (current_price - sl) / current_price * 100
                
                # SWING TARGETS - bigger moves!
                targets = [
                    current_price + (current_price - sl) * 2.5,   # TP1: 2.5R
                    current_price + (current_price - sl) * 4.0,   # TP2: 4R
                    current_price + (current_price - sl) * 6.0,   # TP3: 6R
                ]
                
            elif short_score >= 70 and short_score > long_score + 20:
                signal = 'SHORT'
                score = short_score
                confidence = 'EXTREME 🔥🔥🔥' if short_score >= 85 else ('HIGH 💎' if short_score >= 75 else 'GOOD ✅')
                
                atr = l_d['atr']
                sl_atr = current_price + (2.5 * atr)
                sl_resistance = nearest_resistance['price'] * 1.01 if nearest_resistance else current_price * 1.05
                sl = min(sl_atr, sl_resistance)
                
                risk = (sl - current_price) / current_price * 100
                
                targets = [
                    current_price - (sl - current_price) * 2.5,
                    current_price - (sl - current_price) * 4.0,
                    current_price - (sl - current_price) * 6.0,
                ]
            else:
                return None  # No swing setup
            
            # Calculate reward ratios
            rr = [(abs(tp - current_price) / abs(sl - current_price)) for tp in targets]
            
            # Target percentages
            target_pcts = [(abs(tp - current_price) / current_price * 100) for tp in targets]
            
            # Quality filter for swing trades
            if risk > self.max_risk_percent:
                logger.info(f"❌ {symbol} rejected - risk too high: {risk:.2f}%")
                return None
            
            if target_pcts[0] < self.min_target_percent:
                logger.info(f"❌ {symbol} rejected - target too small: {target_pcts[0]:.2f}%")
                return None
            
            if rr[0] < self.min_reward_ratio:
                logger.info(f"❌ {symbol} rejected - RR too low: {rr[0]:.2f}")
                return None
            
            return {
                'success': True,
                'symbol': symbol.replace('/USDT:USDT', ''),
                'full_symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'score': score,
                'entry': current_price,
                'stop_loss': sl,
                'targets': targets,
                'reward_ratios': rr,
                'target_percentages': target_pcts,
                'risk_percent': risk,
                'reasons': reasons,
                'warnings': warnings,
                'long_score': long_score,
                'short_score': short_score,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'trend_break': trend_analysis,
                'consolidation': consolidation_info,
                'daily_adx': l_d['adx'],
                'daily_rsi': l_d['rsi'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            return None
    
    def should_alert(self, symbol, result):
        """Check if we should alert - stricter for swing (avoid noise)"""
        if result['score'] < self.min_score_threshold:
            return False
        
        # Don't re-alert same pair within 24 hours (swing trades last days)
        if symbol in self.alerted_pairs:
            last_alert = self.alerted_pairs[symbol]
            if datetime.now() - last_alert['time'] < timedelta(hours=24):
                # Only re-alert if signal flipped OR score jumped 20+
                if (last_alert['signal'] == result['signal'] and 
                    result['score'] < last_alert['score'] + 20):
                    return False
        
        return True
    
    def format_swing_alert(self, result, rank=None):
        """Format beautiful swing trade alert - highlight GEMS!"""
        r = result
        emoji = "🚀" if r['signal'] == 'LONG' else "🔻"
        
        rank_text = f"#{rank} " if rank else ""
        
        # Check if this is a GEM (consolidation bounce)
        is_gem = r.get('consolidation', {}).get('is_consolidating', False)
        gem_badge = " 💎 GEM!" if is_gem else ""
        
        msg = f"{'='*50}\n"
        msg += f"{emoji} <b>{rank_text}SWING: {r['symbol']} - {r['confidence']}{gem_badge}</b> {emoji}\n"
        msg += f"{'='*50}\n\n"
        
        msg += f"<b>📊 {r['signal']}</b> | Score: {r['score']:.0f}/100\n"
        msg += f"ADX: {r['daily_adx']:.0f} | RSI: {r['daily_rsi']:.0f}\n\n"
        
        # Highlight consolidation zone if present
        if is_gem:
            cons = r['consolidation']
            msg += f"<b>📦 CONSOLIDATION ZONE:</b>\n"
            msg += f"  Support: ${cons['support_zone']:.6f}\n"
            msg += f"  Resistance: ${cons['resistance_zone']:.6f}\n"
            msg += f"  Duration: {cons['duration_candles']} candles\n\n"
        
        # Key levels
        msg += f"<b>🎯 MAJOR LEVELS:</b>\n"
        if r['nearest_support']:
            msg += f"  Support: ${r['nearest_support']['price']:.4f} ({r['nearest_support']['touches']}x tested)\n"
        if r['nearest_resistance']:
            msg += f"  Resistance: ${r['nearest_resistance']['price']:.4f} ({r['nearest_resistance']['touches']}x tested)\n"
        
        msg += f"\n<b>💰 TRADE SETUP (SWING):</b>\n"
        msg += f"  Entry: ${r['entry']:.6f}\n"
        msg += f"  SL: ${r['stop_loss']:.6f} ({r['risk_percent']:.2f}%)\n"
        
        msg += f"\n<b>🎯 SWING TARGETS:</b>\n"
        for i, (tp, rr, pct) in enumerate(zip(r['targets'], r['reward_ratios'], r['target_percentages']), 1):
            msg += f"  TP{i}: ${tp:.6f} (+{pct:.1f}%, {rr:.1f}R)\n"
        
        msg += f"\n<b>✅ SETUP REASONS:</b>\n"
        for reason in r['reasons'][:6]:
            msg += f"  • {reason}\n"
        
        if r['warnings']:
            msg += f"\n<b>⚠️ CAUTION:</b>\n"
            for warning in r['warnings']:
                msg += f"  {warning}\n"
        
        msg += f"\n<i>⏰ {r['timestamp'].strftime('%Y-%m-%d %H:%M')}</i>"
        msg += f"\n<i>💎 Hold: 3-10 days typically</i>"
        msg += f"\n{'='*50}"
        
        return msg
    
    async def scan_all_pairs(self):
        """Scan for SWING TRADING setups"""
        
        if not self.pairs_to_scan:
            logger.info("📥 Loading pairs for swing trading...")
            await self.load_all_usdt_pairs()
        
        logger.info(f"🔍 SWING SCAN: {len(self.pairs_to_scan)} pairs...")
        
        start_msg = f"🔍 <b>SWING SCAN STARTED</b>\n\nScanning {len(self.pairs_to_scan)} pairs for multi-day setups..."
        await self.send_msg(start_msg)
        
        scan_start_time = datetime.now()
        results = []
        alerts_sent = 0
        
        for i, pair in enumerate(self.pairs_to_scan, 1):
            try:
                if i % 30 == 0:
                    logger.info(f"Progress: {i}/{len(self.pairs_to_scan)}...")
                
                full_symbol = await self.get_symbol_format(pair)
                if not full_symbol:
                    continue
                
                data = await self.fetch_data(full_symbol)
                if not data:
                    continue
                
                result = self.analyze_swing_setup(data, full_symbol)
                
                if result and result['success']:
                    results.append(result)
                    logger.info(f"✅ SWING: {pair} {result['signal']} ({result['score']:.0f})")
                    
                    if self.should_alert(result['full_symbol'], result) and alerts_sent < self.max_alerts_per_scan:
                        alerts_sent += 1
                        msg = self.format_swing_alert(result, rank=alerts_sent)
                        await self.send_msg(msg)
                        
                        # Add to tracking
                        trade_id = f"{result['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        self.active_trades[trade_id] = {
                            'trade_id': trade_id,
                            'symbol': result['symbol'],
                            'full_symbol': result['full_symbol'],
                            'signal': result['signal'],
                            'entry': result['entry'],
                            'stop_loss': result['stop_loss'],
                            'targets': result['targets'],
                            'reward_ratios': result['reward_ratios'],
                            'timestamp': datetime.now(),
                            'tp_hit': [False, False, False],
                            'sl_hit': False
                        }
                        
                        self.alerted_pairs[result['full_symbol']] = {
                            'time': datetime.now(),
                            'signal': result['signal'],
                            'score': result['score']
                        }
                        
                        self.stats['trades_tracked'] += 1
                        self.stats['signals_found'] += 1
                        
                        if result['score'] >= 80:
                            self.stats['high_conviction_signals'] += 1
                
                await asyncio.sleep(0.15)
                
            except Exception as e:
                logger.error(f"Error scanning {pair}: {e}")
                continue
        
        scan_duration = (datetime.now() - scan_start_time).total_seconds()
        
        self.stats['total_scans'] += 1
        self.stats['total_pairs_scanned'] += len(self.pairs_to_scan)
        self.stats['avg_scan_time'] = scan_duration
        self.stats['last_scan_date'] = datetime.now()
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"✅ Swing scan complete! {len(results)} setups in {scan_duration:.1f}s")
        
        # Summary
        elite = [r for r in results if r['score'] >= 80]
        high = [r for r in results if 70 <= r['score'] < 80]
        
        summary = f"✅ <b>SWING SCAN COMPLETE</b>\n\n"
        summary += f"📊 Scanned: {len(self.pairs_to_scan)} pairs\n"
        summary += f"⏱️ Time: {scan_duration/60:.1f} min\n"
        summary += f"✅ Found: {len(results)} swing setups\n"
        summary += f"🔥 Elite (80+): {len(elite)}\n"
        summary += f"💎 High (70-79): {len(high)}\n"
        summary += f"📤 Alerts sent: {alerts_sent}\n\n"
        summary += f"📡 Tracking {len(self.active_trades)} trades"
        
        await self.send_msg(summary)
        
        return results
    
    async def send_msg(self, msg):
        """Send telegram message"""
        try:
            await self.telegram_bot.send_message(chat_id=self.chat_id, text=msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Send error: {e}")
    
    async def send_scan_results(self, results):
        """Results already sent during scan"""
        pass
    
    async def auto_scan_loop(self):
        """Auto-scan loop for swing trading"""
        logger.info(f"🤖 SWING AUTO-SCAN STARTED (every {self.scan_interval/3600:.1f} hours)")
        
        while self.is_scanning:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Starting swing scan cycle...")
                
                results = await self.scan_all_pairs()
                await self.send_scan_results(results)
                
                self.last_scan_time = datetime.now()
                
                next_scan = datetime.now() + timedelta(seconds=self.scan_interval)
                logger.info(f"⏰ Next swing scan at: {next_scan.strftime('%H:%M:%S')}")
                logger.info(f"{'='*60}\n")
                
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                await asyncio.sleep(300)
    
    async def close(self):
        await self.exchange.close()
    
    async def track_trades_loop(self):
        """Monitor swing trades for TP/SL"""
        logger.info("📡 Swing trade tracking started")
        
        while self.is_tracking:
            try:
                if not self.active_trades:
                    await asyncio.sleep(self.price_check_interval)
                    continue
                
                trades_to_remove = []
                
                for trade_id, trade in list(self.active_trades.items()):
                    try:
                        # Swing timeout: 14 days
                        if datetime.now() - trade['timestamp'] > timedelta(days=14):
                            msg = f"⏰ <b>SWING TRADE TIMEOUT (14 DAYS)</b>\n\n"
                            msg += f"<code>{trade_id}</code>\n"
                            msg += f"{trade['symbol']} {trade['signal']}\n\n"
                            msg += f"Consider reviewing position."
                            await self.send_msg(msg)
                            trades_to_remove.append(trade_id)
                            continue
                        
                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        current_price = ticker['last']
                        
                        if 'tp_hit' not in trade:
                            trade['tp_hit'] = [False, False, False]
                        if 'sl_hit' not in trade:
                            trade['sl_hit'] = False
                        
                        # LONG trades
                        if trade['signal'] == 'LONG':
                            for i, (tp, hit) in enumerate(zip(trade['targets'], trade['tp_hit'])):
                                if not hit and current_price >= tp:
                                    profit_pct = ((tp - trade['entry']) / trade['entry']) * 100
                                    
                                    msg = f"🎯 <b>SWING TP{i+1} HIT!</b> 🎯\n\n"
                                    msg += f"<code>{trade_id}</code>\n"
                                    msg += f"<b>{trade['symbol']} LONG</b>\n\n"
                                    msg += f"Entry: ${trade['entry']:.6f}\n"
                                    msg += f"TP{i+1}: ${tp:.6f}\n"
                                    msg += f"Current: ${current_price:.6f}\n"
                                    msg += f"Profit: <b>+{profit_pct:.2f}%</b> ({trade['reward_ratios'][i]:.1f}R)\n\n"
                                    
                                    if i == 0:
                                        msg += "💡 <b>TAKE 40% PROFIT</b>\n"
                                        msg += "📋 Move SL to breakeven"
                                    elif i == 1:
                                        msg += "💡 <b>TAKE 40% PROFIT</b>\n"
                                        msg += "📋 Trail SL to TP1"
                                    else:
                                        msg += "💡 <b>CLOSE REMAINING 20%</b>\n"
                                        msg += "🎊 <b>SWING TRADE COMPLETE!</b>"
                                        trades_to_remove.append(trade_id)
                                    
                                    await self.send_msg(msg)
                                    trade['tp_hit'][i] = True
                                    
                                    if i == 0:
                                        self.stats['tp1_hits'] += 1
                                    elif i == 1:
                                        self.stats['tp2_hits'] += 1
                                    else:
                                        self.stats['tp3_hits'] += 1
                            
                            # Check SL
                            if not trade['sl_hit'] and current_price <= trade['stop_loss']:
                                loss_pct = ((trade['stop_loss'] - trade['entry']) / trade['entry']) * 100
                                
                                msg = f"🛑 <b>SWING STOP HIT</b>\n\n"
                                msg += f"<code>{trade_id}</code>\n"
                                msg += f"<b>{trade['symbol']} LONG</b>\n\n"
                                msg += f"Entry: ${trade['entry']:.6f}\n"
                                msg += f"SL: ${trade['stop_loss']:.6f}\n"
                                msg += f"Current: ${current_price:.6f}\n"
                                msg += f"Loss: <b>{loss_pct:.2f}%</b>\n\n"
                                msg += f"On to the next swing! 💪"
                                
                                await self.send_msg(msg)
                                trade['sl_hit'] = True
                                self.stats['sl_hits'] += 1
                                trades_to_remove.append(trade_id)
                        
                        # SHORT trades
                        else:
                            for i, (tp, hit) in enumerate(zip(trade['targets'], trade['tp_hit'])):
                                if not hit and current_price <= tp:
                                    profit_pct = ((trade['entry'] - tp) / trade['entry']) * 100
                                    
                                    msg = f"🎯 <b>SWING TP{i+1} HIT!</b> 🎯\n\n"
                                    msg += f"<code>{trade_id}</code>\n"
                                    msg += f"<b>{trade['symbol']} SHORT</b>\n\n"
                                    msg += f"Entry: ${trade['entry']:.6f}\n"
                                    msg += f"TP{i+1}: ${tp:.6f}\n"
                                    msg += f"Current: ${current_price:.6f}\n"
                                    msg += f"Profit: <b>+{profit_pct:.2f}%</b> ({trade['reward_ratios'][i]:.1f}R)\n\n"
                                    
                                    if i == 0:
                                        msg += "💡 <b>TAKE 40% PROFIT</b>\n"
                                        msg += "📋 Move SL to breakeven"
                                    elif i == 1:
                                        msg += "💡 <b>TAKE 40% PROFIT</b>\n"
                                        msg += "📋 Trail SL to TP1"
                                    else:
                                        msg += "💡 <b>CLOSE REMAINING 20%</b>\n"
                                        msg += "🎊 <b>SWING TRADE COMPLETE!</b>"
                                        trades_to_remove.append(trade_id)
                                    
                                    await self.send_msg(msg)
                                    trade['tp_hit'][i] = True
                                    
                                    if i == 0:
                                        self.stats['tp1_hits'] += 1
                                    elif i == 1:
                                        self.stats['tp2_hits'] += 1
                                    else:
                                        self.stats['tp3_hits'] += 1
                            
                            if not trade['sl_hit'] and current_price >= trade['stop_loss']:
                                loss_pct = ((current_price - trade['entry']) / trade['entry']) * 100
                                
                                msg = f"🛑 <b>SWING STOP HIT</b>\n\n"
                                msg += f"<code>{trade_id}</code>\n"
                                msg += f"<b>{trade['symbol']} SHORT</b>\n\n"
                                msg += f"Entry: ${trade['entry']:.6f}\n"
                                msg += f"SL: ${trade['stop_loss']:.6f}\n"
                                msg += f"Current: ${current_price:.6f}\n"
                                msg += f"Loss: <b>{loss_pct:.2f}%</b>\n\n"
                                msg += f"On to the next swing! 💪"
                                
                                await self.send_msg(msg)
                                trade['sl_hit'] = True
                                self.stats['sl_hits'] += 1
                                trades_to_remove.append(trade_id)
                        
                    except Exception as e:
                        logger.error(f"Error tracking {trade_id}: {e}")
                        continue
                
                for trade_id in trades_to_remove:
                    if trade_id in self.active_trades:
                        logger.info(f"✅ Stopped tracking: {trade_id}")
                        del self.active_trades[trade_id]
                
                self.stats['active_trades_count'] = len(self.active_trades)
                
                await asyncio.sleep(self.price_check_interval)
                
            except Exception as e:
                logger.error(f"Error in tracking: {e}")
                await asyncio.sleep(60)


class SwingBotCommands:
    def __init__(self, scanner):
        self.scanner = scanner
    
    async def cmd_start(self, update, context):
        msg = "🎯 <b>SWING TRADING SCANNER</b>\n\n"
        msg += "Scans 150 most liquid USDT pairs for MULTI-DAY setups\n\n"
        msg += "<b>🎯 SWING STRATEGY:</b>\n"
        msg += "• Daily + 4H timeframes\n"
        msg += "• Major S/R (3+ touches)\n"
        msg += "• Consolidation breakouts\n"
        msg += "• 5-20% target moves\n"
        msg += "• 1:3 to 1:6 reward/risk\n"
        msg += "• Hold: 3-10 days typically\n\n"
        msg += "<b>📊 COMMANDS:</b>\n"
        msg += "/start_scan - Auto-scan every 4h\n"
        msg += "/stop_scan - Stop scanning\n"
        msg += "/scan_now - Manual scan\n\n"
        msg += "/start_tracking - TP/SL alerts\n"
        msg += "/stop_tracking - Stop alerts\n"
        msg += "/active_trades - View trades\n\n"
        msg += "/status - Scanner status\n"
        msg += "/stats - Statistics\n\n"
        msg += "💎 <b>Quality over quantity!</b>"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    
    async def cmd_start_scan(self, update, context):
        if self.scanner.is_scanning:
            await update.message.reply_text("⚠️ Already scanning!", parse_mode=ParseMode.HTML)
            return
        
        self.scanner.is_scanning = True
        asyncio.create_task(self.scanner.auto_scan_loop())
        
        msg = f"✅ <b>SWING SCANNER STARTED!</b>\n\n"
        msg += f"📊 Pairs: {len(self.scanner.pairs_to_scan) if self.scanner.pairs_to_scan else '150'}\n"
        msg += f"⏰ Every {self.scanner.scan_interval/3600:.1f} hours\n"
        msg += f"🎯 Min score: {self.scanner.min_score_threshold}\n"
        msg += f"📤 Max alerts: {self.scanner.max_alerts_per_scan}\n\n"
        msg += f"First swing scan starting now..."
        
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    
    async def cmd_stop_scan(self, update, context):
        if not self.scanner.is_scanning:
            await update.message.reply_text("⚠️ Not scanning!", parse_mode=ParseMode.HTML)
            return
        
        self.scanner.is_scanning = False
        await update.message.reply_text("🛑 <b>SWING SCANNER STOPPED</b>", parse_mode=ParseMode.HTML)
    
    async def cmd_scan_now(self, update, context):
        await update.message.reply_text("🔍 Starting swing scan...", parse_mode=ParseMode.HTML)
        results = await self.scanner.scan_all_pairs()
    
    async def cmd_status(self, update, context):
        scan_status = "🟢 RUNNING" if self.scanner.is_scanning else "🔴 STOPPED"
        track_status = "🟢 RUNNING" if self.scanner.is_tracking else "🔴 STOPPED"
        
        msg = f"<b>📊 SWING SCANNER STATUS</b>\n\n"
        msg += f"<b>Scanning:</b> {scan_status}\n"
        msg += f"<b>Tracking:</b> {track_status}\n\n"
        msg += f"Pairs: {len(self.scanner.pairs_to_scan)}\n"
        msg += f"Interval: {self.scanner.scan_interval/3600:.1f}h\n"
        msg += f"Min Score: {self.scanner.min_score_threshold}\n"
        msg += f"Active Trades: {len(self.scanner.active_trades)}\n\n"
        
        if self.scanner.last_scan_time:
            time_since = datetime.now() - self.scanner.last_scan_time
            msg += f"Last Scan: {time_since.seconds//3600}h {(time_since.seconds%3600)//60}m ago\n"
        
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    
    async def cmd_stats(self, update, context):
        s = self.scanner.stats
        
        msg = f"<b>📊 SWING TRADING STATS</b>\n\n"
        msg += f"<b>Scanning:</b>\n"
        msg += f"Total Scans: {s['total_scans']}\n"
        msg += f"Signals Found: {s['signals_found']}\n"
        msg += f"Elite Signals (80+): {s['high_conviction_signals']} 🔥\n\n"
        
        msg += f"<b>Trade Results:</b>\n"
        msg += f"Tracked: {s['trades_tracked']}\n"
        msg += f"Active: {s['active_trades_count']}\n"
        msg += f"TP1: {s['tp1_hits']} 🎯\n"
        msg += f"TP2: {s['tp2_hits']} 🎯\n"
        msg += f"TP3: {s['tp3_hits']} 🎯\n"
        msg += f"SL: {s['sl_hits']} 🛑\n"
        
        total_closed = s['tp1_hits'] + s['sl_hits']
        if total_closed > 0:
            win_rate = (s['tp1_hits'] / total_closed) * 100
            msg += f"\nWin Rate: {win_rate:.1f}%\n"
        
        if s['total_scans'] > 0:
            avg_signals = s['signals_found'] / s['total_scans']
            msg += f"Avg Signals/Scan: {avg_signals:.1f}\n"
        
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    
    async def cmd_start_tracking(self, update, context):
        if self.scanner.is_tracking:
            await update.message.reply_text("⚠️ Already tracking!", parse_mode=ParseMode.HTML)
            return
        
        self.scanner.is_tracking = True
        asyncio.create_task(self.scanner.track_trades_loop())
        
        msg = f"✅ <b>SWING TRACKING STARTED!</b>\n\n"
        msg += f"📡 Checking every {self.scanner.price_check_interval}s\n"
        msg += f"Active: {len(self.scanner.active_trades)}"
        
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)
    
    async def cmd_stop_tracking(self, update, context):
        if not self.scanner.is_tracking:
            await update.message.reply_text("⚠️ Not tracking!", parse_mode=ParseMode.HTML)
            return
        
        self.scanner.is_tracking = False
        await update.message.reply_text("🛑 <b>TRACKING STOPPED</b>", parse_mode=ParseMode.HTML)
    
    async def cmd_active_trades(self, update, context):
        trades = self.scanner.active_trades
        
        if not trades:
            await update.message.reply_text("📭 No active swing trades", parse_mode=ParseMode.HTML)
            return
        
        msg = f"📡 <b>ACTIVE SWING TRADES ({len(trades)})</b>\n\n"
        
        for trade_id, trade in list(trades.items())[:15]:
            age_days = (datetime.now() - trade['timestamp']).days
            age_hours = int((datetime.now() - trade['timestamp']).total_seconds() / 3600)
            tp_status = "".join(["✅" if hit else "⏳" for hit in trade['tp_hit']])
            
            msg += f"<b>{trade['symbol']}</b> {trade['signal']}\n"
            msg += f"  Entry: ${trade['entry']:.6f}\n"
            msg += f"  TPs: {tp_status}\n"
            msg += f"  Age: {age_days}d {age_hours%24}h\n\n"
        
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def main():
    TELEGRAM_TOKEN = "8186622122:AAGtQcoh_s7QqIAVACmOYVHLqPX-p6dSNVA"
    TELEGRAM_CHAT_ID = "7500072234"
    
    scanner = SwingTradingScanner(
        telegram_token=TELEGRAM_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID,
        binance_api_key=None,
        binance_secret=None
    )
    
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    commands = SwingBotCommands(scanner)
    
    app.add_handler(CommandHandler("start", commands.cmd_start))
    app.add_handler(CommandHandler("start_scan", commands.cmd_start_scan))
    app.add_handler(CommandHandler("stop_scan", commands.cmd_stop_scan))
    app.add_handler(CommandHandler("scan_now", commands.cmd_scan_now))
    app.add_handler(CommandHandler("status", commands.cmd_status))
    app.add_handler(CommandHandler("stats", commands.cmd_stats))
    app.add_handler(CommandHandler("start_tracking", commands.cmd_start_tracking))
    app.add_handler(CommandHandler("stop_tracking", commands.cmd_stop_tracking))
    app.add_handler(CommandHandler("active_trades", commands.cmd_active_trades))
    
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    
    logger.info("🎯 SWING TRADING BOT ONLINE!")
    
    welcome = "🎯 <b>SWING TRADING SCANNER READY!</b> 🎯\n\n"
    welcome += "✅ Top 150 liquid pairs\n"
    welcome += "✅ Daily + 4H analysis\n"
    welcome += "✅ 5-20% target moves\n"
    welcome += "✅ 1:3 to 1:6 R/R\n"
    welcome += "✅ Scans every 4 hours\n"
    welcome += "✅ Elite setups only (70+ score)\n\n"
    welcome += "<b>🚀 QUICK START:</b>\n"
    welcome += "/start_scan - Start scanning\n"
    welcome += "/start_tracking - Track TPs\n\n"
    welcome += "💎 <b>Quality swing setups incoming!</b>"
    
    await scanner.send_msg(welcome)
    
    # Auto-start tracking
    logger.info("🤖 Auto-starting trade tracking...")
    scanner.is_tracking = True
    tracking_task = asyncio.create_task(scanner.track_trades_loop())
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("⚠️ Shutting down...")
    finally:
        scanner.is_scanning = False
        scanner.is_tracking = False
        if 'tracking_task' in locals():
            tracking_task.cancel()
        await scanner.close()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
