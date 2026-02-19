import asyncio
import ccxt.async_support as ccxt
from telegram import Bot
from telegram.ext import Application, CommandHandler
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')


class iRatersEliteScanner:
    """
    💎 iRATERS ELITE FUTURES SETUP SCANNER
    
    EXACT STRATEGY from @Albert_Adams04 (iRaters):
    
    ══════════════════════════════════════════════════════════════
    TIMEFRAME: 1 HOUR (Binance USDT Perpetual Futures)
    FOCUS: Small/mid-cap gems (volatile, <$200M mcap preferred)
    ══════════════════════════════════════════════════════════════
    
    SETUP REQUIREMENTS (ALL MUST BE PRESENT):
    
    1. Recent pump then pullback (5-20%+ dip)
    2. LIQUIDITY SWEEP: New low wick sweeps below previous significant low
    3. DEMAND ZONE forms: 3-8 candles tight consolidation after sweep
       - Multiple touches of zone low
       - Small bodies, rejection wicks
       - Strong defense of lows
    4. RETEST: Price pulls back into demand zone (entry opportunity)
    
    ENTRY: LONG in demand zone (upper edge preferred)
    STOP LOSS: Below demand zone low (tight 1-3%)
    TARGETS: Measured move 2-4x zone height (10-40%+ on gems)
    
    LEVERAGE: 10-50x (HIGH RISK!)
    DURATION: 1-8 hours (scalp)
    
    OUTPUT FORMAT: Exact "ELITE FUTURES SETUP" style
    """

    def __init__(self, telegram_token, telegram_chat_id,
                 binance_api_key=None, binance_secret=None):
        self.telegram_token = telegram_token
        self.telegram_bot   = Bot(token=telegram_token)
        self.chat_id        = telegram_chat_id
        self.exchange = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # 1H timeframe = scan every 30-60 min
        self.scan_interval        = 1800   # 30 minutes
        self.min_score_threshold  = 75
        self.max_alerts_per_scan  = 8
        self.price_check_interval = 120    # 2 minutes

        self.alerted_pairs  = {}
        self.active_trades  = {}
        self.last_scan_time = None
        self.is_scanning    = False
        self.is_tracking    = False
        self.pairs_to_scan  = []
        self.all_symbols    = []

        self.stats = {
            'total_scans': 0, 'total_pairs_scanned': 0, 'signals_found': 0,
            'tp1_hits': 0, 'tp2_hits': 0, 'tp3_hits': 0, 'sl_hits': 0,
            'active_trades_count': 0, 'last_scan_date': None, 'avg_scan_time': 0
        }

    async def get_symbol_format(self, sym):
        try:
            sym = sym.upper().strip()
            candidates = [f"{sym}/USDT:USDT", f"{sym}USDT/USDT:USDT", sym]
            if sym.endswith('USDT'):
                candidates.insert(0, f"{sym[:-4]}/USDT:USDT")
            for c in candidates:
                if c in self.exchange.symbols:
                    return c
            return None
        except:
            return None

    async def load_all_usdt_pairs(self):
        try:
            logger.info("Loading pairs...")
            await self.exchange.load_markets()
            perps = []
            for symbol, market in self.exchange.markets.items():
                if (market.get('quote') == 'USDT' and market.get('type') == 'swap'
                        and market.get('settle') == 'USDT' and market.get('active', True)):
                    perps.append({
                        'base': market['base'], 'symbol': symbol,
                        'volume': float(market.get('info', {}).get('volume', 0) or 0)
                    })
            perps.sort(key=lambda x: x['volume'], reverse=True)
            
            # Focus on top 100 by volume (includes gems with volume spikes)
            self.pairs_to_scan = [p['base'] for p in perps[:100]]
            self.all_symbols   = [p['symbol'] for p in perps[:100]]
            logger.info(f"Loaded {len(self.pairs_to_scan)} pairs")
            return True
        except Exception as e:
            logger.error(f"Load: {e}")
            self.pairs_to_scan = ['BTC','ETH','BNB','SOL','XRP','ADA','AVAX','DOGE']
            return False

    async def fetch_df(self, symbol, tf, limit):
        ohlcv = await self.exchange.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    async def fetch_data(self, symbol):
        try:
            data = {}
            # 1H primary timeframe
            data['1h'] = await self.fetch_df(symbol, '1h', 200)
            await asyncio.sleep(0.05)
            return data
        except Exception as e:
            logger.error(f"Fetch {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────
    # CORE: iRATERS STRATEGY DETECTION
    # ─────────────────────────────────────────────────

    def detect_recent_pump_pullback(self, df, lookback=50):
        """
        Step 1: Detect if coin had recent pump then sharp pullback.
        Required before liquidity sweep setup.
        """
        recent = df.tail(lookback)
        
        # Find the recent high
        recent_high = recent['high'].max()
        recent_high_idx = recent['high'].idxmax()
        
        # Current price
        current = df['close'].iloc[-1]
        
        # Calculate pullback from high
        pullback_pct = (recent_high - current) / recent_high * 100
        
        # Check if there was an uptrend before the high
        pre_high = df.iloc[:recent_high_idx]
        if len(pre_high) < 20:
            return {'valid': False}
        
        # Simple trend check: price moved up significantly before high
        early_price = pre_high['close'].iloc[-30] if len(pre_high) >= 30 else pre_high['close'].iloc[0]
        pump_pct = (recent_high - early_price) / early_price * 100
        
        # Valid if: pumped at least 15% then pulled back 5-25%
        valid = pump_pct >= 15 and 5 <= pullback_pct <= 30
        
        return {
            'valid': valid,
            'recent_high': recent_high,
            'pump_pct': pump_pct,
            'pullback_pct': pullback_pct,
            'high_idx': recent_high_idx
        }

    def detect_liquidity_sweep(self, df, pump_info, lookback=40):
        """
        Step 2: Detect LIQUIDITY SWEEP.
        = New low wick that sweeps below a significant prior low, then reverses up.
        
        This is THE KEY SIGNAL in iRaters' strategy.
        """
        if not pump_info['valid']:
            return {'swept': False}
        
        high_idx = pump_info['high_idx']
        
        # Look at price action AFTER the pump high
        after_high = df.loc[high_idx:]
        
        if len(after_high) < 10:
            return {'swept': False}
        
        # Find swing lows in the pullback
        swing_lows = []
        for i in range(5, len(after_high) - 3):
            if after_high['low'].iloc[i] == after_high['low'].iloc[i-5:i+4].min():
                swing_lows.append({
                    'idx': i,
                    'price': after_high['low'].iloc[i]
                })
        
        if len(swing_lows) < 2:
            return {'swept': False}
        
        # Check for sweep: later candle makes lower low (sweep) then closes back above
        for i in range(len(swing_lows) - 1):
            prev_low = swing_lows[i]['price']
            
            # Check candles after this low for a sweep
            check_start = swing_lows[i]['idx'] + 1
            check_end = min(len(after_high), swing_lows[i]['idx'] + 15)
            
            for j in range(check_start, check_end):
                candle = after_high.iloc[j]
                
                # Sweep = wick goes below prev low but closes significantly above it
                if (candle['low'] < prev_low * 0.995 and 
                    candle['close'] > prev_low * 1.005):
                    
                    # Found a sweep!
                    sweep_depth = (prev_low - candle['low']) / prev_low * 100
                    candles_ago = len(df) - 1 - (high_idx + j)
                    
                    return {
                        'swept': True,
                        'sweep_low': candle['low'],
                        'swept_level': prev_low,
                        'sweep_depth': sweep_depth,
                        'sweep_idx': high_idx + j,
                        'candles_ago': candles_ago
                    }
        
        return {'swept': False}

    def detect_demand_zone(self, df, sweep_info):
        """
        Step 3: Detect DEMAND ZONE formation after liquidity sweep.
        
        = 3-8 candles of tight consolidation where lows are defended.
        This becomes the beige box / demand zone in charts.
        """
        if not sweep_info['swept']:
            return {'zone': False}
        
        sweep_idx = sweep_info['sweep_idx']
        
        # Look at 3-15 candles AFTER the sweep
        after_sweep = df.iloc[sweep_idx:sweep_idx + 16]
        
        if len(after_sweep) < 4:
            return {'zone': False}
        
        # Try different consolidation windows (3-8 candles)
        best_zone = None
        best_score = 0
        
        for window in [3, 4, 5, 6, 7, 8]:
            if len(after_sweep) < window:
                continue
            
            zone_candles = after_sweep.iloc[:window]
            
            zone_high = zone_candles['high'].max()
            zone_low  = zone_candles['low'].min()
            
            if zone_low == 0:
                continue
            
            # Range of zone
            zone_range_pct = (zone_high - zone_low) / zone_low * 100
            
            # Must be tight (< 8-10%)
            if zone_range_pct > 10:
                continue
            
            # Count touches of zone low (defense)
            touches = sum(1 for low in zone_candles['low'] 
                         if abs(low - zone_low) / zone_low < 0.015)
            
            # Count rejection wicks (low wicks that close higher)
            rejections = sum(1 for i, row in zone_candles.iterrows()
                           if (row['close'] - row['low']) / row['low'] > 0.01)
            
            # Score the zone
            tightness_score = 10 - zone_range_pct
            touch_score = touches * 3
            rejection_score = rejections * 2
            
            score = tightness_score + touch_score + rejection_score
            
            if score > best_score:
                best_score = score
                best_zone = {
                    'zone': True,
                    'zone_low': zone_low,
                    'zone_high': zone_high,
                    'zone_mid': (zone_high + zone_low) / 2,
                    'zone_range_pct': zone_range_pct,
                    'zone_start_idx': sweep_idx,
                    'zone_end_idx': sweep_idx + window,
                    'duration': window,
                    'touches': touches,
                    'rejections': rejections,
                    'score': score
                }
        
        if best_zone:
            return best_zone
        
        return {'zone': False}

    def detect_retest(self, df, demand_zone):
        """
        Step 4: Detect if price is currently RETESTING the demand zone.
        
        = Price pulled back into/near demand zone = ENTRY OPPORTUNITY!
        This is the orange entry box in charts.
        """
        if not demand_zone['zone']:
            return {'retest': False}
        
        zone_low = demand_zone['zone_low']
        zone_high = demand_zone['zone_high']
        zone_end = demand_zone['zone_end_idx']
        
        current = df['close'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        # Check if price broke above zone then came back
        after_zone = df.iloc[zone_end:]
        
        if len(after_zone) < 3:
            return {'retest': False}
        
        # Did price move above zone?
        broke_above = any(c > zone_high * 1.02 for c in after_zone['high'])
        
        if not broke_above:
            # Maybe price is still inside zone forming it
            if zone_low <= current <= zone_high * 1.03:
                return {
                    'retest': True,
                    'retest_type': 'inside_zone',
                    'distance_from_zone_pct': 0
                }
            return {'retest': False}
        
        # Price broke above and now testing back
        # Check if current price is in/near zone
        in_zone = zone_low * 0.97 <= current <= zone_high * 1.03
        
        if in_zone:
            dist = (current - zone_high) / zone_high * 100
            return {
                'retest': True,
                'retest_type': 'pullback_to_zone',
                'distance_from_zone_pct': dist
            }
        
        # Check if approaching zone from above
        approaching = zone_high * 1.03 < current < zone_high * 1.08
        if approaching:
            return {
                'retest': True,
                'retest_type': 'approaching_zone',
                'distance_from_zone_pct': (current - zone_high) / zone_high * 100
            }
        
        return {'retest': False}

    def add_indicators(self, df):
        if len(df) < 50:
            return df
        try:
            df['ema_20']    = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['rsi']       = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['vol_sma']   = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, 1)
            df['atr']       = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=14).average_true_range()
        except:
            pass
        return df

    # ─────────────────────────────────────────────────
    # MAIN ANALYSIS
    # ─────────────────────────────────────────────────

    def analyze_elite_setup(self, data, symbol):
        """
        Full iRaters Elite Futures Setup analysis.
        All 4 requirements must be met.
        """
        try:
            if not data or '1h' not in data:
                return None

            df = self.add_indicators(data['1h'].copy())

            if len(df) < 100:
                return None

            current = df['close'].iloc[-1]
            l1h     = df.iloc[-1]
            rsi     = float(l1h.get('rsi', 50) or 50)
            volr    = float(l1h.get('vol_ratio', 1) or 1)

            # ── STEP 1: Recent pump then pullback ──
            pump = self.detect_recent_pump_pullback(df, lookback=60)
            
            if not pump['valid']:
                return None

            # ── STEP 2: Liquidity sweep ──
            sweep = self.detect_liquidity_sweep(df, pump, lookback=50)
            
            if not sweep['swept']:
                return None

            # ── STEP 3: Demand zone formation ──
            demand = self.detect_demand_zone(df, sweep)
            
            if not demand['zone']:
                return None

            # ── STEP 4: Retest of demand zone ──
            retest = self.detect_retest(df, demand)
            
            if not retest['retest']:
                return None

            # ── ALL 4 CONDITIONS MET! ──
            
            # ── SCORING ──
            score    = 0
            reasons  = []
            warnings = []

            # [A] Pump & pullback quality (0-20 pts)
            if pump['pump_pct'] > 30:
                score += 20
                reasons.append(f"🚀 Strong pump +{pump['pump_pct']:.0f}% before pullback")
            elif pump['pump_pct'] > 20:
                score += 15
                reasons.append(f"🚀 Pump +{pump['pump_pct']:.0f}%")
            else:
                score += 10

            # [B] Liquidity sweep (0-25 pts)
            if sweep['candles_ago'] <= 10:
                score += 25
                reasons.append(f"💥 FRESH liquidity sweep ({sweep['candles_ago']}h ago)")
            elif sweep['candles_ago'] <= 20:
                score += 18
                reasons.append(f"💥 Liquidity sweep ({sweep['candles_ago']}h ago)")
            else:
                score += 10
                warnings.append(f"⚠️ Sweep was {sweep['candles_ago']}h ago")

            # [C] Demand zone quality (0-30 pts)
            if demand['zone_range_pct'] < 3:
                score += 30
                reasons.append(f"📦 TIGHT demand zone ({demand['zone_range_pct']:.1f}%, {demand['duration']} candles)")
            elif demand['zone_range_pct'] < 5:
                score += 25
                reasons.append(f"📦 Strong demand zone ({demand['zone_range_pct']:.1f}%)")
            elif demand['zone_range_pct'] < 8:
                score += 18
                reasons.append(f"📦 Demand zone ({demand['zone_range_pct']:.1f}%)")
            else:
                score += 10
                warnings.append(f"⚠️ Zone is a bit wide ({demand['zone_range_pct']:.1f}%)")

            # Bonus for touches/rejections
            if demand['touches'] >= 3:
                score += 5
                reasons.append(f"🎯 {demand['touches']} touches of zone low")

            # [D] Retest position (0-20 pts)
            if retest['retest_type'] == 'inside_zone':
                score += 20
                reasons.append(f"🎯 INSIDE DEMAND ZONE - prime entry!")
            elif retest['retest_type'] == 'pullback_to_zone':
                score += 18
                reasons.append(f"🎯 Retesting demand zone")
            elif retest['retest_type'] == 'approaching_zone':
                score += 12
                reasons.append(f"🎯 Approaching zone")

            # [E] Volume (0-10 pts)
            if volr > 2.0:
                score += 10
                reasons.append(f"📊 High volume ({volr:.1f}x)")
            elif volr > 1.3:
                score += 5

            # [F] RSI (0-10 pts)
            if 35 <= rsi <= 55:
                score += 10
                reasons.append(f"💎 RSI neutral ({rsi:.0f}) - room to move")
            elif rsi < 35:
                score += 8
                reasons.append(f"💎 RSI oversold ({rsi:.0f})")

            # ── WARNINGS ──
            if rsi > 70:
                warnings.append("⚠️ RSI overbought - risky entry")
                score -= 10
            if demand['zone_range_pct'] > 8:
                warnings.append("⚠️ Zone too wide - less reliable")
                score -= 5

            if score < self.min_score_threshold:
                return None

            # ── TRADE LEVELS (iRaters style) ──
            
            # ENTRY: Upper part of demand zone
            entry = demand['zone_high'] * 0.995  # Just below zone high
            
            # STOP LOSS: Below demand zone low (tight!)
            sl = demand['zone_low'] * 0.985
            risk_pct = (entry - sl) / entry * 100
            
            # If risk too high, adjust
            if risk_pct > 4:
                sl = entry * 0.97
                risk_pct = 3.0

            # TARGETS: Measured move (2-4x zone height)
            zone_height = demand['zone_high'] - demand['zone_low']
            
            tp1 = entry + (zone_height * 1.5)   # 1.5x
            tp2 = entry + (zone_height * 2.5)   # 2.5x
            tp3 = entry + (zone_height * 3.5)   # 3.5x
            tp4 = entry + (zone_height * 4.5)   # 4.5x (moon shot)

            rr   = [abs(t - entry) / abs(sl - entry) for t in [tp1, tp2, tp3, tp4]]
            pcts = [(t - entry) / entry * 100          for t in [tp1, tp2, tp3, tp4]]

            if   score >= 95: conf = 'ELITE 🔥🔥🔥'
            elif score >= 85: conf = 'HIGH 💎💎'
            elif score >= 75: conf = 'GOOD 💎'
            else:             conf = 'WATCH ✅'

            return {
                'success': True,
                'symbol': symbol.replace('/USDT:USDT', ''),
                'full_symbol': symbol,
                'signal': 'LONG',
                'confidence': conf,
                'score': score,
                'entry': entry,
                'stop_loss': sl,
                'risk_percent': risk_pct,
                'targets': [tp1, tp2, tp3, tp4],
                'reward_ratios': rr,
                'target_pcts': pcts,
                'reasons': reasons,
                'warnings': warnings,
                'pump': pump,
                'sweep': sweep,
                'demand': demand,
                'retest': retest,
                'rsi': rsi,
                'vol_ratio': volr,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Analyze {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────
    # FORMATTING (iRaters style)
    # ─────────────────────────────────────────────────

    def format_elite_alert(self, r, rank=None):
        """Format alert EXACTLY like iRaters' posts."""
        rk = f"#{rank} " if rank else ""
        d  = r['demand']
        
        msg  = f"{'═'*46}\n"
        msg += f"🔥 <b>{rk}ELITE FUTURES SETUP → LONG</b> 🔥\n"
        msg += f"{'═'*46}\n\n"

        msg += f"<b>Symbol:</b> {r['symbol']}/USDT (1H)\n"
        msg += f"<b>Score:</b> {r['score']:.0f}/100 • <b>{r['confidence']}</b>\n\n"

        # Demand zone (beige box)
        msg += f"<b>📦 DEMAND ZONE:</b>\n"
        msg += f"  Low:  ${d['zone_low']:.6f}\n"
        msg += f"  High: ${d['zone_high']:.6f}\n"
        msg += f"  Range: {d['zone_range_pct']:.1f}% ({d['duration']} candles)\n\n"

        # Entry zone (orange box)
        msg += f"<b>🎯 ENTRY ZONE:</b>\n"
        msg += f"  ${r['entry']:.6f}\n"
        msg += f"  (Upper edge of demand zone)\n\n"

        # Stop loss
        msg += f"<b>🛑 STOP LOSS:</b>\n"
        msg += f"  ${r['stop_loss']:.6f}\n"
        msg += f"  Risk: {r['risk_percent']:.1f}% (tight!)\n\n"

        # Targets (measured move)
        msg += f"<b>🎯 TARGETS (Measured Move):</b>\n"
        labels = ['1.5x', '2.5x', '3.5x', '4.5x']
        for i, (tp, rr, pct, lbl) in enumerate(
                zip(r['targets'], r['reward_ratios'], r['target_pcts'], labels), 1):
            msg += f"  TP{i}: ${tp:.6f}  (+{pct:.1f}%  {rr:.1f}R)  [{lbl}]\n"

        msg += f"\n<b>📊 SETUP DETAILS:</b>\n"
        msg += f"  • Pumped: +{r['pump']['pump_pct']:.0f}%\n"
        msg += f"  • Swept liquidity {r['sweep']['candles_ago']}h ago\n"
        msg += f"  • Formed demand base ({d['touches']} touches)\n"
        msg += f"  • Now retesting zone → <b>ENTRY!</b>\n"

        msg += f"\n<b>✅ REASONS:</b>\n"
        for rs in r['reasons'][:5]:
            msg += f"  • {rs}\n"

        if r['warnings']:
            msg += f"\n<b>⚠️ WARNINGS:</b>\n"
            for w in r['warnings'][:3]:
                msg += f"  {w}\n"

        msg += f"\n<b>⚡ LEVERAGE:</b> 10-50x (HIGH RISK!)\n"
        msg += f"<b>⏱️ DURATION:</b> 1-8 hours (scalp)\n"

        msg += f"\n<i>🔥 Price swept liquidity → formed demand base → retesting → expecting bounce & impulse up!</i>\n"
        msg += f"\n<b>⚠️ DISCLAIMER:</b> DYOR, high risk, not financial advice\n"
        msg += f"\n<i>⏰ {r['timestamp'].strftime('%Y-%m-%d %H:%M')}</i>"
        msg += f"\n{'═'*46}"
        return msg

    def should_alert(self, symbol, result):
        if result['score'] < self.min_score_threshold:
            return False
        if symbol in self.alerted_pairs:
            last = self.alerted_pairs[symbol]
            # 6 hour cooldown for 1H setups
            if datetime.now() - last['time'] < timedelta(hours=6):
                if result['score'] < last['score'] + 15:
                    return False
        return True

    async def send_msg(self, msg):
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id, text=msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Send: {e}")

    async def scan_all_pairs(self):
        if not self.pairs_to_scan:
            await self.load_all_usdt_pairs()

        logger.info(f"🔥 ELITE SETUP SCAN: {len(self.pairs_to_scan)} pairs")
        await self.send_msg(
            f"🔍 <b>ELITE FUTURES SETUP SCAN</b>\n\n"
            f"Scanning {len(self.pairs_to_scan)} pairs (1H)\n"
            f"Pattern: Sweep → Demand Zone → Retest → LONG"
        )

        t0 = datetime.now(); results = []; alerts = 0

        for i, pair in enumerate(self.pairs_to_scan, 1):
            try:
                if i % 20 == 0:
                    logger.info(f"{i}/{len(self.pairs_to_scan)}")
                sym = await self.get_symbol_format(pair)
                if not sym:
                    continue
                data = await self.fetch_data(sym)
                if not data:
                    continue
                result = self.analyze_elite_setup(data, sym)
                if result and result['success']:
                    results.append(result)
                    logger.info(f"🔥 {pair} ELITE  score={result['score']:.0f}")
                    if self.should_alert(result['full_symbol'], result) and alerts < self.max_alerts_per_scan:
                        alerts += 1
                        await self.send_msg(self.format_elite_alert(result, rank=alerts))
                        tid = f"{result['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        self.active_trades[tid] = {
                            'trade_id': tid, 'symbol': result['symbol'],
                            'full_symbol': result['full_symbol'], 'signal': 'LONG',
                            'entry': result['entry'], 'stop_loss': result['stop_loss'],
                            'targets': result['targets'], 'reward_ratios': result['reward_ratios'],
                            'timestamp': datetime.now(),
                            'tp_hit': [False, False, False, False], 'sl_hit': False,
                        }
                        self.alerted_pairs[result['full_symbol']] = {
                            'time': datetime.now(), 'score': result['score']
                        }
                        self.stats['signals_found'] += 1
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"{pair}: {e}")

        dur = (datetime.now() - t0).total_seconds()
        self.stats['total_scans'] += 1
        self.stats['total_pairs_scanned'] += len(self.pairs_to_scan)
        self.stats['avg_scan_time'] = dur
        self.stats['last_scan_date'] = datetime.now()
        self.last_scan_time = datetime.now()

        results.sort(key=lambda x: x['score'], reverse=True)
        elite = [r for r in results if r['score'] >= 90]
        high  = [r for r in results if 80 <= r['score'] < 90]
        good  = [r for r in results if 75 <= r['score'] < 80]

        summ  = f"✅ <b>SCAN COMPLETE</b>\n\n"
        summ += f"📊 {len(self.pairs_to_scan)} pairs\n"
        summ += f"⏱️ {dur/60:.1f} min\n"
        summ += f"🔥 Elite (90+): {len(elite)}\n"
        summ += f"💎 High (80-89): {len(high)}\n"
        summ += f"✅ Good (75-79): {len(good)}\n"
        summ += f"📤 Alerts: {alerts}\n\n"
        summ += f"📡 Tracking: {len(self.active_trades)}"
        await self.send_msg(summ)
        return results

    async def auto_scan_loop(self):
        logger.info(f"Elite scan every {self.scan_interval//60}m")
        while self.is_scanning:
            try:
                await self.scan_all_pairs()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Loop: {e}")
                await asyncio.sleep(300)

    async def track_trades_loop(self):
        logger.info("Tracking...")
        while self.is_tracking:
            try:
                if not self.active_trades:
                    await asyncio.sleep(self.price_check_interval)
                    continue
                to_remove = []
                for tid, trade in list(self.active_trades.items()):
                    try:
                        if datetime.now() - trade['timestamp'] > timedelta(hours=16):
                            await self.send_msg(f"⏰ {trade['symbol']} - 16h timeout")
                            to_remove.append(tid); continue
                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price  = ticker['last']
                        
                        for i, (tp, hit) in enumerate(zip(trade['targets'], trade['tp_hit'])):
                            if not hit and price >= tp:
                                pnl = (price - trade['entry']) / trade['entry'] * 100
                                msg = f"🎯 <b>TP{i+1} HIT!</b>\n{trade['symbol']}\n+{pnl:.1f}% ({trade['reward_ratios'][i]:.1f}R)\n\n"
                                if i == 0:
                                    msg += "💡 Take 30% profit • Move SL to breakeven"
                                elif i == 1:
                                    msg += "💡 Take 30% profit • Trail SL to TP1"
                                elif i == 2:
                                    msg += "💡 Take 30% profit • Trail SL"
                                elif i == 3:
                                    msg += "💡 Close remaining • 🎊 ELITE WIN!"
                                await self.send_msg(msg)
                                trade['tp_hit'][i] = True
                                self.stats[f'tp{i+1}_hits'] += 1
                                if i == 3: to_remove.append(tid)
                        
                        if not trade['sl_hit'] and price <= trade['stop_loss']:
                            loss = (trade['stop_loss'] - trade['entry']) / trade['entry'] * 100
                            await self.send_msg(f"🛑 <b>STOP HIT</b>\n{trade['symbol']}\n-{loss:.1f}%\n\nCut & next setup!")
                            trade['sl_hit'] = True
                            self.stats['sl_hits'] += 1
                            to_remove.append(tid)
                    except:
                        pass
                for tid in to_remove:
                    self.active_trades.pop(tid, None)
                self.stats['active_trades_count'] = len(self.active_trades)
                await asyncio.sleep(self.price_check_interval)
            except:
                await asyncio.sleep(120)

    async def close(self):
        await self.exchange.close()


class BotCmds:
    def __init__(self, s):
        self.s = s

    async def cmd_start(self, u, c):
        msg  = "🔥 <b>iRATERS ELITE FUTURES SCANNER</b>\n\n"
        msg += "<b>Strategy: @Albert_Adams04 (iRaters)</b>\n\n"
        msg += "Pattern:\n"
        msg += "1️⃣ Recent pump + pullback\n"
        msg += "2️⃣ Liquidity sweep below\n"
        msg += "3️⃣ Demand zone forms (3-8 candles)\n"
        msg += "4️⃣ Retest → LONG entry\n\n"
        msg += "🎯 Targets: 10-40%+ (2-4x measured move)\n"
        msg += "⚡ Leverage: 10-50x (HIGH RISK!)\n"
        msg += "⏱️ Duration: 1-8 hours\n\n"
        msg += "📊 Timeframe: 1H Binance perpetuals\n\n"
        msg += "/start_scan\n/start_tracking\n/status\n/stats"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_start_scan(self, u, c):
        if self.s.is_scanning:
            await u.message.reply_text("⚠️ Already running!", parse_mode=ParseMode.HTML)
            return
        self.s.is_scanning = True
        asyncio.create_task(self.s.auto_scan_loop())
        await u.message.reply_text(
            f"🔥 <b>ELITE SCANNER ON!</b>\n\nEvery {self.s.scan_interval//60}min\nScanning...",
            parse_mode=ParseMode.HTML)

    async def cmd_stop_scan(self, u, c):
        self.s.is_scanning = False
        await u.message.reply_text("🛑 STOPPED", parse_mode=ParseMode.HTML)

    async def cmd_scan_now(self, u, c):
        await u.message.reply_text("🔍 Scanning...", parse_mode=ParseMode.HTML)
        await self.s.scan_all_pairs()

    async def cmd_start_tracking(self, u, c):
        if self.s.is_tracking:
            await u.message.reply_text("⚠️ Already tracking!", parse_mode=ParseMode.HTML)
            return
        self.s.is_tracking = True
        asyncio.create_task(self.s.track_trades_loop())
        await u.message.reply_text("🔥 TRACKING ELITE SETUPS!", parse_mode=ParseMode.HTML)

    async def cmd_stop_tracking(self, u, c):
        self.s.is_tracking = False
        await u.message.reply_text("🛑 STOPPED", parse_mode=ParseMode.HTML)

    async def cmd_status(self, u, c):
        scan = "🟢" if self.s.is_scanning else "🔴"
        trk  = "🟢" if self.s.is_tracking else "🔴"
        msg  = f"<b>STATUS</b>\n\nScan: {scan}\nTrack: {trk}\n\n"
        msg += f"Pairs: {len(self.s.pairs_to_scan)}\nActive: {len(self.s.active_trades)}"
        if self.s.last_scan_time:
            mins = int((datetime.now() - self.s.last_scan_time).total_seconds() // 60)
            msg += f"\nLast: {mins}m ago"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_stats(self, u, c):
        s = self.s.stats
        msg = f"<b>ELITE STATS</b>\n\nScans: {s['total_scans']}\nSignals: {s['signals_found']}\n\n"
        msg += f"TP1: {s['tp1_hits']}\nTP2: {s['tp2_hits']}\nTP3: {s['tp3_hits']}\nSL: {s['sl_hits']}"
        t = s['tp1_hits'] + s['sl_hits']
        if t > 0:
            msg += f"\n\nWin: {s['tp1_hits']/t*100:.1f}%"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def main():
    TOKEN = "7731521911:AAFnus-fDivEwoKqrtwZXMmKEj5BU1EhQn4"
    CHAT  = "7500072234"

    scanner = iRatersEliteScanner(TOKEN, CHAT)
    app     = Application.builder().token(TOKEN).build()
    cmds    = BotCmds(scanner)

    for cmd, fn in [
        ("start", cmds.cmd_start), ("start_scan", cmds.cmd_start_scan),
        ("stop_scan", cmds.cmd_stop_scan), ("scan_now", cmds.cmd_scan_now),
        ("start_tracking", cmds.cmd_start_tracking),
        ("stop_tracking", cmds.cmd_stop_tracking),
        ("status", cmds.cmd_status), ("stats", cmds.cmd_stats),
    ]:
        app.add_handler(CommandHandler(cmd, fn))

    await app.initialize(); await app.start(); await app.updater.start_polling()
    logger.info("🔥 iRATERS ELITE SCANNER ONLINE!")

    welcome  = "🔥 <b>iRATERS ELITE FUTURES SCANNER READY!</b>\n\n"
    welcome += "<b>Strategy: @Albert_Adams04</b>\n\n"
    welcome += "✅ 1H Binance perpetuals\n"
    welcome += "✅ Liquidity sweep detection\n"
    welcome += "✅ Demand zone formation\n"
    welcome += "✅ Retest entry signals\n"
    welcome += "✅ 10-40%+ targets\n"
    welcome += "✅ High leverage scalps\n\n"
    welcome += "/start_scan\n/start_tracking\n\n"
    welcome += "🔥 ELITE FUTURES SETUPS INCOMING!"
    await scanner.send_msg(welcome)

    scanner.is_tracking = True
    tracking_task = asyncio.create_task(scanner.track_trades_loop())

    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        scanner.is_scanning = False
        scanner.is_tracking = False
        tracking_task.cancel()
        await scanner.close()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
