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


class SMC15MinScanner:
    """
    💎 15-MINUTE SMC SCANNER
    
    THE ACTUAL STRATEGY from @free_fx_pro (BAN 15min chart confirmed):
    
    TIMEFRAME: 15 MINUTE (not 1H!)
    
    FULL SMC METHODOLOGY:
    1. Map Premium / Equilibrium / Discount zones
    2. Detect liquidity sweeps (sellside/buyside)
    3. Identify CHoCH (Change of Character) - first sign of reversal
    4. Confirm BOS (Break of Structure) - trend change confirmed
    5. Find order blocks (last down candle before move up)
    6. Enter in discount on pullback to order block
    7. SL below sweep low
    8. TP at premium zone / previous high
    
    SCAN FREQUENCY: Every 5-10 minutes (FAST for 15min timeframe)
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

        # FAST SCANNING for 15min timeframe
        self.scan_interval        = 600    # 10 minutes
        self.min_score_threshold  = 70
        self.max_alerts_per_scan  = 10
        self.price_check_interval = 60     # 1 minute

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
            self.pairs_to_scan = [p['base'] for p in perps[:150]]  # Top 150 only
            self.all_symbols   = [p['symbol'] for p in perps[:150]]
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
            # 15min primary + 1H for context
            data['15m'] = await self.fetch_df(symbol, '15m', 300)
            await asyncio.sleep(0.03)
            data['1h']  = await self.fetch_df(symbol, '1h', 100)
            await asyncio.sleep(0.03)
            return data
        except Exception as e:
            logger.error(f"Fetch {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────
    # SMC CORE FUNCTIONS (15min optimized)
    # ─────────────────────────────────────────────────

    def map_premium_discount(self, df, lookback=200):
        """Map Premium/Equilibrium/Discount zones on 15min."""
        recent = df.tail(lookback)
        swing_high = recent['high'].max()
        swing_low  = recent['low'].min()
        eq = (swing_high + swing_low) / 2
        
        range_size = swing_high - swing_low
        premium_start   = swing_high - (range_size * 0.25)
        discount_end    = swing_low  + (range_size * 0.25)
        
        return {
            'swing_high': swing_high, 'swing_low': swing_low,
            'equilibrium': eq, 'premium_start': premium_start,
            'discount_end': discount_end, 'range_size': range_size
        }

    def find_structure_points(self, df, window=10):
        """Find swing highs and swing lows for structure."""
        highs, lows = [], []
        
        for i in range(window, len(df) - window):
            # Swing high
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                highs.append({'idx': i, 'price': df['high'].iloc[i]})
            # Swing low
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                lows.append({'idx': i, 'price': df['low'].iloc[i]})
        
        return {'highs': highs, 'lows': lows}

    def detect_liquidity_sweep(self, df, structure, lookback=80):
        """
        Detect LIQUIDITY SWEEP on 15min.
        = Price wicks below recent swing low then closes back above.
        """
        lows = structure['lows']
        if not lows:
            return {'swept': False}
        
        # Find most recent swing low
        recent_lows = [l for l in lows if l['idx'] > len(df) - lookback]
        if not recent_lows:
            return {'swept': False}
        
        # Get the lowest swing low as key level
        key_low = min(recent_lows, key=lambda x: x['price'])
        key_price = key_low['price']
        key_idx = key_low['idx']
        
        # Check candles AFTER this low for sweep
        after = df.iloc[key_idx:]
        
        for i in range(len(after)):
            candle = after.iloc[i]
            # Sweep = wick below then close above
            if candle['low'] < key_price * 0.997 and candle['close'] > key_price * 1.002:
                candles_ago = len(df) - 1 - (key_idx + i)
                return {
                    'swept': True,
                    'sweep_low': candle['low'],
                    'key_level': key_price,
                    'sweep_depth': (key_price - candle['low']) / key_price * 100,
                    'candles_ago': candles_ago,
                    'sweep_idx': key_idx + i
                }
        
        return {'swept': False}

    def detect_choch(self, df, structure, sweep_info):
        """
        Detect CHoCH (Change of Character) on 15min.
        = After downtrend, price makes HIGHER HIGH.
        First sign of reversal.
        """
        if not sweep_info.get('swept'):
            return {'choch': False}
        
        sweep_idx = sweep_info.get('sweep_idx', len(df) - 20)
        highs = structure['highs']
        
        # Get highs BEFORE sweep
        highs_before = [h for h in highs if h['idx'] < sweep_idx]
        if len(highs_before) < 2:
            return {'choch': False}
        
        # Recent high before sweep
        recent_high_before = highs_before[-1]['price']
        
        # Check if price after sweep made HIGHER high
        after_sweep = df.iloc[sweep_idx:]
        highest_after = after_sweep['high'].max()
        
        if highest_after > recent_high_before * 1.003:  # 0.3% buffer
            candles_ago = len(df) - 1 - after_sweep['high'].idxmax()
            return {
                'choch': True,
                'choch_high': highest_after,
                'broke_high': recent_high_before,
                'candles_ago': candles_ago
            }
        
        return {'choch': False}

    def detect_bos(self, df, structure, choch_info):
        """
        Detect BOS (Break of Structure) on 15min.
        = Confirms new uptrend by breaking ANOTHER high.
        """
        if not choch_info.get('choch'):
            return {'bos': False}
        
        highs = structure['highs']
        if len(highs) < 3:
            return {'bos': False}
        
        # Check if recent price broke multiple highs
        current = df['close'].iloc[-1]
        recent_high = highs[-1]['price'] if highs else current * 0.9
        
        bos_broken = current > recent_high
        
        return {
            'bos': bos_broken,
            'bos_level': recent_high if bos_broken else None
        }

    def find_order_block(self, df, sweep_info):
        """
        Find ORDER BLOCK on 15min.
        = Last bearish candle before bullish move.
        This is where smart money bought.
        """
        if not sweep_info.get('swept'):
            return {'ob': False}
        
        sweep_idx = sweep_info.get('sweep_idx', len(df) - 10)
        
        # Look for last red candle before the move up
        after_sweep = df.iloc[sweep_idx:sweep_idx + 20]
        
        for i in range(len(after_sweep) - 1):
            current = after_sweep.iloc[i]
            next_candle = after_sweep.iloc[i + 1]
            
            # Red candle followed by green candle
            if (current['close'] < current['open'] and 
                next_candle['close'] > next_candle['open']):
                
                ob_high = current['high']
                ob_low  = current['low']
                ob_mid  = (ob_high + ob_low) / 2
                
                return {
                    'ob': True,
                    'ob_high': ob_high,
                    'ob_low': ob_low,
                    'ob_mid': ob_mid,
                    'ob_idx': sweep_idx + i
                }
        
        return {'ob': False}

    def add_indicators(self, df):
        if len(df) < 50:
            return df
        try:
            df['ema_20']    = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50']    = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['rsi']       = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['vol_sma']   = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, 1)
            df['atr']       = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=14).average_true_range()
        except:
            pass
        return df

    # ─────────────────────────────────────────────────
    # MAIN ANALYSIS (15MIN)
    # ─────────────────────────────────────────────────

    def analyze_smc_15m(self, data, symbol):
        """Full 15min SMC analysis."""
        try:
            if not data or '15m' not in data:
                return None

            df15m = self.add_indicators(data['15m'].copy())
            df1h  = self.add_indicators(data['1h'].copy())

            if len(df15m) < 100:
                return None

            current = df15m['close'].iloc[-1]
            l15m    = df15m.iloc[-1]
            l1h     = df1h.iloc[-1]
            rsi     = float(l15m.get('rsi', 50) or 50)
            volr    = float(l15m.get('vol_ratio', 1) or 1)

            # ── [1] Map zones ──
            zones = self.map_premium_discount(df15m, lookback=200)

            # ── [2] Find structure ──
            structure = self.find_structure_points(df15m, window=8)

            # ── [3] Liquidity sweep ──
            sweep = self.detect_liquidity_sweep(df15m, structure, lookback=100)

            # ── [4] CHoCH ──
            choch = self.detect_choch(df15m, structure, sweep)

            # ── [5] BOS ──
            bos = self.detect_bos(df15m, structure, choch)

            # ── [6] Order block ──
            ob = self.find_order_block(df15m, sweep)

            # ── [7] Check if price in discount ──
            in_discount = current < zones['equilibrium']
            dist_pct = (zones['equilibrium'] - current) / zones['equilibrium'] * 100 if in_discount else 0

            # ── SCORING (15min optimized) ──
            score = 0
            reasons = []
            warnings = []

            # [A] Liquidity sweep (0-35 pts) - CRITICAL
            if sweep['swept']:
                ca = sweep['candles_ago']
                if ca <= 10:  # Within 2.5 hours on 15min
                    score += 35
                    reasons.append(f"💥 LIQUIDITY SWEEP! ({ca}x15min ago, -{sweep['sweep_depth']:.1f}%)")
                elif ca <= 20:
                    score += 25
                    reasons.append(f"💥 Liquidity sweep ({ca}x15min ago)")
                else:
                    score += 12
            else:
                warnings.append("⚠️ No liquidity sweep detected")
                score -= 15

            # [B] CHoCH (0-30 pts) - REVERSAL SIGNAL
            if choch['choch']:
                score += 30
                reasons.append(f"🔄 CHoCH CONFIRMED - Trend reversing!")
            else:
                warnings.append("⚠️ No CHoCH - reversal not confirmed")
                score -= 10

            # [C] BOS (0-20 pts) - STRUCTURE BREAK
            if bos['bos']:
                score += 20
                reasons.append(f"📈 BOS - Structure broken bullish!")

            # [D] Order block (0-15 pts)
            if ob['ob']:
                score += 15
                # Check if price near OB
                dist_to_ob = abs(current - ob['ob_mid']) / ob['ob_mid'] * 100
                if dist_to_ob < 1:
                    score += 5
                    reasons.append(f"📦 AT ORDER BLOCK - prime entry zone!")
                else:
                    reasons.append(f"📦 Order block identified")

            # [E] Discount zone (0-15 pts)
            if in_discount:
                if dist_pct > 10:
                    score += 15
                    reasons.append(f"🔵 Deep discount ({dist_pct:.1f}% below EQ)")
                elif dist_pct > 3:
                    score += 10
                    reasons.append(f"🔵 In discount zone")

            # [F] RSI (0-10 pts)
            if rsi < 35:
                score += 10
                reasons.append(f"💎 RSI oversold ({rsi:.0f})")
            elif rsi < 45:
                score += 6

            # [G] Volume (0-10 pts)
            if volr > 2.0:
                score += 10
                reasons.append(f"📊 High volume ({volr:.1f}x)")
            elif volr > 1.5:
                score += 5

            # [H] 1H context (0-10 pts)
            rsi_1h = float(l1h.get('rsi', 50) or 50)
            if rsi_1h < 40:
                score += 10
                reasons.append(f"💎 1H RSI oversold ({rsi_1h:.0f})")

            # ── WARNINGS ──
            if rsi > 70:
                warnings.append("⚠️ RSI overbought on 15min")
                score -= 12
            if not in_discount:
                warnings.append("⚠️ Price not in discount - risky entry")
                score -= 10
            if sweep.get('candles_ago', 999) > 40:
                warnings.append("⚠️ Sweep is old (>10 hours)")
                score -= 8

            if score < self.min_score_threshold:
                return None

            # ── TRADE LEVELS ──
            entry = current

            # SL: Below sweep low
            if sweep['swept']:
                sl = sweep['sweep_low'] * 0.995
            else:
                sl = zones['swing_low'] * 0.98

            risk_pct = (entry - sl) / entry * 100
            if risk_pct > 8:
                sl = entry * 0.95
                risk_pct = 5.0

            # TP: Equilibrium → Previous high → Premium
            tp1 = zones['equilibrium']
            tp2 = (zones['equilibrium'] + zones['swing_high']) / 2
            tp3 = zones['swing_high'] * 0.99

            rr   = [abs(t - entry) / abs(sl - entry) for t in [tp1, tp2, tp3]]
            pcts = [(t - entry) / entry * 100          for t in [tp1, tp2, tp3]]

            if   score >= 90: conf = 'ELITE 🔥🔥🔥'
            elif score >= 80: conf = 'HIGH 💎💎'
            elif score >= 70: conf = 'GOOD 💎'
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
                'targets': [tp1, tp2, tp3],
                'reward_ratios': rr,
                'target_pcts': pcts,
                'reasons': reasons,
                'warnings': warnings,
                'zones': zones,
                'sweep': sweep,
                'choch': choch,
                'bos': bos,
                'ob': ob,
                'rsi': rsi,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Analyze {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────
    # FORMATTING
    # ─────────────────────────────────────────────────

    def format_alert(self, r, rank=None):
        rk = f"#{rank} " if rank else ""
        z  = r['zones']
        
        msg  = f"{'═'*46}\n"
        msg += f"💎 <b>{rk}15MIN SMC: {r['symbol']} — {r['confidence']}</b> 💎\n"
        msg += f"{'═'*46}\n\n"

        msg += f"<b>LONG</b>  Score: {r['score']:.0f}/100\n"
        msg += f"RSI: {r['rsi']:.0f}\n\n"

        # Zones
        msg += f"<b>🗺️ SMC ZONES (15m):</b>\n"
        msg += f"  🔴 Premium:  ${z['premium_start']:.6f}+\n"
        msg += f"  ⚖️ EQ:       ${z['equilibrium']:.6f}\n"
        msg += f"  🔵 Discount: <${z['equilibrium']:.6f}\n"
        msg += f"  📍 Current:  ${r['entry']:.6f}\n\n"

        # SMC confirmations
        msg += f"<b>🎯 SMC SIGNALS:</b>\n"
        if r['sweep']['swept']:
            msg += f"  ✅ Liq. Sweep ({r['sweep']['candles_ago']}x15min ago)\n"
        else:
            msg += f"  ⏳ Liq. Sweep\n"

        if r['choch']['choch']:
            msg += f"  ✅ CHoCH\n"
        else:
            msg += f"  ⏳ CHoCH\n"

        if r['bos']['bos']:
            msg += f"  ✅ BOS\n"
        else:
            msg += f"  ⏳ BOS\n"

        if r['ob']['ob']:
            msg += f"  ✅ Order Block\n"

        msg += f"\n<b>💰 TRADE:</b>\n"
        msg += f"  Entry: ${r['entry']:.6f}\n"
        msg += f"  SL:    ${r['stop_loss']:.6f}  (-{r['risk_percent']:.1f}%)\n\n"

        msg += f"<b>🎯 TARGETS:</b>\n"
        labels = ['EQ', 'Mid', 'High']
        for i, (tp, rr, pct, lbl) in enumerate(
                zip(r['targets'], r['reward_ratios'], r['target_pcts'], labels), 1):
            msg += f"  TP{i}: ${tp:.6f}  (+{pct:.1f}%  {rr:.1f}R)  [{lbl}]\n"

        msg += f"\n<b>✅ REASONS:</b>\n"
        for rs in r['reasons'][:6]:
            msg += f"  • {rs}\n"

        if r['warnings']:
            msg += f"\n<b>⚠️ WARNINGS:</b>\n"
            for w in r['warnings'][:3]:
                msg += f"  {w}\n"

        msg += f"\n<i>⏰ {r['timestamp'].strftime('%H:%M')}</i>"
        msg += f"\n<i>⚡ 15min SMC: Sweep → CHoCH → BOS → 🚀</i>"
        msg += f"\n{'═'*46}"
        return msg

    def should_alert(self, symbol, result):
        if result['score'] < self.min_score_threshold:
            return False
        if symbol in self.alerted_pairs:
            last = self.alerted_pairs[symbol]
            # Only 2 hour cooldown for 15min (moves fast)
            if datetime.now() - last['time'] < timedelta(hours=2):
                if result['score'] < last['score'] + 20:
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

        logger.info(f"💎 15MIN SMC SCAN: {len(self.pairs_to_scan)} pairs")
        await self.send_msg(
            f"🔍 <b>15MIN SMC SCAN</b>\n\n"
            f"Scanning {len(self.pairs_to_scan)} pairs\n"
            f"Pattern: Sweep → CHoCH → BOS → Entry"
        )

        t0 = datetime.now(); results = []; alerts = 0

        for i, pair in enumerate(self.pairs_to_scan, 1):
            try:
                if i % 30 == 0:
                    logger.info(f"{i}/{len(self.pairs_to_scan)}")
                sym = await self.get_symbol_format(pair)
                if not sym:
                    continue
                data = await self.fetch_data(sym)
                if not data:
                    continue
                result = self.analyze_smc_15m(data, sym)
                if result and result['success']:
                    results.append(result)
                    logger.info(f"💎 {pair}  score={result['score']:.0f}")
                    if self.should_alert(result['full_symbol'], result) and alerts < self.max_alerts_per_scan:
                        alerts += 1
                        await self.send_msg(self.format_alert(result, rank=alerts))
                        tid = f"{result['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        self.active_trades[tid] = {
                            'trade_id': tid, 'symbol': result['symbol'],
                            'full_symbol': result['full_symbol'], 'signal': 'LONG',
                            'entry': result['entry'], 'stop_loss': result['stop_loss'],
                            'targets': result['targets'], 'reward_ratios': result['reward_ratios'],
                            'timestamp': datetime.now(),
                            'tp_hit': [False, False, False], 'sl_hit': False,
                        }
                        self.alerted_pairs[result['full_symbol']] = {
                            'time': datetime.now(), 'score': result['score']
                        }
                        self.stats['signals_found'] += 1
                await asyncio.sleep(0.08)
            except Exception as e:
                logger.error(f"{pair}: {e}")

        dur = (datetime.now() - t0).total_seconds()
        self.stats['total_scans'] += 1
        self.stats['total_pairs_scanned'] += len(self.pairs_to_scan)
        self.stats['avg_scan_time'] = dur
        self.stats['last_scan_date'] = datetime.now()
        self.last_scan_time = datetime.now()

        results.sort(key=lambda x: x['score'], reverse=True)
        elite = [r for r in results if r['score'] >= 85]
        good  = [r for r in results if 70 <= r['score'] < 85]

        summ  = f"✅ <b>SCAN DONE</b>\n\n"
        summ += f"📊 {len(self.pairs_to_scan)} pairs\n"
        summ += f"⏱️ {dur/60:.1f} min\n"
        summ += f"💎 Elite (85+): {len(elite)}\n"
        summ += f"✅ Good (70-84): {len(good)}\n"
        summ += f"📤 Alerts: {alerts}\n\n"
        summ += f"📡 Tracking: {len(self.active_trades)}"
        await self.send_msg(summ)
        return results

    async def auto_scan_loop(self):
        logger.info(f"15min scan every {self.scan_interval//60}m")
        while self.is_scanning:
            try:
                await self.scan_all_pairs()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Loop: {e}")
                await asyncio.sleep(120)

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
                        if datetime.now() - trade['timestamp'] > timedelta(days=3):
                            await self.send_msg(f"⏰ {trade['symbol']}")
                            to_remove.append(tid); continue
                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price  = ticker['last']
                        for i, (tp, hit) in enumerate(zip(trade['targets'], trade['tp_hit'])):
                            if not hit and price >= tp:
                                pnl = (tp - trade['entry']) / trade['entry'] * 100
                                msg = f"🎯 TP{i+1}!\n{trade['symbol']}\n+{pnl:.1f}% ({trade['reward_ratios'][i]:.1f}R)"
                                await self.send_msg(msg)
                                trade['tp_hit'][i] = True
                                self.stats[f'tp{i+1}_hits'] += 1
                                if i == 2: to_remove.append(tid)
                        if not trade['sl_hit'] and price <= trade['stop_loss']:
                            loss = (trade['stop_loss'] - trade['entry']) / trade['entry'] * 100
                            await self.send_msg(f"🛑 SL\n{trade['symbol']}\n{loss:.1f}%")
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
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


class BotCmds:
    def __init__(self, s):
        self.s = s

    async def cmd_start(self, u, c):
        msg  = "⚡ <b>15MIN SMC SCANNER</b>\n\n"
        msg += "<b>Timeframe: 15 MINUTE</b>\n"
        msg += "<b>Scan: Every 10 min</b>\n\n"
        msg += "Strategy:\n"
        msg += "1️⃣ Liquidity sweep\n2️⃣ CHoCH\n3️⃣ BOS\n4️⃣ Discount entry\n5️⃣ 🚀\n\n"
        msg += "/start_scan\n/start_tracking\n/status\n/stats"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_start_scan(self, u, c):
        if self.s.is_scanning:
            await u.message.reply_text("⚠️ Running!", parse_mode=ParseMode.HTML)
            return
        self.s.is_scanning = True
        asyncio.create_task(self.s.auto_scan_loop())
        await u.message.reply_text(
            f"✅ <b>15MIN SCANNER ON!</b>\n\nEvery {self.s.scan_interval//60}min\nScanning...",
            parse_mode=ParseMode.HTML)

    async def cmd_stop_scan(self, u, c):
        self.s.is_scanning = False
        await u.message.reply_text("🛑 STOPPED", parse_mode=ParseMode.HTML)

    async def cmd_scan_now(self, u, c):
        await u.message.reply_text("🔍 Scanning...", parse_mode=ParseMode.HTML)
        await self.s.scan_all_pairs()

    async def cmd_start_tracking(self, u, c):
        if self.s.is_tracking:
            await u.message.reply_text("⚠️ Running!", parse_mode=ParseMode.HTML)
            return
        self.s.is_tracking = True
        asyncio.create_task(self.s.track_trades_loop())
        await u.message.reply_text("✅ TRACKING!", parse_mode=ParseMode.HTML)

    async def cmd_stop_tracking(self, u, c):
        self.s.is_tracking = False
        await u.message.reply_text("🛑 STOPPED", parse_mode=ParseMode.HTML)

    async def cmd_status(self, u, c):
        scan = "🟢" if self.s.is_scanning else "🔴"
        trk  = "🟢" if self.s.is_tracking else "🔴"
        msg  = f"Scan: {scan}\nTrack: {trk}\nPairs: {len(self.s.pairs_to_scan)}\nActive: {len(self.s.active_trades)}"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_stats(self, u, c):
        s = self.s.stats
        msg = f"Scans: {s['total_scans']}\nSignals: {s['signals_found']}\n"
        msg += f"TP1: {s['tp1_hits']}\nTP2: {s['tp2_hits']}\nTP3: {s['tp3_hits']}\nSL: {s['sl_hits']}"
        t = s['tp1_hits'] + s['sl_hits']
        if t > 0:
            msg += f"\n\nWin: {s['tp1_hits']/t*100:.1f}%"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def main():
    TOKEN = "8186622122:AAGtQcoh_s7QqIAVACmOYVHLqPX-p6dSNVA"
    CHAT  = "7500072234"

    scanner = SMC15MinScanner(TOKEN, CHAT)
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
    logger.info("⚡ 15MIN SMC SCANNER ONLINE!")

    welcome  = "⚡ <b>15MIN SMC READY!</b>\n\n"
    welcome += "<b>Timeframe: 15 MINUTE</b>\n"
    welcome += "<b>Scan: Every 10 min</b>\n\n"
    welcome += "✅ Liquidity sweeps\n✅ CHoCH detection\n✅ BOS confirmation\n"
    welcome += "✅ Order blocks\n✅ Fast alerts\n\n"
    welcome += "/start_scan\n/start_tracking"
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
