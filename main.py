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


class ConsolidationBreakoutScanner:
    """
    💎 CONSOLIDATION BREAKOUT SCANNER
    
    THE ACTUAL PATTERN from @free_fx_pro (analyzed 12 charts):
    
    ✅ Most common setup: CONSOLIDATION BREAKOUT
       - GUN, NAORIS, KITE, WLFI, ESP, BAN
       - Price consolidates in tight horizontal range (15-40 candles)
       - Breaks above consolidation cleanly
       - Pumps 15-40%+
    
    ✅ Less common: Liquidity sweep reversal
       - ORCA, RAYSOL, ARIA
       - We'll detect these too
    
    PRIMARY FOCUS: Consolidation breakouts (80% of his calls)
    
    Strategy:
    1. Detect tight horizontal consolidation (range <7%, 15+ candles)
    2. Wait for clean break above consolidation high
    3. Confirm with volume spike (optional but preferred)
    4. Enter on breakout or pullback to breakout level
    5. SL below consolidation box
    6. TP = measured move (consolidation height added to breakout)
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

        self.scan_interval        = 1800   # 30 min (faster for breakouts)
        self.min_score_threshold  = 65
        self.max_alerts_per_scan  = 8
        self.price_check_interval = 120

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
        except Exception as e:
            logger.error(f"Symbol: {e}")
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
            self.pairs_to_scan = [p['base'] for p in perps]
            self.all_symbols   = [p['symbol'] for p in perps]
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
            data['1h'] = await self.fetch_df(symbol, '1h', 200)
            await asyncio.sleep(0.05)
            return data
        except Exception as e:
            logger.error(f"Fetch {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────
    # CORE: CONSOLIDATION DETECTION
    # ─────────────────────────────────────────────────

    def detect_consolidation(self, df, lookback=60):
        """
        Detect TIGHT HORIZONTAL CONSOLIDATION.
        This is the beige box in all his charts.
        
        Requirements (from analyzing his charts):
        - At least 15-20 candles
        - Range < 7-8% (tight)
        - Price bouncing between clear high/low
        - Recently formed (within last 40 candles)
        """
        consolidations = []

        # Try different window sizes (15 to 40 candles)
        for window in [15, 20, 25, 30, 35, 40]:
            # Look at recent price action
            for start in range(max(0, len(df) - lookback), len(df) - window, 2):
                end = start + window
                if end > len(df):
                    break

                zone = df.iloc[start:end]
                
                hi = zone['high'].max()
                lo = zone['low'].min()
                
                if lo == 0:
                    continue

                # Calculate range
                rng_pct = (hi - lo) / lo * 100

                # Must be TIGHT consolidation
                if rng_pct > 8:
                    continue

                # Calculate touches to high/low
                touches_high = sum(1 for h in zone['high'] if abs(h - hi) / hi < 0.01)
                touches_low  = sum(1 for l in zone['low']  if abs(l - lo) / lo < 0.01)
                total_touches = touches_high + touches_low

                # Quality score
                tightness_score = 8 - rng_pct  # Tighter = better
                touch_score     = total_touches * 2
                duration_score  = window / 2
                recency_score   = (len(df) - end) / len(df) * 20  # More recent = better
                
                score = tightness_score + touch_score + duration_score + recency_score

                consolidations.append({
                    'start':   start,
                    'end':     end,
                    'high':    hi,
                    'low':     lo,
                    'mid':     (hi + lo) / 2,
                    'range_pct': rng_pct,
                    'duration': window,
                    'touches':  total_touches,
                    'score':    score
                })

        if not consolidations:
            return None

        # Return best consolidation
        consolidations.sort(key=lambda x: x['score'], reverse=True)
        return consolidations[0]

    def detect_breakout(self, df, consolidation):
        """
        Detect if price has broken ABOVE consolidation.
        
        From charts:
        - Clean break above consolidation high
        - Not just a wick - needs close above
        - Ideally with volume spike
        """
        if consolidation is None:
            return {'breakout': False}

        cons_high = consolidation['high']
        cons_end  = consolidation['end']

        # Look at candles AFTER consolidation
        after_cons = df.iloc[cons_end:]

        if len(after_cons) < 1:
            return {'breakout': False}

        # Find first candle that CLOSES above consolidation
        for i in range(len(after_cons)):
            candle = after_cons.iloc[i]
            
            # Must close above (not just wick)
            if candle['close'] > cons_high * 1.005:  # 0.5% buffer for clean break
                breakout_idx  = cons_end + i
                candles_since = len(df) - 1 - breakout_idx
                
                # Check if breakout is recent
                if candles_since > 10:
                    continue  # Too old

                return {
                    'breakout':       True,
                    'breakout_price': candle['close'],
                    'breakout_idx':   breakout_idx,
                    'candles_ago':    candles_since,
                    'breakout_pct':   (candle['close'] - cons_high) / cons_high * 100
                }

        # No breakout yet - check if price is AT consolidation high (about to break)
        current = df['close'].iloc[-1]
        if cons_high * 0.995 <= current <= cons_high * 1.005:
            return {
                'breakout':    False,
                'at_resistance': True,
                'distance':    (current - cons_high) / cons_high * 100
            }

        return {'breakout': False}

    def detect_liquidity_sweep(self, df, consolidation):
        """
        Also detect sweep pattern (like ORCA, RAYSOL, ARIA).
        This is secondary but still catches some setups.
        """
        if consolidation is None:
            return {'swept': False}

        cons_low = consolidation['low']
        cons_end = consolidation['end']

        # Look DURING or slightly after consolidation
        check_zone = df.iloc[max(0, cons_end - 10):min(len(df), cons_end + 15)]

        for i in range(len(check_zone)):
            candle = check_zone.iloc[i]
            
            # Wick below consolidation but closes back inside/above
            if candle['low'] < cons_low * 0.995 and candle['close'] > cons_low * 1.005:
                return {
                    'swept':       True,
                    'sweep_low':   candle['low'],
                    'sweep_depth': (cons_low - candle['low']) / cons_low * 100,
                    'candles_ago': len(df) - 1 - (cons_end - 10 + i)
                }

        return {'swept': False}

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
        except Exception as e:
            logger.error(f"Indicators: {e}")
        return df

    # ─────────────────────────────────────────────────
    # MAIN ANALYSIS
    # ─────────────────────────────────────────────────

    def analyze_breakout(self, data, symbol):
        """
        Analyze for consolidation breakout pattern.
        This is the PRIMARY pattern from his charts.
        """
        try:
            if not data or '1h' not in data:
                return None

            df = self.add_indicators(data['1h'].copy())

            if len(df) < 50:
                return None

            current = df['close'].iloc[-1]
            l1h     = df.iloc[-1]
            rsi     = float(l1h.get('rsi', 50) or 50)
            volr    = float(l1h.get('vol_ratio', 1) or 1)
            atr     = float(l1h.get('atr', current * 0.02) or current * 0.02)

            # ── [1] Detect consolidation zone ──
            consolidation = self.detect_consolidation(df, lookback=80)

            if consolidation is None:
                return None

            # ── [2] Detect breakout ──
            breakout = self.detect_breakout(df, consolidation)

            # ── [3] Also check for sweep (less common) ──
            sweep = self.detect_liquidity_sweep(df, consolidation)

            # ── SCORING ──
            score    = 0
            reasons  = []
            warnings = []

            # [A] Consolidation quality (0-25 pts)
            cons_range  = consolidation['range_pct']
            cons_duration = consolidation['duration']
            cons_touches  = consolidation['touches']

            if cons_range < 4:
                score += 25
                reasons.append(f"📦 ULTRA TIGHT consolidation ({cons_range:.1f}%, {cons_duration} candles)")
            elif cons_range < 6:
                score += 20
                reasons.append(f"📦 Tight consolidation ({cons_range:.1f}%, {cons_duration} candles)")
            elif cons_range < 8:
                score += 12
                reasons.append(f"📦 Consolidation zone ({cons_range:.1f}%, {cons_duration} candles)")

            # Bonus for multiple touches
            if cons_touches >= 6:
                score += 8
                reasons.append(f"🎯 {cons_touches} touches to consolidation edges")

            # [B] BREAKOUT (0-40 pts) - MOST IMPORTANT
            if breakout['breakout']:
                ca = breakout['candles_ago']
                bp = breakout['breakout_pct']
                
                if ca <= 2:
                    score += 40
                    reasons.append(f"🚀 FRESH BREAKOUT! ({ca}h ago, +{bp:.1f}% above consolidation)")
                elif ca <= 5:
                    score += 30
                    reasons.append(f"🚀 Recent breakout ({ca}h ago)")
                elif ca <= 8:
                    score += 18
                    reasons.append(f"🚀 Breakout confirmed ({ca}h ago)")
            elif breakout.get('at_resistance'):
                score += 15
                reasons.append(f"⚡ AT CONSOLIDATION HIGH - breakout imminent")
            else:
                # No breakout yet
                warnings.append("⚠️ No breakout yet - price still in consolidation")
                score -= 15

            # [C] Volume on breakout (0-15 pts)
            if breakout['breakout']:
                bo_idx = breakout['breakout_idx']
                if bo_idx < len(df):
                    bo_vol = df.iloc[bo_idx]['volume']
                    avg_vol = df['volume'].tail(50).mean()
                    vol_mult = bo_vol / avg_vol if avg_vol > 0 else 1

                    if vol_mult > 2.0:
                        score += 15
                        reasons.append(f"📊 HUGE VOLUME on breakout ({vol_mult:.1f}x avg)")
                    elif vol_mult > 1.5:
                        score += 10
                        reasons.append(f"📊 Strong volume ({vol_mult:.1f}x avg)")

            # [D] Liquidity sweep bonus (0-15 pts) - secondary pattern
            if sweep['swept']:
                score += 15
                reasons.append(f"💥 Liquidity sweep detected before breakout")

            # [E] RSI (0-10 pts)
            if 40 <= rsi <= 60:
                score += 10
                reasons.append(f"💎 RSI healthy ({rsi:.0f}) - room to run")
            elif rsi < 40:
                score += 7
                reasons.append(f"💎 RSI low ({rsi:.0f})")

            # [F] Consolidation duration (0-10 pts)
            if cons_duration >= 30:
                score += 10
                reasons.append(f"⏱️ Long consolidation ({cons_duration}h) - big energy")
            elif cons_duration >= 20:
                score += 6

            # [G] Price position (0-5 pts)
            # Best if price just broke out or is at top of consolidation
            dist_from_high = (current - consolidation['high']) / consolidation['high'] * 100
            if -0.5 <= dist_from_high <= 2:
                score += 5

            # ── WARNINGS ──
            if rsi > 70:
                warnings.append("⚠️ RSI overbought")
                score -= 10
            if breakout['breakout'] and breakout['candles_ago'] > 8:
                warnings.append("⚠️ Breakout is getting old")
                score -= 8

            if score < self.min_score_threshold:
                return None

            # ── TRADE LEVELS (Measured Move) ──
            entry = current

            # SL: Below consolidation box
            sl = consolidation['low'] * 0.98
            risk_pct = (entry - sl) / entry * 100

            if risk_pct > 12:
                sl = entry * 0.92
                risk_pct = 8.0

            # TP: Measured move (height of consolidation added to breakout)
            cons_height = consolidation['high'] - consolidation['low']
            
            tp1 = consolidation['high'] + (cons_height * 1.0)  # 1x measured move
            tp2 = consolidation['high'] + (cons_height * 1.8)  # 1.8x
            tp3 = consolidation['high'] + (cons_height * 3.0)  # 3x

            rr   = [abs(t - entry) / abs(sl - entry) for t in [tp1, tp2, tp3]]
            pcts = [(t - entry) / entry * 100          for t in [tp1, tp2, tp3]]

            if   score >= 85: conf = 'ELITE 🔥🔥🔥'
            elif score >= 75: conf = 'HIGH 💎💎'
            elif score >= 65: conf = 'GOOD 💎'
            else:             conf = 'WATCH ✅'

            return {
                'success':       True,
                'symbol':        symbol.replace('/USDT:USDT', ''),
                'full_symbol':   symbol,
                'signal':        'LONG',
                'confidence':    conf,
                'score':         score,
                'entry':         entry,
                'stop_loss':     sl,
                'risk_percent':  risk_pct,
                'targets':       [tp1, tp2, tp3],
                'reward_ratios': rr,
                'target_pcts':   pcts,
                'reasons':       reasons,
                'warnings':      warnings,
                'consolidation': consolidation,
                'breakout':      breakout,
                'sweep':         sweep,
                'rsi':           rsi,
                'vol_ratio':     volr,
                'timestamp':     datetime.now()
            }

        except Exception as e:
            logger.error(f"Analyze {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────
    # FORMATTING
    # ─────────────────────────────────────────────────

    def format_alert(self, r, rank=None):
        rk   = f"#{rank} " if rank else ""
        cons = r['consolidation']
        bo   = r['breakout']
        
        msg  = f"{'═'*46}\n"
        msg += f"💎 <b>{rk}BREAKOUT: {r['symbol']} — {r['confidence']}</b> 💎\n"
        msg += f"{'═'*46}\n\n"

        msg += f"<b>📊 LONG</b>  |  Score: {r['score']:.0f}/100\n"
        msg += f"RSI: {r['rsi']:.0f}  |  Vol: {r['vol_ratio']:.1f}x\n\n"

        # Consolidation info
        msg += f"<b>📦 CONSOLIDATION ZONE:</b>\n"
        msg += f"  High: ${cons['high']:.6f}\n"
        msg += f"  Low:  ${cons['low']:.6f}\n"
        msg += f"  Range: {cons['range_pct']:.1f}%  ({cons['duration']} candles)\n"
        msg += f"  Touches: {cons['touches']}\n\n"

        # Breakout status
        msg += f"<b>🚀 BREAKOUT STATUS:</b>\n"
        if bo['breakout']:
            msg += f"  ✅ BROKE OUT {bo['candles_ago']}h ago\n"
            msg += f"  📈 Breakout price: ${bo['breakout_price']:.6f}\n"
            msg += f"  💪 +{bo['breakout_pct']:.1f}% above consolidation\n"
        elif bo.get('at_resistance'):
            msg += f"  ⚡ AT RESISTANCE - ready to break!\n"
        else:
            msg += f"  ⏳ Watching for breakout\n"

        if r['sweep']['swept']:
            msg += f"  💥 Sweep detected (bonus confirmation)\n"

        msg += f"\n<b>💰 TRADE (Measured Move):</b>\n"
        msg += f"  Entry: ${r['entry']:.6f}\n"
        msg += f"  SL:    ${r['stop_loss']:.6f}  (-{r['risk_percent']:.1f}%)\n\n"

        msg += f"<b>🎯 TARGETS:</b>\n"
        labels = ['1x MM', '1.8x MM', '3x MM']
        for i, (tp, rr, pct, lbl) in enumerate(
                zip(r['targets'], r['reward_ratios'], r['target_pcts'], labels), 1):
            msg += f"  TP{i}: ${tp:.6f}  (+{pct:.1f}%  {rr:.1f}R)  [{lbl}]\n"

        msg += f"\n<b>✅ REASONS:</b>\n"
        for rs in r['reasons']:
            msg += f"  • {rs}\n"

        if r['warnings']:
            msg += f"\n<b>⚠️ WARNINGS:</b>\n"
            for w in r['warnings']:
                msg += f"  {w}\n"

        msg += f"\n<i>⏰ {r['timestamp'].strftime('%Y-%m-%d %H:%M')}</i>"
        msg += f"\n<i>💎 Consolidation → Breakout → Pump!</i>"
        msg += f"\n{'═'*46}"
        return msg

    def should_alert(self, symbol, result):
        if result['score'] < self.min_score_threshold:
            return False
        if symbol in self.alerted_pairs:
            last = self.alerted_pairs[symbol]
            if datetime.now() - last['time'] < timedelta(hours=3):
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

        logger.info(f"💎 BREAKOUT SCAN: {len(self.pairs_to_scan)} pairs")
        await self.send_msg(
            f"🔍 <b>CONSOLIDATION BREAKOUT SCAN</b>\n\n"
            f"Scanning {len(self.pairs_to_scan)} pairs\n"
            f"Pattern: Tight consolidation → Clean breakout"
        )

        t0 = datetime.now(); results = []; alerts = 0

        for i, pair in enumerate(self.pairs_to_scan, 1):
            try:
                if i % 50 == 0:
                    logger.info(f"{i}/{len(self.pairs_to_scan)}")
                sym = await self.get_symbol_format(pair)
                if not sym:
                    continue
                data = await self.fetch_data(sym)
                if not data:
                    continue
                result = self.analyze_breakout(data, sym)
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
                await asyncio.sleep(0.12)
            except Exception as e:
                logger.error(f"Scan {pair}: {e}")

        dur = (datetime.now() - t0).total_seconds()
        self.stats['total_scans'] += 1
        self.stats['total_pairs_scanned'] += len(self.pairs_to_scan)
        self.stats['avg_scan_time'] = dur
        self.stats['last_scan_date'] = datetime.now()
        self.last_scan_time = datetime.now()

        results.sort(key=lambda x: x['score'], reverse=True)
        elite = [r for r in results if r['score'] >= 80]
        good  = [r for r in results if 65 <= r['score'] < 80]

        summ  = f"✅ <b>SCAN COMPLETE</b>\n\n"
        summ += f"📊 Scanned: {len(self.pairs_to_scan)}\n"
        summ += f"⏱️ Time: {dur/60:.1f} min\n"
        summ += f"💎 Elite (80+): {len(elite)}\n"
        summ += f"✅ Good (65-79): {len(good)}\n"
        summ += f"📤 Alerts: {alerts}\n\n"
        summ += f"📡 Tracking: {len(self.active_trades)}"
        await self.send_msg(summ)
        return results

    async def auto_scan_loop(self):
        logger.info(f"Auto-scan every {self.scan_interval//60}m")
        while self.is_scanning:
            try:
                await self.scan_all_pairs()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Loop: {e}")
                await asyncio.sleep(120)

    async def track_trades_loop(self):
        logger.info("Tracking started")
        while self.is_tracking:
            try:
                if not self.active_trades:
                    await asyncio.sleep(self.price_check_interval)
                    continue
                to_remove = []
                actions = [
                    "💡 Take 40%\n📋 Move SL to breakeven",
                    "💡 Take 40%\n📋 Trail SL to TP1",
                    "💡 Close 20%\n🎊 COMPLETE!"
                ]
                for tid, trade in list(self.active_trades.items()):
                    try:
                        if datetime.now() - trade['timestamp'] > timedelta(days=7):
                            await self.send_msg(f"⏰ {trade['symbol']}\n<code>{tid}</code>")
                            to_remove.append(tid); continue
                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price  = ticker['last']
                        for i, (tp, hit) in enumerate(zip(trade['targets'], trade['tp_hit'])):
                            if not hit and price >= tp:
                                pnl = (tp - trade['entry']) / trade['entry'] * 100
                                msg = f"🎯 <b>TP{i+1}!</b>\n\n<code>{tid}</code>\n"
                                msg += f"<b>{trade['symbol']}</b>\n\n"
                                msg += f"Entry: ${trade['entry']:.6f}\n"
                                msg += f"TP{i+1}: ${tp:.6f}\n"
                                msg += f"Profit: <b>+{pnl:.2f}%</b> ({trade['reward_ratios'][i]:.1f}R)\n\n"
                                msg += actions[i]
                                await self.send_msg(msg)
                                trade['tp_hit'][i] = True
                                self.stats[f'tp{i+1}_hits'] += 1
                                if i == 2: to_remove.append(tid)
                        if not trade['sl_hit'] and price <= trade['stop_loss']:
                            loss = (trade['stop_loss'] - trade['entry']) / trade['entry'] * 100
                            msg = f"🛑 <b>STOP</b>\n\n<code>{tid}</code>\n"
                            msg += f"<b>{trade['symbol']}</b>\n\n"
                            msg += f"Entry: ${trade['entry']:.6f}\nSL: ${trade['stop_loss']:.6f}\n"
                            msg += f"Loss: <b>{loss:.2f}%</b>\n\nNext one! 💪"
                            await self.send_msg(msg)
                            trade['sl_hit'] = True
                            self.stats['sl_hits'] += 1
                            to_remove.append(tid)
                    except Exception as e:
                        logger.error(f"Track {tid}: {e}")
                for tid in to_remove:
                    self.active_trades.pop(tid, None)
                self.stats['active_trades_count'] = len(self.active_trades)
                await asyncio.sleep(self.price_check_interval)
            except Exception as e:
                logger.error(f"Track: {e}")
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


class BotCommands:
    def __init__(self, scanner):
        self.s = scanner

    async def cmd_start(self, update, context):
        msg  = "💎 <b>CONSOLIDATION BREAKOUT SCANNER</b>\n\n"
        msg += "<b>Pattern (analyzed from @free_fx_pro):</b>\n\n"
        msg += "1️⃣ 📦 Tight horizontal consolidation (15-40 candles, <8% range)\n"
        msg += "2️⃣ 🚀 Clean breakout above consolidation\n"
        msg += "3️⃣ 📊 Volume spike on breakout (preferred)\n"
        msg += "4️⃣ 💰 Enter on breakout or pullback\n"
        msg += "5️⃣ 🛑 SL below consolidation box\n"
        msg += "6️⃣ 🎯 TP = Measured move (1x-3x)\n\n"
        msg += "<b>COMMANDS:</b>\n"
        msg += "/start_scan      - Auto-scan every 30min\n"
        msg += "/stop_scan\n"
        msg += "/scan_now\n"
        msg += "/start_tracking\n"
        msg += "/stop_tracking\n"
        msg += "/active_trades\n"
        msg += "/status\n"
        msg += "/stats\n\n"
        msg += "💎 <b>Same as @free_fx_pro!</b>"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_start_scan(self, update, context):
        if self.s.is_scanning:
            await update.message.reply_text("⚠️ Already scanning!", parse_mode=ParseMode.HTML)
            return
        self.s.is_scanning = True
        asyncio.create_task(self.s.auto_scan_loop())
        n = len(self.s.pairs_to_scan) if self.s.pairs_to_scan else '300+'
        await update.message.reply_text(
            f"✅ <b>SCANNER STARTED!</b>\n\nPairs: {n}\nEvery {self.s.scan_interval//60}min\n\nScanning now...",
            parse_mode=ParseMode.HTML)

    async def cmd_stop_scan(self, update, context):
        self.s.is_scanning = False
        await update.message.reply_text("🛑 <b>STOPPED</b>", parse_mode=ParseMode.HTML)

    async def cmd_scan_now(self, update, context):
        await update.message.reply_text("🔍 Scanning...", parse_mode=ParseMode.HTML)
        await self.s.scan_all_pairs()

    async def cmd_start_tracking(self, update, context):
        if self.s.is_tracking:
            await update.message.reply_text("⚠️ Already tracking!", parse_mode=ParseMode.HTML)
            return
        self.s.is_tracking = True
        asyncio.create_task(self.s.track_trades_loop())
        await update.message.reply_text(
            f"✅ <b>TRACKING!</b>\n\nEvery {self.s.price_check_interval}s\nActive: {len(self.s.active_trades)}",
            parse_mode=ParseMode.HTML)

    async def cmd_stop_tracking(self, update, context):
        self.s.is_tracking = False
        await update.message.reply_text("🛑 <b>STOPPED</b>", parse_mode=ParseMode.HTML)

    async def cmd_status(self, update, context):
        scan = "🟢" if self.s.is_scanning else "🔴"
        trk  = "🟢" if self.s.is_tracking else "🔴"
        msg  = f"<b>STATUS</b>\n\nScan: {scan}\nTrack: {trk}\n\n"
        msg += f"Pairs: {len(self.s.pairs_to_scan)}\nInterval: {self.s.scan_interval//60}min\n"
        msg += f"Active: {len(self.s.active_trades)}"
        if self.s.last_scan_time:
            mins = int((datetime.now() - self.s.last_scan_time).total_seconds() // 60)
            msg += f"\nLast: {mins}m ago"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_stats(self, update, context):
        s = self.s.stats
        msg = "<b>STATS</b>\n\n"
        msg += f"Scans: {s['total_scans']}\nSignals: {s['signals_found']}\n\n"
        msg += f"TP1: {s['tp1_hits']} 🎯\nTP2: {s['tp2_hits']} 🎯\nTP3: {s['tp3_hits']} 🎯\nSL: {s['sl_hits']} 🛑"
        t = s['tp1_hits'] + s['sl_hits']
        if t > 0:
            msg += f"\n\nWin: {s['tp1_hits']/t*100:.1f}%"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_active_trades(self, update, context):
        trades = self.s.active_trades
        if not trades:
            await update.message.reply_text("📭 No trades", parse_mode=ParseMode.HTML)
            return
        msg = f"📡 <b>ACTIVE ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:15]:
            h   = int((datetime.now() - t['timestamp']).total_seconds() / 3600)
            tps = "".join(["✅" if hit else "⏳" for hit in t['tp_hit']])
            msg += f"<b>{t['symbol']}</b>  ${t['entry']:.6f}  {tps}  {h}h\n"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def main():
    TOKEN = "8186622122:AAGtQcoh_s7QqIAVACmOYVHLqPX-p6dSNVA"
    CHAT  = "7500072234"

    scanner = ConsolidationBreakoutScanner(TOKEN, CHAT)
    app     = Application.builder().token(TOKEN).build()
    cmds    = BotCommands(scanner)

    for cmd, fn in [
        ("start", cmds.cmd_start), ("start_scan", cmds.cmd_start_scan),
        ("stop_scan", cmds.cmd_stop_scan), ("scan_now", cmds.cmd_scan_now),
        ("start_tracking", cmds.cmd_start_tracking), ("stop_tracking", cmds.cmd_stop_tracking),
        ("status", cmds.cmd_status), ("stats", cmds.cmd_stats),
        ("active_trades", cmds.cmd_active_trades),
    ]:
        app.add_handler(CommandHandler(cmd, fn))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logger.info("💎 BREAKOUT SCANNER ONLINE!")

    welcome  = "💎 <b>CONSOLIDATION BREAKOUT SCANNER!</b> 💎\n\n"
    welcome += "<b>Pattern: Tight Box → Clean Breakout → 🚀</b>\n\n"
    welcome += "✅ All USDT perpetuals\n✅ 1H analysis\n"
    welcome += "✅ Scans every 30min\n✅ Live TP/SL tracking\n\n"
    welcome += "/start_scan\n/start_tracking\n\n"
    welcome += "💎 Same as @free_fx_pro!"
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
