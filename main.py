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


class GemScanner:
    """
    💎 GEM SCANNER
    
    THE EXACT PATTERN FROM SCREENSHOTS (BAS, ARIA, RAYSOL):
    1. Coin sits in a FLAT HORIZONTAL BASE for many candles
    2. Coin PUMPS big from base (20-100%+)
    3. Coin RETRACES back down to the original base
    4. Base HOLDS as support again
    5. BOT ALERTS -> BUY the bounce!

    Primary timeframe: 1H
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

        self.scan_interval        = 3600   # 1 hour
        self.min_score_threshold  = 60
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

    # ────────────────────────────────────────────────────────
    # EXCHANGE
    # ────────────────────────────────────────────────────────

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
            logger.error(f"Symbol fmt: {e}")
            return None

    async def load_all_usdt_pairs(self):
        try:
            logger.info("Loading USDT pairs...")
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
            logger.error(f"Load pairs: {e}")
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
            data['1h'] = await self.fetch_df(symbol, '1h', 300)
            await asyncio.sleep(0.05)
            data['4h'] = await self.fetch_df(symbol, '4h', 100)
            await asyncio.sleep(0.05)
            return data
        except Exception as e:
            logger.error(f"Fetch {symbol}: {e}")
            return None

    # ────────────────────────────────────────────────────────
    # PATTERN DETECTION
    # ────────────────────────────────────────────────────────

    def detect_flat_base(self, df):
        """Find the flattest horizontal zone in price history."""
        best        = None
        best_score  = 0

        for window in [15, 20, 25, 30, 40]:
            if len(df) < window + 20:
                continue
            # Don't look at the very last candles - base must have happened before pump
            search_end = len(df) - 10
            for start in range(0, search_end - window, 3):
                end = start + window
                w   = df.iloc[start:end]
                hi  = w['high'].max()
                lo  = w['low'].min()
                if lo == 0:
                    continue
                rng = (hi - lo) / lo * 100
                if rng > 7:
                    continue
                # Score: tighter is better, more candles is better
                score = (7 - rng) * (window / 20)
                if score > best_score:
                    best_score = score
                    best = {
                        'low':       lo,
                        'high':      hi,
                        'mid':       (hi + lo) / 2,
                        'range_pct': rng,
                        'start_idx': start,
                        'end_idx':   end,
                        'duration':  window
                    }
        return best

    def detect_pump(self, df, base):
        """After the base, did price pump at least 15%?"""
        if base is None:
            return None
        after = df.iloc[base['end_idx']:]
        if len(after) < 3:
            return None
        pump_high = after['high'].max()
        pump_pct  = (pump_high - base['mid']) / base['mid'] * 100
        if pump_pct < 15:
            return None
        return {'pump_high': pump_high, 'pump_pct': pump_pct}

    def detect_retrace_to_base(self, df, base, pump):
        """Has price retraced back to the original base zone?"""
        if base is None or pump is None:
            return None
        current = df['close'].iloc[-1]
        # Price must be near/at the base top (within 3% above it)
        zone_top    = base['high'] * 1.03
        zone_bottom = base['low']  * 0.96
        if not (zone_bottom <= current <= zone_top):
            return None
        retrace_pct = (pump['pump_high'] - current) / pump['pump_high'] * 100
        if retrace_pct < 25:
            return None
        dist_top = (current - base['high']) / base['high'] * 100
        dist_low = (current - base['low'])  / base['low']  * 100
        return {
            'current':     current,
            'retrace_pct': retrace_pct,
            'dist_top':    dist_top,
            'dist_low':    dist_low,
            'at_base_top': abs(dist_top) < 2.0,
            'inside_base': dist_top < 0,
        }

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
        except Exception as e:
            logger.error(f"Indicator: {e}")
        return df

    # ────────────────────────────────────────────────────────
    # ANALYSIS
    # ────────────────────────────────────────────────────────

    def analyze_gem(self, data, symbol):
        try:
            if not data or '1h' not in data:
                return None

            df1h = self.add_indicators(data['1h'].copy())
            df4h = self.add_indicators(data['4h'].copy())

            if len(df1h) < 60:
                return None

            base    = self.detect_flat_base(df1h)
            if base is None:
                return None

            pump    = self.detect_pump(df1h, base)
            if pump is None:
                return None

            retrace = self.detect_retrace_to_base(df1h, base, pump)
            if retrace is None:
                return None

            current = retrace['current']
            l1h     = df1h.iloc[-1]
            l4h     = df4h.iloc[-1]
            rsi     = float(l1h.get('rsi', 50) or 50)
            volr    = float(l1h.get('vol_ratio', 1) or 1)

            score    = 0
            reasons  = []
            warnings = []

            # [1] Pump size: bigger pump = bigger bounce potential (0-25 pts)
            pp = min(25, int(pump['pump_pct'] / 4))
            score += pp
            reasons.append(f"🚀 Pumped +{pump['pump_pct']:.0f}% from base")

            # [2] Where is price in the base zone (0-25 pts)
            if retrace['at_base_top']:
                score += 25
                reasons.append("💎 AT BASE TOP - Perfect re-entry!")
            elif retrace['inside_base']:
                score += 20
                reasons.append("🎯 Inside base zone")
            else:
                score += 12
                reasons.append("📍 Near base zone")

            # [3] How tight the base is (0-20 pts)
            rng = base['range_pct']
            if rng < 2.5:
                score += 20
                reasons.append(f"📦 ULTRA TIGHT base ({rng:.1f}%, {base['duration']} candles)")
            elif rng < 4.0:
                score += 15
                reasons.append(f"📦 Tight base ({rng:.1f}%, {base['duration']} candles)")
            elif rng < 6.5:
                score += 8
                reasons.append(f"📦 Base zone ({rng:.1f}%, {base['duration']} candles)")

            # [4] Retrace depth - sweet spot 40-70% (0-15 pts)
            rtr = retrace['retrace_pct']
            if 40 <= rtr <= 70:
                score += 15
                reasons.append(f"📉 Healthy {rtr:.0f}% retrace from high")
            elif 25 <= rtr < 40 or 70 < rtr <= 85:
                score += 8
                reasons.append(f"📉 {rtr:.0f}% retrace from high")
            else:
                score += 3

            # [5] RSI position (0-10 pts)
            if 30 <= rsi <= 55:
                score += 10
                reasons.append(f"💎 RSI healthy ({rsi:.0f})")
            elif rsi < 30:
                score += 8
                reasons.append(f"💎 RSI oversold ({rsi:.0f})")
            elif rsi < 65:
                score += 4

            # [6] Low-volume retrace = healthy (0-10 pts)
            if volr < 0.7:
                score += 10
                reasons.append(f"📊 Low vol retrace ({volr:.1f}x) - healthy pullback")
            elif volr < 1.0:
                score += 5

            # [7] 4H uptrend bonus (0-10 pts)
            try:
                e20 = float(l4h.get('ema_20', 0) or 0)
                e50 = float(l4h.get('ema_50', 0) or 0)
                if e20 > e50 > 0:
                    score += 10
                    reasons.append("✅ 4H uptrend - bigger timeframe bullish")
            except Exception:
                pass

            # Penalties
            if rsi > 72:
                warnings.append("⚠️ RSI still elevated")
                score -= 8
            if rtr > 88:
                warnings.append("⚠️ Very deep retrace - base may break")
                score -= 5

            if score < self.min_score_threshold:
                return None

            # Trade levels
            entry    = current
            atr      = float(l1h.get('atr', current * 0.02) or current * 0.02)
            sl       = base['low'] * 0.965
            risk_pct = (entry - sl) / entry * 100
            if risk_pct > 12:
                sl       = entry * 0.92
                risk_pct = 8.0

            move = entry - sl
            tp1  = entry + move * 1.5
            tp2  = entry + move * 3.0
            tp3  = min(entry + move * 5.0, pump['pump_high'] * 1.1)

            rr   = [abs(t - entry) / abs(sl - entry) for t in [tp1, tp2, tp3]]
            pcts = [(t - entry) / entry * 100          for t in [tp1, tp2, tp3]]

            if   score >= 85: conf = 'ELITE 🔥🔥🔥'
            elif score >= 75: conf = 'HIGH 💎💎'
            elif score >= 65: conf = 'GOOD 💎'
            else:             conf = 'WATCH ✅'

            return {
                'success': True,
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
                'base':          base,
                'pump':          pump,
                'retrace':       retrace,
                'rsi':           rsi,
                'vol_ratio':     volr,
                'timestamp':     datetime.now()
            }

        except Exception as e:
            logger.error(f"Analyze {symbol}: {e}")
            return None

    # ────────────────────────────────────────────────────────
    # FORMATTING
    # ────────────────────────────────────────────────────────

    def format_alert(self, r, rank=None):
        rk   = f"#{rank} " if rank else ""
        msg  = f"{'═'*46}\n"
        msg += f"💎 <b>{rk}GEM: {r['symbol']} — {r['confidence']}</b> 💎\n"
        msg += f"{'═'*46}\n\n"
        msg += f"<b>LONG</b>  Score: {r['score']:.0f}/100\n"
        msg += f"RSI: {r['rsi']:.0f}  Vol: {r['vol_ratio']:.1f}x\n\n"
        msg += f"<b>🔍 PATTERN:</b>\n"
        msg += f"  📦 Base: ${r['base']['low']:.6f} – ${r['base']['high']:.6f}\n"
        msg += f"  📦 {r['base']['range_pct']:.1f}% range  |  {r['base']['duration']} candles\n"
        msg += f"  🚀 Pump high: ${r['pump']['pump_high']:.6f}  (+{r['pump']['pump_pct']:.0f}%)\n"
        msg += f"  📉 Retrace: {r['retrace']['retrace_pct']:.0f}% back to base\n\n"
        msg += f"<b>💰 TRADE:</b>\n"
        msg += f"  Entry: ${r['entry']:.6f}\n"
        msg += f"  SL:    ${r['stop_loss']:.6f}  (-{r['risk_percent']:.1f}%)\n\n"
        msg += f"<b>🎯 TARGETS:</b>\n"
        for i, (tp, rr, pct) in enumerate(zip(r['targets'], r['reward_ratios'], r['target_pcts']), 1):
            msg += f"  TP{i}: ${tp:.6f}  (+{pct:.1f}%  {rr:.1f}R)\n"
        msg += f"\n<b>✅ REASONS:</b>\n"
        for rs in r['reasons']:
            msg += f"  • {rs}\n"
        if r['warnings']:
            msg += f"\n<b>⚠️ WARNINGS:</b>\n"
            for w in r['warnings']:
                msg += f"  {w}\n"
        msg += f"\n<i>⏰ {r['timestamp'].strftime('%Y-%m-%d %H:%M')}</i>"
        msg += f"\n<i>💎 Base → Pump → Retrace → Bounce</i>"
        msg += f"\n{'═'*46}"
        return msg

    def should_alert(self, symbol, result):
        if result['score'] < self.min_score_threshold:
            return False
        if symbol in self.alerted_pairs:
            last = self.alerted_pairs[symbol]
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

    # ────────────────────────────────────────────────────────
    # SCAN
    # ────────────────────────────────────────────────────────

    async def scan_all_pairs(self):
        if not self.pairs_to_scan:
            await self.load_all_usdt_pairs()

        logger.info(f"💎 GEM SCAN: {len(self.pairs_to_scan)} pairs")
        await self.send_msg(
            f"🔍 <b>GEM SCAN STARTED</b>\n\n"
            f"Scanning {len(self.pairs_to_scan)} pairs\n"
            f"Pattern: Base → Pump → Retrace → BUY"
        )

        t0      = datetime.now()
        results = []
        alerts  = 0

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

                result = self.analyze_gem(data, sym)
                if result and result['success']:
                    results.append(result)
                    logger.info(f"💎 {pair}  score={result['score']:.0f}")

                    if (self.should_alert(result['full_symbol'], result)
                            and alerts < self.max_alerts_per_scan):
                        alerts += 1
                        await self.send_msg(self.format_alert(result, rank=alerts))

                        tid = f"{result['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        self.active_trades[tid] = {
                            'trade_id': tid,
                            'symbol': result['symbol'],
                            'full_symbol': result['full_symbol'],
                            'signal': 'LONG',
                            'entry': result['entry'],
                            'stop_loss': result['stop_loss'],
                            'targets': result['targets'],
                            'reward_ratios': result['reward_ratios'],
                            'timestamp': datetime.now(),
                            'tp_hit': [False, False, False],
                            'sl_hit': False,
                        }
                        self.alerted_pairs[result['full_symbol']] = {
                            'time': datetime.now(), 'score': result['score']
                        }
                        self.stats['signals_found'] += 1

                await asyncio.sleep(0.12)

            except Exception as e:
                logger.error(f"Scan {pair}: {e}")

        dur = (datetime.now() - t0).total_seconds()
        self.stats['total_scans']          += 1
        self.stats['total_pairs_scanned']  += len(self.pairs_to_scan)
        self.stats['avg_scan_time']         = dur
        self.stats['last_scan_date']        = datetime.now()
        self.last_scan_time                 = datetime.now()

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

    # ────────────────────────────────────────────────────────
    # LOOPS
    # ────────────────────────────────────────────────────────

    async def auto_scan_loop(self):
        logger.info(f"Auto-scan every {self.scan_interval//60}m")
        while self.is_scanning:
            try:
                await self.scan_all_pairs()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Scan loop: {e}")
                await asyncio.sleep(120)

    async def track_trades_loop(self):
        logger.info("Tracking started")
        while self.is_tracking:
            try:
                if not self.active_trades:
                    await asyncio.sleep(self.price_check_interval)
                    continue

                to_remove = []
                actions   = [
                    "💡 Take 50% profit\n📋 Move SL to breakeven",
                    "💡 Take 30% profit\n📋 Trail SL to TP1",
                    "💡 Take final 20%\n🎊 <b>TRADE COMPLETE!</b>"
                ]

                for tid, trade in list(self.active_trades.items()):
                    try:
                        if datetime.now() - trade['timestamp'] > timedelta(days=7):
                            await self.send_msg(
                                f"⏰ Timeout: {trade['symbol']}\n<code>{tid}</code>")
                            to_remove.append(tid)
                            continue

                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price  = ticker['last']

                        for i, (tp, hit) in enumerate(zip(trade['targets'], trade['tp_hit'])):
                            if not hit and price >= tp:
                                pnl  = (tp - trade['entry']) / trade['entry'] * 100
                                msg  = f"🎯 <b>TP{i+1} HIT!</b>\n\n"
                                msg += f"<code>{tid}</code>\n"
                                msg += f"<b>{trade['symbol']}</b>\n\n"
                                msg += f"Entry:  ${trade['entry']:.6f}\n"
                                msg += f"TP{i+1}:  ${tp:.6f}\n"
                                msg += f"Profit: <b>+{pnl:.2f}%</b>  ({trade['reward_ratios'][i]:.1f}R)\n\n"
                                msg += actions[i]
                                await self.send_msg(msg)
                                trade['tp_hit'][i] = True
                                self.stats[f'tp{i+1}_hits'] += 1
                                if i == 2:
                                    to_remove.append(tid)

                        if not trade['sl_hit'] and price <= trade['stop_loss']:
                            loss = (trade['stop_loss'] - trade['entry']) / trade['entry'] * 100
                            msg  = f"🛑 <b>STOP HIT</b>\n\n"
                            msg += f"<code>{tid}</code>\n"
                            msg += f"<b>{trade['symbol']}</b>\n\n"
                            msg += f"Entry:  ${trade['entry']:.6f}\n"
                            msg += f"SL:     ${trade['stop_loss']:.6f}\n"
                            msg += f"Loss:   <b>{loss:.2f}%</b>\n\nCut and move on! 💪"
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
                logger.error(f"Track loop: {e}")
                await asyncio.sleep(60)

    async def close(self):
        await self.exchange.close()


# ────────────────────────────────────────────────────────
# BOT COMMANDS
# ────────────────────────────────────────────────────────

class GemBotCommands:
    def __init__(self, scanner):
        self.s = scanner

    async def cmd_start(self, update, context):
        msg  = "💎 <b>GEM SCANNER</b>\n\n"
        msg += "<b>Pattern:</b> Flat Base → Pump → Retrace → BUY!\n\n"
        msg += "1️⃣ Coin sits in TIGHT FLAT BASE\n"
        msg += "2️⃣ Coin PUMPS hard from base\n"
        msg += "3️⃣ Coin RETRACES back to base\n"
        msg += "4️⃣ 💎 BOT ALERTS YOU to buy!\n"
        msg += "5️⃣ 🚀 Ride the next pump!\n\n"
        msg += "<b>COMMANDS:</b>\n"
        msg += "/start_scan      - Auto-scan every hour\n"
        msg += "/stop_scan       - Stop scanning\n"
        msg += "/scan_now        - Scan immediately\n"
        msg += "/start_tracking  - Live TP/SL alerts\n"
        msg += "/stop_tracking   - Stop TP/SL alerts\n"
        msg += "/active_trades   - View open trades\n"
        msg += "/status          - Bot status\n"
        msg += "/stats           - Statistics\n\n"
        msg += "💎 <b>Let's catch those gems!</b>"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_start_scan(self, update, context):
        if self.s.is_scanning:
            await update.message.reply_text("⚠️ Already scanning!", parse_mode=ParseMode.HTML)
            return
        self.s.is_scanning = True
        asyncio.create_task(self.s.auto_scan_loop())
        n   = len(self.s.pairs_to_scan) if self.s.pairs_to_scan else '300+'
        msg = (f"✅ <b>GEM SCANNER STARTED!</b>\n\n"
               f"Pairs: {n}\nEvery {self.s.scan_interval//60} min\n"
               f"Min score: {self.s.min_score_threshold}\n\nFirst scan starting now...")
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_stop_scan(self, update, context):
        self.s.is_scanning = False
        await update.message.reply_text("🛑 <b>SCANNER STOPPED</b>", parse_mode=ParseMode.HTML)

    async def cmd_scan_now(self, update, context):
        await update.message.reply_text("🔍 Manual gem scan...", parse_mode=ParseMode.HTML)
        await self.s.scan_all_pairs()

    async def cmd_start_tracking(self, update, context):
        if self.s.is_tracking:
            await update.message.reply_text("⚠️ Already tracking!", parse_mode=ParseMode.HTML)
            return
        self.s.is_tracking = True
        asyncio.create_task(self.s.track_trades_loop())
        msg = (f"✅ <b>TRACKING STARTED!</b>\n\n"
               f"Checking every {self.s.price_check_interval}s\n"
               f"Active: {len(self.s.active_trades)}")
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_stop_tracking(self, update, context):
        self.s.is_tracking = False
        await update.message.reply_text("🛑 <b>TRACKING STOPPED</b>", parse_mode=ParseMode.HTML)

    async def cmd_status(self, update, context):
        scan  = "🟢 RUNNING" if self.s.is_scanning else "🔴 STOPPED"
        track = "🟢 RUNNING" if self.s.is_tracking else "🔴 STOPPED"
        msg   = f"<b>GEM SCANNER STATUS</b>\n\n"
        msg  += f"Scan:   {scan}\n"
        msg  += f"Track:  {track}\n\n"
        msg  += f"Pairs:  {len(self.s.pairs_to_scan)}\n"
        msg  += f"Interval: {self.s.scan_interval//60}min\n"
        msg  += f"Active trades: {len(self.s.active_trades)}\n"
        if self.s.last_scan_time:
            mins = int((datetime.now() - self.s.last_scan_time).total_seconds() // 60)
            msg += f"\nLast scan: {mins}m ago"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_stats(self, update, context):
        s   = self.s.stats
        msg = "<b>GEM STATS</b>\n\n"
        msg += f"Scans:  {s['total_scans']}\n"
        msg += f"Pairs:  {s['total_pairs_scanned']}\n"
        msg += f"Gems:   {s['signals_found']}\n\n"
        msg += f"TP1: {s['tp1_hits']} 🎯\n"
        msg += f"TP2: {s['tp2_hits']} 🎯\n"
        msg += f"TP3: {s['tp3_hits']} 🎯\n"
        msg += f"SL:  {s['sl_hits']} 🛑"
        t = s['tp1_hits'] + s['sl_hits']
        if t > 0:
            msg += f"\n\nWin rate: {s['tp1_hits']/t*100:.1f}%"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_active_trades(self, update, context):
        trades = self.s.active_trades
        if not trades:
            await update.message.reply_text("📭 No active trades", parse_mode=ParseMode.HTML)
            return
        msg = f"📡 <b>ACTIVE TRADES ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:15]:
            h   = int((datetime.now() - t['timestamp']).total_seconds() / 3600)
            tps = "".join(["✅" if hit else "⏳" for hit in t['tp_hit']])
            msg += f"<b>{t['symbol']}</b>  ${t['entry']:.6f}  TPs:{tps}  {h}h\n"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────

async def main():
    TELEGRAM_TOKEN   = "8186622122:AAGtQcoh_s7QqIAVACmOYVHLqPX-p6dSNVA"
    TELEGRAM_CHAT_ID = "7500072234"

    scanner = GemScanner(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    app     = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds    = GemBotCommands(scanner)

    handlers = [
        ("start",          cmds.cmd_start),
        ("start_scan",     cmds.cmd_start_scan),
        ("stop_scan",      cmds.cmd_stop_scan),
        ("scan_now",       cmds.cmd_scan_now),
        ("start_tracking", cmds.cmd_start_tracking),
        ("stop_tracking",  cmds.cmd_stop_tracking),
        ("status",         cmds.cmd_status),
        ("stats",          cmds.cmd_stats),
        ("active_trades",  cmds.cmd_active_trades),
    ]
    for cmd, fn in handlers:
        app.add_handler(CommandHandler(cmd, fn))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    logger.info("💎 GEM SCANNER ONLINE!")

    welcome  = "💎 <b>GEM SCANNER READY!</b> 💎\n\n"
    welcome += "<b>Pattern: Base → Pump → Retrace → BUY</b>\n\n"
    welcome += "✅ All USDT perpetuals\n"
    welcome += "✅ 1H timeframe\n"
    welcome += "✅ Live TP/SL tracking\n\n"
    welcome += "/start_scan — begin scanning\n"
    welcome += "/start_tracking — track TPs\n\n"
    welcome += "💎 <b>Gems incoming!</b>"
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
