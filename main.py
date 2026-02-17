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


class SMCGemScanner:
    """
    💎 SMC GEM SCANNER - Exact Strategy From Screenshots
    
    ENTRY LOGIC (confirmed from ORCA, RAYSOL, ARIA charts):
    
    Step 1: Price in strong DOWNTREND (series of lower lows)
    Step 2: Price hits KEY HORIZONTAL LEVEL (previous structure / EQL)
    Step 3: LIQUIDITY SWEEP - price wicks below that level (stop hunt)
    Step 4: CHoCH (Change of Character) - first higher high = reversal signal
    Step 5: Price is in DISCOUNT zone (below 50% equilibrium of swing range)
    Step 6: BOS (Break of Structure) confirms bullish
    Step 7: ENTRY on retest of key level from above OR at equilibrium
    Step 8: SL below the liquidity sweep wick
    Step 9: Target = Premium zone / previous highs

    Timeframe: 1H primary
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
        self.min_score_threshold  = 62
        self.max_alerts_per_scan  = 8
        self.price_check_interval = 120    # 2 min

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

    # ─────────────────────────────────────────────────
    # EXCHANGE HELPERS
    # ─────────────────────────────────────────────────

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
            data['1h']  = await self.fetch_df(symbol, '1h', 300)
            await asyncio.sleep(0.05)
            data['4h']  = await self.fetch_df(symbol, '4h', 100)
            await asyncio.sleep(0.05)
            return data
        except Exception as e:
            logger.error(f"Fetch {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────
    # SMC CORE FUNCTIONS
    # ─────────────────────────────────────────────────

    def find_swing_range(self, df, lookback=100):
        """
        Find the significant swing high and swing low
        to define the PREMIUM / EQUILIBRIUM / DISCOUNT zones.
        Like the indicator in his screenshots.
        """
        recent = df.tail(lookback)
        swing_high = recent['high'].max()
        swing_low  = recent['low'].min()
        equilibrium = (swing_high + swing_low) / 2

        # Premium = top 25%, Discount = bottom 25%
        range_size      = swing_high - swing_low
        premium_start   = swing_high - (range_size * 0.25)
        discount_end    = swing_low  + (range_size * 0.25)

        return {
            'swing_high':    swing_high,
            'swing_low':     swing_low,
            'equilibrium':   equilibrium,
            'premium_start': premium_start,
            'discount_end':  discount_end,
            'range_size':    range_size
        }

    def detect_downtrend(self, df, lookback=40):
        """
        Confirm the prior downtrend (series of lower highs + lower lows).
        This is required before a valid CHoCH can happen.
        """
        recent = df.tail(lookback)
        highs  = recent['high'].values
        lows   = recent['low'].values

        # Count lower highs
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        # Count lower lows
        lower_lows  = sum(1 for i in range(1, len(lows))  if lows[i]  < lows[i-1])

        total_candles   = lookback - 1
        lh_ratio        = lower_highs / total_candles
        ll_ratio        = lower_lows  / total_candles

        # Strong downtrend = majority of candles making lower structure
        is_downtrend    = lh_ratio > 0.50 and ll_ratio > 0.45

        # Also check: recent price significantly below where it started
        price_drop_pct  = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100

        return {
            'is_downtrend':   is_downtrend,
            'lh_ratio':       lh_ratio,
            'll_ratio':       ll_ratio,
            'price_drop_pct': price_drop_pct
        }

    def detect_liquidity_sweep(self, df, key_level, lookback=30):
        """
        Detect a LIQUIDITY SWEEP below a key level.
        = Price wicks below the key level then closes back above it.
        This is the stop hunt that precedes the big move up.
        
        Key in all 3 charts: price briefly dips below support then snaps back.
        """
        recent = df.tail(lookback)

        for i in range(len(recent) - 1, max(len(recent) - 15, 0), -1):
            candle = recent.iloc[i]
            # Wick went below key level but CLOSED above it (or near it)
            if (candle['low'] < key_level * 0.995 and
                    candle['close'] > key_level * 0.985):
                # How deep was the sweep?
                sweep_depth_pct = (key_level - candle['low']) / key_level * 100
                candles_ago     = len(recent) - 1 - i

                return {
                    'swept':          True,
                    'sweep_low':      candle['low'],
                    'sweep_depth':    sweep_depth_pct,
                    'candles_ago':    candles_ago,
                    'sweep_idx':      i
                }

        return {'swept': False}

    def detect_choch(self, df, sweep_info, lookback=20):
        """
        Detect CHoCH (Change of Character) AFTER the liquidity sweep.
        CHoCH = price makes a HIGHER HIGH after a series of lower highs.
        This is the first sign of reversal after the sweep.
        """
        if not sweep_info.get('swept'):
            return {'choch': False}

        # Look at candles AFTER the sweep
        sweep_idx = sweep_info.get('sweep_idx', len(df) - 5)
        after_sweep = df.iloc[sweep_idx:]

        if len(after_sweep) < 3:
            return {'choch': False}

        # Find if price made a higher high after sweep
        highs = after_sweep['high'].values
        for i in range(2, len(highs)):
            if highs[i] > highs[i-1] and highs[i-1] > highs[i-2]:
                # Higher highs forming = CHoCH
                choch_price = highs[i]
                candles_ago = len(after_sweep) - 1 - i
                return {
                    'choch':       True,
                    'choch_price': choch_price,
                    'candles_ago': candles_ago
                }

        # Also check: simply if current price is above sweep low + making new high
        if len(highs) >= 2 and highs[-1] > highs[-2]:
            return {
                'choch':       True,
                'choch_price': highs[-1],
                'candles_ago': 0
            }

        return {'choch': False}

    def detect_bos(self, df, lookback=50):
        """
        Detect BOS (Break of Structure).
        BOS = price breaks above a previous swing high = bullish confirmation.
        In his charts this appears AFTER CHoCH and confirms the new uptrend.
        """
        recent = df.tail(lookback)

        # Find previous swing high (before recent lows)
        mid_point   = len(recent) // 2
        first_half  = recent.iloc[:mid_point]
        second_half = recent.iloc[mid_point:]

        prev_high   = first_half['high'].max()
        current     = recent['close'].iloc[-1]

        bos_broken  = current > prev_high
        bos_margin  = (current - prev_high) / prev_high * 100 if bos_broken else 0

        return {
            'bos':        bos_broken,
            'prev_high':  prev_high,
            'bos_margin': bos_margin
        }

    def find_key_level(self, df, lookback=100):
        """
        Find the KEY HORIZONTAL LEVEL that price is respecting.
        In his charts = yellow horizontal line = previous support/EQL.
        This is where the liquidity sweep happens.
        """
        recent = df.tail(lookback)

        # Find clusters of equal lows (EQL) - the yellow line in his charts
        lows = recent['low'].values
        best_level = None
        best_touches = 0

        for i in range(len(lows)):
            level    = lows[i]
            touches  = 0
            for j in range(len(lows)):
                if abs(lows[j] - level) / level < 0.008:  # within 0.8%
                    touches += 1
            if touches > best_touches:
                best_touches = touches
                best_level   = level

        # Also check recent swing lows
        swing_lows = []
        for i in range(5, len(recent) - 5):
            if recent['low'].iloc[i] == recent['low'].iloc[i-5:i+6].min():
                swing_lows.append(recent['low'].iloc[i])

        if not best_level and swing_lows:
            best_level   = min(swing_lows)
            best_touches = 1

        return {
            'level':   best_level,
            'touches': best_touches
        }

    def is_price_in_discount(self, current_price, swing_range):
        """Check if price is in DISCOUNT zone (below equilibrium)."""
        eq  = swing_range['equilibrium']
        low = swing_range['swing_low']
        if low == 0:
            return False
        # Price below equilibrium = discount
        pct_below_eq = (eq - current_price) / eq * 100
        return pct_below_eq > 0  # Any amount below EQ = discount

    def add_indicators(self, df):
        if len(df) < 50:
            return df
        try:
            df['ema_20']    = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50']    = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema_200']   = ta.trend.EMAIndicator(df['close'], window=min(200, len(df)-1)).ema_indicator()
            df['rsi']       = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['vol_sma']   = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / df['vol_sma'].replace(0, 1)
            df['atr']       = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=14).average_true_range()
        except Exception as e:
            logger.error(f"Indicator err: {e}")
        return df

    # ─────────────────────────────────────────────────
    # MAIN ANALYSIS
    # ─────────────────────────────────────────────────

    def analyze_smc(self, data, symbol):
        """
        Full SMC analysis matching exactly what's in the screenshots.
        ORCA, RAYSOL, ARIA all use the same sequence.
        """
        try:
            if not data or '1h' not in data:
                return None

            df1h = self.add_indicators(data['1h'].copy())
            df4h = self.add_indicators(data['4h'].copy())

            if len(df1h) < 80:
                return None

            current = df1h['close'].iloc[-1]
            l1h     = df1h.iloc[-1]
            l4h     = df4h.iloc[-1]
            rsi     = float(l1h.get('rsi', 50) or 50)
            atr     = float(l1h.get('atr', current * 0.02) or current * 0.02)

            # ── [1] Get swing range → Premium/EQ/Discount ──
            swing_range = self.find_swing_range(df1h, lookback=120)

            # ── [2] Must be in DISCOUNT zone ──
            in_discount = self.is_price_in_discount(current, swing_range)
            discount_pct = (swing_range['equilibrium'] - current) / swing_range['equilibrium'] * 100

            # Allow slight above equilibrium but not in premium
            if current > swing_range['premium_start']:
                return None  # Price in premium - not our setup

            # ── [3] Check downtrend preceded this ──
            trend_info = self.detect_downtrend(df1h, lookback=60)

            # ── [4] Find key level ──
            key_level_info = self.find_key_level(df1h, lookback=100)
            key_level = key_level_info['level']

            if key_level is None:
                return None

            # ── [5] Detect liquidity sweep ──
            sweep = self.detect_liquidity_sweep(df1h, key_level, lookback=40)

            # ── [6] Detect CHoCH after sweep ──
            choch = self.detect_choch(df1h, sweep, lookback=20)

            # ── [7] Detect BOS ──
            bos = self.detect_bos(df1h, lookback=60)

            # ── SCORING ──
            score    = 0
            reasons  = []
            warnings = []

            # [A] Price in Discount zone (0-20 pts) — ESSENTIAL
            if in_discount:
                if discount_pct > 15:
                    score += 20
                    reasons.append(f"🔵 DEEP DISCOUNT zone ({discount_pct:.1f}% below EQ)")
                elif discount_pct > 5:
                    score += 15
                    reasons.append(f"🔵 DISCOUNT zone ({discount_pct:.1f}% below EQ)")
                else:
                    score += 8
                    reasons.append(f"🔵 Near Equilibrium (slightly discounted)")
            else:
                # Near equilibrium is still ok
                eq_dist = abs(current - swing_range['equilibrium']) / swing_range['equilibrium'] * 100
                if eq_dist < 5:
                    score += 5
                    reasons.append(f"⚖️ Near Equilibrium ({eq_dist:.1f}% from EQ)")

            # [B] Liquidity sweep detected (0-25 pts) — KEY SIGNAL
            if sweep['swept']:
                ca = sweep['candles_ago']
                if ca <= 5:
                    score += 25
                    reasons.append(f"💥 FRESH LIQUIDITY SWEEP ({ca} candles ago, -{sweep['sweep_depth']:.1f}%)")
                elif ca <= 12:
                    score += 18
                    reasons.append(f"💥 Liquidity sweep ({ca} candles ago)")
                elif ca <= 20:
                    score += 10
                    reasons.append(f"💥 Recent sweep ({ca} candles ago)")
            else:
                # Check if current price is just AT key level (about to sweep)
                dist_to_key = (current - key_level) / key_level * 100
                if 0 <= dist_to_key < 1.5:
                    score += 8
                    reasons.append(f"⚡ Price AT key level (sweep imminent?)")

            # [C] CHoCH detected (0-20 pts) — CONFIRMS REVERSAL
            if choch['choch']:
                ca = choch.get('candles_ago', 5)
                if ca <= 3:
                    score += 20
                    reasons.append(f"🔄 FRESH CHoCH - Trend reversal confirmed!")
                elif ca <= 8:
                    score += 15
                    reasons.append(f"🔄 CHoCH confirmed ({ca} candles ago)")
                else:
                    score += 8
                    reasons.append(f"🔄 CHoCH present")

            # [D] Prior downtrend (0-15 pts) — CONTEXT
            if trend_info['is_downtrend']:
                drop = abs(trend_info['price_drop_pct'])
                if drop > 30:
                    score += 15
                    reasons.append(f"📉 Strong downtrend before reversal (-{drop:.0f}%)")
                elif drop > 15:
                    score += 10
                    reasons.append(f"📉 Downtrend present (-{drop:.0f}%)")
                else:
                    score += 5
            elif trend_info['price_drop_pct'] < -10:
                score += 8
                reasons.append(f"📉 Price dropped ({abs(trend_info['price_drop_pct']):.0f}% from recent high)")

            # [E] BOS (Break of Structure) (0-15 pts) — FINAL CONFIRMATION
            if bos['bos']:
                score += 15
                reasons.append(f"✅ BOS - Structure broken bullish (+{bos['bos_margin']:.1f}%)")

            # [F] RSI position (0-10 pts)
            if rsi < 35:
                score += 10
                reasons.append(f"💎 RSI oversold ({rsi:.0f}) — bounce fuel ready")
            elif rsi < 45:
                score += 7
                reasons.append(f"💎 RSI low ({rsi:.0f}) — healthy for bounce")
            elif rsi < 55:
                score += 4

            # [G] Volume surge on recent candle (0-10 pts)
            vol_ratio = float(l1h.get('vol_ratio', 1) or 1)
            if vol_ratio > 2.0:
                score += 10
                reasons.append(f"📊 HIGH VOLUME ({vol_ratio:.1f}x) — institutional activity")
            elif vol_ratio > 1.3:
                score += 5
                reasons.append(f"📊 Above avg volume ({vol_ratio:.1f}x)")

            # [H] 4H context (0-10 pts)
            try:
                e20_4h = float(l4h.get('ema_20', 0) or 0)
                e50_4h = float(l4h.get('ema_50', 0) or 0)
                rsi_4h = float(l4h.get('rsi', 50) or 50)
                if rsi_4h < 40:
                    score += 10
                    reasons.append(f"💎 4H RSI oversold ({rsi_4h:.0f}) — bigger picture supports bounce")
                elif rsi_4h < 50:
                    score += 5
            except Exception:
                pass

            # ── WARNINGS ──
            if rsi > 65:
                warnings.append("⚠️ RSI elevated on 1H — momentum may stall")
                score -= 8
            if not sweep['swept'] and not choch['choch']:
                warnings.append("⚠️ No sweep or CHoCH yet — wait for confirmation")
                score -= 10
            if current > swing_range['equilibrium'] * 1.05:
                warnings.append("⚠️ Price above equilibrium — not ideal discount entry")
                score -= 10

            if score < self.min_score_threshold:
                return None

            # ── TRADE LEVELS (matching his style) ──
            entry   = current

            # SL: below the liquidity sweep low (or below key level)
            if sweep['swept']:
                sl = sweep['sweep_low'] * 0.985   # just below sweep wick
            else:
                sl = key_level * 0.97

            risk_pct = (entry - sl) / entry * 100
            if risk_pct > 12:
                sl       = entry * 0.91
                risk_pct = 9.0
            if risk_pct < 1:
                sl       = entry * 0.96
                risk_pct = 4.0

            # Targets: Equilibrium → Previous high → Premium zone
            eq     = swing_range['equilibrium']
            ph     = swing_range['swing_high']
            prem   = swing_range['premium_start']

            move   = entry - sl
            tp1    = max(entry + move * 1.5, eq)         # At minimum to equilibrium
            tp2    = max(entry + move * 3.0, (eq + ph) / 2)  # Halfway to old high
            tp3    = max(entry + move * 5.0, ph * 0.98)  # Near/at previous high

            rr     = [abs(t - entry) / abs(sl - entry) for t in [tp1, tp2, tp3]]
            pcts   = [(t - entry) / entry * 100          for t in [tp1, tp2, tp3]]

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
                'swing_range':   swing_range,
                'sweep':         sweep,
                'choch':         choch,
                'bos':           bos,
                'trend':         trend_info,
                'rsi':           rsi,
                'vol_ratio':     vol_ratio,
                'timestamp':     datetime.now()
            }

        except Exception as e:
            logger.error(f"SMC analyze {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────
    # FORMATTING
    # ─────────────────────────────────────────────────

    def format_alert(self, r, rank=None):
        rk   = f"#{rank} " if rank else ""
        sr   = r['swing_range']
        msg  = f"{'═'*46}\n"
        msg += f"💎 <b>{rk}SMC GEM: {r['symbol']} — {r['confidence']}</b> 💎\n"
        msg += f"{'═'*46}\n\n"

        msg += f"<b>📊 LONG</b>  |  Score: {r['score']:.0f}/100\n"
        msg += f"RSI: {r['rsi']:.0f}  |  Vol: {r['vol_ratio']:.1f}x\n\n"

        # SMC Zone info
        msg += f"<b>🗺️ SMC ZONES (1H):</b>\n"
        msg += f"  🔴 Premium:     ${sr['premium_start']:.6f}+\n"
        msg += f"  ⚖️ Equilibrium: ${sr['equilibrium']:.6f}\n"
        msg += f"  🔵 Discount:    Below ${sr['equilibrium']:.6f}\n"
        msg += f"  📍 Current:     ${r['entry']:.6f}"

        # Show which zone price is in
        if r['entry'] > sr['premium_start']:
            msg += " 🔴 PREMIUM\n\n"
        elif r['entry'] > sr['equilibrium']:
            msg += " ⚖️ NEAR EQ\n\n"
        else:
            pct_disc = (sr['equilibrium'] - r['entry']) / sr['equilibrium'] * 100
            msg += f" 🔵 DISCOUNT ({pct_disc:.1f}% below EQ)\n\n"

        # Sweep & CHoCH status
        msg += f"<b>🔍 SMC SIGNALS:</b>\n"
        if r['sweep']['swept']:
            msg += f"  💥 Liq. Sweep: ✅ ({r['sweep']['candles_ago']}h ago, -{r['sweep']['sweep_depth']:.1f}%)\n"
        else:
            msg += f"  💥 Liq. Sweep: ⏳ Watching...\n"

        if r['choch']['choch']:
            msg += f"  🔄 CHoCH:      ✅ Confirmed\n"
        else:
            msg += f"  🔄 CHoCH:      ⏳ Not yet\n"

        if r['bos']['bos']:
            msg += f"  📈 BOS:        ✅ Broken (+{r['bos']['bos_margin']:.1f}%)\n"
        else:
            msg += f"  📈 BOS:        ⏳ Watching\n"

        msg += f"\n<b>💰 TRADE:</b>\n"
        msg += f"  Entry: ${r['entry']:.6f}\n"
        msg += f"  SL:    ${r['stop_loss']:.6f}  (-{r['risk_percent']:.1f}%)\n\n"

        msg += f"<b>🎯 TARGETS:</b>\n"
        labels = ['EQ Zone', 'Mid-Range', 'Prev High']
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
        msg += f"\n<i>💎 SMC: Sweep → CHoCH → Discount Entry → Premium Target</i>"
        msg += f"\n{'═'*46}"
        return msg

    def should_alert(self, symbol, result):
        if result['score'] < self.min_score_threshold:
            return False
        if symbol in self.alerted_pairs:
            last = self.alerted_pairs[symbol]
            if datetime.now() - last['time'] < timedelta(hours=4):
                if result['score'] < last['score'] + 15:
                    return False
        return True

    async def send_msg(self, msg):
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id, text=msg, parse_mode=ParseMode.HTML)
        except Exception as e:
            logger.error(f"Send: {e}")

    # ─────────────────────────────────────────────────
    # SCAN
    # ─────────────────────────────────────────────────

    async def scan_all_pairs(self):
        if not self.pairs_to_scan:
            await self.load_all_usdt_pairs()

        logger.info(f"💎 SMC SCAN: {len(self.pairs_to_scan)} pairs")
        await self.send_msg(
            f"🔍 <b>SMC GEM SCAN STARTED</b>\n\n"
            f"Scanning {len(self.pairs_to_scan)} pairs\n"
            f"Strategy: Liquidity Sweep → CHoCH → Discount Entry"
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
                result = self.analyze_smc(data, sym)
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

        summ  = f"✅ <b>SMC SCAN COMPLETE</b>\n\n"
        summ += f"📊 Scanned: {len(self.pairs_to_scan)}\n"
        summ += f"⏱️ Time: {dur/60:.1f} min\n"
        summ += f"💎 Elite SMC (80+): {len(elite)}\n"
        summ += f"✅ Good SMC (62-79): {len(good)}\n"
        summ += f"📤 Alerts: {alerts}\n\n"
        summ += f"📡 Tracking: {len(self.active_trades)}"
        await self.send_msg(summ)
        return results

    async def auto_scan_loop(self):
        logger.info(f"SMC auto-scan every {self.scan_interval//60}m")
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
                actions = [
                    "💡 Take 40% — Move SL to breakeven",
                    "💡 Take 40% — Trail SL to TP1",
                    "💡 Close 20% — 🎊 TRADE COMPLETE!"
                ]
                for tid, trade in list(self.active_trades.items()):
                    try:
                        if datetime.now() - trade['timestamp'] > timedelta(days=7):
                            await self.send_msg(f"⏰ Timeout: {trade['symbol']}\n<code>{tid}</code>")
                            to_remove.append(tid); continue
                        ticker = await self.exchange.fetch_ticker(trade['full_symbol'])
                        price  = ticker['last']
                        for i, (tp, hit) in enumerate(zip(trade['targets'], trade['tp_hit'])):
                            if not hit and price >= tp:
                                pnl  = (tp - trade['entry']) / trade['entry'] * 100
                                msg  = f"🎯 <b>TP{i+1} HIT!</b>\n\n<code>{tid}</code>\n"
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
                            msg  = f"🛑 <b>STOP HIT</b>\n\n<code>{tid}</code>\n"
                            msg += f"<b>{trade['symbol']}</b>\n\n"
                            msg += f"Entry: ${trade['entry']:.6f}\nSL: ${trade['stop_loss']:.6f}\n"
                            msg += f"Loss: <b>{loss:.2f}%</b>\n\nCut & move on! 💪"
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


# ─────────────────────────────────────────────────
# BOT COMMANDS
# ─────────────────────────────────────────────────

class SMCBotCommands:
    def __init__(self, scanner):
        self.s = scanner

    async def cmd_start(self, update, context):
        msg  = "💎 <b>SMC GEM SCANNER</b>\n\n"
        msg += "<b>Strategy (from @free_fx_pro):</b>\n\n"
        msg += "1️⃣ 📉 Prior downtrend exists\n"
        msg += "2️⃣ 💥 Liquidity SWEEP below key level\n"
        msg += "3️⃣ 🔄 CHoCH confirms reversal\n"
        msg += "4️⃣ 🔵 Price in DISCOUNT zone\n"
        msg += "5️⃣ 📈 BOS confirms bullish structure\n"
        msg += "6️⃣ 🚀 Enter → Target PREMIUM zone!\n\n"
        msg += "<b>COMMANDS:</b>\n"
        msg += "/start_scan      - Auto-scan hourly\n"
        msg += "/stop_scan       - Stop\n"
        msg += "/scan_now        - Scan now\n"
        msg += "/start_tracking  - TP/SL alerts\n"
        msg += "/stop_tracking   - Stop alerts\n"
        msg += "/active_trades   - Open trades\n"
        msg += "/status          - Status\n"
        msg += "/stats           - Statistics\n\n"
        msg += "💎 <b>Same strategy as @free_fx_pro!</b>"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_start_scan(self, update, context):
        if self.s.is_scanning:
            await update.message.reply_text("⚠️ Already scanning!", parse_mode=ParseMode.HTML)
            return
        self.s.is_scanning = True
        asyncio.create_task(self.s.auto_scan_loop())
        n = len(self.s.pairs_to_scan) if self.s.pairs_to_scan else '300+'
        await update.message.reply_text(
            f"✅ <b>SMC SCANNER STARTED!</b>\n\nPairs: {n}\nEvery {self.s.scan_interval//60}min\n\nFirst scan now...",
            parse_mode=ParseMode.HTML)

    async def cmd_stop_scan(self, update, context):
        self.s.is_scanning = False
        await update.message.reply_text("🛑 <b>SCANNER STOPPED</b>", parse_mode=ParseMode.HTML)

    async def cmd_scan_now(self, update, context):
        await update.message.reply_text("🔍 SMC scan starting...", parse_mode=ParseMode.HTML)
        await self.s.scan_all_pairs()

    async def cmd_start_tracking(self, update, context):
        if self.s.is_tracking:
            await update.message.reply_text("⚠️ Already tracking!", parse_mode=ParseMode.HTML)
            return
        self.s.is_tracking = True
        asyncio.create_task(self.s.track_trades_loop())
        await update.message.reply_text(
            f"✅ <b>TRACKING STARTED!</b>\n\nEvery {self.s.price_check_interval}s\nActive: {len(self.s.active_trades)}",
            parse_mode=ParseMode.HTML)

    async def cmd_stop_tracking(self, update, context):
        self.s.is_tracking = False
        await update.message.reply_text("🛑 <b>TRACKING STOPPED</b>", parse_mode=ParseMode.HTML)

    async def cmd_status(self, update, context):
        scan = "🟢 ON" if self.s.is_scanning else "🔴 OFF"
        trk  = "🟢 ON" if self.s.is_tracking else "🔴 OFF"
        msg  = f"<b>SMC SCANNER STATUS</b>\n\nScan: {scan}\nTrack: {trk}\n\n"
        msg += f"Pairs: {len(self.s.pairs_to_scan)}\nInterval: {self.s.scan_interval//60}min\n"
        msg += f"Active trades: {len(self.s.active_trades)}"
        if self.s.last_scan_time:
            mins = int((datetime.now() - self.s.last_scan_time).total_seconds() // 60)
            msg += f"\nLast scan: {mins}m ago"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_stats(self, update, context):
        s = self.s.stats
        msg = "<b>SMC SCANNER STATS</b>\n\n"
        msg += f"Scans: {s['total_scans']}\nPairs: {s['total_pairs_scanned']}\nGems: {s['signals_found']}\n\n"
        msg += f"TP1: {s['tp1_hits']} 🎯\nTP2: {s['tp2_hits']} 🎯\nTP3: {s['tp3_hits']} 🎯\nSL: {s['sl_hits']} 🛑"
        t = s['tp1_hits'] + s['sl_hits']
        if t > 0:
            msg += f"\n\nWin rate: {s['tp1_hits']/t*100:.1f}%"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_active_trades(self, update, context):
        trades = self.s.active_trades
        if not trades:
            await update.message.reply_text("📭 No active trades", parse_mode=ParseMode.HTML)
            return
        msg = f"📡 <b>ACTIVE SMC TRADES ({len(trades)})</b>\n\n"
        for tid, t in list(trades.items())[:15]:
            h   = int((datetime.now() - t['timestamp']).total_seconds() / 3600)
            tps = "".join(["✅" if hit else "⏳" for hit in t['tp_hit']])
            msg += f"<b>{t['symbol']}</b>  ${t['entry']:.6f}  {tps}  {h}h\n"
        await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

async def main():
    TELEGRAM_TOKEN   = "8186622122:AAGtQcoh_s7QqIAVACmOYVHLqPX-p6dSNVA"
    TELEGRAM_CHAT_ID = "7500072234"

    scanner = SMCGemScanner(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    app     = Application.builder().token(TELEGRAM_TOKEN).build()
    cmds    = SMCBotCommands(scanner)

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
    logger.info("💎 SMC GEM SCANNER ONLINE!")

    welcome  = "💎 <b>SMC GEM SCANNER READY!</b> 💎\n\n"
    welcome += "<b>Strategy: Sweep → CHoCH → Discount → 🚀</b>\n\n"
    welcome += "✅ All USDT perpetuals\n✅ 1H + 4H analysis\n"
    welcome += "✅ Liquidity sweep detection\n✅ CHoCH & BOS detection\n"
    welcome += "✅ Premium/Discount/EQ zones\n\n"
    welcome += "/start_scan — begin!\n/start_tracking — track TPs\n\n"
    welcome += "💎 Same strategy as @free_fx_pro!"
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
