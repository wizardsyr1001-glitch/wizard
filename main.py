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

    def detect_displacement_candle(self, df, sweep_info, lookback=15):
        """
        Detect STRONG BULLISH DISPLACEMENT CANDLE after liquidity sweep.
        Per user strategy: Large body, above average range.
        This is THE confirmation candle.
        """
        if not sweep_info.get('swept'):
            return {'displacement': False}

        sweep_idx = sweep_info.get('sweep_idx', len(df) - 10)
        after_sweep = df.iloc[sweep_idx:]

        if len(after_sweep) < 2:
            return {'displacement': False}

        # Calculate average candle body and range
        avg_body  = abs(df['close'] - df['open']).tail(50).mean()
        avg_range = (df['high'] - df['low']).tail(50).mean()

        # Look for strong bullish candle (large body, bullish close)
        for i in range(len(after_sweep)):
            candle = after_sweep.iloc[i]
            body   = abs(candle['close'] - candle['open'])
            rng    = candle['high'] - candle['low']
            
            # Must be bullish
            is_bullish = candle['close'] > candle['open']
            
            # Must be large (>1.5x avg body OR >2x avg range)
            is_large = body > avg_body * 1.5 or rng > avg_range * 2.0
            
            # Body should be majority of range (strong close)
            body_pct = (body / rng * 100) if rng > 0 else 0
            
            if is_bullish and is_large and body_pct > 60:
                return {
                    'displacement':  True,
                    'candle_idx':    sweep_idx + i,
                    'candles_ago':   len(after_sweep) - 1 - i,
                    'body_size':     body / avg_body,
                    'range_size':    rng / avg_range,
                    'displacement_price': candle['close']
                }
        
        return {'displacement': False}

    def detect_fvg(self, df, displacement_info, lookback=10):
        """
        Detect Fair Value Gap (FVG) after displacement candle.
        FVG = gap between candle 1 high and candle 3 low (or vice versa).
        Shows institutional fast move leaving inefficiency.
        """
        if not displacement_info.get('displacement'):
            return {'fvg': False}

        disp_idx = displacement_info.get('candle_idx', len(df) - 5)
        
        # Check if there's a gap after displacement
        if disp_idx + 3 > len(df):
            return {'fvg': False}

        # Bullish FVG: candle[i-1].high < candle[i+1].low
        for i in range(max(disp_idx - 2, 0), min(disp_idx + 3, len(df) - 2)):
            c1 = df.iloc[i]
            c2 = df.iloc[i + 1]  # displacement candle potentially
            c3 = df.iloc[i + 2]
            
            # Bullish FVG
            if c1['high'] < c3['low']:
                fvg_size = c3['low'] - c1['high']
                fvg_pct  = fvg_size / c1['high'] * 100
                return {
                    'fvg':      True,
                    'fvg_low':  c1['high'],
                    'fvg_high': c3['low'],
                    'fvg_size_pct': fvg_pct
                }
        
        return {'fvg': False}

    def detect_mss(self, df, sweep_info, lookback=50):
        """
        Detect MSS (Market Structure Shift).
        MSS = Break of most recent lower high (becomes higher high).
        This confirms trend reversal per user strategy.
        """
        if not sweep_info.get('swept'):
            return {'mss': False}

        sweep_idx = sweep_info.get('sweep_idx', len(df) - 10)
        
        # Find the most recent lower high BEFORE the sweep
        before_sweep = df.iloc[max(0, sweep_idx - lookback):sweep_idx]
        
        if len(before_sweep) < 5:
            return {'mss': False}

        # Find swing highs
        swing_highs = []
        for i in range(3, len(before_sweep) - 3):
            if before_sweep['high'].iloc[i] == before_sweep['high'].iloc[i-3:i+4].max():
                swing_highs.append(before_sweep['high'].iloc[i])
        
        if len(swing_highs) < 2:
            return {'mss': False}

        # Most recent lower high
        recent_lh = swing_highs[-1]
        current   = df['close'].iloc[-1]
        
        # Did we break above it?
        mss_broken = current > recent_lh
        mss_margin = (current - recent_lh) / recent_lh * 100 if mss_broken else 0
        
        return {
            'mss':        mss_broken,
            'recent_lh':  recent_lh,
            'mss_margin': mss_margin
        }

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

            # ── [6] Detect DISPLACEMENT CANDLE after sweep ──
            displacement = self.detect_displacement_candle(df1h, sweep, lookback=15)

            # ── [7] Detect FVG (Fair Value Gap) ──
            fvg = self.detect_fvg(df1h, displacement, lookback=10)

            # ── [8] Detect MSS (Market Structure Shift) ──
            mss = self.detect_mss(df1h, sweep, lookback=50)

            # ── [9] Volume spike on displacement (optional confirmation) ──
            volume_spike = False
            if displacement.get('displacement'):
                disp_idx = displacement['candle_idx']
                if disp_idx < len(df1h):
                    disp_candle_vol = df1h.iloc[disp_idx]['volume']
                    avg_vol = df1h['volume'].tail(50).mean()
                    volume_spike = disp_candle_vol > avg_vol * 1.5

            # ── [10] RSI bullish divergence (optional) ──
            rsi_divergence = False
            if sweep.get('swept') and rsi < 40:
                # Simple check: price made lower low but RSI higher low
                sweep_idx = sweep['sweep_idx']
                if sweep_idx > 10:
                    sweep_low_price = df1h.iloc[sweep_idx]['low']
                    sweep_low_rsi   = df1h.iloc[sweep_idx]['rsi']
                    
                    # Look for previous low
                    prev_section = df1h.iloc[max(0, sweep_idx - 40):sweep_idx]
                    if len(prev_section) > 0:
                        prev_low_price = prev_section['low'].min()
                        prev_low_idx   = prev_section['low'].idxmin()
                        prev_low_rsi   = prev_section.loc[prev_low_idx, 'rsi']
                        
                        # Bullish divergence: lower price but higher RSI
                        if sweep_low_price < prev_low_price and sweep_low_rsi > prev_low_rsi:
                            rsi_divergence = True

            # ── SCORING (User's Exact Strategy) ──
            score    = 0
            reasons  = []
            warnings = []

            # [A] Equal lows / multi-touch support (0-15 pts)
            touches = key_level_info['touches']
            if touches >= 3:
                score += 15
                reasons.append(f"📍 EQUAL LOWS - {touches}x touches at ${key_level:.6f}")
            elif touches == 2:
                score += 10
                reasons.append(f"📍 Strong support - {touches}x touches")
            elif touches == 1:
                score += 5

            # [B] **LIQUIDITY SWEEP** (0-30 pts) — CRITICAL PER USER STRATEGY
            if sweep['swept']:
                ca = sweep['candles_ago']
                if ca <= 5:
                    score += 30
                    reasons.append(f"💥 LIQUIDITY SWEEP! ({ca}h ago, -{sweep['sweep_depth']:.1f}%) ✅")
                elif ca <= 12:
                    score += 22
                    reasons.append(f"💥 Liquidity sweep ({ca}h ago)")
                elif ca <= 20:
                    score += 12
                    reasons.append(f"💥 Recent sweep ({ca}h ago)")
            else:
                # Penalty if no sweep yet
                warnings.append("⚠️ No liquidity sweep detected yet")
                score -= 10

            # [C] **DISPLACEMENT CANDLE** (0-25 pts) — USER STRATEGY RULE #3
            if displacement['displacement']:
                body_mult  = displacement['body_size']
                range_mult = displacement['range_size']
                score += 25
                reasons.append(f"🚀 DISPLACEMENT CANDLE! ({body_mult:.1f}x body, {range_mult:.1f}x range) ✅")
            else:
                warnings.append("⚠️ No displacement candle after sweep")
                score -= 8

            # [D] **MSS (Market Structure Shift)** (0-25 pts) — USER STRATEGY RULE #4
            if mss['mss']:
                score += 25
                reasons.append(f"📈 MSS CONFIRMED! Broke lower high by +{mss['mss_margin']:.1f}% ✅")
            else:
                warnings.append("⚠️ MSS not confirmed - structure not shifted yet")
                score -= 10

            # [E] Price in Discount (0-10 pts)
            if in_discount:
                if discount_pct > 10:
                    score += 10
                    reasons.append(f"🔵 Deep discount ({discount_pct:.1f}% below EQ)")
                else:
                    score += 6
                    reasons.append(f"🔵 Discount zone")

            # [F] **VOLUME SPIKE** on displacement (0-10 pts) — USER OPTIONAL CONFIRMATION
            if volume_spike:
                score += 10
                reasons.append(f"📊 VOLUME SPIKE on displacement ✅")

            # [G] **RSI DIVERGENCE** (0-10 pts) — USER OPTIONAL CONFIRMATION
            if rsi_divergence:
                score += 10
                reasons.append(f"📈 RSI BULLISH DIVERGENCE ✅")
            elif rsi < 35:
                score += 6
                reasons.append(f"💎 RSI oversold ({rsi:.0f})")

            # [H] **FVG (Fair Value Gap)** (0-10 pts) — USER OPTIONAL CONFIRMATION
            if fvg['fvg']:
                score += 10
                reasons.append(f"⚡ FVG DETECTED ({fvg['fvg_size_pct']:.1f}% gap) ✅")

            # [I] Prior downtrend (0-10 pts)
            if trend_info['is_downtrend']:
                drop = abs(trend_info['price_drop_pct'])
                if drop > 25:
                    score += 10
                    reasons.append(f"📉 Strong downtrend before (-{drop:.0f}%)")
                else:
                    score += 5

            # [J] 4H context (0-5 pts)
            try:
                rsi_4h = float(l4h.get('rsi', 50) or 50)
                if rsi_4h < 40:
                    score += 5
                    reasons.append(f"💎 4H RSI oversold ({rsi_4h:.0f})")
            except Exception:
                pass

            # ── WARNINGS ──
            if rsi > 65:
                warnings.append("⚠️ RSI high on 1H")
                score -= 8
            if current > swing_range['premium_start']:
                warnings.append("⚠️ Price in premium - not discount entry")
                score -= 15

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
                'displacement':  displacement,
                'mss':           mss,
                'fvg':           fvg,
                'trend':         trend_info,
                'rsi':           rsi,
                'rsi_divergence': rsi_divergence,
                'volume_spike':  volume_spike,
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

        # Sweep & Confirmations status
        msg += f"<b>🎯 STRATEGY CONFIRMATIONS:</b>\n"
        
        # Rule 1 & 2: Equal lows + Liquidity sweep
        if r['sweep']['swept']:
            msg += f"  ✅ Liquidity Sweep ({r['sweep']['candles_ago']}h ago)\n"
        else:
            msg += f"  ⏳ Liquidity Sweep - waiting\n"

        # Rule 3: Displacement candle
        if r['displacement']['displacement']:
            msg += f"  ✅ Displacement Candle ({r['displacement']['body_size']:.1f}x body)\n"
        else:
            msg += f"  ⏳ Displacement Candle - waiting\n"

        # Rule 4: MSS
        if r['mss']['mss']:
            msg += f"  ✅ MSS (Market Structure Shift)\n"
        else:
            msg += f"  ⏳ MSS - waiting\n"

        # Optional confirmations
        msg += f"\n<b>💎 OPTIONAL CONFIRMATIONS:</b>\n"
        if r.get('volume_spike'):
            msg += f"  ✅ Volume Spike\n"
        if r.get('rsi_divergence'):
            msg += f"  ✅ RSI Bullish Divergence\n"
        if r['fvg']['fvg']:
            msg += f"  ✅ Fair Value Gap ({r['fvg']['fvg_size_pct']:.1f}%)\n"

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
