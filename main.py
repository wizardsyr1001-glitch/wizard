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


class EquilibriumRetestScanner:
    """
    💎 EQUILIBRIUM RETEST SCANNER
    
    THE EXACT PATTERN from @free_fx_pro (BAN, ESP, WLFI, KITE confirmed):
    
    ══════════════════════════════════════════════════════════════
    LONG SETUP (what we see in ALL 4 screenshots):
    ══════════════════════════════════════════════════════════════
    
    1. Price falls from PREMIUM (red zone) → crosses EQUILIBRIUM (yellow line)
    2. Price enters DISCOUNT zone (blue zone)
    3. Price RETESTS equilibrium 2-3 times from below
       - Each retest = price touches EQ but gets rejected back down
       - This confirms EQ is strong resistance
    4. Price HOLDS in discount zone (doesn't break back up)
    5. 🚀 ENTRY LONG in discount - target back to EQ/Premium
    
    ══════════════════════════════════════════════════════════════
    SHORT SETUP (opposite):
    ══════════════════════════════════════════════════════════════
    
    1. Price rises from DISCOUNT → crosses EQUILIBRIUM
    2. Price enters PREMIUM zone
    3. Price retests EQ 2-3 times from above
    4. Price holds in premium
    5. 🚀 ENTRY SHORT in premium - target back to EQ/Discount
    
    TIMEFRAME: 15 minute
    SCAN FREQUENCY: Every 5 minutes (to catch retests quickly)
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

        # Fast scanning for 15min timeframe
        self.scan_interval        = 300    # 5 minutes (to catch retests)
        self.min_score_threshold  = 70
        self.max_alerts_per_scan  = 10
        self.price_check_interval = 60

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
            # Top 120 pairs - quality focus
            self.pairs_to_scan = [p['base'] for p in perps[:120]]
            self.all_symbols   = [p['symbol'] for p in perps[:120]]
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
            data['15m'] = await self.fetch_df(symbol, '15m', 200)
            await asyncio.sleep(0.04)
            data['1h']  = await self.fetch_df(symbol, '1h', 100)
            await asyncio.sleep(0.04)
            return data
        except Exception as e:
            logger.error(f"Fetch {symbol}: {e}")
            return None

    # ─────────────────────────────────────────────────
    # CORE: EQUILIBRIUM & ZONE DETECTION
    # ─────────────────────────────────────────────────

    def calculate_zones(self, df, lookback=150):
        """
        Calculate Premium / Equilibrium / Discount zones.
        This is what creates the red/yellow/blue zones in his charts.
        """
        recent = df.tail(lookback)
        
        swing_high = recent['high'].max()
        swing_low  = recent['low'].min()
        
        # Equilibrium = 50% of range
        equilibrium = (swing_high + swing_low) / 2
        
        # Premium zone = top 25%
        range_size = swing_high - swing_low
        premium_start = swing_high - (range_size * 0.25)
        
        # Discount zone = bottom 25%
        discount_end = swing_low + (range_size * 0.25)
        
        return {
            'swing_high':    swing_high,
            'swing_low':     swing_low,
            'equilibrium':   equilibrium,
            'premium_start': premium_start,
            'discount_end':  discount_end,
            'range_size':    range_size
        }

    def detect_eq_cross_to_discount(self, df, zones, lookback=50):
        """
        Detect when price crossed FROM premium INTO discount.
        This is step 1 of the pattern.
        """
        eq = zones['equilibrium']
        recent = df.tail(lookback)
        
        # Find the cross point
        cross_idx = None
        for i in range(len(recent) - 1):
            prev_close = recent['close'].iloc[i]
            curr_close = recent['close'].iloc[i + 1]
            
            # Crossed from above EQ to below EQ
            if prev_close > eq and curr_close < eq:
                cross_idx = i + 1
                break
        
        if cross_idx is None:
            return {'crossed': False}
        
        # How long ago was the cross?
        candles_since_cross = len(recent) - 1 - cross_idx
        
        return {
            'crossed': True,
            'cross_idx': cross_idx,
            'candles_ago': candles_since_cross,
            'cross_price': recent['close'].iloc[cross_idx]
        }

    def detect_eq_cross_to_premium(self, df, zones, lookback=50):
        """
        Detect when price crossed FROM discount INTO premium.
        For SHORT setups.
        """
        eq = zones['equilibrium']
        recent = df.tail(lookback)
        
        cross_idx = None
        for i in range(len(recent) - 1):
            prev_close = recent['close'].iloc[i]
            curr_close = recent['close'].iloc[i + 1]
            
            # Crossed from below EQ to above EQ
            if prev_close < eq and curr_close > eq:
                cross_idx = i + 1
                break
        
        if cross_idx is None:
            return {'crossed': False}
        
        candles_since_cross = len(recent) - 1 - cross_idx
        
        return {
            'crossed': True,
            'cross_idx': cross_idx,
            'candles_ago': candles_since_cross,
            'cross_price': recent['close'].iloc[cross_idx]
        }

    def count_eq_retests_from_discount(self, df, zones, cross_info):
        """
        Count how many times price RETESTED equilibrium from discount side.
        
        THIS IS THE KEY PATTERN!
        
        A retest = price came UP from discount, touched/approached EQ,
                   but got REJECTED back down into discount.
        
        From screenshots: we see 2-3 clear retests before the pump.
        """
        if not cross_info['crossed']:
            return {'retests': 0, 'retest_points': []}
        
        eq = zones['equilibrium']
        cross_idx = cross_info['cross_idx']
        
        # Look at candles AFTER the cross into discount
        after_cross = df.iloc[cross_idx:]
        
        retests = []
        eq_tolerance = eq * 0.015  # Within 1.5% of EQ counts as touch
        
        for i in range(1, len(after_cross)):
            candle = after_cross.iloc[i]
            prev_candle = after_cross.iloc[i-1]
            
            # Check if this candle tested EQ from below
            # Retest = high came near/touched EQ but close stayed below
            if (candle['high'] >= eq - eq_tolerance and 
                candle['close'] < eq):
                
                # Make sure it came from below (previous candle was lower)
                if prev_candle['close'] < eq - eq_tolerance:
                    retests.append({
                        'idx': cross_idx + i,
                        'high': candle['high'],
                        'close': candle['close'],
                        'distance_from_eq': (eq - candle['high']) / eq * 100
                    })
        
        # Remove duplicate retests (consecutive candles touching EQ count as 1)
        clean_retests = []
        if retests:
            clean_retests.append(retests[0])
            for rt in retests[1:]:
                # Only add if it's at least 3 candles away from last retest
                if rt['idx'] - clean_retests[-1]['idx'] >= 3:
                    clean_retests.append(rt)
        
        return {
            'retests': len(clean_retests),
            'retest_points': clean_retests
        }

    def count_eq_retests_from_premium(self, df, zones, cross_info):
        """
        Count retests from premium side (for SHORT setups).
        Opposite logic.
        """
        if not cross_info['crossed']:
            return {'retests': 0, 'retest_points': []}
        
        eq = zones['equilibrium']
        cross_idx = cross_info['cross_idx']
        
        after_cross = df.iloc[cross_idx:]
        retests = []
        eq_tolerance = eq * 0.015
        
        for i in range(1, len(after_cross)):
            candle = after_cross.iloc[i]
            prev_candle = after_cross.iloc[i-1]
            
            # Retest from above = low came near EQ but close stayed above
            if (candle['low'] <= eq + eq_tolerance and 
                candle['close'] > eq):
                
                if prev_candle['close'] > eq + eq_tolerance:
                    retests.append({
                        'idx': cross_idx + i,
                        'low': candle['low'],
                        'close': candle['close'],
                        'distance_from_eq': (candle['low'] - eq) / eq * 100
                    })
        
        clean_retests = []
        if retests:
            clean_retests.append(retests[0])
            for rt in retests[1:]:
                if rt['idx'] - clean_retests[-1]['idx'] >= 3:
                    clean_retests.append(rt)
        
        return {
            'retests': len(clean_retests),
            'retest_points': clean_retests
        }

    def check_holding_in_zone(self, df, zones, cross_info, zone_type='discount'):
        """
        Check if price is HOLDING in the zone (not breaking back through EQ).
        This confirms the setup is still valid.
        """
        if not cross_info['crossed']:
            return False
        
        eq = zones['equilibrium']
        
        # Check last 5-10 candles
        recent = df.tail(10)
        
        if zone_type == 'discount':
            # Should be holding BELOW equilibrium
            closes_below = sum(1 for c in recent['close'] if c < eq)
            return closes_below >= 7  # At least 7 of last 10 candles below EQ
        else:  # premium
            # Should be holding ABOVE equilibrium
            closes_above = sum(1 for c in recent['close'] if c > eq)
            return closes_above >= 7

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

    def analyze_eq_retest(self, data, symbol):
        """
        Full equilibrium retest analysis.
        Detects BOTH long and short setups.
        """
        try:
            if not data or '15m' not in data:
                return None

            df15m = self.add_indicators(data['15m'].copy())
            df1h  = self.add_indicators(data['1h'].copy())

            if len(df15m) < 80:
                return None

            current = df15m['close'].iloc[-1]
            l15m    = df15m.iloc[-1]
            l1h     = df1h.iloc[-1]
            rsi     = float(l15m.get('rsi', 50) or 50)
            rsi_1h  = float(l1h.get('rsi', 50) or 50)
            volr    = float(l15m.get('vol_ratio', 1) or 1)

            # ── [1] Calculate zones ──
            zones = self.calculate_zones(df15m, lookback=150)

            # ── [2] Check for LONG setup (discount) ──
            cross_to_discount = self.detect_eq_cross_to_discount(df15m, zones, lookback=60)
            
            long_setup = None
            if cross_to_discount['crossed']:
                # Count retests from discount
                retests_discount = self.count_eq_retests_from_discount(
                    df15m, zones, cross_to_discount)
                
                # Check if holding in discount
                holding_discount = self.check_holding_in_zone(
                    df15m, zones, cross_to_discount, 'discount')
                
                if retests_discount['retests'] >= 2 and holding_discount:
                    long_setup = {
                        'signal': 'LONG',
                        'cross': cross_to_discount,
                        'retests': retests_discount,
                        'holding': holding_discount
                    }

            # ── [3] Check for SHORT setup (premium) ──
            cross_to_premium = self.detect_eq_cross_to_premium(df15m, zones, lookback=60)
            
            short_setup = None
            if cross_to_premium['crossed']:
                retests_premium = self.count_eq_retests_from_premium(
                    df15m, zones, cross_to_premium)
                
                holding_premium = self.check_holding_in_zone(
                    df15m, zones, cross_to_premium, 'premium')
                
                if retests_premium['retests'] >= 2 and holding_premium:
                    short_setup = {
                        'signal': 'SHORT',
                        'cross': cross_to_premium,
                        'retests': retests_premium,
                        'holding': holding_premium
                    }

            # Pick the better setup (or both if both exist)
            setup = None
            if long_setup and not short_setup:
                setup = long_setup
            elif short_setup and not long_setup:
                setup = short_setup
            elif long_setup and short_setup:
                # Both exist - pick the one with more retests
                if long_setup['retests']['retests'] >= short_setup['retests']['retests']:
                    setup = long_setup
                else:
                    setup = short_setup

            if not setup:
                return None

            # ── SCORING ──
            score    = 0
            reasons  = []
            warnings = []
            signal   = setup['signal']

            # [A] Number of retests (0-40 pts) - MOST IMPORTANT
            num_retests = setup['retests']['retests']
            if num_retests >= 4:
                score += 40
                reasons.append(f"🎯 {num_retests} EQ RETESTS - Perfect setup!")
            elif num_retests == 3:
                score += 35
                reasons.append(f"🎯 {num_retests} EQ retests - Strong!")
            elif num_retests == 2:
                score += 25
                reasons.append(f"🎯 {num_retests} EQ retests - Good")

            # [B] Holding in zone (0-25 pts)
            if setup['holding']:
                score += 25
                zone_name = "DISCOUNT" if signal == 'LONG' else "PREMIUM"
                reasons.append(f"📍 HOLDING in {zone_name} zone")

            # [C] Time since cross (0-20 pts) - prefer recent crosses
            candles_ago = setup['cross']['candles_ago']
            if candles_ago <= 15:  # Within ~4 hours
                score += 20
                reasons.append(f"⏱️ Fresh cross ({candles_ago}x15min ago)")
            elif candles_ago <= 25:
                score += 12
                reasons.append(f"⏱️ Recent cross ({candles_ago}x15min ago)")
            else:
                score += 5
                warnings.append(f"⚠️ Cross was {candles_ago}x15min ago - getting old")

            # [D] RSI alignment (0-15 pts)
            if signal == 'LONG':
                if rsi < 35:
                    score += 15
                    reasons.append(f"💎 RSI oversold ({rsi:.0f}) - bounce ready")
                elif rsi < 45:
                    score += 10
                    reasons.append(f"💎 RSI low ({rsi:.0f})")
                elif rsi > 60:
                    warnings.append(f"⚠️ RSI high ({rsi:.0f}) for long")
                    score -= 10
            else:  # SHORT
                if rsi > 65:
                    score += 15
                    reasons.append(f"💎 RSI overbought ({rsi:.0f}) - drop ready")
                elif rsi > 55:
                    score += 10
                    reasons.append(f"💎 RSI high ({rsi:.0f})")
                elif rsi < 40:
                    warnings.append(f"⚠️ RSI low ({rsi:.0f}) for short")
                    score -= 10

            # [E] Volume (0-10 pts)
            if volr > 1.5:
                score += 10
                reasons.append(f"📊 Strong volume ({volr:.1f}x)")
            elif volr > 1.0:
                score += 5

            # [F] 1H RSI context (0-10 pts)
            if signal == 'LONG' and rsi_1h < 40:
                score += 10
                reasons.append(f"💎 1H RSI oversold ({rsi_1h:.0f})")
            elif signal == 'SHORT' and rsi_1h > 60:
                score += 10
                reasons.append(f"💎 1H RSI overbought ({rsi_1h:.0f})")

            # [G] Zone position (0-10 pts)
            if signal == 'LONG':
                # Deeper in discount = better
                dist_pct = (zones['equilibrium'] - current) / zones['equilibrium'] * 100
                if dist_pct > 8:
                    score += 10
                    reasons.append(f"🔵 Deep in discount ({dist_pct:.1f}% below EQ)")
                elif dist_pct > 3:
                    score += 5
            else:  # SHORT
                dist_pct = (current - zones['equilibrium']) / zones['equilibrium'] * 100
                if dist_pct > 8:
                    score += 10
                    reasons.append(f"🔴 Deep in premium ({dist_pct:.1f}% above EQ)")
                elif dist_pct > 3:
                    score += 5

            if score < self.min_score_threshold:
                return None

            # ── TRADE LEVELS ──
            entry = current
            eq    = zones['equilibrium']

            if signal == 'LONG':
                # SL below discount zone
                sl = zones['swing_low'] * 0.985
                # TPs: EQ → mid to high → high
                tp1 = eq
                tp2 = (eq + zones['swing_high']) / 2
                tp3 = zones['swing_high'] * 0.99
            else:  # SHORT
                # SL above premium zone
                sl = zones['swing_high'] * 1.015
                # TPs: EQ → mid to low → low
                tp1 = eq
                tp2 = (eq + zones['swing_low']) / 2
                tp3 = zones['swing_low'] * 1.01

            risk_pct = abs(entry - sl) / entry * 100
            if risk_pct > 10:
                if signal == 'LONG':
                    sl = entry * 0.94
                else:
                    sl = entry * 1.06
                risk_pct = 6.0

            rr   = [abs(t - entry) / abs(sl - entry) for t in [tp1, tp2, tp3]]
            pcts = [(t - entry) / entry * 100          for t in [tp1, tp2, tp3]]
            # Fix signs for SHORT
            if signal == 'SHORT':
                pcts = [-p for p in pcts]

            if   score >= 90: conf = 'ELITE 🔥🔥🔥'
            elif score >= 80: conf = 'HIGH 💎💎'
            elif score >= 70: conf = 'GOOD 💎'
            else:             conf = 'WATCH ✅'

            return {
                'success': True,
                'symbol': symbol.replace('/USDT:USDT', ''),
                'full_symbol': symbol,
                'signal': signal,
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
                'setup': setup,
                'rsi': rsi,
                'vol_ratio': volr,
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
        s  = r['setup']
        
        msg  = f"{'═'*46}\n"
        msg += f"💎 <b>{rk}EQ RETEST: {r['symbol']} — {r['confidence']}</b> 💎\n"
        msg += f"{'═'*46}\n\n"

        msg += f"<b>{r['signal']}</b>  Score: {r['score']:.0f}/100\n"
        msg += f"RSI: {r['rsi']:.0f}\n\n"

        # Zones
        msg += f"<b>🗺️ ZONES (15m):</b>\n"
        msg += f"  🔴 Premium:  ${z['premium_start']:.6f}+\n"
        msg += f"  ⚖️ EQ:       ${z['equilibrium']:.6f}\n"
        msg += f"  🔵 Discount: <${z['discount_end']:.6f}\n"
        msg += f"  📍 Current:  ${r['entry']:.6f}\n\n"

        # Pattern details
        msg += f"<b>🎯 PATTERN:</b>\n"
        msg += f"  • Crossed to {'DISCOUNT' if r['signal']=='LONG' else 'PREMIUM'} {s['cross']['candles_ago']}x15min ago\n"
        msg += f"  • <b>{s['retests']['retests']} EQ RETESTS</b> ✅\n"
        msg += f"  • Holding in zone ✅\n\n"

        msg += f"<b>💰 TRADE:</b>\n"
        msg += f"  Entry: ${r['entry']:.6f}\n"
        msg += f"  SL:    ${r['stop_loss']:.6f}  ({r['risk_percent']:.1f}%)\n\n"

        msg += f"<b>🎯 TARGETS:</b>\n"
        labels = ['EQ', 'Mid', 'Extreme']
        for i, (tp, rr, pct, lbl) in enumerate(
                zip(r['targets'], r['reward_ratios'], r['target_pcts'], labels), 1):
            msg += f"  TP{i}: ${tp:.6f}  ({pct:+.1f}%  {rr:.1f}R)  [{lbl}]\n"

        msg += f"\n<b>✅ REASONS:</b>\n"
        for rs in r['reasons'][:6]:
            msg += f"  • {rs}\n"

        if r['warnings']:
            msg += f"\n<b>⚠️ WARNINGS:</b>\n"
            for w in r['warnings'][:3]:
                msg += f"  {w}\n"

        msg += f"\n<i>⏰ {r['timestamp'].strftime('%H:%M')}</i>"
        msg += f"\n<i>💎 Cross → {s['retests']['retests']} Retests → Hold → 🚀</i>"
        msg += f"\n{'═'*46}"
        return msg

    def should_alert(self, symbol, result):
        if result['score'] < self.min_score_threshold:
            return False
        if symbol in self.alerted_pairs:
            last = self.alerted_pairs[symbol]
            # 3 hour cooldown
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

        logger.info(f"💎 EQ RETEST SCAN: {len(self.pairs_to_scan)} pairs")
        await self.send_msg(
            f"🔍 <b>EQ RETEST SCAN</b>\n\n"
            f"Scanning {len(self.pairs_to_scan)} pairs\n"
            f"Pattern: Cross → Retests → Hold → 🚀"
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
                result = self.analyze_eq_retest(data, sym)
                if result and result['success']:
                    results.append(result)
                    logger.info(f"💎 {pair} {result['signal']}  score={result['score']:.0f}")
                    if self.should_alert(result['full_symbol'], result) and alerts < self.max_alerts_per_scan:
                        alerts += 1
                        await self.send_msg(self.format_alert(result, rank=alerts))
                        tid = f"{result['symbol']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                        self.active_trades[tid] = {
                            'trade_id': tid, 'symbol': result['symbol'],
                            'full_symbol': result['full_symbol'], 'signal': result['signal'],
                            'entry': result['entry'], 'stop_loss': result['stop_loss'],
                            'targets': result['targets'], 'reward_ratios': result['reward_ratios'],
                            'timestamp': datetime.now(),
                            'tp_hit': [False, False, False], 'sl_hit': False,
                        }
                        self.alerted_pairs[result['full_symbol']] = {
                            'time': datetime.now(), 'score': result['score']
                        }
                        self.stats['signals_found'] += 1
                await asyncio.sleep(0.07)
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
        logger.info(f"EQ scan every {self.scan_interval//60}m")
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
                        
                        # Check TPs
                        for i, (tp, hit) in enumerate(zip(trade['targets'], trade['tp_hit'])):
                            if not hit:
                                hit_condition = (price >= tp if trade['signal'] == 'LONG' else price <= tp)
                                if hit_condition:
                                    pnl = abs((price - trade['entry']) / trade['entry'] * 100)
                                    msg = f"🎯 TP{i+1}!\n{trade['symbol']} {trade['signal']}\n+{pnl:.1f}% ({trade['reward_ratios'][i]:.1f}R)"
                                    await self.send_msg(msg)
                                    trade['tp_hit'][i] = True
                                    self.stats[f'tp{i+1}_hits'] += 1
                                    if i == 2: to_remove.append(tid)
                        
                        # Check SL
                        if not trade['sl_hit']:
                            sl_condition = (price <= trade['stop_loss'] if trade['signal'] == 'LONG' 
                                          else price >= trade['stop_loss'])
                            if sl_condition:
                                loss = abs((trade['stop_loss'] - trade['entry']) / trade['entry'] * 100)
                                await self.send_msg(f"🛑 SL\n{trade['symbol']} {trade['signal']}\n-{loss:.1f}%")
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
        msg  = "💎 <b>EQUILIBRIUM RETEST SCANNER</b>\n\n"
        msg += "<b>The EXACT pattern from @free_fx_pro</b>\n\n"
        msg += "Pattern:\n"
        msg += "1️⃣ Cross into zone\n"
        msg += "2️⃣ 2-3+ EQ retests\n"
        msg += "3️⃣ Hold in zone\n"
        msg += "4️⃣ 🚀 PUMP!\n\n"
        msg += "15min charts • Scan every 5min\n\n"
        msg += "/start_scan\n/start_tracking\n/status\n/stats"
        await u.message.reply_text(msg, parse_mode=ParseMode.HTML)

    async def cmd_start_scan(self, u, c):
        if self.s.is_scanning:
            await u.message.reply_text("⚠️ Running!", parse_mode=ParseMode.HTML)
            return
        self.s.is_scanning = True
        asyncio.create_task(self.s.auto_scan_loop())
        await u.message.reply_text(
            f"✅ <b>EQ SCANNER ON!</b>\n\nEvery {self.s.scan_interval//60}min\nScanning...",
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
        if self.s.last_scan_time:
            mins = int((datetime.now() - self.s.last_scan_time).total_seconds() // 60)
            msg += f"\nLast: {mins}m ago"
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

    scanner = EquilibriumRetestScanner(TOKEN, CHAT)
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
    logger.info("💎 EQ RETEST SCANNER ONLINE!")

    welcome  = "💎 <b>EQ RETEST SCANNER READY!</b>\n\n"
    welcome += "<b>Pattern from @free_fx_pro:</b>\n"
    welcome += "Cross → 2-3 Retests → Hold → 🚀\n\n"
    welcome += "✅ 15min charts\n✅ Scan every 5min\n"
    welcome += "✅ Both LONG & SHORT\n✅ Fast alerts\n\n"
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
