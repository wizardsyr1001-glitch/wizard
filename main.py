"""
OB Scanner Bot v5 — LuxAlgo Order Block Strategy
Validated: 90.0% WR | SHORT-only | age ≤3h or ≥9h | 200 pairs
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import ta
import logging
from telegram import Bot
from telegram.constants import ParseMode

# ============================================================
# CREDENTIALS
# ============================================================
TELEGRAM_TOKEN   = "8034062612:AAEJYbPA8sMODYvqvt8U-5mM7c3Y3-GOYtM"
TELEGRAM_CHAT_ID = "7500072234"

# ============================================================
# CONFIG — validated backtest parameters, do not change
# ============================================================
TOP_N_PAIRS     = 200
MIN_VOLUME_USDT = 1_000_000
SCAN_INTERVAL   = 3600       # 1H candle close
COOLDOWN_HOURS  = 24

OB_LENGTH       = 5
OB_MAX_AGE      = 12         # outer limit — inner dead zone filtered separately
OB_AGE_MIN_A    = 1          # FRESH window: age 2-3h (100% WR in backtest)
OB_AGE_MAX_A    = 3
OB_AGE_MIN_B    = 9          # MATURE window: age 9-12h (76%+ WR, survived mitigation)
OB_AGE_MAX_B    = 12         # age 4-8h = dead zone (42% WR) — skip entirely
MITIGATION      = 'Wick'

STRUCT_BARS     = 120
DISCOUNT_MAX    = 40
PREMIUM_MIN     = 60

SL_BUFFER_MULT  = 0.2
TP1_RR          = 1.5
TP2_RR          = 3.0
TP3_RR          = 5.0
TRADE_MONITOR_BARS = 48      # watch active trades for 48h
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('ob_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

TIER_COLORS = {'PREMIUM': '🔴', 'HIGH': '🟡', 'GOOD': '🟢'}


class OBScanner:
    def __init__(self):
        # Bybit — no geo-blocking, USDT perpetual futures, ccxt-compatible
        self.exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # USDT perpetual = linear on Bybit
                'adjustForTimeDifference': True,
            }
        })
        self.bot       = Bot(token=TELEGRAM_TOKEN)
        self.cooldowns = {}   # symbol → datetime of last signal
        self.scan_count = 0

        # Active trade tracking for TP/SL alerts
        # key: symbol, value: dict with signal data + tp1_hit flag
        self.active_trades = {}

    # ─── DATA ───────────────────────────────────────────────

    async def get_top_pairs(self):
        try:
            await self.exchange.load_markets()
            tickers = await self.exchange.fetch_tickers()
            pairs = []
            for symbol, market in self.exchange.markets.items():
                # Bybit linear: USDT perpetual swaps only
                if not market.get('active', False):
                    continue
                if market.get('quote') != 'USDT':
                    continue
                if market.get('type') != 'swap':
                    continue
                if not market.get('linear', True):
                    continue
                ticker = tickers.get(symbol, {}) or {}
                vol = ticker.get('quoteVolume', 0) or 0
                if vol > MIN_VOLUME_USDT:
                    pairs.append((symbol, vol))
            pairs.sort(key=lambda x: x[1], reverse=True)
            selected = [p[0] for p in pairs[:TOP_N_PAIRS]]
            logger.info(f"Found {len(selected)} USDT perp pairs on Bybit")
            return selected
        except Exception as e:
            logger.error(f"get_top_pairs: {e}")
            return []

    async def fetch_ohlcv(self, symbol, bars=300):
        try:
            data = await self.exchange.fetch_ohlcv(symbol, '1h', limit=bars)
            if not data or len(data) < 100:
                return None
            df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.drop_duplicates('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception:
            return None

    def add_indicators(self, df):
        try:
            df = df.copy()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            df['hl2'] = (df['high'] + df['low']) / 2
            return df
        except Exception:
            return None

    # ─── LUXALGO OB DETECTION ───────────────────────────────

    def detect_obs(self, df):
        n, L = len(df), OB_LENGTH
        obs, os = [], None
        for i in range(L, n - L):
            vol_slice = df['volume'].iloc[i-L : i+L+1]
            is_vpivot = len(vol_slice) == 2*L+1 and df['volume'].iloc[i] == vol_slice.max()
            upper = df['high'].iloc[i-L+1 : i+1].max()
            lower = df['low'].iloc[i-L+1  : i+1].min()
            if df['high'].iloc[i-L] > upper: os = 0
            elif df['low'].iloc[i-L] < lower: os = 1
            if is_vpivot and os is not None:
                ob_bar = i - L
                if os == 1:
                    obs.append({'type':'BULL_OB','ob_top':float(df['hl2'].iloc[ob_bar]),
                                'ob_btm':float(df['low'].iloc[ob_bar]),
                                'ob_avg':float((df['hl2'].iloc[ob_bar]+df['low'].iloc[ob_bar])/2),
                                'formed_at':ob_bar,'valid':True})
                else:
                    obs.append({'type':'BEAR_OB','ob_top':float(df['high'].iloc[ob_bar]),
                                'ob_btm':float(df['hl2'].iloc[ob_bar]),
                                'ob_avg':float((df['high'].iloc[ob_bar]+df['hl2'].iloc[ob_bar])/2),
                                'formed_at':ob_bar,'valid':True})
        return obs

    def apply_mitigation(self, obs, df):
        L, n = OB_LENGTH, len(df)
        for ob in obs:
            if not ob['valid']: continue
            for i in range(ob['formed_at']+1, n):
                start = max(0, i-L+1)
                t_bull = df['low'].iloc[start:i+1].min()
                t_bear = df['high'].iloc[start:i+1].max()
                if ob['type']=='BULL_OB' and t_bull < ob['ob_btm']:
                    ob['valid']=False; break
                elif ob['type']=='BEAR_OB' and t_bear > ob['ob_top']:
                    ob['valid']=False; break
        return obs

    # ─── MARKET STRUCTURE ───────────────────────────────────

    def find_swings(self, df, lb=5):
        n=len(df); sh=[False]*n; sl=[False]*n
        for i in range(lb, n-lb):
            if df['high'].iloc[i]==df['high'].iloc[i-lb:i+lb+1].max(): sh[i]=True
            if df['low'].iloc[i]==df['low'].iloc[i-lb:i+lb+1].min():  sl[i]=True
        return sh, sl

    def get_structure(self, df, sh, sl):
        i=len(df)-1; start=max(0,i-STRUCT_BARS)
        sh_bars=[j for j in range(start,i) if sh[j]]
        sl_bars=[j for j in range(start,i) if sl[j]]
        if len(sh_bars)<2 or len(sl_bars)<2: return 'NEUTRAL',None,None
        sh1,sh2=df['high'].iloc[sh_bars[-1]],df['high'].iloc[sh_bars[-2]]
        sl1,sl2=df['low'].iloc[sl_bars[-1]],df['low'].iloc[sl_bars[-2]]
        if sh1>sh2 and sl1>sl2: return 'BULLISH',sh1,sl1
        if sh1<sh2 and sl1<sl2: return 'BEARISH',sh1,sl1
        return 'NEUTRAL',sh1,sl1

    def get_pd(self, last_sh, last_sl, price):
        if last_sh is None or last_sl is None: return 'NEUTRAL',50.0
        rng=last_sh-last_sl
        if rng<=0: return 'NEUTRAL',50.0
        pos=max(0.0,min(100.0,(price-last_sl)/rng*100))
        if pos<=DISCOUNT_MAX: return 'DISCOUNT',round(pos,1)
        if pos>=PREMIUM_MIN:  return 'PREMIUM', round(pos,1)
        return 'EQUILIBRIUM',round(pos,1)

    # ─── SIGNAL CHECK ───────────────────────────────────────

    def check_signal(self, df, obs, structure, pd_level, last_sh, last_sl):
        n=len(df); i=n-1
        cur=df.iloc[i]; prev=df.iloc[i-1]
        if pd.isna(cur['atr']) or cur['atr']==0: return None
        atr=cur['atr']

        for ob in obs:
            if not ob['valid']: continue
            age=i-ob['formed_at']
            # Age window filter: fresh (2-3h) OR mature (9-12h) — skip 4-8h dead zone
            in_fresh  = OB_AGE_MIN_A < age <= OB_AGE_MAX_A
            in_mature = OB_AGE_MIN_B <= age <= OB_AGE_MAX_B
            if not (in_fresh or in_mature): continue

            # SHORT ONLY — LONG disabled (bear market: LONG WR consistently 57-60%)
            if ob['type']=='BEAR_OB' and structure=='BEARISH' and pd_level=='PREMIUM':
                touched   = prev['high']>=ob['ob_btm'] and prev['close']<=ob['ob_top']*1.005
                confirmed = cur['close']<cur['open'] and cur['close']<ob['ob_avg'] and cur['close']<prev['close']
                if touched and confirmed:
                    entry=cur['close']; sl=ob['ob_top']+atr*SL_BUFFER_MULT
                    risk=max(sl-entry, atr*0.05)
                    tier='PREMIUM' if age<=6 else 'HIGH' if age<=9 else 'GOOD'
                    return {'direction':'SHORT','entry':entry,'sl':round(sl,8),
                            'tp1':round(entry-risk*TP1_RR,8),'tp2':round(entry-risk*TP2_RR,8),
                            'tp3':round(entry-risk*TP3_RR,8),'risk_pct':round(risk/entry*100,2),
                            'ob_type':ob['type'],'ob_top':round(ob['ob_top'],8),'ob_btm':round(ob['ob_btm'],8),
                            'ob_age':age,'structure':structure,'pd_level':pd_level,
                            'pd_pct':self.get_pd(last_sh,last_sl,entry)[1],'atr':round(atr,8),'tier':tier}
        return None

    # ─── TP/SL MONITORING ───────────────────────────────────

    async def monitor_active_trades(self):
        """
        Check all active trades against current price.
        Rules:
          - TP1 hit → send TP1 alert, mark tp1_hit=True
          - TP2 hit → send TP2 alert
          - TP3 hit → send TP3 alert, close trade
          - SL hit  → only alert if tp1_hit is False (protected trade = no SL alert)
        """
        to_remove = []

        for symbol, trade in list(self.active_trades.items()):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                price  = ticker['last']
                if not price: continue

                d         = trade['direction']
                tp1_hit   = trade.get('tp1_hit', False)
                tp2_hit   = trade.get('tp2_hit', False)
                bars_open = trade.get('bars_open', 0) + 1
                trade['bars_open'] = bars_open

                pair = symbol.replace('/USDT:USDT','')

                def fmt(p):
                    if p>1000: return f"{p:,.2f}"
                    elif p>1:  return f"{p:.4f}"
                    elif p>0.01: return f"{p:.6f}"
                    else: return f"{p:.8f}"

                if d == 'LONG':
                    # TP checks (in order)
                    if not tp1_hit and price >= trade['tp1']:
                        trade['tp1_hit'] = True
                        await self._send(
                            f"✅ *TP1 HIT* 🎯\n"
                            f"*{pair}* LONG\n"
                            f"Price: `{fmt(price)}` → TP1: `{fmt(trade['tp1'])}`\n"
                            f"🛡️ SL moved to breakeven — protected"
                        )
                    elif tp1_hit and not tp2_hit and price >= trade['tp2']:
                        trade['tp2_hit'] = True
                        await self._send(
                            f"✅✅ *TP2 HIT* 🎯🎯\n"
                            f"*{pair}* LONG\n"
                            f"Price: `{fmt(price)}` → TP2: `{fmt(trade['tp2'])}`\n"
                            f"🔥 Riding to TP3: `{fmt(trade['tp3'])}`"
                        )
                    elif tp2_hit and price >= trade['tp3']:
                        await self._send(
                            f"🏆 *TP3 HIT — FULL TARGET* 🏆\n"
                            f"*{pair}* LONG\n"
                            f"Price: `{fmt(price)}` → TP3: `{fmt(trade['tp3'])}`\n"
                            f"Entry: `{fmt(trade['entry'])}` | R:R = 5.0"
                        )
                        to_remove.append(symbol)
                    # SL check — ONLY alert if TP1 was never hit
                    elif not tp1_hit and price <= trade['sl']:
                        await self._send(
                            f"🛑 *STOPPED OUT*\n"
                            f"*{pair}* LONG\n"
                            f"Price: `{fmt(price)}` hit SL: `{fmt(trade['sl'])}`\n"
                            f"Loss: -{trade['risk_pct']:.2f}%"
                        )
                        to_remove.append(symbol)
                    # TP1 hit but price came back to entry (protected, no alert)
                    elif tp1_hit and not tp2_hit and price <= trade['entry']:
                        logger.info(f"{pair} LONG: TP1 hit, price returned to entry (protected, no alert)")
                        to_remove.append(symbol)

                else:  # SHORT
                    if not tp1_hit and price <= trade['tp1']:
                        trade['tp1_hit'] = True
                        await self._send(
                            f"✅ *TP1 HIT* 🎯\n"
                            f"*{pair}* SHORT\n"
                            f"Price: `{fmt(price)}` → TP1: `{fmt(trade['tp1'])}`\n"
                            f"🛡️ SL moved to breakeven — protected"
                        )
                    elif tp1_hit and not tp2_hit and price <= trade['tp2']:
                        trade['tp2_hit'] = True
                        await self._send(
                            f"✅✅ *TP2 HIT* 🎯🎯\n"
                            f"*{pair}* SHORT\n"
                            f"Price: `{fmt(price)}` → TP2: `{fmt(trade['tp2'])}`\n"
                            f"🔥 Riding to TP3: `{fmt(trade['tp3'])}`"
                        )
                    elif tp2_hit and price <= trade['tp3']:
                        await self._send(
                            f"🏆 *TP3 HIT — FULL TARGET* 🏆\n"
                            f"*{pair}* SHORT\n"
                            f"Price: `{fmt(price)}` → TP3: `{fmt(trade['tp3'])}`\n"
                            f"Entry: `{fmt(trade['entry'])}` | R:R = 5.0"
                        )
                        to_remove.append(symbol)
                    elif not tp1_hit and price >= trade['sl']:
                        await self._send(
                            f"🛑 *STOPPED OUT*\n"
                            f"*{pair}* SHORT\n"
                            f"Price: `{fmt(price)}` hit SL: `{fmt(trade['sl'])}`\n"
                            f"Loss: -{trade['risk_pct']:.2f}%"
                        )
                        to_remove.append(symbol)
                    elif tp1_hit and not tp2_hit and price >= trade['entry']:
                        logger.info(f"{pair} SHORT: TP1 hit, price returned to entry (protected, no alert)")
                        to_remove.append(symbol)

                # Auto-expire after TRADE_MONITOR_BARS hours
                if bars_open >= TRADE_MONITOR_BARS and symbol not in to_remove:
                    to_remove.append(symbol)
                    logger.info(f"{pair}: trade expired after {bars_open}h monitoring")

            except Exception as e:
                logger.error(f"monitor error {symbol}: {e}")

        for sym in to_remove:
            self.active_trades.pop(sym, None)

    # ─── TELEGRAM HELPERS ───────────────────────────────────

    async def _send(self, text):
        """Send a plain signal/alert message — no status noise"""
        try:
            await self.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=text,
                parse_mode=ParseMode.MARKDOWN
            )
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def format_signal(self, symbol, signal):
        pair  = symbol.replace('/USDT:USDT','')
        d     = signal['direction']
        tier  = signal['tier']
        emoji = TIER_COLORS.get(tier,'🟢')
        arrow = '📈' if d=='LONG' else '📉'
        struct= '🏗 Bullish' if signal['structure']=='BULLISH' else '🏗 Bearish'
        pd_str= f"{'🔻 Discount' if signal['pd_level']=='DISCOUNT' else '🔺 Premium'} {signal['pd_pct']}%"

        def fmt(p):
            if p>1000: return f"{p:,.2f}"
            elif p>1:  return f"{p:.4f}"
            elif p>0.01: return f"{p:.6f}"
            else: return f"{p:.8f}"

        return (
            f"{emoji} *{tier}* {arrow} *{pair}* · {d}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📦 {signal['ob_type']} · {signal['ob_age']}h old\n"
            f"Zone `{fmt(signal['ob_btm'])}` — `{fmt(signal['ob_top'])}`\n\n"
            f"🎯 Entry  `{fmt(signal['entry'])}`\n"
            f"🛑 SL     `{fmt(signal['sl'])}` ({signal['risk_pct']:.2f}%)\n"
            f"✅ TP1    `{fmt(signal['tp1'])}` (1.5R)\n"
            f"✅ TP2    `{fmt(signal['tp2'])}` (3.0R)\n"
            f"✅ TP3    `{fmt(signal['tp3'])}` (5.0R)\n\n"
            f"📊 {struct} · {pd_str}\n"
            f"⏰ {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%H:%M UTC')}"
        )

    # ─── SCAN ONE PAIR ──────────────────────────────────────

    async def scan_pair(self, symbol):
        try:
            last = self.cooldowns.get(symbol)
            if last and (datetime.now(timezone.utc).replace(tzinfo=None)-last).total_seconds() < COOLDOWN_HOURS*3600:
                return None

            df = await self.fetch_ohlcv(symbol, bars=300)
            if df is None: return None

            df = self.add_indicators(df)
            if df is None: return None
            df.dropna(subset=['atr','hl2'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            if len(df) < OB_LENGTH*2+10: return None

            obs = self.detect_obs(df)
            obs = self.apply_mitigation(obs, df)
            sh, sl = self.find_swings(df)

            structure, last_sh, last_sl = self.get_structure(df, sh, sl)
            if structure=='NEUTRAL': return None

            pd_level, _ = self.get_pd(last_sh, last_sl, df.iloc[-1]['close'])
            if pd_level in ('EQUILIBRIUM','NEUTRAL'): return None

            return self.check_signal(df, obs, structure, pd_level, last_sh, last_sl)

        except Exception as e:
            logger.error(f"scan_pair {symbol}: {e}")
            return None

    # ─── MAIN SCAN LOOP ─────────────────────────────────────

    async def run_scan(self):
        self.scan_count += 1
        logger.info(f"Scan #{self.scan_count} | {datetime.now(timezone.utc).replace(tzinfo=None).strftime('%H:%M UTC')}")

        # Monitor active trades first
        if self.active_trades:
            await self.monitor_active_trades()

        pairs   = await self.get_top_pairs()
        signals = []

        for i, symbol in enumerate(pairs, 1):
            signal = await self.scan_pair(symbol)
            if signal:
                signals.append((symbol, signal))
                self.cooldowns[symbol] = datetime.now(timezone.utc).replace(tzinfo=None)
                # Register for TP/SL monitoring
                self.active_trades[symbol] = {**signal, 'tp1_hit':False, 'tp2_hit':False, 'bars_open':0}
                logger.info(f"Signal: {symbol} {signal['direction']} {signal['tier']}")
            if i % 25 == 0:
                logger.info(f"  {i}/{len(pairs)} pairs scanned | {len(signals)} signals")
            await asyncio.sleep(0.15)

        logger.info(f"Scan #{self.scan_count} done: {len(signals)} signals")

        # Send signal cards — PREMIUM first
        order = {'PREMIUM':0,'HIGH':1,'GOOD':2}
        signals.sort(key=lambda x: order.get(x[1]['tier'],9))

        for symbol, signal in signals:
            msg = self.format_signal(symbol, signal)
            await self._send(msg)

    async def run(self):
        logger.info("OB Scanner Bot v4 started")
        logger.info(f"Pairs: {TOP_N_PAIRS} | OB_LENGTH: {OB_LENGTH} | Max Age: {OB_MAX_AGE}h | Exchange: Bybit")
        logger.info("TP1 hit = SL alert suppressed (protected trade)")

        while True:
            try:
                await self.run_scan()
            except Exception as e:
                logger.error(f"Scan loop error: {e}")
            logger.info(f"Next scan in {SCAN_INTERVAL//60}m")
            await asyncio.sleep(SCAN_INTERVAL)

    async def close(self):
        await self.exchange.close()


async def main():
    scanner = OBScanner()
    try:
        await scanner.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    finally:
        await scanner.close()


if __name__ == "__main__":
    asyncio.run(main())
