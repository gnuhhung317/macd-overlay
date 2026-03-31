import sys, os
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
sys.path.append(str(BASE_DIR))

from ml.backtest_sniper import BacktestConfig, run_backtest_with_config

def verify_live_trades():
    print("Running Backtest on Bitget Data (2026-03-14 -> 2026-03-22)...")
    config = BacktestConfig(
        start_date='2026-03-14',
        end_date='2026-03-22',
        exchange='bitget',
        initial_capital=100.0,
        risk_per_trade=0.04
    )
    
    potential_signals, price_db, trades, equity_curve = run_backtest_with_config(config)
    
    # Let's list the expected trades from the user
    user_trades = [
        ("KMNOUSDT", "2026-03-21 18:36:23"),
        ("AKTUSDT", "2026-03-21 18:01:46"),
        ("AKTUSDT", "2026-03-21 16:25:30"),
        ("AKTUSDT", "2026-03-20 15:01:53"),
        ("SAHARAUSDT", "2026-03-20 08:02:24"),
        ("EIGENUSDT", "2026-03-20 13:07:20"),
        ("MASKUSDT", "2026-03-20 09:04:30"),
        ("CFXUSDT", "2026-03-20 07:16:11"),
        ("AXSUSDT", "2026-03-20 08:02:22"),
        ("CYSUSDT", "2026-03-19 01:01:47"),
        ("HOLOUSDT", "2026-03-19 16:01:56"),
        ("OLUSDT", "2026-03-19 08:06:09"),
        ("AINUSDT", "2026-03-19 08:01:42"),
        ("AKTUSDT", "2026-03-19 09:03:16"),
        ("UAIUSDT", "2026-03-19 16:01:53"),
        ("HYPEUSDT", "2026-03-19 01:01:46"),
        ("BICOUSDT", "2026-03-17 17:05:16"),
        ("IRUSDT", "2026-03-17 11:02:11"),
        ("XANUSDT", "2026-03-15 22:08:11"),
        ("TAOUSDT", "2026-03-16 17:02:51"),
    ]
    
    print("\n--- User Trades vs Backtest Signals ---")
    
    # Filter backtest potential signals for these symbols
    # We want to see if the backtest generated a signal at the exact hour for these coins
    for sym, live_time_str in user_trades:
        live_dt = datetime.strptime(live_time_str, "%Y-%m-%d %H:%M:%S")
        
        # In Sniper bot, a trade executed at 18:36:23 was likely triggered by the 18:00:00 or 17:00:00 candle signal
        # and entered via limit order. So the signal time would be live_time floored to hour, or live_time - 1 hour floored.
        # But wait, limit_order matches the previous candle's close + atr offset. 
        # A limit order filled during the 18:00 candle hour means the signal was generated AT 17:00 or 18:00 (since we wait 2 bars max).
        
        # Let's search all generated potentials for this symbol around this date
        sym_signals = [sig for sig in potential_signals if sig['symbol'] == sym]
        
        # Find matching signal around this time (±24h for leeway)
        found = []
        for sig in sym_signals:
            if sig['type'] == 'SHORT':
                time_diff = abs((sig['timestamp'] - live_dt).total_seconds()) / 3600.0
                if time_diff <= 24:
                    found.append(sig)
                    
        if found:
            for s in found:
                # check if it turned into a filled trade in the backtest
                filled = [t for t in trades if t.symbol == sym and t.signal_time == s['timestamp']]
                status = f"FILLED ({filled[0].entry_time})" if filled else "SIGNAL ONLY (NOT FILLED IN BT)"
                print(f"[MATCH] {sym} | Live Fill: {live_time_str} | BT Signal: {s['timestamp']} ({s['prob']:.4f}) -> {status}")
        else:
            print(f"[MISSING] {sym} at {live_time_str} -> Backtest had NO SHORT signals nearby!")

if __name__ == '__main__':
    verify_live_trades()
