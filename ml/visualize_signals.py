import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from backtest_sniper import BacktestConfig, load_assets, backtest_symbol
import random
# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
OUTPUT_DIR = BASE_DIR / "ml" / "signal_charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_charts(threshold=0.95, max_charts=30):
    print(f"🚀 Analyzing signals with confidence > {threshold}...")
    
    clf, features, _ = load_assets(override_threshold=threshold)
    config = BacktestConfig(
        start_date="2025-01-01",
        end_date="2025-06-01",
        exchange='binance',
        threshold=threshold
    )
    
    symbols_dir = BASE_DIR / "data" / "processed" / "symbols_v3"
    symbol_files = list(symbols_dir.glob("*.parquet"))
    random.shuffle(symbol_files)
    
    stats = {
        'TP_HIT': 0,
        'SL_HIT': 0,
        'TIMEOUT_WIN': 0,
        'TIMEOUT_LOSS': 0,
        'TOTAL': 0
    }
    
    charts_generated = 0
    
    for file_path in symbol_files:
        symbol = file_path.stem.replace("USDT", "")
        signals, df = backtest_symbol(file_path, features, clf, threshold, config)
        
        if not signals or len(signals) == 0:
            continue
            
        for sig in signals:
            stats['TOTAL'] += 1
            ts = sig['timestamp']
            entry_price = sig['close']
            atr = sig['atr_val']
            
            sl_price = entry_price - (1.5 * atr)
            tp_price = entry_price + (3.0 * atr)
            
            # Simulate Trade
            idx = df[df['timestamp'] == ts].index[0]
            outcome = 'TIMEOUT'
            exit_price = entry_price
            exit_time = ts
            
            # Look ahead up to 48 bars
            for h in range(1, 49):
                future_idx = idx + h
                if future_idx >= len(df): break
                
                row = df.iloc[future_idx]
                # SL check first (conservative)
                if row['low'] <= sl_price:
                    outcome = 'SL_HIT'
                    exit_price = sl_price
                    exit_time = row['timestamp']
                    break
                # TP check
                if row['high'] >= tp_price:
                    outcome = 'TP_HIT'
                    exit_price = tp_price
                    exit_time = row['timestamp']
                    break
            
            if outcome == 'TIMEOUT':
                final_row = df.iloc[min(idx + 48, len(df)-1)]
                exit_price = final_row['close']
                exit_time = final_row['timestamp']
                if exit_price > entry_price:
                    outcome = 'TIMEOUT_WIN'
                else:
                    outcome = 'TIMEOUT_LOSS'
            
            stats[outcome] += 1
            
            # Chart Generation
            if charts_generated < max_charts:
                # Find the index of the signal in the original df
                prob = sig['prob_long']
                
                # Get window for chart
                start_idx = max(0, idx - 24)
                end_idx = min(len(df) - 1, idx + 72)
                window = df.iloc[start_idx:end_idx].copy()
                
                plt.figure(figsize=(12, 6))
                plt.plot(window['timestamp'], window['close'], label='Close Price', color='black', alpha=0.7)
                
                # Mark Entry, SL, TP
                plt.scatter(ts, entry_price, color='blue', marker='^', s=100, label=f'ENTRY (Prob: {prob:.2f})')
                plt.axhline(y=sl_price, color='red', linestyle='--', alpha=0.3, label='SL (1.5 ATR)')
                plt.axhline(y=tp_price, color='green', linestyle='--', alpha=0.3, label='TP (3.0 ATR)')
                
                # Mark Exit
                exit_color = 'green' if 'WIN' in outcome or 'TP' in outcome else 'red'
                plt.scatter(exit_time, exit_price, color=exit_color, marker='x', s=100, label=f'EXIT: {outcome}')
                
                plt.title(f"Reversal Signal: {symbol} | Prob: {prob:.2f} | Result: {outcome}")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                file_name = f"{symbol}_{ts.strftime('%Y%m%d_%H%M')}_{outcome}.png"
                plt.savefig(OUTPUT_DIR / file_name)
                plt.close()
                charts_generated += 1
                print(f"✅ Generated chart: {file_name}")

    print(f"\n{'='*40}")
    print(f"📊 BACKTEST SUMMARY (Threshold: {threshold})")
    print(f"{'='*40}")
    print(f"Total Signals: {stats['TOTAL']}")
    if stats['TOTAL'] > 0:
        win_rate = (stats['TP_HIT'] + stats['TIMEOUT_WIN']) / stats['TOTAL'] * 100
        print(f"Win Rate (TP + Timeout Win): {win_rate:.2f}%")
        print(f" - TP Hits: {stats['TP_HIT']} ({stats['TP_HIT']/stats['TOTAL']*100:.1f}%)")
        print(f" - SL Hits: {stats['SL_HIT']} ({stats['SL_HIT']/stats['TOTAL']*100:.1f}%)")
        print(f" - Timeout Wins: {stats['TIMEOUT_WIN']} ({stats['TIMEOUT_WIN']/stats['TOTAL']*100:.1f}%)")
        print(f" - Timeout Losses: {stats['TIMEOUT_LOSS']} ({stats['TIMEOUT_LOSS']/stats['TOTAL']*100:.1f}%)")
    print(f"{'='*40}")
    print(f"🏁 Charts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_charts(threshold=0.8, max_charts=0)
