import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.backtest_3stage import ThreeStageBacktester, BacktestConfig

DATA_DIR = project_root / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

def print_table(results):
    print("\n| Period | CB Threshold | Cooldown Bars | Total Trades | Win Rate | Final Equity | Max Drawdown |")
    print("|--------|--------------|---------------|--------------|----------|--------------|--------------|")
    for r in results:
        cb_str = f"{r['cb']:.0%}"
        wr_str = f"{r['win_rate']:.1%}"
        eq_str = f"${r['equity']:.2f}"
        dd_str = f"{r['drawdown']:.1%}"
        print(f"| {r['period']:<15} | {cb_str:<12} | {r['cooldown']:<13} | {r['trades']:<12} | {wr_str:<8} | {eq_str:<12} | {dd_str:<12} |")

def main():
    print("🚀 Starting Circuit Breaker Parameter Grid Search...")
    data_path = PROCESSED_DIR / 'features_1d_full.parquet'
    
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        return
        
    df = pd.read_parquet(data_path)
    df = df.sort_values('timestamp')
    print(f"Loaded {len(df):,} rows.")
    
    # Define test periods
    periods = {
        "2023 Bull": ("2023-01-01", "2024-01-01"),
        "2024 Chop/Bear": ("2024-04-01", "2024-10-01"),
        "2025+ Bull": ("2024-11-01", "2026-03-01"),
        "Full History": (None, None)
    }
    
    cb_pcts = [0.0, 0.30]
    cooldowns = [0, 1]
    
    all_results = []
    
    # Initialize models once to save massive load time
    print("⏳ Loading machine learning models once...")
    base_backtester = ThreeStageBacktester(BacktestConfig())
    
    for period_name, (start, end) in periods.items():
        print(f"\n==========================================")
        print(f"📅 Testing Period: {period_name} ({start} to {end})")
        print(f"==========================================")
        
        df_test = df.copy()
        if start: df_test = df_test[df_test['timestamp'] >= pd.to_datetime(start)]
        if end: df_test = df_test[df_test['timestamp'] <= pd.to_datetime(end)]
        
        if len(df_test) == 0:
            print("No data for this period.")
            continue
            
        print(f"Data shape: {len(df_test):,} rows.")
        
        for cb in cb_pcts:
            for cd in cooldowns:
                # Skip cooldown variations if CB is disabled
                if cb == 0.0 and cd > 0:
                    continue
                    
                print(f"  Testing CB={cb:.0%} | Cooldown={cd} bars...", end=" ", flush=True)
                
                config = BacktestConfig(
                    initial_capital=100.0,
                    risk_per_trade=0.01,
                    leverage=20,
                    margin_mode='ISOLATED',
                    use_scanner_filter=True,
                    max_bars=10,
                    global_cb_pct=cb,
                    global_cb_cooldown=cd
                )
                        
                base_backtester.config = config
                
                try:
                    # Use the main entry point method
                    result = base_backtester.run_backtest(df_test, verbose=False)
                    
                    trades = len(result.trades)
                    if hasattr(result, 'total_trades'): 
                        base_backtester._calculate_metrics(result) # Calculate metrics properly
                    
                    win_rate = result.win_rate if hasattr(result, 'win_rate') else 0.0
                    equity = result.equity_curve[-1] if hasattr(result, 'equity_curve') and len(result.equity_curve) > 0 else 100.0
                    drawdown = result.max_drawdown if hasattr(result, 'max_drawdown') else 0.0
                        
                    print(f"Equity: ${equity:.2f} | DD: {drawdown:.1%} | Trades: {trades}")
                    
                    all_results.append({
                        'period': period_name,
                        'cb': cb,
                        'cooldown': cd,
                        'trades': trades,
                        'win_rate': win_rate,
                        'equity': equity,
                        'drawdown': drawdown
                    })
                except Exception as e:
                    print(f"ERROR: {e}")
                    
    print("\n\n🏆 OPTIMIZATION RESULTS 🏆")
    print_table(all_results)
    
if __name__ == "__main__":
    main()
