#!/usr/bin/env python3
"""
Grid Search for MACD Overlay Backtest Stability Analysis.
Tests different Entry Zones over various historical market cycles.
"""
import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Add parent directory to path to allow importing from ml
sys.path.append(str(Path(__file__).parent.parent))

from ml.backtest_3stage import ThreeStageBacktester, BacktestConfig

# Define Historical Periods
PERIODS = {
    "2021 Mega Bull": ("2020-10-01", "2021-05-15"),
    "2022 Crypto Winter": ("2021-11-15", "2022-12-31"),
    "2023 Recovery": ("2023-01-01", "2023-12-31"),
    "2024-2025 ETF Bull": ("2024-01-01", "2025-01-31"),
    "Out-of-Sample (Feb 26)": ("2026-02-01", "2026-02-22"),
    "Full (2020-2026)": ("2020-01-01", "2026-02-22")
}

# Define Zone Configurations
ZONE_CONFIGS = {
    "ALL": ["GOOD ENTRY", "DISCOUNT", "DEEP MERGE"],
    "GOOD_DEEP": ["GOOD ENTRY", "DEEP MERGE"],
    "GOOD_DISCOUNT": ["GOOD ENTRY", "DISCOUNT"],
    "GOOD_ONLY": ["GOOD ENTRY"],
    "DISCOUNT_ONLY": ["DISCOUNT"],
    "DISCOUNT_DEEP": ["DISCOUNT", "DEEP MERGE"],
    "DEEP_ONLY": ["DEEP MERGE"]
}

def run_stability_grid_search(args, offsets=[0, 2, 4, 6, 8]):
    print(f"🚀 Starting Multi-Start Grid Search (Leverage: {args.leverage}x, Mode: {args.margin_mode}, TF: {args.timeframe}, Offsets: {offsets})")
    
    # Load data once to speed up
    data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / f'features_{args.timeframe}_full.parquet'
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return
    
    df_full = pd.read_parquet(data_path)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
    
    all_results = []
    
    periods_to_run = PERIODS.copy()
    if getattr(args, 'start', None) and getattr(args, 'end', None):
        periods_to_run = {f"Custom ({args.start} to {args.end})": (args.start, args.end)}
        
    for p_name, (start_dt, end_dt) in periods_to_run.items():
        print(f"\n--- Testing Period: {p_name} ({start_dt} to {end_dt}) ---")
        
        start_ts = pd.to_datetime(start_dt)
        end_ts = pd.to_datetime(end_dt)
        
        for z_name, zones in ZONE_CONFIGS.items():
            print(f"  Testing Zones: {z_name} across {len(offsets)} offsets...")
            
            sub_runs = []
            for offset_days in offsets:
                # Calculate offset start
                current_start = start_ts + pd.Timedelta(days=offset_days)
                if current_start >= end_ts: continue
                
                # Filter data for this run
                mask = (df_full['timestamp'] >= current_start) & (df_full['timestamp'] <= end_ts)
                df_period = df_full.loc[mask].copy()
                
                if df_period.empty: continue
                
                config = BacktestConfig(
                    initial_capital=args.capital,
                    risk_per_trade=args.risk,
                    entry_threshold=args.threshold,
                    fee_rate=args.fee,
                    slippage=args.slippage,
                    leverage=args.leverage,
                    timeframe=args.timeframe,
                    margin_mode=args.margin_mode,
                    use_kelly=args.kelly,
                    fixed_position_size=args.fixed_size,
                    position_size_usd=args.size_usd,
                    max_open_trades=args.max_positions,
                    use_trailing_stop=args.trailing,
                    trailing_start_pct=args.trailing_start,
                    trailing_step_pct=args.trailing_step,
                    use_scanner_filter=args.use_scanner,
                    scanner_mae=args.scanner_mae,
                    scanner_mfe=args.scanner_mfe,
                    scanner_lookback_days=args.scanner_lookback,
                    allowed_zones=zones,
                    start_date=current_start.strftime("%Y-%m-%d"),
                    end_date=end_dt
                )
                
                backtester = ThreeStageBacktester(config)
                result = backtester.run_backtest(df_period, verbose=False)
                
                sub_runs.append({
                    "TR": result.total_return,
                    "MDD": result.max_drawdown,
                    "WinRate": result.win_rate,
                    "Trades": result.total_trades,
                    "Liquidations": len([t for t in result.trades if t.exit_reason == 'LIQUIDATED']),
                    "Sharpe": result.sharpe_ratio
                })
            
            if not sub_runs: continue
            
            # Aggregate Results
            all_results.append({
                "Period": p_name,
                "Zones": z_name,
                "TR (%)": f"{sum(r['TR'] for r in sub_runs)/len(sub_runs) * 100:.1f}%",
                "MDD (%)": f"{sum(r['MDD'] for r in sub_runs)/len(sub_runs) * 100:.1f}%",
                "Avg WR": f"{sum(r['WinRate'] for r in sub_runs)/len(sub_runs) * 100:.1f}%",
                "Avg Trades": round(sum(r['Trades'] for r in sub_runs)/len(sub_runs), 1),
                "Avg Liq": round(sum(r['Liquidations'] for r in sub_runs)/len(sub_runs), 1),
                "Avg Sharpe": round(sum(r['Sharpe'] for r in sub_runs)/len(sub_runs), 2),
                "Sharpe Var": round(pd.Series([r['Sharpe'] for r in sub_runs]).std(), 3)
            })
            
    # Save Report
    report_df = pd.DataFrame(all_results)
    report_path = Path(__file__).parent.parent / 'output' / f'grid_search_{args.timeframe}_{args.leverage}x_{args.margin_mode}.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Grid Search Stability Report\n\n")
        f.write(f"- **Leverage**: {args.leverage}x\n")
        f.write(f"- **Margin Mode**: {args.margin_mode}\n")
        f.write(f"- **Timeframe**: {args.timeframe}\n\n")
        f.write(report_df.to_markdown(index=False))
        
    print(f"\n✅ Grid Search complete! Report saved to: {report_path}")
    return report_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=100.0)
    parser.add_argument('--risk', type=float, default=0.01)
    parser.add_argument('--threshold', type=float, default=0.65)
    parser.add_argument('--fee', type=float, default=0.001)
    parser.add_argument('--slippage', type=float, default=0.0005)
    parser.add_argument('--kelly', action='store_true')
    parser.add_argument('--fixed-size', action='store_true')
    parser.add_argument('--size-usd', type=float, default=1000)
    parser.add_argument("--leverage", type=float, default=20)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--margin-mode", type=str, default="ISOLATED", choices=["ISOLATED", "CROSS"])
    parser.add_argument("--timeframe", type=str, default="1d")
    parser.add_argument('--trailing', action='store_true')
    parser.add_argument('--trailing-start', type=float, default=0.1)
    parser.add_argument('--trailing-step', type=float, default=0.05)
    parser.add_argument('--use-scanner', action='store_true')
    parser.add_argument('--scanner-mae', type=float, default=0.04)
    parser.add_argument('--scanner-mfe', type=float, default=0.12)
    parser.add_argument('--scanner-lookback', type=int, default=6)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()
    
    run_stability_grid_search(args)
