import pandas as pd
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from backtest_tmfs_v2 import run_backtest_event_driven, BacktestConfig, calculate_metrics, resample_data
import argparse

def process_symbol(symbol_file, start_date=None, end_date=None, timeframe="1h"):
    symbol = symbol_file.stem.replace("_USDT", "")
    try:
        df = pd.read_parquet(symbol_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if start_date:
            start_ts = pd.to_datetime(start_date).tz_localize(df['timestamp'].dt.tz)
            df = df[df['timestamp'] >= start_ts]
        if end_date:
            end_ts = pd.to_datetime(end_date).tz_localize(df['timestamp'].dt.tz)
            df = df[df['timestamp'] <= end_ts]
            
        df = df.reset_index(drop=True)
        
        if timeframe != "1h":
            df = resample_data(df, timeframe)
            
        if len(df) < 100: # Skip if too little data
            return None
            
        config = BacktestConfig(
            symbol=symbol,
            qty_pct=0.4 # Matching user's latest change
        )
        
        trades, final_equity, equity_curve = run_backtest_event_driven(df, config)
        metrics = calculate_metrics(trades, equity_curve)
        
        if not metrics:
            return None
            
        metrics['Symbol'] = symbol
        return metrics
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, help="Limit number of symbols to test")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--timeframe", type=str, default="1h", choices=["1h", "4h", "8h", "12h", "1d"], help="Timeframe to resample to")
    args = parser.parse_args()

    data_dir = Path("data/ohlcv")
    symbol_files = list(data_dir.glob("*.parquet"))
    
    if args.limit:
        symbol_files = symbol_files[:args.limit]
        
    print(f"Found {len(symbol_files)} symbol files. Starting batch backtest with {args.workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_symbol, f, args.start, args.end, args.timeframe): f for f in symbol_files}
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                print(f"Completed {res['Symbol']}: Win Rate {res['Win Rate']}, Net Profit {res['Net Profit %']}")

    if results:
        results_df = pd.DataFrame(results)
        # Move Symbol to first column
        cols = ['Symbol'] + [c for c in results_df.columns if c != 'Symbol']
        results_df = results_df[cols]
        
        output_file = "batch_backtest_report.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nBatch backtest complete. Results saved to {output_file}")
        
        # Summary metrics
        # Clean numeric columns for sorting
        def clean_pct(x):
            try: return float(str(x).replace('%', ''))
            except: return 0.0
            
        def clean_val(x):
            try: return float(str(x).replace('$', ''))
            except: return 0.0

        results_df['Net Profit % Num'] = results_df['Net Profit %'].apply(clean_pct)
        
        print("\nTop 10 Performers (Net Profit %):")
        print(results_df.sort_values('Net Profit % Num', ascending=False).head(10)[['Symbol', 'Total Trades', 'Win Rate', 'Net Profit %']])
        
        print("\nSummary Statistics:")
        print(f"Total Symbols Tested: {len(results)}")
        print(f"Average Net Profit %: {results_df['Net Profit % Num'].mean():.2f}%")
        print(f"Profitable Symbols  : {len(results_df[results_df['Net Profit % Num'] > 0])}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
