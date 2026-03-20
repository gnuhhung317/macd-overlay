import pandas as pd
from datetime import datetime
from ml.config import get_timeframe_config
from ml.backtest_3stage import ThreeStageBacktester, BacktestConfig, PROCESSED_DIR

def run_debug():
    tf_val = '1d'
    tf_config = get_timeframe_config(tf_val)
    config = BacktestConfig(
        initial_capital=300.0,
        risk_per_trade=0.02, # default
        leverage=20.0,
        timeframe=tf_val,
        max_bars=tf_config.max_bars,
        max_open_trades=13,
        entry_threshold=0.6,
        use_scanner_filter=True,
        start_date='2025-01-03',
        end_date='2025-03-22'
    )
    
    data_path = PROCESSED_DIR / f'features_{tf_val}_full.parquet'
    df = pd.read_parquet(data_path).sort_values('timestamp')
    df = df[df['timestamp'] >= pd.to_datetime('2025-01-03')]
    df = df[df['timestamp'] <= pd.to_datetime('2025-03-22')]

    backtester = ThreeStageBacktester(config)
    result = backtester.run_backtest(df, verbose=False)
    
    print("BACKTEST DAILY OPEN POSITIONS MAX:", max(result.daily_open_positions.values()) if result.daily_open_positions else 0)
    
    # GUI calculation
    timestamps = sorted(df['timestamp'].unique())
    gui_max = 0
    t_max = None
    
    counts = {}
    for t in timestamps:
        open_trades = [tr for tr in result.trades if tr.entry_time <= t and (not tr.exit_time or tr.exit_time > t)]
        counts[t] = len(open_trades)
        if len(open_trades) > gui_max:
            gui_max = len(open_trades)
            t_max = t
            
    print("GUI MAX OPEN POSITIONS:", gui_max, "AT", t_max)
    
    # Analyze the trades at t_max
    open_trades = [tr for tr in result.trades if tr.entry_time <= t_max and (not tr.exit_time or tr.exit_time > t_max)]
    print(f"Trades open at {t_max} (Total: {len(open_trades)}):")
    for tr in sorted(open_trades, key=lambda x: x.entry_time):
        print(f"  {tr.symbol} {tr.direction} Entry: {tr.entry_time} Exit: {tr.exit_time} Reason: {tr.exit_reason}")

if __name__ == '__main__':
    run_debug()
