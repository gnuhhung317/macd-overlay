import pandas as pd
from backtest_3stage import ThreeStageBacktester, BacktestConfig
from plot_time_equity import create_daily_equity_curve
from pathlib import Path

def main():
    data_path = Path('../bitget-data/processed/features_1d_full.parquet')
    if not data_path.exists():
        data_path = Path('bitget-data/processed/features_1d_full.parquet')
        
    df = pd.read_parquet(data_path)

    config = BacktestConfig(
        initial_capital=100.0,
        timeframe='1d',
        margin_mode='ISOLATED',
        leverage=20.0,
        risk_per_trade=0.01,
        entry_threshold=0.6,
        max_open_trades=10,
        use_scanner_filter=True,
        use_trailing_stop=True,
        trailing_start_pct=0.1,  # Default from plot_time_equity.py
        trailing_step_pct=0.05   # Default from plot_time_equity.py
    )

    bt = ThreeStageBacktester(config)

    df_warmup = df[(df['timestamp'] >= '2026-01-05') & (df['timestamp'] <= '2026-02-23')].copy()

    price_columns = ['timestamp', 'close']
    if 'symbol' in df_warmup.columns:
        price_columns.insert(0, 'symbol')
    if 'open' in df_warmup.columns:
        price_columns.extend(['open', 'high', 'low'])
    price_data = df_warmup[price_columns].copy()

    print("Running backtest...")
    result, daily_equity_df, _ = create_daily_equity_curve(df_warmup, bt, '2026-01-05', '2026-02-23', price_data)

    print(f"Total trades: {len(result.trades)}")

    # Reproduce Plotter's counting logic
    trade_events = []
    for t in result.trades:
        trade_events.append({'date': t.entry_time.date(), 'type': 'entry', 'id': id(t), 'symbol': t.symbol})
        if t.exit_time:
            trade_events.append({'date': t.exit_time.date(), 'type': 'exit', 'id': id(t), 'symbol': t.symbol})
            
    trade_df = pd.DataFrame(trade_events)
    
    open_positions_track = {}
    max_count = 0
    max_date = None
    
    for date in pd.date_range('2026-01-05', '2026-02-23', freq='D'):
        current_date = date.date()
        day_events = trade_df[trade_df['date'] == current_date] if not trade_df.empty else pd.DataFrame()
        
        for _, event in day_events.iterrows():
            if event['type'] == 'entry':
                open_positions_track[event['id']] = event['symbol']
            elif event['type'] == 'exit':
                if event['id'] in open_positions_track:
                    del open_positions_track[event['id']]
        
        count = len(open_positions_track)
        if count > max_count:
            max_count = count
            max_date = current_date
            
        if current_date == pd.to_datetime('2026-02-15').date():
            print(f"\nCalculated open positions on {current_date} directly: {count}")
            for tid, sym in open_positions_track.items():
                t = next(x for x in result.trades if id(x) == tid)
                print(f'   - {sym} Entry: {t.entry_time.date()} Exit: {t.exit_time.date() if t.exit_time else "None"} Reason: {t.exit_reason}')

    print(f"\nMaximum open positions at any time point was {max_count} on {max_date}")

if __name__ == '__main__':
    main()
