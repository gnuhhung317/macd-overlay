import itertools
import random
import time
from research_pipeline import (
    DonchianBreakoutStrategy, 
    DonchianAdxStrategy, 
    BacktestConfig, 
    run_backtest_event_driven, 
    calculate_metrics, 
    log_experiment
)
import pandas as pd
from pathlib import Path

def run_infinite_loop():
    symbol = "SOLUSDT"
    data_path = Path("data/ohlcv") / f"{symbol}_USDT.parquet"
    if not data_path.exists():
        print("Data not found")
        return
        
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    split_idx = int(len(df) * 0.7)

    iteration = 5
    while True:
        print(f"\n--- Starting Iteration {iteration} ---")
        
        # Randomize strategy
        strat_choice = random.choice(['DonchianBreakout', 'DonchianAdx'])
        bb_period = random.choice([20, 30, 40, 50, 60, 80])
        sl_mult = random.choice([2.0, 3.0, 4.0, 5.0])
        
        config = BacktestConfig(symbol=symbol, bb_period=bb_period, sl_atr_mult=sl_mult)
        
        if strat_choice == 'DonchianBreakout':
            strategy = DonchianBreakoutStrategy()
            params_str = f"period={bb_period}, sl={sl_mult}"
        else:
            strategy = DonchianAdxStrategy()
            adx_thres = random.choice([20, 25, 30])
            params_str = f"period={bb_period}, adx_filter>{adx_thres}, sl={sl_mult}"
            
            # Monkey patch adx thres
            def calc_sig_override(self, df, config, adx_t=adx_thres):
                df['long_trigger'] = (df['high'] > df['dc_upper']) & (df['adx'] > adx_t)
                df['short_trigger'] = (df['low'] < df['dc_lower']) & (df['adx'] > adx_t)
                df['exit_signal_long'] = df['low'] < df['dc_exit_long']
                df['exit_signal_short'] = df['high'] > df['dc_exit_short']
                return df
            strategy.calculate_signals = calc_sig_override.__get__(strategy, DonchianAdxStrategy)
            
        print(f"Strategy: {strat_choice}, Params: {params_str}")
        
        # Calculate indicators and signals on full dataset to prevent lookahead issues at boundaries
        df_full = strategy.calculate_indicators(df.copy(), config)
        df_full = strategy.calculate_signals(df_full, config)
        
        df_is = df_full.iloc[:split_idx].copy().reset_index(drop=True)
        df_oos = df_full.iloc[split_idx:].copy().reset_index(drop=True)

        print("Running IS...")
        t_is, eq_is, curve_is = run_backtest_event_driven(df_is, config)
        metrics_is = calculate_metrics(t_is, curve_is)
        
        print("Running OOS...")
        t_oos, eq_oos, curve_oos = run_backtest_event_driven(df_oos, config)
        metrics_oos = calculate_metrics(t_oos, curve_oos)
        
        log_experiment(f"{strat_choice}_Random", metrics_is, metrics_oos, params_str)
        print(f"Iteration {iteration} Complete! Logged.")
        print(f"IS Sharpe: {metrics_is['Sharpe Ratio']:.2f} | OOS Sharpe: {metrics_oos['Sharpe Ratio']:.2f}")
        
        iteration += 1
        time.sleep(1) # Let the CPU breathe

if __name__ == "__main__":
    run_infinite_loop()
