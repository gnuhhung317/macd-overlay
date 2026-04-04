import pandas as pd
import numpy as np
import os
import csv
from research_pipeline import BaseStrategy, BacktestConfig, TradeSide, run_backtest_event_driven, calculate_metrics

class StructuralDivergenceStrategy(BaseStrategy):
    """
    Mean Reversion / Fakeout Fade Strategy
    Capitalizing on Price-OI Divergences
    """
    def calculate_indicators(self, df, config):
        df = df.copy()
        
        # Calculate 24h rolling max/min (shift by 1 to prevent lookahead)
        df['Price_Rolling_Max_24h'] = df['high'].rolling(window=24).max().shift(1)
        df['Price_Rolling_Min_24h'] = df['low'].rolling(window=24).min().shift(1)
        
        # Calculate 24h rolling max/min for OI (shift by 1 to prevent lookahead)
        if 'sum_open_interest' in df.columns:
            df['OI_Rolling_Max_24h'] = df['sum_open_interest'].rolling(window=24).max().shift(1)
            df['OI_Rolling_Min_24h'] = df['sum_open_interest'].rolling(window=24).min().shift(1)
            
            # Rate of Change (4 period)
            df['Close_ROC_4'] = df['close'].pct_change(periods=4)
            df['OI_ROC_4'] = df['sum_open_interest'].pct_change(periods=4)
            
            # Liquidation proxy
            # Sudden huge drop in OI (e.g., -5% in 1H) with price moving wildly
            df['OI_Change_1H'] = df['sum_open_interest'].pct_change()
        
        if 'fundingRate' in df.columns:
            # 3-period rolling ROC for Funding
            # Since funding is often very small, difference is better than pct_change
            df['Funding_Diff_3'] = df['fundingRate'].diff(periods=3)
        
        # ATR for Stop Loss sizing
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df

    def calculate_signals(self, df, config):
        df = df.copy()
        df['long_trigger'] = False
        df['short_trigger'] = False
        df['exit_signal_long'] = False
        df['exit_signal_short'] = False

        # Make sure columns exist
        if 'sum_open_interest' not in df.columns:
            return df
            
        for i in range(25, len(df)):
            close_price = df.loc[i, 'close']
            
            # Fakeout Breakout Upwards (Short Trigger)
            # Price reaches new high but OI is dropping
            if close_price >= df.loc[i, 'Price_Rolling_Max_24h']:
                if df.loc[i, 'Close_ROC_4'] > 0 and df.loc[i, 'OI_ROC_4'] < -0.01:
                    df.loc[i, 'short_trigger'] = True
                    
            # Fakeout Breakdown Downwards (Long Trigger)
            # Price breaks to new low but OI is dropping
            elif close_price <= df.loc[i, 'Price_Rolling_Min_24h']:
                if df.loc[i, 'Close_ROC_4'] < 0 and df.loc[i, 'OI_ROC_4'] < -0.01:
                    df.loc[i, 'long_trigger'] = True
                    
        return df

def walk_forward_backtest(df, strategy_class, symbol, sl_mult=3.0, tp_perc=0.03):
    # Sliding window config
    window_length = 4320
    test_length = 720
    
    strategy = strategy_class()
    config = BacktestConfig(symbol=symbol, sl_atr_mult=sl_mult, tp_perc=tp_perc)
    df = strategy.calculate_indicators(df, config)
    df = strategy.calculate_signals(df, config)
    
    # Clean rows with NaN indicators
    df = df.dropna(subset=['atr', 'OI_ROC_4']).reset_index(drop=True)
    
    total_rows = len(df)
    
    is_sharpes = []
    
    # To properly evaluate Walk-Forward OOS performance, 
    # we must concatenate the entire out-of-sample equity curve over all folds,
    # rather than averaging the Sharpe Ratios of independent short periods.
    all_oos_trades = []
    global_oos_curve = [config.initial_capital]
    
    for start_idx in range(0, total_rows - window_length - test_length, test_length):
        end_is = start_idx + window_length
        end_oos = end_is + test_length
        
        df_is = df.iloc[start_idx:end_is].copy().reset_index(drop=True)
        df_oos = df.iloc[end_is:end_oos].copy().reset_index(drop=True)
        
        t_is, eq_is, curve_is = run_backtest_event_driven(df_is, config)
        metric_is = calculate_metrics(t_is, curve_is)
        
        t_oos, eq_oos, curve_oos = run_backtest_event_driven(df_oos, config)
        
        # Only log valid splits
        if len(t_is) > 0 and len(t_oos) > 0:
            is_sharpes.append(metric_is.get('Sharpe Ratio', 0))
            
            # Stitch the OOS equity curve by computing period returns and compounding
            ret_oos = pd.Series(curve_oos).pct_change().dropna()
            for r in ret_oos:
                global_oos_curve.append(global_oos_curve[-1] * (1 + r))
            
            all_oos_trades.extend(t_oos)
            
    avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
    
    # Calculate OOS metrics on the stitched globally continuous curve
    global_oos_metrics = calculate_metrics(all_oos_trades, global_oos_curve)
    avg_oos_sharpe = global_oos_metrics.get('Sharpe Ratio', 0.0)
    avg_oos_dd = global_oos_metrics.get('Max Drawdown', 0.0)
    
    params_str = f"sl={sl_mult},tp={tp_perc}"
    log_results("StructuralDivergence", avg_is_sharpe, avg_oos_sharpe, avg_oos_dd, params_str)

def log_results(name, is_sharpe, oos_sharpe, oos_dd, params):
    file_exists = os.path.isfile("experiments_log.csv")
    with open("experiments_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Strategy Name", "IS Sharpe", "OOS Sharpe", "OOS Max DD", "Params"])
        writer.writerow([name, round(is_sharpe,3), round(oos_sharpe,3), round(oos_dd,3), params])
        
if __name__ == "__main__":
    print("Starting Structural Divergence Research")
    
    symbol_bare = 'BTCUSDT'
    # load files
    try:
        df_ohlcv = pd.read_parquet(f'data/ohlcv/{symbol_bare}_USDT.parquet')
        df_deriv = pd.read_parquet(f'data/derivatives/{symbol_bare}.parquet')
        df_fund = pd.read_parquet(f'data/funding/{symbol_bare}_USDT.parquet')
        
        # Align timestamps
        df_fund['timestamp'] = df_fund['timestamp'].dt.floor('h')
        
        # Take mean if there are multiple entries for the same hour
        df_fund = df_fund.groupby('timestamp').mean().reset_index()
        df_deriv = df_deriv.groupby('timestamp').last().reset_index()
        
        # Merge
        df = pd.merge(df_ohlcv, df_deriv, on='timestamp', how='inner')
        df = pd.merge(df, df_fund, on='timestamp', how='left')
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Data ready: {len(df)} rows")
        
        # Explore multiple parameters continuously
        for sl_mult in [2.0, 3.0, 4.0, 5.0]:
            for tp_perc in [0.02, 0.03, 0.04, 0.05, 0.08]:
                print(f"Testing Structural Divergence with SL={sl_mult}, TP={tp_perc}...")
                walk_forward_backtest(df, StructuralDivergenceStrategy, symbol_bare, sl_mult, tp_perc)
                
        print("Done. Log updated.")
        
    except Exception as e:
        print("Error during research pipeline:", e)
