import pandas as pd
import numpy as np
import os
import csv
from research_pipeline import BaseStrategy, BacktestConfig, TradeSide, run_backtest_event_driven, calculate_metrics

class EfficiencySqueezeStrategy(BaseStrategy):
    """
    Efficiency Thrust & Volatility Squeeze Breakout Strategy (Trend Continuation)
    
    Hypothesis: 
    Markets alternate between low volatility (compression) and high volatility (expansion).
    If the market is emerging from a severe volatility squeeze (Bollinger Band Width Z-Score < -1.5)
    and simultaneously exhibits a highly efficient directional move (Kaufman Efficiency Ratio > 0.6) 
    backed by volume, this is the start of a new macro trend. 
    """
    def __init__(self, er_thresh=0.6, vol_thresh=1.5, squeeze_z=-1.0):
        self.er_thresh = er_thresh
        self.vol_thresh = vol_thresh
        self.squeeze_z = squeeze_z

    def calculate_indicators(self, df, config):
        df = df.copy()
        
        # 1. Kaufman Efficiency Ratio (ER) - 10 Periods
        # ER = Abs(Price Change) / Sum of absolute period-to-period price changes
        df['change_10'] = df['close'] - df['close'].shift(10)
        df['abs_diff'] = np.abs(df['close'] - df['close'].shift(1)).rolling(10).sum()
        df['er'] = np.abs(df['change_10']) / df['abs_diff'].replace(0, np.nan)
        df['direction'] = np.sign(df['change_10'])
        
        # 2. Volatility Squeeze (Bollinger Band Width Z-Score)
        std_20 = df['close'].rolling(20).std()
        df['bb_width'] = std_20 * 4 
        bb_mean = df['bb_width'].rolling(20).mean()
        bb_std = df['bb_width'].rolling(20).std().replace(0, np.nan)
        df['bb_width_z'] = (df['bb_width'] - bb_mean) / bb_std
        
        # Squeeze Active if recently in compression
        df['squeeze_active'] = df['bb_width_z'].rolling(5).min() < self.squeeze_z
        
        # 3. Volume Thrust
        vol_sma_20 = df['volume'].rolling(20).mean().replace(0, np.nan)
        df['vol_ratio'] = df['volume'] / vol_sma_20
        
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

        if 'er' not in df.columns:
            return df
            
        # LONG Trigger
        long_cond = (df['squeeze_active']) & (df['er'] > self.er_thresh) & (df['direction'] == 1) & (df['vol_ratio'] > self.vol_thresh)
        df.loc[long_cond, 'long_trigger'] = True
        
        # SHORT Trigger
        short_cond = (df['squeeze_active']) & (df['er'] > self.er_thresh) & (df['direction'] == -1) & (df['vol_ratio'] > self.vol_thresh)
        df.loc[short_cond, 'short_trigger'] = True
        
        # Exit: When trend loses efficiency
        df['exit_signal_long'] = (df['er'] < 0.2) & (df['direction'] == -1)
        df['exit_signal_short'] = (df['er'] < 0.2) & (df['direction'] == 1)
                    
        return df

def walk_forward_backtest(df, strategy_class, symbol, sl_mult=3.0, tp_perc=0.05, er_t=0.6, vol_t=1.5):
    window_length = 4320
    test_length = 720
    
    strategy = strategy_class(er_thresh=er_t, vol_thresh=vol_t)
    config = BacktestConfig(symbol=symbol, sl_atr_mult=sl_mult, tp_perc=tp_perc)
    df_eval = strategy.calculate_indicators(df, config)
    df_eval = strategy.calculate_signals(df_eval, config)
    
    df_eval = df_eval.dropna(subset=['atr', 'er', 'bb_width_z']).reset_index(drop=True)
    total_rows = len(df_eval)
    
    is_sharpes = []
    all_oos_trades = []
    global_oos_curve = [config.initial_capital]
    
    for start_idx in range(0, total_rows - window_length - test_length, test_length):
        end_is = start_idx + window_length
        end_oos = end_is + test_length
        
        df_is = df_eval.iloc[start_idx:end_is].copy().reset_index(drop=True)
        df_oos = df_eval.iloc[end_is:end_oos].copy().reset_index(drop=True)
        
        t_is, eq_is, curve_is = run_backtest_event_driven(df_is, config)
        metric_is = calculate_metrics(t_is, curve_is)
        
        t_oos, eq_oos, curve_oos = run_backtest_event_driven(df_oos, config)
        
        if len(t_is) > 0 and len(t_oos) > 0:
            is_sharpes.append(metric_is.get('Sharpe Ratio', 0))
            ret_oos = pd.Series(curve_oos).pct_change().dropna()
            for r in ret_oos:
                global_oos_curve.append(global_oos_curve[-1] * (1 + r))
            all_oos_trades.extend(t_oos)
            
    avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
    global_oos_metrics = calculate_metrics(all_oos_trades, global_oos_curve)
    avg_oos_sharpe = global_oos_metrics.get('Sharpe Ratio', 0.0)
    avg_oos_dd = global_oos_metrics.get('Max Drawdown', 0.0)
    
    params_str = f"ER={er_t},Vol={vol_t},sl={sl_mult},tp={tp_perc}"
    log_results("Efficiency_Squeeze", avg_is_sharpe, avg_oos_sharpe, avg_oos_dd, params_str)

def log_results(name, is_sharpe, oos_sharpe, oos_dd, params):
    file_exists = os.path.isfile("experiments_log.csv")
    with open("experiments_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Strategy Name", "IS Sharpe", "OOS Sharpe", "OOS Max DD", "Params"])
        writer.writerow([name, round(is_sharpe,3), round(oos_sharpe,3), round(oos_dd,3), params])
        
if __name__ == "__main__":
    print("Starting Efficiency Thrust + Squeeze Strategy Research (Iteration 5)")
    
    symbol_bare = 'BTCUSDT'
    try:
        df_ohlcv = pd.read_parquet(f'data/ohlcv/{symbol_bare}_USDT.parquet')
        df_deriv = pd.read_parquet(f'data/derivatives/{symbol_bare}.parquet')
        
        # This strategy only strictly uses Price & Volume & OI
        df_deriv = df_deriv.groupby('timestamp').last().reset_index()
        
        df = pd.merge(df_ohlcv, df_deriv, on='timestamp', how='inner')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Data ready: {len(df)} rows. Commencing Walk-Forward.")
        
        for er_t in [0.5, 0.6, 0.7]:
            for vol_t in [1.5, 2.0]:
                for sl_mult in [2.0, 3.0]:
                    for tp_perc in [0.05, 0.08, 0.12]:
                        print(f"Testing ER={er_t}, VolRatio={vol_t}, SL={sl_mult}, TP={tp_perc}...")
                        walk_forward_backtest(df, EfficiencySqueezeStrategy, symbol_bare, sl_mult, tp_perc, er_t, vol_t)
                        
        print("Done. Efficiency Squeeze Log updated.")
        
    except Exception as e:
        print("Error during pipeline:", e)