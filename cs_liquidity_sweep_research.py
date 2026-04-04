import pandas as pd
import numpy as np
import os
import csv
from research_pipeline import BaseStrategy, BacktestConfig, TradeSide, run_backtest_event_driven, calculate_metrics

class LiquiditySweepStrategy(BaseStrategy):
    """
    Liquidity Sweep & FVG Imbalance (Smart Money Concept)
    
    Hypothesis: 
    When price violently sweeps below a low or above a high (creating a huge wick) 
    accompanied by abnormally high volume AND Open Interest drops (showing liquidations/stop hunts), 
    this marks an institutional reversal point.
    """
    def __init__(self, vol_z_thresh=1.0, oi_drop_thresh=-0.005):
        self.vol_z_thresh = vol_z_thresh
        self.oi_drop_thresh = oi_drop_thresh

    def calculate_indicators(self, df, config):
        df = df.copy()
        
        # Safe variables
        close_safe = df['close'].replace(0, np.nan)
        hl_range = (df['high'] - df['low']).replace(0, np.nan)
        body = np.abs(df['close'] - df['open'])
        
        # Shadows
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Volume Z-Score (20 period)
        vol_sma = df['volume'].rolling(20).mean().replace(0, np.nan)
        vol_std = df['volume'].rolling(20).std().replace(0, np.nan)
        df['vol_z'] = (df['volume'] - vol_sma) / vol_std
        
        # Liquidity Sweep Logic
        # Wick must be > 2x body, > 50% of the whole candle range, and volume must be spiking
        is_bull_sweep = (lower_shadow > 2 * body) & (lower_shadow > 0.5 * hl_range) & (df['vol_z'] > self.vol_z_thresh)
        is_bear_sweep = (upper_shadow > 2 * body) & (upper_shadow > 0.5 * hl_range) & (df['vol_z'] > self.vol_z_thresh)
        
        df['bull_sweep'] = is_bull_sweep
        df['bear_sweep'] = is_bear_sweep
        
        # OI Drop Verification (Liquidation proof)
        if 'sum_open_interest' in df.columns:
            df['oi_pct_change'] = df['sum_open_interest'].pct_change()
        else:
            df['oi_pct_change'] = 0.0
            
        # ATR Setup
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

        if 'vol_z' not in df.columns:
            return df
            
        # Long trigger: Bullish sweep occurred AND Open interest dropped (liquidating late shorts)
        long_cond = df['bull_sweep'] & (df['oi_pct_change'] < self.oi_drop_thresh)
        df.loc[long_cond, 'long_trigger'] = True
        
        # Short trigger: Bearish sweep occurred AND Open interest dropped (liquidating late longs)
        short_cond = df['bear_sweep'] & (df['oi_pct_change'] < self.oi_drop_thresh)
        df.loc[short_cond, 'short_trigger'] = True
                    
        return df

def walk_forward_backtest(df, strategy_class, symbol, sl_mult=3.0, tp_perc=0.03, vol_z_thresh=1.0, oi_drop_thresh=-0.005):
    window_length = 4320
    test_length = 720
    
    strategy = strategy_class(vol_z_thresh=vol_z_thresh, oi_drop_thresh=oi_drop_thresh)
    config = BacktestConfig(symbol=symbol, sl_atr_mult=sl_mult, tp_perc=tp_perc)
    df_eval = strategy.calculate_indicators(df, config)
    df_eval = strategy.calculate_signals(df_eval, config)
    
    df_eval = df_eval.dropna(subset=['atr', 'vol_z']).reset_index(drop=True)
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
    
    params_str = f"volZ={vol_z_thresh},oiDrop={oi_drop_thresh},sl={sl_mult},tp={tp_perc}"
    log_results("SM_LiquiditySweep", avg_is_sharpe, avg_oos_sharpe, avg_oos_dd, params_str)

def log_results(name, is_sharpe, oos_sharpe, oos_dd, params):
    file_exists = os.path.isfile("experiments_log.csv")
    with open("experiments_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Strategy Name", "IS Sharpe", "OOS Sharpe", "OOS Max DD", "Params"])
        writer.writerow([name, round(is_sharpe,3), round(oos_sharpe,3), round(oos_dd,3), params])
        
if __name__ == "__main__":
    print("Starting Institutional Liquidity Sweep Research (Iteration 3)")
    
    symbol_bare = 'BTCUSDT'
    try:
        df_ohlcv = pd.read_parquet(f'data/ohlcv/{symbol_bare}_USDT.parquet')
        df_deriv = pd.read_parquet(f'data/derivatives/{symbol_bare}.parquet')
        
        df_deriv = df_deriv.groupby('timestamp').last().reset_index()
        
        df = pd.merge(df_ohlcv, df_deriv, on='timestamp', how='inner')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Data ready: {len(df)} rows. Grid searching Sweep + OI factors.")
        
        for vol_z in [1.0, 1.5, 2.0]:
            for oi_drop in [-0.002, -0.005, -0.01]:
                for sl_mult in [1.5, 2.0, 3.0]:
                    for tp_perc in [0.02, 0.04, 0.08]:
                        print(f"Testing Sweep volZ={vol_z}, oiDrop={oi_drop}, SL={sl_mult}, TP={tp_perc}...")
                        walk_forward_backtest(df, LiquiditySweepStrategy, symbol_bare, sl_mult, tp_perc, vol_z, oi_drop)
                        
        print("Done. Liquidity Sweep Log updated.")
        
    except Exception as e:
        print("Error during research pipeline:", e)