import pandas as pd
import numpy as np
import os
import csv
from research_pipeline import BaseStrategy, BacktestConfig, run_backtest_event_driven, calculate_metrics

class SFAIEdgeStrategy(BaseStrategy):
    """
    Shadow Funding Asymmetry Index (SFAI) Edge Strategy
    
    Hypothesis: 
    When market positioning (Funding Rate) opposes the micro-structural 
    price rejection (Wick Shadows) concurrently with high volume, liquidity is trapped.
    
    - If SFAI is highly positive -> Upper shadow is large + funding is positive. 
      (Retail longs are trapped buying the top into a rejection wick on high volume).
      -> Trigger SHORT.
      
    - If SFAI is highly negative -> Lower shadow is large + funding is negative. 
      (Retail shorts are trapped selling the bottom into a rejection wick on high volume).
      -> Trigger LONG.
    """
    def __init__(self, z_threshold=2.0):
        self.z_threshold = z_threshold

    def calculate_indicators(self, df, config):
        df = df.copy()
        
        # Safe close to prevent division by zero
        close_safe = df['close'].replace(0, np.nan)
        
        # Calculate Micro-structure (Shadows & Body)
        upper_shadow_pct = (df['high'] - df[['open', 'close']].max(axis=1)) / close_safe
        lower_shadow_pct = (df[['open', 'close']].min(axis=1) - df['low']) / close_safe
        body_size = (np.abs(df['close'] - df['open']) / close_safe).replace(0, 0.0001)  # avoid inf SFAI
        
        shadow_diff = upper_shadow_pct - lower_shadow_pct
        
        # Calculate Volume component
        vol_sma_20 = df['volume'].rolling(20, min_periods=1).mean().replace(0, np.nan)
        vol_ratio = df['volume'] / vol_sma_20
        
        # Ensure funding exists
        if 'fundingRate' not in df.columns:
            df['fundingRate'] = 0.0
            
        funding = df['fundingRate'].ffill().fillna(0)
        
        # Calculate Base SFAI
        # Concept: (Wick Rejection Bias / Body) * Direction of Retail * Force of Retail * Volume Validity
        # SHIFT ALL INGREDIENTS BY 1 TO AVOID LOOKAHEAD: Wait, SFAI uses CURRENT candle, 
        # but trade executes NEXT candle logic in run_backtest_event_driven (using prev_row). 
        # So we can calculate based on current row.
        
        sfai = (shadow_diff / body_size) * np.sign(funding) * (np.abs(funding)**1.2) * vol_ratio
        
        # Calculate SFAI Z-Score (120 periods = 5 days)
        sfai_roll = sfai.rolling(120, min_periods=20)
        df['sfai_zscore'] = (sfai - sfai_roll.mean()) / sfai_roll.std().replace(0, np.nan)
        
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

        if 'sfai_zscore' not in df.columns:
            return df
            
        # SFAI positive -> Trapped Longs -> SHORT trigger
        df.loc[df['sfai_zscore'] > self.z_threshold, 'short_trigger'] = True
        
        # SFAI negative -> Trapped Shorts -> LONG trigger
        df.loc[df['sfai_zscore'] < -self.z_threshold, 'long_trigger'] = True
                    
        return df

def walk_forward_backtest(df, strategy_class, symbol, sl_mult=3.0, tp_perc=0.03, z_thresh=2.0):
    window_length = 4320
    test_length = 720
    
    strategy = strategy_class(z_threshold=z_thresh)
    config = BacktestConfig(symbol=symbol, sl_atr_mult=sl_mult, tp_perc=tp_perc)
    df = strategy.calculate_indicators(df, config)
    df = strategy.calculate_signals(df, config)
    
    df = df.dropna(subset=['atr', 'sfai_zscore']).reset_index(drop=True)
    total_rows = len(df)
    
    is_sharpes = []
    
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
    
    params_str = f"SFAI_Z={z_thresh},sl={sl_mult},tp={tp_perc}"
    log_results("SFAI_TrappedLiquidity", avg_is_sharpe, avg_oos_sharpe, avg_oos_dd, params_str)

def log_results(name, is_sharpe, oos_sharpe, oos_dd, params):
    file_exists = os.path.isfile("experiments_log.csv")
    with open("experiments_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Strategy Name", "IS Sharpe", "OOS Sharpe", "OOS Max DD", "Params"])
        writer.writerow([name, round(is_sharpe,3), round(oos_sharpe,3), round(oos_dd,3), params])
        
if __name__ == "__main__":
    print("Starting SFAI Edge Strategy Research (Iteration 2)")
    
    symbol_bare = 'SOLUSDT'
    try:
        df_ohlcv = pd.read_parquet(f'data/ohlcv/{symbol_bare}_USDT.parquet')
        df_deriv = pd.read_parquet(f'data/derivatives/{symbol_bare}.parquet')
        df_fund = pd.read_parquet(f'data/funding/{symbol_bare}_USDT.parquet')
        
        df_fund['timestamp'] = df_fund['timestamp'].dt.floor('h')
        df_fund = df_fund.groupby('timestamp').mean().reset_index()
        df_deriv = df_deriv.groupby('timestamp').last().reset_index()
        
        df = pd.merge(df_ohlcv, df_deriv, on='timestamp', how='inner')
        df = pd.merge(df, df_fund, on='timestamp', how='left')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Data ready: {len(df)} rows. Grid searching SFAI Z-Score + SL/TP parameters.")
        
        for z in [2.0, 2.5, 3.0]:
            for sl_mult in [2.0, 3.0, 4.0]:
                for tp_perc in [0.03, 0.05, 0.08]:
                    print(f"Testing SFAI Z={z}, SL={sl_mult}, TP={tp_perc}...")
                    walk_forward_backtest(df, SFAIEdgeStrategy, symbol_bare, sl_mult, tp_perc, z)
                    
        print("Done. SFAI Experiment Log updated.")
        
    except Exception as e:
        print("Error during research pipeline:", e)