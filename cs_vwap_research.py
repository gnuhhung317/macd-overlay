import pandas as pd
import numpy as np
import os
import csv
from research_pipeline import BaseStrategy, BacktestConfig, TradeSide, run_backtest_event_driven, calculate_metrics

class VWAPRubberBandStrategy(BaseStrategy):
    """
    VWAP Rubber Band Mean Reversion Strategy
    
    Hypothesis: 
    Institutional algorithms use VWAP as fair value. When price stretches too far 
    from the rolling anchor VWAP (high Z-Score) while retail funding rate is aggressively 
    supporting the stretch, a mean-reversion snapback is imminent. 
    """
    def __init__(self, vwap_z_thresh=2.5):
        self.vwap_z_thresh = vwap_z_thresh

    def calculate_indicators(self, df, config):
        df = df.copy()
        
        # Safe close & Typical Price
        typ = (df['high'] + df['low'] + df['close']) / 3.0
        
        # Calculate segmented VWAP (Instead of indefinite cumulative, rolling 120H / 5-day VWAP)
        # Using rolling sum for numerator and denominator
        rolling_pv = (typ * df['volume']).rolling(window=120, min_periods=20).sum()
        rolling_v = df['volume'].rolling(window=120, min_periods=20).sum().replace(0, np.nan)
        df['vwap_120'] = rolling_pv / rolling_v
        
        # VWAP Distance Percentage
        df['vwap_dist_pct'] = (df['close'] - df['vwap_120']) / df['vwap_120'].replace(0, np.nan)
        
        # Z-Score of the distance (over 5 days)
        dist_mean = df['vwap_dist_pct'].rolling(120, min_periods=20).mean()
        dist_std = df['vwap_dist_pct'].rolling(120, min_periods=20).std().replace(0, np.nan)
        df['vwap_z'] = (df['vwap_dist_pct'] - dist_mean) / dist_std
        
        if 'fundingRate' not in df.columns:
            df['fundingRate'] = 0.0
            
        df['funding'] = df['fundingRate'].ffill().fillna(0)
        
        # Funding 24H Mean
        df['fund_ma'] = df['funding'].rolling(24, min_periods=1).mean()
        
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

        if 'vwap_z' not in df.columns:
            return df
            
        # SHORT Trigger: Price is stretched far above VWAP AND funding is heavily positive
        short_cond = (df['vwap_z'] > self.vwap_z_thresh) & (df['funding'] > df['fund_ma']) & (df['funding'] > 0)
        df.loc[short_cond, 'short_trigger'] = True
        
        # LONG Trigger: Price is stretched far below VWAP AND funding is heavily negative
        long_cond = (df['vwap_z'] < -self.vwap_z_thresh) & (df['funding'] < df['fund_ma']) & (df['funding'] < 0)
        df.loc[long_cond, 'long_trigger'] = True
        
        # Adaptive Exit: Reverting to the 120H VWAP mean
        df['exit_signal_long'] = df['close'] >= df['vwap_120']
        df['exit_signal_short'] = df['close'] <= df['vwap_120']
                    
        return df

def walk_forward_backtest(df, strategy_class, symbol, sl_mult=3.0, tp_perc=0.03, vwap_z=2.5):
    window_length = 4320
    test_length = 720
    
    strategy = strategy_class(vwap_z_thresh=vwap_z)
    config = BacktestConfig(symbol=symbol, sl_atr_mult=sl_mult, tp_perc=tp_perc)
    df_eval = strategy.calculate_indicators(df, config)
    df_eval = strategy.calculate_signals(df_eval, config)
    
    df_eval = df_eval.dropna(subset=['atr', 'vwap_z']).reset_index(drop=True)
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
    
    params_str = f"vwapZ={vwap_z},sl={sl_mult},tp={tp_perc}"
    log_results("VWAP_RubberBand", avg_is_sharpe, avg_oos_sharpe, avg_oos_dd, params_str)

def log_results(name, is_sharpe, oos_sharpe, oos_dd, params):
    file_exists = os.path.isfile("experiments_log.csv")
    with open("experiments_log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Strategy Name", "IS Sharpe", "OOS Sharpe", "OOS Max DD", "Params"])
        writer.writerow([name, round(is_sharpe,3), round(oos_sharpe,3), round(oos_dd,3), params])
        
if __name__ == "__main__":
    print("Starting VWAP Rubber Band Edge Research (Iteration 4)")
    
    symbol_bare = 'BTCUSDT'
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
        
        print(f"Data ready: {len(df)} rows.")
        
        for z in [2.0, 2.5, 3.0]:
            for sl_mult in [2.0, 3.0]:
                for tp_perc in [0.03, 0.05, 0.1]:
                    print(f"Testing VWAP_Z={z}, SL={sl_mult}, TP={tp_perc}...")
                    walk_forward_backtest(df, VWAPRubberBandStrategy, symbol_bare, sl_mult, tp_perc, z)
                    
        print("Done. VWAP Log updated.")
        
    except Exception as e:
        print("Error during pipeline:", e)