import pandas as pd
from research_pipeline import *
import numpy as np

class MacdEmaStrategy(BaseStrategy):
    def calculate_indicators(self, df, config):
        df['ema_fast'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=200, adjust=False).mean()
        
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        df['atr'] = calculate_atr(df, 14)
        
        for col in ['ema_fast', 'ema_slow', 'macd', 'macd_signal', 'macd_hist', 'atr']:
            df[col] = df[col].shift(1)
            
        return df

    def calculate_signals(self, df, config):
        macd_cross_up = (df['macd_hist'] > 0) & (df['macd_hist'].shift(1) <= 0)
        macd_cross_down = (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) >= 0)
        
        df['long_trigger'] = (df['close'] > df['ema_slow']) & macd_cross_up & (df['macd'] < 0)
        df['short_trigger'] = (df['close'] < df['ema_slow']) & macd_cross_down & (df['macd'] > 0)
        
        df['exit_signal_long'] = macd_cross_down
        df['exit_signal_short'] = macd_cross_up
        return df

def test_macd():
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        file_path = f'data/processed/symbols_v3/{symbol}.parquet'
        try: df = pd.read_parquet(file_path)
        except: continue
        
        config = BacktestConfig(symbol=symbol, bb_period=20, sl_atr_mult=2.0)
        strategy = MacdEmaStrategy()
        df_copy = strategy.calculate_indicators(df, config)
        df_copy = strategy.calculate_signals(df_copy, config)
        
        split_idx = int(len(df_copy) * 0.7)
        df_is = df_copy.iloc[:split_idx].copy().reset_index(drop=True)
        df_oos = df_copy.iloc[split_idx:].copy().reset_index(drop=True)
        
        t_is, eq_is, curve_is = run_backtest_event_driven(df_is, config)
        metrics_is = calculate_metrics(t_is, curve_is)
        print(f'{symbol} IS -> Sharpe: {metrics_is["Sharpe Ratio"]:.2f}, DD: {metrics_is["Max Drawdown"]*100:.2f}%')
        
        t_oos, eq_oos, curve_oos = run_backtest_event_driven(df_oos, config)
        metrics_oos = calculate_metrics(t_oos, curve_oos)
        s = metrics_oos["Sharpe Ratio"]
        d = metrics_oos["Max Drawdown"]*100
        print(f'{symbol} OOS -> Sharpe: {s:.2f}, DD: {d:.2f}%')

test_macd()
