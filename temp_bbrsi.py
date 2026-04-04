import pandas as pd
from research_pipeline import *
import numpy as np

class BBRSIStrategy(BaseStrategy):
    def calculate_indicators(self, df, config):
        length = config.bb_period
        
        df['sma'] = df['close'].rolling(length).mean()
        std = df['close'].rolling(length).std()
        df['bb_upper'] = df['sma'] + (2.5 * std)  
        df['bb_lower'] = df['sma'] - (2.5 * std)  
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        df['atr'] = calculate_atr(df, 14)
        
        df['sma200'] = df['close'].rolling(200).mean()
        
        for col in ['sma', 'bb_upper', 'bb_lower', 'rsi', 'atr', 'sma200']:
            df[col] = df[col].shift(1)
            
        return df

    def calculate_signals(self, df, config):
        df['long_trigger'] = (df['close'] < df['bb_lower']) & (df['rsi'] < 30) & (df['close'] > df['sma200'])
        df['short_trigger'] = (df['close'] > df['bb_upper']) & (df['rsi'] > 70) & (df['close'] < df['sma200'])
        
        df['exit_signal_long'] = (df['close'] > df['sma'])
        df['exit_signal_short'] = (df['close'] < df['sma'])
        return df

def test_bbrsi():
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        file_path = f'data/processed/symbols_v3/{symbol}.parquet'
        try: df = pd.read_parquet(file_path)
        except: continue
        
        config = BacktestConfig(symbol=symbol, bb_period=20, sl_atr_mult=3.0)
        strategy = BBRSIStrategy()
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

test_bbrsi()
