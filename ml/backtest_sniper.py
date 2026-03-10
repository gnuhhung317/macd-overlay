import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import joblib
import matplotlib.pyplot as plt

@dataclass
class BacktestConfig:
    initial_capital: float = 1000.0
    leverage: float = 5.0
    risk_per_trade: float = 0.02 
    max_open_trades: int = 5
    fee_rate: float = 0.0006
    sl_atr: float = 2.0
    tp_atr: float = 4.0
    horizon: int = 48
    # Đường dẫn file của bạn
    data_path: str = r"data\processed\features_1h_btc_context.parquet"

class SniperChronoBacktester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.balance = config.initial_capital
        self.positions = []
        self.trade_history = []
        self._load_models()

    def _load_models(self):
        base_ml = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy\ml\training\models\1h")
        self.meta = joblib.load(base_ml / "ensemble_meta.joblib")
        self.clf = joblib.load(base_ml / "ensemble_lgbm_tabular.joblib")
        self.features = self.meta.get('features', [])

    def prepare_features(self, df):
        """Tính toán tất cả features mà Model Sniper yêu cầu"""
        df = df.copy()
        
        # Đảm bảo tính toán các cột mà logic Scanner v2.0 yêu cầu
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # ATR (14) - Cực kỳ quan trọng cho SL/TP và Features
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = df['tr'].rolling(14).mean()
        
        # Các cột bị thiếu gây ra lỗi KeyError
        df['dist_to_ema50_atr'] = (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
        df['vol_acceleration'] = df['volume'] / (df['volume'].shift(1) + 1e-9)
        
        # Các filter khác từ Scanner
        vol_sma = df['volume'].rolling(20).mean().shift(1)
        df['vol_ratio'] = df['volume'] / (vol_sma + 1e-9)
        
        return df, vol_sma

    def get_signals(self, df, vol_sma):
        """Logic bóp cò dựa trên Ignition Bar + AI Scoring"""
        # Tầng 1: Ignition Filter (Logic Scanner v2.0)
        c1 = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
        c2 = ((df['close'] - df['open']) / df['open']) > 0.015
        c3 = (df['volume'] > vol_sma * 1.5) & (df['volume'] < vol_sma * 4.0)
        c4 = (df['rsi_14'] >= 55) & (df['rsi_14'] <= 72)
        
        ignition_mask = c1 & c2 & c3 & c4
        signals = pd.Series('WAIT', index=df.index)
        
        if not ignition_mask.any():
            return signals

        # Tầng 2: AI Scoring - Chỉ chạy trên các nến thỏa mãn Ignition
        # Đảm bảo CHỈ lấy các cột features mà model yêu cầu
        X = df.loc[ignition_mask, self.features].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        probas = self.clf.predict_proba(X)
        
        # Map kết quả trở lại signals
        # Giả sử Model: 0: WAIT, 1: LONG, 2: SHORT
        results = np.full(len(X), 'WAIT', dtype=object)
        results[probas[:, 1] > 0.6] = 'LONG'
        results[probas[:, 2] > 0.6] = 'SHORT'
        
        signals.loc[ignition_mask] = results
        return signals

    def run(self, df):
        df, vol_sma = self.prepare_features(df)
        signals = self.get_signals(df, vol_sma)
        
        equity_curve = []
        print(f"🚀 Sniper Backtesting: {len(df)} bars...")

        for i in range(len(df)):
            row = df.iloc[i]
            
            # 1. Update & Check Exit
            self._update_positions(row)

            # 2. Check Entry
            sig = signals.iloc[i]
            if sig != 'WAIT' and len(self.positions) < self.config.max_open_trades:
                self._open_position(row, sig)

            equity_curve.append({'timestamp': row['timestamp'], 'balance': self.balance})

        return pd.DataFrame(equity_curve), pd.DataFrame(self.trade_history)

    def _open_position(self, row, side):
        entry_price = row['close']
        atr = row['atr_14']
        if np.isnan(atr): atr = entry_price * 0.02 # Fallback
        
        risk_amt = self.balance * self.config.risk_per_trade
        sl_dist = self.config.sl_atr * atr
        
        # Tính Size dựa trên khoảng cách SL để khống chế rủi ro
        pos_size_usd = (risk_amt / sl_dist) * entry_price
        margin_req = pos_size_usd / self.config.leverage
        
        # Giới hạn margin không vượt quá balance
        if margin_req > self.balance * 0.9:
            margin_req = self.balance * 0.9
            pos_size_usd = margin_req * self.config.leverage

        self.balance -= (pos_size_usd * self.config.fee_rate) # Trừ phí entry
        
        sl_price = entry_price - sl_dist if side == 'LONG' else entry_price + sl_dist
        tp_price = entry_price + (self.config.tp_atr * atr) if side == 'LONG' else entry_price - (self.config.tp_atr * atr)

        self.positions.append({
            'side': side,
            'entry_price': entry_price,
            'size_usd': pos_size_usd,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'entry_time': row['timestamp'],
            'bars': 0
        })

    def _update_positions(self, row):
        for pos in self.positions[:]:
            pos['bars'] += 1
            exit_price = 0
            reason = ""

            # Check SL-First
            if pos['side'] == 'LONG':
                if row['low'] <= pos['sl_price']:
                    exit_price, reason = pos['sl_price'], "SL"
                elif row['high'] >= pos['tp_price']:
                    exit_price, reason = pos['tp_price'], "TP"
            else: # SHORT
                if row['high'] >= pos['sl_price']:
                    exit_price, reason = pos['sl_price'], "SL"
                elif row['low'] <= pos['tp_price']:
                    exit_price, reason = pos['tp_price'], "TP"

            if not reason and pos['bars'] >= self.config.horizon:
                exit_price, reason = row['close'], "TIMEOUT"

            if reason:
                pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] if pos['side'] == 'LONG' else (pos['entry_price'] - exit_price) / pos['entry_price']
                pnl_usd = pos['size_usd'] * pnl_pct
                fee = pos['size_usd'] * (1 + pnl_pct) * self.config.fee_rate
                
                self.balance += pnl_usd - fee
                self.trade_history.append({
                    'time': row['timestamp'],
                    'side': pos['side'],
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd - fee,
                    'reason': reason
                })
                self.positions.remove(pos)

if __name__ == "__main__":
    config = BacktestConfig()
    df = pd.read_parquet(config.data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    # Chỉ test data mới từ 2025
    df_test = df[df['timestamp'] >= '2025-01-01'].copy()
    
    bt = SniperChronoBacktester(config)
    eq_curve, trades = bt.run(df_test)
    
    if not trades.empty:
        print(f"\nWin Rate: {(trades['pnl_usd'] > 0).mean():.2%}")
        print(f"Final Balance: ${bt.balance:.2f}")
        eq_curve.plot(x='timestamp', y='balance')
        plt.show()