#!/usr/bin/env python3
"""
3-Stage ML Trading System (Decision Triad)

Stage 1: Entry Filter - Predicts if crossover is good entry (Classification)
Stage 2: SL Predictor - Predicts optimal stop loss level (Regression)
Stage 3: TP Predictor - Predicts optimal take profit level (Regression)

The 3 stages work together to:
1. Filter bad entries (Stage 1)
2. Calculate dynamic Risk (SL) based on volatility (Stage 2)
3. Calculate dynamic Reward (TP) based on momentum (Stage 3)
4. Compute dynamic Risk-Reward Ratio for position sizing
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib

MODEL_DIR = Path(__file__).parent / 'models'


class ThreeStageMLSystem:
    """
    2-Stage ML Trading System
    
    Stage 1: Entry Filter (Classification)
        - Input: Market features at crossover
        - Output: Probability of good entry [0, 1]
        
    Stage 2: SL Predictor (Regression)  
        - Input: Market features at entry
        - Output: Optimal SL percentage [0.5%, 10%]
    """
    
    def __init__(
        self,
        entry_model_path: str = None,
        sl_model_path: str = None,
        entry_threshold: float = 0.5,
        default_sl: float = 0.02,
        default_tp_ratio: float = 2.0  # TP = SL * ratio
    ):
        self.entry_model = None
        self.entry_scaler = None
        self.entry_features = None
        
        self.sl_model = None
        self.sl_scaler = None
        self.sl_features = None
        
        self.entry_threshold = entry_threshold
        self.default_sl = default_sl
        self.default_tp_ratio = default_tp_ratio
        
        # Load models
        if entry_model_path:
            self.load_entry_model(entry_model_path)
        if sl_model_path:
            self.load_sl_model(sl_model_path)
    
    def load_entry_model(self, path: str):
        """Load Stage 1 Entry Filter model."""
        if not Path(path).exists():
            print(f"Entry model not found: {path}")
            return
        
        data = joblib.load(path)
        self.entry_model = data['model']
        self.entry_scaler = data.get('scaler')
        self.entry_features = data['feature_names']
        print(f"✓ Loaded Entry Filter: {len(self.entry_features)} features")
    
    def load_sl_model(self, path: str):
        """Load Stage 2 SL Predictor model."""
        if not Path(path).exists():
            print(f"SL model not found: {path}")
            return
        
        data = joblib.load(path)
        self.sl_model = data['model']
        self.sl_scaler = data.get('scaler')
        self.sl_features = data['feature_names']
        print(f"✓ Loaded SL Predictor: {len(self.sl_features)} features")
    
    def _prepare_features(self, features: pd.DataFrame, feature_names: list, scaler) -> np.ndarray:
        """Prepare features for prediction."""
        X = pd.DataFrame()
        for col in feature_names:
            if col in features.columns:
                X[col] = features[col].values if hasattr(features[col], 'values') else [features[col]]
            else:
                X[col] = 0
        
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        if scaler is not None:
            X = scaler.transform(X)
        
        return X
    
    def predict_entry(self, features: pd.DataFrame) -> Tuple[bool, float]:
        """
        Stage 1: Predict if entry is good.
        
        Returns:
            (should_enter, confidence)
        """
        if self.entry_model is None:
            return True, 0.5
        
        X = self._prepare_features(features, self.entry_features, self.entry_scaler)
        proba = self.entry_model.predict_proba(X)[0, 1]
        should_enter = proba >= self.entry_threshold
        
        return should_enter, proba
    
    def predict_sl(self, features: pd.DataFrame, min_sl: float = 0.005, max_sl: float = 0.10) -> float:
        """
        Stage 2: Predict optimal SL.
        
        Returns:
            SL as decimal (e.g., 0.02 for 2%)
        """
        if self.sl_model is None:
            return self.default_sl
        
        X = self._prepare_features(features, self.sl_features, self.sl_scaler)
        sl = self.sl_model.predict(X)[0]
        return np.clip(sl, min_sl, max_sl)
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Full 2-stage prediction.
        
        Returns:
            {
                'should_enter': bool,
                'entry_confidence': float,
                'sl_pct': float,
                'tp_pct': float,
                'risk_reward': float
            }
        """
        # Stage 1: Entry
        should_enter, confidence = self.predict_entry(features)
        
        # Stage 2: SL (only if entering)
        if should_enter:
            sl_pct = self.predict_sl(features)
        else:
            sl_pct = self.default_sl
        
        # Calculate TP based on SL
        tp_pct = sl_pct * self.default_tp_ratio
        
        return {
            'should_enter': should_enter,
            'entry_confidence': confidence,
            'sl_pct': sl_pct,
            'tp_pct': tp_pct,
            'risk_reward': self.default_tp_ratio
        }
    
    def set_entry_threshold(self, threshold: float):
        """Set confidence threshold for entry."""
        self.entry_threshold = threshold
    
    def set_tp_ratio(self, ratio: float):
        """Set TP/SL ratio."""
        self.default_tp_ratio = ratio


def backtest_2stage_ml(
    df: pd.DataFrame,
    system: TwoStageMLSystem,
    initial_capital: float = 10000,
    position_size: float = 0.1,  # 10% of capital per trade
    max_bars: int = 10
) -> Dict:
    """
    Backtest 2-stage ML system.
    
    Returns detailed results per trade and overall metrics.
    """
    results = []
    capital = initial_capital
    
    # Get crossover rows
    crossovers = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    
    for idx in crossovers.index:
        row = df.loc[idx]
        row_df = df.loc[[idx]]
        
        # Get prediction
        pred = system.predict(row_df)
        
        entry_price = row['close']
        is_long = row['macd_cross_up'] == 1
        
        # Record trade
        trade = {
            'timestamp': row['timestamp'] if 'timestamp' in row else idx,
            'symbol': row['symbol'] if 'symbol' in row else 'UNKNOWN',
            'is_long': is_long,
            'entry_price': entry_price,
            'should_enter': pred['should_enter'],
            'entry_confidence': pred['entry_confidence'],
            'predicted_sl_pct': pred['sl_pct'],
            'predicted_tp_pct': pred['tp_pct'],
        }
        
        if not pred['should_enter']:
            trade['result'] = 'FILTERED'
            trade['pnl'] = 0
            trade['pnl_pct'] = 0
            results.append(trade)
            continue
        
        # Calculate TP/SL prices
        if is_long:
            tp_price = entry_price * (1 + pred['tp_pct'])
            sl_price = entry_price * (1 - pred['sl_pct'])
        else:
            tp_price = entry_price * (1 - pred['tp_pct'])
            sl_price = entry_price * (1 + pred['sl_pct'])
        
        trade['tp_price'] = tp_price
        trade['sl_price'] = sl_price
        
        # Simulate trade
        pos_in_df = df.index.get_loc(idx)
        exit_price = entry_price
        exit_reason = 'TIMEOUT'
        bars_held = max_bars
        
        for j in range(1, min(max_bars + 1, len(df) - pos_in_df)):
            future_idx = df.index[pos_in_df + j]
            future_high = df.loc[future_idx, 'high']
            future_low = df.loc[future_idx, 'low']
            
            if is_long:
                if future_high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP_HIT'
                    bars_held = j
                    break
                elif future_low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL_HIT'
                    bars_held = j
                    break
            else:
                if future_low <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP_HIT'
                    bars_held = j
                    break
                elif future_high >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL_HIT'
                    bars_held = j
                    break
        else:
            # Timeout - exit at last close
            if pos_in_df + max_bars < len(df):
                exit_price = df.iloc[pos_in_df + max_bars]['close']
        
        # Calculate PnL
        if is_long:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        trade_capital = capital * position_size
        pnl = trade_capital * pnl_pct
        capital += pnl
        
        trade['exit_price'] = exit_price
        trade['result'] = exit_reason
        trade['bars_held'] = bars_held
        trade['pnl'] = pnl
        trade['pnl_pct'] = pnl_pct
        trade['capital_after'] = capital
        
        results.append(trade)
    
    # Calculate metrics
    df_results = pd.DataFrame(results)
    
    # Filter executed trades
    executed = df_results[df_results['result'] != 'FILTERED']
    
    if len(executed) == 0:
        return {
            'trades': df_results,
            'total_trades': 0,
            'filtered_trades': len(df_results),
            'executed_trades': 0
        }
    
    wins = executed[executed['pnl'] > 0]
    losses = executed[executed['pnl'] <= 0]
    
    metrics = {
        'trades': df_results,
        'total_signals': len(df_results),
        'filtered_trades': len(df_results[df_results['result'] == 'FILTERED']),
        'executed_trades': len(executed),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(executed) if len(executed) > 0 else 0,
        'total_pnl': executed['pnl'].sum(),
        'total_pnl_pct': executed['pnl_pct'].sum(),
        'avg_pnl_pct': executed['pnl_pct'].mean(),
        'avg_win_pct': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
        'avg_loss_pct': losses['pnl_pct'].mean() if len(losses) > 0 else 0,
        'final_capital': capital,
        'return_pct': (capital - initial_capital) / initial_capital,
        'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
    }
    
    # SL prediction accuracy
    if 'predicted_sl_pct' in executed.columns and 'max_adverse_excursion' in df.columns:
        executed_with_mae = executed.merge(
            df[['timestamp', 'max_adverse_excursion']].dropna(),
            on='timestamp',
            how='left'
        )
        if 'max_adverse_excursion' in executed_with_mae.columns:
            valid_mae = executed_with_mae.dropna(subset=['max_adverse_excursion'])
            if len(valid_mae) > 0:
                sl_error = (valid_mae['predicted_sl_pct'] - valid_mae['max_adverse_excursion']).abs().mean()
                metrics['sl_prediction_mae'] = sl_error
    
    return metrics


def compare_strategies(df: pd.DataFrame, system: TwoStageMLSystem) -> Dict:
    """
    Compare 3 strategies:
    1. Baseline: No ML, fixed 3% TP / 1.5% SL
    2. ML Entry Only: Entry filter + fixed TP/SL  
    3. Full 2-Stage ML: Entry filter + dynamic SL
    """
    results = {}
    
    # Strategy 1: Baseline (no ML)
    print("\n" + "="*60)
    print("Strategy 1: Baseline (No ML, TP=3%, SL=1.5%)")
    print("="*60)
    
    baseline_system = TwoStageMLSystem(
        entry_threshold=0.0,  # Accept all
        default_sl=0.015,
        default_tp_ratio=2.0  # 3% TP
    )
    results['baseline'] = backtest_2stage_ml(df, baseline_system)
    print_backtest_summary(results['baseline'], "Baseline")
    
    # Strategy 2: ML Entry Only
    print("\n" + "="*60)
    print("Strategy 2: ML Entry Filter + Fixed TP/SL")
    print("="*60)
    
    entry_only_system = TwoStageMLSystem(
        entry_model_path=str(MODEL_DIR / 'entry_filter.joblib'),
        entry_threshold=0.5,
        default_sl=0.015,
        default_tp_ratio=2.0
    )
    results['ml_entry'] = backtest_2stage_ml(df, entry_only_system)
    print_backtest_summary(results['ml_entry'], "ML Entry")
    
    # Strategy 3: Full 2-Stage ML
    print("\n" + "="*60)
    print("Strategy 3: Full 2-Stage ML (Entry + Dynamic SL)")
    print("="*60)
    
    full_system = TwoStageMLSystem(
        entry_model_path=str(MODEL_DIR / 'entry_filter.joblib'),
        sl_model_path=str(MODEL_DIR / 'sl_predictor.joblib'),
        entry_threshold=0.5,
        default_tp_ratio=2.0
    )
    results['full_ml'] = backtest_2stage_ml(df, full_system)
    print_backtest_summary(results['full_ml'], "Full 2-Stage ML")
    
    # Comparison table
    print("\n" + "="*60)
    print("Strategy Comparison")
    print("="*60)
    print(f"{'Strategy':<25} {'Trades':>8} {'Win%':>8} {'Total PnL':>12} {'Profit Factor':>14}")
    print("-"*70)
    
    for name, res in results.items():
        if res.get('executed_trades', 0) > 0:
            print(f"{name:<25} {res['executed_trades']:>8} {res['win_rate']:>7.1%} {res['total_pnl_pct']:>11.1%} {res['profit_factor']:>14.2f}")
    
    return results


def print_backtest_summary(metrics: Dict, name: str = ''):
    """Print backtest summary."""
    if metrics.get('executed_trades', 0) == 0:
        print(f"No trades executed!")
        return
    
    print(f"\n{name} Results:")
    print(f"  Total signals: {metrics['total_signals']}")
    print(f"  Filtered: {metrics['filtered_trades']} ({metrics['filtered_trades']/metrics['total_signals']:.1%})")
    print(f"  Executed: {metrics['executed_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Total PnL: {metrics['total_pnl_pct']:.2%}")
    print(f"  Avg PnL/Trade: {metrics['avg_pnl_pct']:.2%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Final Capital: ${metrics['final_capital']:,.2f}")


if __name__ == '__main__':
    from pathlib import Path
    import argparse
    
    parser = argparse.ArgumentParser(description='2-Stage ML Trading System')
    parser.add_argument('--data', type=str, default=None, help='Path to data')
    parser.add_argument('--threshold', type=float, default=0.5, help='Entry threshold')
    
    args = parser.parse_args()
    
    DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
    
    data_path = args.data
    if data_path is None:
        data_path = DATA_DIR / 'features_1d_full.parquet'
        if not data_path.exists():
            data_path = DATA_DIR / 'features_1d_test.parquet'
    
    if not Path(data_path).exists():
        print(f"Data not found: {data_path}")
        exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Initialize 2-stage system
    system = TwoStageMLSystem(
        entry_model_path=str(MODEL_DIR / 'entry_filter.joblib'),
        sl_model_path=str(MODEL_DIR / 'sl_predictor.joblib'),
        entry_threshold=args.threshold
    )
    
    # Compare strategies
    results = compare_strategies(df, system)
