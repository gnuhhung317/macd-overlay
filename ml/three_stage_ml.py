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
    3-Stage ML Trading System (Decision Triad)
    
    Stage 1: Entry Filter (Classification)
        - Input: Market features at crossover
        - Output: Probability of good entry [0, 1]
        - Drivers: Overall market conditions
        
    Stage 2: SL Predictor (Regression)  
        - Input: Market features at entry
        - Output: Optimal SL percentage [0.5%, 10%]
        - Drivers: Volatility, noise, ATR
        
    Stage 3: TP Predictor (Regression)
        - Input: Market features at entry
        - Output: Optimal TP percentage [1%, 15%]
        - Drivers: Momentum, trend strength, volume
    
    Dynamic Risk-Reward:
        RR = TP / SL (computed per trade, not fixed!)
    """
    
    def __init__(
        self,
        entry_model_path: str = None,
        sl_model_path: str = None,
        tp_model_path: str = None,
        entry_threshold: float = 0.5,
        min_rr_ratio: float = 1.0,  # Minimum RR to take trade
        default_sl: float = 0.02,
        default_tp: float = 0.04
    ):
        # Stage 1: Entry Filter
        self.entry_model = None
        self.entry_scaler = None
        self.entry_features = None
        
        # Stage 2: SL Predictor
        self.sl_model = None
        self.sl_scaler = None
        self.sl_features = None
        
        # Stage 3: TP Predictor
        self.tp_model = None
        self.tp_scaler = None
        self.tp_features = None
        
        # Thresholds
        self.entry_threshold = entry_threshold
        self.min_rr_ratio = min_rr_ratio
        self.default_sl = default_sl
        self.default_tp = default_tp
        
        # Load models
        if entry_model_path:
            self.load_entry_model(entry_model_path)
        if sl_model_path:
            self.load_sl_model(sl_model_path)
        if tp_model_path:
            self.load_tp_model(tp_model_path)
    
    def load_entry_model(self, path: str):
        """Load Stage 1 Entry Filter model."""
        if not Path(path).exists():
            print(f"Entry model not found: {path}")
            return
        
        data = joblib.load(path)
        self.entry_model = data['model']
        self.entry_scaler = data.get('scaler')
        self.entry_features = data['feature_names']
        print(f"✓ Stage 1 - Entry Filter: {len(self.entry_features)} features")
    
    def load_sl_model(self, path: str):
        """Load Stage 2 SL Predictor model."""
        if not Path(path).exists():
            print(f"SL model not found: {path}")
            return
        
        data = joblib.load(path)
        self.sl_model = data['model']
        self.sl_scaler = data.get('scaler')
        self.sl_features = data['feature_names']
        print(f"✓ Stage 2 - SL Predictor: {len(self.sl_features)} features")
    
    def load_tp_model(self, path: str):
        """Load Stage 3 TP Predictor model."""
        if not Path(path).exists():
            print(f"TP model not found: {path}")
            return
        
        data = joblib.load(path)
        self.tp_model = data['model']
        self.tp_scaler = data.get('scaler')
        self.tp_features = data['feature_names']
        print(f"✓ Stage 3 - TP Predictor: {len(self.tp_features)} features")
    
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
        """Stage 1: Predict if entry is good."""
        if self.entry_model is None:
            return True, 0.5
        
        X = self._prepare_features(features, self.entry_features, self.entry_scaler)
        proba = self.entry_model.predict_proba(X)[0, 1]
        should_enter = proba >= self.entry_threshold
        
        return should_enter, proba
    
    def predict_sl(self, features: pd.DataFrame, min_sl: float = 0.005, max_sl: float = 0.10) -> float:
        """Stage 2: Predict optimal SL."""
        if self.sl_model is None:
            return self.default_sl
        
        X = self._prepare_features(features, self.sl_features, self.sl_scaler)
        sl = self.sl_model.predict(X)[0]
        return np.clip(sl, min_sl, max_sl)
    
    def predict_tp(self, features: pd.DataFrame, min_tp: float = 0.01, max_tp: float = 0.15) -> float:
        """Stage 3: Predict optimal TP."""
        if self.tp_model is None:
            return self.default_tp
        
        X = self._prepare_features(features, self.tp_features, self.tp_scaler)
        tp = self.tp_model.predict(X)[0]
        return np.clip(tp, min_tp, max_tp)
    
    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Full 3-stage prediction with dynamic Risk-Reward.
        
        Returns:
            {
                'should_enter': bool,
                'entry_confidence': float,
                'sl_pct': float,
                'tp_pct': float,
                'risk_reward': float,
                'filter_reason': str  # Why trade was rejected
            }
        """
        result = {
            'should_enter': False,
            'entry_confidence': 0.0,
            'sl_pct': self.default_sl,
            'tp_pct': self.default_tp,
            'risk_reward': self.default_tp / self.default_sl,
            'filter_reason': ''
        }
        
        # Stage 1: Entry Filter
        should_enter, confidence = self.predict_entry(features)
        result['entry_confidence'] = confidence
        
        if not should_enter:
            result['filter_reason'] = f'Low confidence ({confidence:.2%} < {self.entry_threshold:.0%})'
            return result
        
        # Stage 2: SL Prediction
        sl_pct = self.predict_sl(features)
        result['sl_pct'] = sl_pct
        
        # Stage 3: TP Prediction
        tp_pct = self.predict_tp(features)
        result['tp_pct'] = tp_pct
        
        # Calculate dynamic Risk-Reward
        rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0
        result['risk_reward'] = rr_ratio
        
        # Final decision: Check if RR is acceptable
        if rr_ratio < self.min_rr_ratio:
            result['filter_reason'] = f'Poor RR ({rr_ratio:.2f} < {self.min_rr_ratio:.1f})'
            return result
        
        result['should_enter'] = True
        return result
    
    def get_position_size(self, base_size: float, risk_reward: float, confidence: float) -> float:
        """
        Calculate position size based on RR and confidence.
        Higher RR and confidence = larger position
        """
        # Base multiplier from RR (capped at 2x)
        rr_multiplier = min(risk_reward / 2.0, 2.0)
        
        # Confidence multiplier (0.5x to 1.5x)
        conf_multiplier = 0.5 + confidence
        
        return base_size * rr_multiplier * conf_multiplier
    
    def set_entry_threshold(self, threshold: float):
        self.entry_threshold = threshold
    
    def set_min_rr(self, min_rr: float):
        self.min_rr_ratio = min_rr


# Backwards compatibility alias
TwoStageMLSystem = ThreeStageMLSystem


def backtest_3stage_ml(
    df: pd.DataFrame,
    system: ThreeStageMLSystem,
    initial_capital: float = 10000,
    position_size: float = 0.1,  # 10% of capital per trade
    max_bars: int = 10,
    dynamic_sizing: bool = False  # Use dynamic position sizing
) -> Dict:
    """
    Backtest 3-stage ML system with dynamic TP/SL.
    """
    results = []
    capital = initial_capital
    
    # Get crossover rows
    crossovers = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    
    for idx in crossovers.index:
        row = df.loc[idx]
        row_df = df.loc[[idx]]
        
        # Get 3-stage prediction
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
            'risk_reward': pred['risk_reward'],
            'filter_reason': pred.get('filter_reason', '')
        }
        
        if not pred['should_enter']:
            trade['result'] = 'FILTERED'
            trade['pnl'] = 0
            trade['pnl_pct'] = 0
            results.append(trade)
            continue
        
        # Calculate TP/SL prices using ML predictions
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
            if pos_in_df + max_bars < len(df):
                exit_price = df.iloc[pos_in_df + max_bars]['close']
        
        # Calculate PnL
        if is_long:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        # Position sizing
        if dynamic_sizing:
            trade_size = system.get_position_size(
                position_size, pred['risk_reward'], pred['entry_confidence']
            )
        else:
            trade_size = position_size
        
        trade_capital = capital * trade_size
        pnl = trade_capital * pnl_pct
        capital += pnl
        
        trade['exit_price'] = exit_price
        trade['result'] = exit_reason
        trade['bars_held'] = bars_held
        trade['pnl'] = pnl
        trade['pnl_pct'] = pnl_pct
        trade['position_size'] = trade_size
        trade['capital_after'] = capital
        
        results.append(trade)
    
    # Calculate metrics
    df_results = pd.DataFrame(results)
    
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
    
    # Separate filter reasons
    filtered = df_results[df_results['result'] == 'FILTERED']
    low_conf = filtered[filtered['filter_reason'].str.contains('confidence', case=False, na=False)]
    poor_rr = filtered[filtered['filter_reason'].str.contains('RR', case=False, na=False)]
    
    metrics = {
        'trades': df_results,
        'total_signals': len(df_results),
        'filtered_trades': len(filtered),
        'filtered_low_confidence': len(low_conf),
        'filtered_poor_rr': len(poor_rr),
        'executed_trades': len(executed),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(executed) if len(executed) > 0 else 0,
        'total_pnl': executed['pnl'].sum(),
        'total_pnl_pct': executed['pnl_pct'].sum(),
        'avg_pnl_pct': executed['pnl_pct'].mean(),
        'avg_win_pct': wins['pnl_pct'].mean() if len(wins) > 0 else 0,
        'avg_loss_pct': losses['pnl_pct'].mean() if len(losses) > 0 else 0,
        'avg_rr': executed['risk_reward'].mean(),
        'avg_sl_pct': executed['predicted_sl_pct'].mean(),
        'avg_tp_pct': executed['predicted_tp_pct'].mean(),
        'final_capital': capital,
        'return_pct': (capital - initial_capital) / initial_capital,
        'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
    }
    
    return metrics


def compare_strategies(df: pd.DataFrame) -> Dict:
    """
    Compare 4 strategies:
    1. Baseline: No ML, fixed 3% TP / 1.5% SL
    2. ML Entry Only: Entry filter + fixed TP/SL  
    3. ML Entry + Dynamic SL: Entry filter + SL predictor + fixed TP ratio
    4. Full 3-Stage ML: Entry filter + SL predictor + TP predictor
    """
    results = {}
    
    # Strategy 1: Baseline (no ML)
    print("\n" + "="*60)
    print("Strategy 1: Baseline (No ML, TP=3%, SL=1.5%)")
    print("="*60)
    
    baseline_system = ThreeStageMLSystem(
        entry_threshold=0.0,  # Accept all
        min_rr_ratio=0.0,
        default_sl=0.015,
        default_tp=0.03
    )
    results['baseline'] = backtest_3stage_ml(df, baseline_system)
    print_backtest_summary(results['baseline'], "Baseline")
    
    # Strategy 2: ML Entry Only
    print("\n" + "="*60)
    print("Strategy 2: ML Entry Filter + Fixed TP/SL")
    print("="*60)
    
    entry_only_system = ThreeStageMLSystem(
        entry_model_path=str(MODEL_DIR / 'entry_filter.joblib'),
        entry_threshold=0.5,
        min_rr_ratio=0.0,
        default_sl=0.015,
        default_tp=0.03
    )
    results['ml_entry'] = backtest_3stage_ml(df, entry_only_system)
    print_backtest_summary(results['ml_entry'], "ML Entry")
    
        # Strategy 2: ML Entry Only
    print("\n" + "="*60)
    print("Strategy 2.1: ML Entry Filter + Fixed TP/SL")
    print("="*60)
    
    entry_only_system = ThreeStageMLSystem(
        entry_model_path=str(MODEL_DIR / 'entry_filter.joblib'),
        entry_threshold=0.4,
        min_rr_ratio=2,
        default_sl=0.1,
        default_tp=0.05
    )
    results['ml_entry'] = backtest_3stage_ml(df, entry_only_system)
    print_backtest_summary(results['ml_entry'], "ML Entry")


    # Strategy 3: ML Entry + Dynamic SL
    print("\n" + "="*60)
    print("Strategy 3: ML Entry + Dynamic SL (Fixed 2:1 RR)")
    print("="*60)
    
    entry_sl_system = ThreeStageMLSystem(
        entry_model_path=str(MODEL_DIR / 'entry_filter.joblib'),
        sl_model_path=str(MODEL_DIR / 'sl_predictor.joblib'),
        entry_threshold=0.5,
        min_rr_ratio=0.0,
        default_tp=0.06  # Will be overridden by 2x SL
    )
    # For this strategy, TP = 2 * SL
    results['ml_entry_sl'] = backtest_3stage_ml(df, entry_sl_system)
    print_backtest_summary(results['ml_entry_sl'], "ML Entry + SL")
    
    # Strategy 4: Full 3-Stage ML
    print("\n" + "="*60)
    print("Strategy 4: Full 3-Stage ML (Entry + SL + TP)")
    print("="*60)
    
    full_system = ThreeStageMLSystem(
        entry_model_path=str(MODEL_DIR / 'entry_filter.joblib'),
        sl_model_path=str(MODEL_DIR / 'sl_predictor.joblib'),
        tp_model_path=str(MODEL_DIR / 'tp_predictor.joblib'),
        entry_threshold=0.5,
        min_rr_ratio=1.0  # Require RR >= 1
    )
    results['full_3stage'] = backtest_3stage_ml(df, full_system)
    print_backtest_summary(results['full_3stage'], "Full 3-Stage ML")
    
    # Strategy 5: Full 3-Stage ML with Dynamic Sizing
    print("\n" + "="*60)
    print("Strategy 5: Full 3-Stage ML + Dynamic Position Sizing")
    print("="*60)
    
    results['full_3stage_dynamic'] = backtest_3stage_ml(df, full_system, dynamic_sizing=True)
    print_backtest_summary(results['full_3stage_dynamic'], "3-Stage + Dynamic Size")
    
    # Comparison table
    print("\n" + "="*70)
    print("Strategy Comparison")
    print("="*70)
    print(f"{'Strategy':<30} {'Trades':>7} {'Win%':>7} {'Avg RR':>7} {'PnL%':>8} {'PF':>6}")
    print("-"*70)
    
    for name, res in results.items():
        if res.get('executed_trades', 0) > 0:
            avg_rr = res.get('avg_rr', 2.0)
            print(f"{name:<30} {res['executed_trades']:>7} {res['win_rate']:>6.1%} {avg_rr:>7.2f} {res['total_pnl_pct']:>7.1%} {res['profit_factor']:>6.2f}")
    
    return results


def print_backtest_summary(metrics: Dict, name: str = ''):
    """Print backtest summary."""
    if metrics.get('executed_trades', 0) == 0:
        print(f"No trades executed!")
        return
    
    print(f"\n{name} Results:")
    print(f"  Total signals: {metrics['total_signals']}")
    print(f"  Filtered: {metrics['filtered_trades']} ({metrics['filtered_trades']/metrics['total_signals']:.1%})")
    if 'filtered_low_confidence' in metrics:
        print(f"    - Low confidence: {metrics['filtered_low_confidence']}")
        print(f"    - Poor RR: {metrics['filtered_poor_rr']}")
    print(f"  Executed: {metrics['executed_trades']}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Avg RR: {metrics.get('avg_rr', 0):.2f}")
    print(f"  Avg SL: {metrics.get('avg_sl_pct', 0):.2%}")
    print(f"  Avg TP: {metrics.get('avg_tp_pct', 0):.2%}")
    print(f"  Total PnL: {metrics['total_pnl_pct']:.2%}")
    print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"  Final Capital: ${metrics['final_capital']:,.2f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='3-Stage ML Trading System')
    parser.add_argument('--data', type=str, default=None, help='Path to data')
    parser.add_argument('--threshold', type=float, default=0.5, help='Entry threshold')
    parser.add_argument('--min-rr', type=float, default=1.0, help='Minimum RR ratio')
    
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
    
    # Compare all strategies
    results = compare_strategies(df)
