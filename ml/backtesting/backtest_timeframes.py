#!/usr/bin/env python3
"""
Multi-Timeframe Backtester

Runs backtest for each timeframe using their respective trained models.
Compares performance across timeframes and generates reports.
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SUPPORTED_TIMEFRAMES, get_timeframe_config
from backtest_3stage import ThreeStageBacktester, BacktestConfig, BacktestResult, plot_equity_curve, plot_backtest_trades

# Paths
ML_DIR = Path(__file__).parent.parent
DATA_DIR = ML_DIR.parent / 'bitget-data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = ML_DIR / 'models'
RESULTS_DIR = ML_DIR / 'results'


class TimeframeBacktester(ThreeStageBacktester):
    """Backtester that loads models for specific timeframe"""
    
    def __init__(self, timeframe: str, config: BacktestConfig = None):
        self.timeframe = timeframe
        self.config = config or BacktestConfig()
        
        # Set model directory to timeframe-specific
        self.model_dir = MODELS_DIR / timeframe
        
        self.entry_model = None
        self.sl_model = None
        self.tp_model = None
        self.entry_scaler = None
        self.sl_scaler = None
        self.tp_scaler = None
        self.entry_features = None
        self.sl_features = None
        self.tp_features = None
        
        self._load_models()
    
    def _load_models(self):
        """Load models for this timeframe"""
        import joblib
        
        # Stage 1: Entry Filter
        entry_path = self.model_dir / 'entry_filter.joblib'
        if entry_path.exists():
            data = joblib.load(entry_path)
            self.entry_model = data['model']
            self.entry_scaler = data.get('scaler')
            self.entry_features = data['feature_names']
            print(f"✓ {self.timeframe} Entry Filter loaded")
        else:
            print(f"⚠️ {self.timeframe} Entry Filter not found")
        
        # Stage 2: SL Predictor
        sl_path = self.model_dir / 'sl_predictor.joblib'
        if sl_path.exists():
            data = joblib.load(sl_path)
            self.sl_model = data['model']
            self.sl_scaler = data.get('scaler')
            self.sl_features = data['feature_names']
            print(f"✓ {self.timeframe} SL Predictor loaded")
        else:
            print(f"⚠️ {self.timeframe} SL Predictor not found")
        
        # Stage 3: TP Predictor
        tp_path = self.model_dir / 'tp_predictor.joblib'
        if tp_path.exists():
            data = joblib.load(tp_path)
            self.tp_model = data['model']
            self.tp_scaler = data.get('scaler')
            self.tp_features = data['feature_names']
            self.tp_predict_rr = data.get('predict_rr', False)
            print(f"✓ {self.timeframe} TP Predictor loaded")
        else:
            print(f"⚠️ {self.timeframe} TP Predictor not found")


def run_timeframe_backtest(timeframe: str, config: BacktestConfig = None) -> BacktestResult:
    """Run backtest for a specific timeframe"""
    print(f"\n{'='*60}")
    print(f"Backtesting {timeframe}")
    print('='*60)
    
    # Load data
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return None
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows")
    
    # Filter for test set
    if config and (hasattr(config, 'start_date') and config.start_date or hasattr(config, 'end_date') and config.end_date):
        df_test = df.copy()
        if hasattr(config, 'start_date') and config.start_date:
            start_dt = pd.Timestamp(config.start_date)
            # Handle timezone naive/aware comparison
            if df_test['timestamp'].iloc[0].tz is not None and start_dt.tz is None:
                start_dt = start_dt.tz_localize('UTC')
            elif df_test['timestamp'].iloc[0].tz is None and start_dt.tz is not None:
                start_dt = start_dt.tz_localize(None)
            
            df_test = df_test[df_test['timestamp'] >= start_dt]
            
        if hasattr(config, 'end_date') and config.end_date:
            end_dt = pd.Timestamp(config.end_date)
            # Handle timezone naive/aware comparison
            if df_test['timestamp'].iloc[0].tz is not None and end_dt.tz is None:
                end_dt = end_dt.tz_localize('UTC')
            elif df_test['timestamp'].iloc[0].tz is None and end_dt.tz is not None:
                end_dt = end_dt.tz_localize(None)
                
            df_test = df_test[df_test['timestamp'] <= end_dt]
            
        if df_test.empty:
            print(f"⚠️ No data found in range. Dates: {config.start_date} to {config.end_date}")
            return None
    else:
        # Default: Use last 20% for testing
        test_start_idx = int(len(df) * 0.8)
        df_test = df.iloc[test_start_idx:].copy()
        
    print(f"Test period: {len(df_test):,} rows")
    
    # Get timeframe config
    tf_config = get_timeframe_config(timeframe)
    
    # Create backtest config
    if config is None:
        config = BacktestConfig(
            entry_threshold=tf_config.entry_threshold,
            leverage=tf_config.default_leverage,
            max_bars=tf_config.max_bars
        )
    
    # Run backtest
    backtester = TimeframeBacktester(timeframe, config)
    result = backtester.run_backtest(df_test, verbose=False)
    
    print(f"   Trades: {result.total_trades}, Win Rate: {result.win_rate:.1%}")
    print(f"   Return: {result.total_return:.1%}, Max DD: {result.max_drawdown:.1%}")
    print(f"   Sharpe: {result.sharpe_ratio:.2f}, PF: {result.profit_factor:.2f}")
    
    return result, df_test


def compare_timeframes(config: BacktestConfig = None) -> Dict[str, BacktestResult]:
    """Compare backtest results across all timeframes"""
    results = {}
    
    print("\n" + "="*70)
    print("MULTI-TIMEFRAME BACKTEST COMPARISON")
    print("="*70)
    
    for tf in SUPPORTED_TIMEFRAMES:
        # Check if models exist
        model_dir = MODELS_DIR / tf
        if not model_dir.exists():
            print(f"\n⚠️ Skipping {tf}: No models found")
            continue
        
        # Check if data exists
        data_path = PROCESSED_DIR / f'features_{tf}_full.parquet'
        if not data_path.exists():
            print(f"\n⚠️ Skipping {tf}: No data found")
            continue
        
        result, _ = run_timeframe_backtest(tf, config)
        if result:
            results[tf] = result
    
    if not results:
        print("No valid results!")
        return {}
    
    # Summary table
    print("\n" + "="*90)
    print("TIMEFRAME COMPARISON SUMMARY")
    print("="*90)
    print(f"\n{'Timeframe':<10} {'Trades':>8} {'Win%':>8} {'Return':>12} {'MaxDD':>10} {'Sharpe':>8} {'PF':>8}")
    print("-"*90)
    
    for tf in SUPPORTED_TIMEFRAMES:
        if tf not in results:
            continue
        res = results[tf]
        print(f"{tf:<10} {res.total_trades:>8} {res.win_rate:>7.1%} "
              f"{res.total_return:>11.1%} {res.max_drawdown:>9.1%} "
              f"{res.sharpe_ratio:>7.2f} {res.profit_factor:>7.2f}")
    
    return results


def plot_timeframe_comparison(results: Dict[str, BacktestResult], save_path: str = None):
    """Plot comparison across timeframes"""
    if not results:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    timeframes = list(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(timeframes)))
    
    # 1. Equity Curves
    ax1 = axes[0, 0]
    for i, (tf, res) in enumerate(results.items()):
        if res.equity_curve:
            ax1.plot(res.equity_curve, label=tf, linewidth=2, color=colors[i])
    ax1.set_title('Equity Curves by Timeframe', fontweight='bold')
    ax1.set_xlabel('Trade #')
    ax1.set_ylabel('Equity ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns Bar
    ax2 = axes[0, 1]
    returns = [results[tf].total_return * 100 for tf in timeframes]
    bars = ax2.bar(timeframes, returns, color=colors)
    ax2.set_title('Total Return by Timeframe', fontweight='bold')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, ret in zip(bars, returns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{ret:.0f}%', ha='center', fontweight='bold')
    
    # 3. Win Rate Bar
    ax3 = axes[0, 2]
    win_rates = [results[tf].win_rate * 100 for tf in timeframes]
    bars = ax3.bar(timeframes, win_rates, color=colors)
    ax3.set_title('Win Rate by Timeframe', fontweight='bold')
    ax3.set_ylabel('Win Rate (%)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    for bar, wr in zip(bars, win_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{wr:.1f}%', ha='center', fontweight='bold')
    
    # 4. Sharpe Ratio
    ax4 = axes[1, 0]
    sharpes = [results[tf].sharpe_ratio for tf in timeframes]
    bars = ax4.bar(timeframes, sharpes, color=colors)
    ax4.set_title('Sharpe Ratio by Timeframe', fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Good')
    ax4.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Excellent')
    for bar, sr in zip(bars, sharpes):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{sr:.2f}', ha='center', fontweight='bold')
    
    # 5. Max Drawdown
    ax5 = axes[1, 1]
    max_dds = [results[tf].max_drawdown * 100 for tf in timeframes]
    bars = ax5.bar(timeframes, max_dds, color=colors)
    ax5.set_title('Max Drawdown by Timeframe', fontweight='bold')
    ax5.set_ylabel('Max Drawdown (%)')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, dd in zip(bars, max_dds):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{dd:.1f}%', ha='center', fontweight='bold')
    
    # 6. Trade Count
    ax6 = axes[1, 2]
    trades = [results[tf].total_trades for tf in timeframes]
    bars = ax6.bar(timeframes, trades, color=colors)
    ax6.set_title('Trade Count by Timeframe', fontweight='bold')
    ax6.set_ylabel('Number of Trades')
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, trades):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{t}', ha='center', fontweight='bold')
    
    plt.suptitle('Multi-Timeframe Backtest Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Chart saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Multi-Timeframe Backtester')
    parser.add_argument('timeframe', nargs='?', default='all',
                       help='Timeframe to backtest (1h, 4h, 8h, 12h, 1d, all)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--fixed-size', action='store_true', help='Use fixed position size')
    parser.add_argument('--size-usd', type=float, default=1000, help='Fixed position size')
    parser.add_argument('--leverage', type=float, default=None, help='Override leverage')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Build config
    config = BacktestConfig(
        initial_capital=args.capital,
        fixed_position_size=args.fixed_size,
        position_size_usd=args.size_usd
    )
    
    if hasattr(args, 'start_date'):
        config.start_date = args.start_date
    if hasattr(args, 'end_date'):
        config.end_date = args.end_date
        
    if args.leverage:
        config.leverage = args.leverage
    
    if args.timeframe == 'all':
        results = compare_timeframes(config)
        if results:
            plot_timeframe_comparison(
                results, 
                save_path=str(ML_DIR.parent / 'backtest_timeframe_comparison.png')
            )
    else:
        result, df_test = run_timeframe_backtest(args.timeframe, config)
        if result:
            # Plot results
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Equity Curve
            plot_path = RESULTS_DIR / f'backtest_{args.timeframe}.png'
            plot_equity_curve({args.timeframe: result}, title=f"Backtest {args.timeframe}", save_path=str(plot_path))
            print(f"📊 Equity Chart saved to: {plot_path}")
            
            # Trade Setup Chart
            trade_plot_path = RESULTS_DIR / f'backtest_trades_{args.timeframe}.png'
            plot_backtest_trades(df_test, result.trades, title=f"Backtest Trades {args.timeframe}", save_path=str(trade_plot_path))
            print(f"🕯️ Trade Chart saved to: {trade_plot_path}")
            
            # Show summary
            print(f"Trades: {result.total_trades} | Return: {result.total_return:.1%} | Win Rate: {result.win_rate:.1%}")


if __name__ == '__main__':
    main()
