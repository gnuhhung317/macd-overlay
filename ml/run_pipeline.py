#!/usr/bin/env python3
"""
Main Pipeline: Run all ML training steps
"""
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data_pipeline import build_dataset, generate_labels, save_processed_data, OHLCV_DIR
from ml.backtester import MACDBacktester, run_full_backtest, analyze_by_regime, RESULTS_DIR
from ml.train_entry_filter import prepare_training_data, train_models, evaluate_model, save_model

import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    print("="*70)
    print("  MACD ML PIPELINE - Full Training")
    print("="*70)
    
    # ===== Step 1: Data Pipeline =====
    print("\n" + "="*70)
    print("  STEP 1: Data Pipeline - Load & Feature Engineering")
    print("="*70)
    
    # Get all symbols with enough data
    symbols = [f.stem.replace('_USDT', '') for f in OHLCV_DIR.glob('*.parquet')]
    # Filter quarterly
    symbols = [s for s in symbols if not any(x in s for x in ['-26', '-25', '-24'])]
    print(f"Found {len(symbols)} symbols")
    
    # Build dataset
    df = build_dataset(symbols, min_days=180)  # Minimum 6 months
    
    if df.empty:
        print("No data! Exiting.")
        return
    
    # Generate labels
    print("\nGenerating labels...")
    df = generate_labels(df, tp_pct=0.03, sl_pct=0.015, max_bars=10)
    
    # Save
    save_processed_data(df, 'features_1d_full.parquet')
    
    # ===== Step 2: Backtest Baseline =====
    print("\n" + "="*70)
    print("  STEP 2: Backtest - Baseline Performance")
    print("="*70)
    
    backtester = MACDBacktester(tp_pct=0.03, sl_pct=0.015, max_bars=10)
    
    all_trades = []
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        result = backtester.run(symbol_df, symbol)
        all_trades.extend(result.trades)
    
    final_result = backtester._calculate_results(all_trades)
    
    print(f"\nBaseline Results (TP=3%, SL=1.5%):")
    print(f"  Total Trades: {final_result.total_trades}")
    print(f"  Win Rate: {final_result.win_rate:.2%}")
    print(f"  Sharpe Ratio: {final_result.sharpe_ratio:.2f}")
    print(f"  Profit Factor: {final_result.profit_factor:.2f}")
    print(f"  Max Drawdown: {final_result.max_drawdown:.2%}")
    
    # ===== Step 3: TP/SL Optimization =====
    print("\n" + "="*70)
    print("  STEP 3: TP/SL Optimization Grid Search")
    print("="*70)
    
    grid_results = run_full_backtest(
        df,
        tp_range=[0.02, 0.03, 0.04, 0.05, 0.06],
        sl_range=[0.01, 0.015, 0.02, 0.025, 0.03]
    )
    
    RESULTS_DIR.mkdir(exist_ok=True)
    grid_results.to_csv(RESULTS_DIR / 'tp_sl_optimization_full.csv', index=False)
    
    # Best by Sharpe
    best = grid_results.loc[grid_results['sharpe_ratio'].idxmax()]
    print(f"\nBest Configuration (by Sharpe):")
    print(f"  TP: {best['tp_pct']:.1%}, SL: {best['sl_pct']:.1%}")
    print(f"  Win Rate: {best['win_rate']:.2%}")
    print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {best['profit_factor']:.2f}")
    
    # Best by Profit Factor
    best_pf = grid_results.loc[grid_results['profit_factor'].idxmax()]
    print(f"\nBest Configuration (by Profit Factor):")
    print(f"  TP: {best_pf['tp_pct']:.1%}, SL: {best_pf['sl_pct']:.1%}")
    print(f"  Win Rate: {best_pf['win_rate']:.2%}")
    print(f"  Profit Factor: {best_pf['profit_factor']:.2f}")
    
    # ===== Step 4: ML Entry Filter =====
    print("\n" + "="*70)
    print("  STEP 4: ML Entry Filter Training")
    print("="*70)
    
    # Prepare data
    X, y = prepare_training_data(df)
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Train
    results = train_models(X_train_scaled, y_train)
    
    # Evaluate
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    best_model = results[best_model_name]['model']
    
    eval_results = evaluate_model(best_model, X_test_scaled, y_test, best_model_name)
    
    # Save model
    save_model(best_model, scaler, list(X.columns), 'entry_filter_full.joblib')
    
    # ===== Step 5: Backtest with ML Filter =====
    print("\n" + "="*70)
    print("  STEP 5: Backtest with ML Entry Filter")
    print("="*70)
    
    # Get predictions
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Test different thresholds
    print("\nPerformance at different ML confidence thresholds:")
    print("-"*60)
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        # Filter entries by ML prediction
        mask = y_proba >= threshold
        filtered_trades = X_test[mask].index.tolist()
        
        if len(filtered_trades) == 0:
            continue
        
        # Calculate metrics on filtered trades
        filtered_y = y_test[mask]
        win_rate = filtered_y.mean()
        coverage = len(filtered_y) / len(y_test)
        
        print(f"Threshold {threshold:.0%}: Win Rate={win_rate:.2%}, Coverage={coverage:.2%}, Trades={len(filtered_y)}")
    
    # ===== Summary =====
    print("\n" + "="*70)
    print("  TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"""
Dataset:
  - Symbols: {df['symbol'].nunique()}
  - Total rows: {len(df)}
  - Crossovers analyzed: {len(X)}
  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}

Baseline Strategy (MACD Crossover):
  - Win Rate: {final_result.win_rate:.2%}
  - Sharpe Ratio: {final_result.sharpe_ratio:.2f}

Best TP/SL:
  - TP: {best['tp_pct']:.1%}, SL: {best['sl_pct']:.1%}
  - Sharpe: {best['sharpe_ratio']:.2f}

ML Entry Filter:
  - Best Model: {best_model_name}
  - Test ROC-AUC: {eval_results['roc_auc']:.4f}

Files saved:
  - data/processed/features_1d_full.parquet
  - ml/models/entry_filter_full.joblib
  - ml/results/tp_sl_optimization_full.csv
""")


if __name__ == '__main__':
    main()
