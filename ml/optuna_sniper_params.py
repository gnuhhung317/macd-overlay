import optuna
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import math

# Add current directory to sys.path to allow imports from ml package
sys.path.append(os.getcwd())

# Import from the backtest core
from ml.backtest_sniper import (
    BacktestConfig, Trade, TradeState, 
    load_assets, backtest_symbol, run_portfolio_simulation
)

# Configuration for Optuna
START_DATE = '2025-01-01'
INITIAL_CAPITAL = 100.0
RISK_PER_TRADE = 0.05
MAX_OPEN_TRADES = 10
LEVERAGE = 20.0

def precompute_data():
    """Scan all symbols once and collect potential signals and price data."""
    print("🚀 Precomputing signals and price data (Stage 1 & 2)...")
    config = BacktestConfig(start_date=START_DATE)
    clf, features, threshold = load_assets()
    
    if clf is None:
        print("❌ Error: Could not load assets.")
        sys.exit(1)
        
    base_dir = Path(os.getcwd())
    symbols_dir = base_dir / "data" / "processed" / "symbols_v3"
    all_files = list(symbols_dir.glob("*.parquet"))
    
    all_signals = []
    full_price_db = {}
    
    for i, file_path in enumerate(all_files):
        if i % 100 == 0:
            print(f"Scanning {i}/{len(all_files)} symbols...")
        res, ohlcv = backtest_symbol(file_path, features, clf, threshold, config)
        if res:
            all_signals.extend(res)
            full_price_db[Path(file_path).stem.replace('_USDT', '').replace('USDT', '')] = ohlcv
            
    print(f"✅ Precomputation complete. Found {len(all_signals)} potential signals.")
    return all_signals, full_price_db

def objective(trial, signals, full_price_db, config_base):
    # Search Space
    params = {
        'long_atr_offset': trial.suggest_float('long_atr_offset', -1.2, 0.2),
        'short_atr_offset': trial.suggest_float('short_atr_offset', -0.2, 1.2),
        # 'tp_mult_long': trial.suggest_float('tp_mult_long', 1.0, 5.0),
        # 'sl_mult_long': trial.suggest_float('sl_mult_long', 0.5, 3.0),
    }
    
    config = BacktestConfig(
        start_date=config_base.start_date,
        initial_capital=config_base.initial_capital,
        risk_per_trade=config_base.risk_per_trade,
        max_open_trades=config_base.max_open_trades,
        leverage=config_base.leverage,
        **params
    )
    
    trades, equity_curve, _ = run_portfolio_simulation(signals, full_price_db, config)
    
    # Gradient trừng phạt mượt mà cho các trường hợp không có lệnh
    if not trades or len(equity_curve) < 2:
        return -100.0

    report_df = pd.DataFrame([vars(t) for t in trades])
    report_df = report_df[report_df['result'] != 'MISSED']
    
    # RẤT TỐT: Đoạn này tạo gradient để Optuna biết hướng tìm thêm lệnh
    min_trades = 15
    if len(report_df) < min_trades: 
        return -50.0 + (len(report_df) / 100.0)
        
    equity_series = pd.DataFrame(equity_curve, columns=['time', 'val']).set_index('time')['val']
    daily_equity = equity_series.resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    
    # Base Metrics
    std = daily_returns.std()
    sharpe = (daily_returns.mean() / (std + 1e-9)) * np.sqrt(365) if std > 0 else 0
    
    roll_max = equity_series.cummax()
    drawdowns = (equity_series - roll_max) / (roll_max + 1e-9)
    max_dd = abs(drawdowns.min()) # Biến đổi DD thành số dương để dễ tính toán
    
    days = max((equity_series.index[-1] - equity_series.index[0]).days, 1)
    years = days / 365.25
    
    # ĐÃ SỬA: Tính chuẩn CAGR (Lãi kép)
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1
    
    # ĐÃ SỬA: Loại bỏ hàm abs() bao ngoài cagr. Âm là âm, dương là dương!
    calmar = cagr / max(max_dd, 0.01)
    
    # Tối ưu hóa: Trừng phạt nếu PnL âm (Không cho Optuna lươn lẹo)
    if cagr < 0:
        return cagr * 100 # Phóng đại độ âm để nó học cách né
    
    # ĐÃ SỬA: Trừng phạt Rủi ro phi tuyến tính (Exponential Penalty)
    # Không dùng if/else chặn -1e9 nữa. Drawdown càng cao, điểm càng bị ép về 0.
    # Với DD = 30% -> penalty = 0.22; DD = 50% -> penalty = 0.08; DD = 60% -> penalty = 0.04
    dd_penalty = math.exp(-max_dd * 5.0) 
    
    # Kết hợp điểm (Có thể tùy chỉnh trọng số tùy gu của bạn)
    raw_score = (sharpe * 0.4) + (calmar * 0.6)
    
    # Điểm cuối cùng bị bào mòn bởi rủi ro drawdown
    final_score = raw_score * dd_penalty
    
    return final_score

def run_test_on_range(signals, full_price_db, config):
    """Run a single backtest for a specific range and return key metrics."""
    trades, equity_curve, _ = run_portfolio_simulation(signals, full_price_db, config)
    if not trades or len(equity_curve) < 2:
        return 0.0, 0.0, 0, 0.0
        
    report_df = pd.DataFrame([vars(t) for t in trades])
    report_df = report_df[report_df['result'] != 'MISSED']
    
    if report_df.empty:
        return 0.0, 0.0, 0, 0.0
        
    equity_series = pd.DataFrame(equity_curve, columns=['time', 'val']).set_index('time')['val']
    daily_equity = equity_series.resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    sharpe = (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(365) if not daily_returns.empty else 0
    total_ret = ((equity_series.iloc[-1] / equity_series.iloc[0]) - 1) * 100
    max_dd = ((equity_series - equity_series.cummax()) / equity_series.cummax()).min() * 100
    
    return sharpe, total_ret, len(report_df), max_dd

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optuna Sniper with Walk-Forward Validation")
    parser.add_argument('--trials', type=int, default=50, help='Number of trials per fold')
    args = parser.parse_args()

    # 1. Precompute
    all_signals, full_price_db = precompute_data()
    
    # Adjust folds for actual 2025-2026 context
    actual_folds = [
        ('2025-01-01', '2025-05-01', '2025-05-01', '2025-06-01'),
        ('2025-03-01', '2025-07-01', '2025-07-01', '2025-08-01'),
        ('2025-06-01', '2025-10-01', '2025-10-01', '2025-11-01'),
        ('2025-09-01', '2026-01-01', '2026-01-01', '2026-02-01'),
    ]

    results = []
    config_base = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        max_open_trades=MAX_OPEN_TRADES,
        leverage=LEVERAGE
    )

    print(f"\n🚀 Starting Walk-Forward Optimization ({len(actual_folds)} folds)...")
    
    for i, (is_start, is_end, oos_start, oos_end) in enumerate(actual_folds):
        print(f"\n--- 📂 FOLD {i+1} | IS: {is_start} to {is_end} | OOS: {oos_start} to {oos_end} ---")
        
        # Filter signals for IS and OOS
        is_ts_start = pd.to_datetime(is_start)
        is_ts_end = pd.to_datetime(is_end)
        oos_ts_start = pd.to_datetime(oos_start)
        oos_ts_end = pd.to_datetime(oos_end)
        
        is_signals = [s for s in all_signals if is_ts_start <= s['timestamp'] < is_ts_end]
        oos_signals = [s for s in all_signals if oos_ts_start <= s['timestamp'] < oos_ts_end]
        
        if not is_signals:
            print(f"⚠️ No signals in IS period. Skipping fold.")
            continue
            
        # Optimize IS
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, is_signals, full_price_db, config_base), n_trials=args.trials)
        
        best_params = study.best_params
        
        # Prepare fold config
        fold_config_dict = vars(config_base).copy()
        fold_config_dict.update(best_params)
        fold_config = BacktestConfig(**fold_config_dict)
        
        is_sharpe, is_ret, is_cnt, is_dd = run_test_on_range(is_signals, full_price_db, fold_config)
        
        # Test OOS
        oos_sharpe, oos_ret, oos_cnt, oos_dd = run_test_on_range(oos_signals, full_price_db, fold_config)
        
        print(f"📈 IS Result: Sharpe {is_sharpe:.2f} | Return {is_ret:.2f}% | Trades {is_cnt} | MaxDD {is_dd:.2f}%")
        print(f"📉 OOS Result: Sharpe {oos_sharpe:.2f} | Return {oos_ret:.2f}% | Trades {oos_cnt} | MaxDD {oos_dd:.2f}%")
        
        results.append({
            'fold': i+1,
            'is_range': f"{is_start} to {is_end}",
            'oos_range': f"{oos_start} to {oos_end}",
            'is_sharpe': is_sharpe,
            'oos_sharpe': oos_sharpe,
            'oos_return': oos_ret,
            'oos_dd': oos_dd,
            'params': best_params
        })

    # Summary report
    print("\n" + "🏁" + "="*60 + "🏁")
    print(f"{'FOLD':<5} | {'IS Sharpe':<10} | {'OOS Sharpe':<10} | {'OOS Ret%':<10} | {'OOS DD%':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['fold']:<5} | {r['is_sharpe']:<10.2f} | {r['oos_sharpe']:<10.2f} | {r['oos_return']:<10.2f} | {r['oos_dd']:<10.2f}")
    
    # ==========================================================
    # QUAN TƯ DUY: PARAMETER ENSEMBLING (MEDIAN)
    # ==========================================================
    if results:
        print("\n📊 ENSEMBLED PARAMETERS (MEDIAN ACROSS ALL FOLDS):")
        
        # Lấy danh sách các keys tham số từ Fold đầu tiên
        param_keys = results[0]['params'].keys()
        final_median_params = {}
        
        for key in param_keys:
            # Rút trích toàn bộ giá trị của tham số này trên tất cả các Fold
            values = [r['params'][key] for r in results]
            
            # Tính Trung vị (Median) để triệt tiêu nhiễu
            median_val = np.median(values)
            final_median_params[key] = float(median_val)
            
            print(f"  {key:<20}: {final_median_params[key]:.4f}")
            
        # Lưu file Median Params để chạy bot Live/Paper Trading
        output_path = Path("ml") / "optuna_best_params_wfv_median.joblib"
        os.makedirs(output_path.parent, exist_ok=True) # Đảm bảo thư mục tồn tại
        joblib.dump(final_median_params, output_path)
        print(f"\n✅ Cập nhật hoàn tất! Tham số trung vị đã được lưu vào: {output_path}")
        
    print("="*65)

if __name__ == "__main__":
    main()