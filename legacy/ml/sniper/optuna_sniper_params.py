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
RISK_PER_TRADE = 0.1
MAX_OPEN_TRADES = 5
LEVERAGE = 20.0

def precompute_data(force_recompute=False):
    """Scan symbols once, cache the result, load from cache next time."""
    cache_path = Path(os.getcwd()) / "data" / "cache" / "precomputed_signals.joblib"
    
    # Tư duy mở: Nếu cache tồn tại và không bị ép tính lại, load luôn cho nhanh
    if not force_recompute and cache_path.exists():
        print(f"♻️ Bỏ qua quét dữ liệu! Đang load từ Cache: {cache_path}...")
        return joblib.load(cache_path)

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
    
    # GHI CACHE: Lưu lại thành quả để lần sau dùng
    os.makedirs(cache_path.parent, exist_ok=True)
    joblib.dump((all_signals, full_price_db), cache_path)
    print(f"💾 Đã lưu cache tại {cache_path}")
    
    return all_signals, full_price_db

def objective(trial, signals, full_price_db, config_base):
    # Search Space
    params = {
        # 'long_atr_offset': trial.suggest_float('long_atr_offset', -2, 0.2),
        # 'short_atr_offset': trial.suggest_float('short_atr_offset', -0.2, 1.2),
        # 'max_open_trades': trial.suggest_int('max_open_trades', 5, 25),
        # 'risk_per_trade': trial.suggest_float('risk_per_trade', 0.02, 0.10),
        # 'tp_mult_long': trial.suggest_float('tp_mult_long', 1.0, 5.0),
        # 'sl_mult_long': trial.suggest_float('sl_mult_long', 0.5, 3.0),
        # 'tp_mult_short': trial.suggest_float('tp_mult_short', 1.0, 5.0),
        # 'sl_mult_short': trial.suggest_float('sl_mult_short', 0.5, 3.0),
        'max_bars_hold': trial.suggest_int('max_bars_hold', 10, 50),
    }
    
    config = BacktestConfig(
        start_date=config_base.start_date,
        initial_capital=config_base.initial_capital,
        leverage=config_base.leverage,
        **params
    )
    
    trades, equity_curve, _ = run_portfolio_simulation(signals, full_price_db, config)
    
    # 1. Tránh văng lỗi, nhưng không gán số âm khổng lồ mù quáng
    if not trades or len(equity_curve) < 2:
        return -1.0 # Base loss để TPE biết hướng này tệ, không tốn time đào sâu

    report_df = pd.DataFrame([vars(t) for t in trades])
    report_df = report_df[report_df['result'] != 'MISSED']
    num_trades = len(report_df)
    
    if num_trades == 0:
        return -1.0
        
    # 2. Tiền xử lý dữ liệu Equity
    equity_series = pd.DataFrame(equity_curve, columns=['time', 'val']).set_index('time')['val']
    daily_equity = equity_series.resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().dropna()
    
    if daily_returns.empty:
        return -1.0

    days = max((equity_series.index[-1] - equity_series.index[0]).days, 1)
    years = days / 365.25
    
    # 3. Tính Metrics Cốt Lõi
    cagr = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1
    
    roll_max = equity_series.cummax()
    drawdowns = (roll_max - equity_series) / roll_max 
    max_dd = float(drawdowns.max())
    
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std()
    sortino = (daily_returns.mean() / (downside_std + 1e-9)) * np.sqrt(365) if downside_std > 0 else 0.0

    # ---------------------------------------------------------
    # 4. HỆ THỐNG PENALTY MƯỢT (SMOOTH CONTINUOUS PENALTIES)
    # ---------------------------------------------------------
    
    # A. Ngưỡng động (Dynamic Threshold) cho số lượng lệnh
    # Tự động điều chỉnh theo độ dài của Fold thay vì hardcode số 30
    # Kỳ vọng tối thiểu: Trung bình 1.5 lệnh / tuần
    expected_min_trades = max(5.0, (days / 7.0) * 1.5) 
    
    # Hàm Sigmoid phạt mượt: Ít lệnh -> tiệm cận 0 (bóp điểm). Vượt kỳ vọng -> tiệm cận 1.
    # Hệ số k=0.3 kiểm soát độ dốc của hàm Sigmoid.
    trade_factor = 1.0 / (1.0 + math.exp(-0.3 * (num_trades - expected_min_trades)))
    
    # B. Penalty Drawdown (Exponential Decay)
    safe_dd_limit = 0.25  # Vùng an toàn 25%
    k_factor = 18.0       # Tốc độ bào mòn điểm số khi vượt safe limit
    excess_dd = max(0.0, max_dd - safe_dd_limit)
    dd_penalty_factor = math.exp(-k_factor * excess_dd)
    
    # ---------------------------------------------------------
    # 5. ASYMMETRIC SCORING (CHỐNG LỖI TOÁN HỌC KHI TÍNH PHẠT)
    # ---------------------------------------------------------
    
    if cagr > 0:
        # Nếu hệ thống CÓ LÃI: Điểm là sự kết hợp giữa CAGR và Sortino, 
        # sau đó bị "bóp" lại bằng các hệ số Penalty (nhân với số < 1).
        raw_score = (cagr * 0.6) + (sortino * 0.4)
        final_score = raw_score * trade_factor * dd_penalty_factor
    else:
        # LỖI CHẾT NGƯỜI Ở BẢN CŨ: Nếu CAGR âm (-0.5) mà đem NHÂN với penalty factor (0.1),
        # kết quả ra -0.05 (Tức là tăng điểm, làm TPE hiểu lầm là DD càng cao thì đỡ tệ hơn).
        # Khắc phục: Khi hệ thống LỖ, ta dùng phép TRỪ (Cộng dồn hình phạt).
        
        trade_shortfall_penalty = max(0.0, expected_min_trades - num_trades) / expected_min_trades
        # Lỗ cơ bản (cagr) TRỪ ĐI hình phạt DD (nhân đôi trọng số) TRỪ ĐI hình phạt thiếu lệnh
        final_score = cagr - (excess_dd * 2.5) - (trade_shortfall_penalty * 0.5)

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
    parser.add_argument('--force-recompute', action='store_true', help='Force recomputation of signals')
    args = parser.parse_args()

    # 1. Precompute
    all_signals, full_price_db = precompute_data(force_recompute=args.force_recompute)
    
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
        # Tạo Database SQLite để lưu trạng thái Optuna vĩnh viễn
        db_path = f"sqlite:///optuna_wfv.db"
        study_name = f"fold_{i+1}_{is_start}_to_{is_end}"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=db_path,
            load_if_exists=True, # QUAN TRỌNG: Load lại nếu đã từng chạy
            direction='maximize'
        )
        
        # Chỉ chạy thêm số lượng trials còn thiếu, không chạy lại từ đầu
        completed_trials = len(study.trials)
        remaining_trials = max(0, args.trials - completed_trials)
        
        if remaining_trials > 0:
            print(f"🔄 Tiếp tục chạy thêm {remaining_trials} trials cho Fold {i+1}...")
            # Sử dụng n_jobs=-1 để chạy đa luồng (tận dụng hết số core CPU bạn có)
            study.optimize(lambda t: objective(t, is_signals, full_price_db, config_base), 
                           n_trials=remaining_trials, 
                           n_jobs=-1) 
        else:
            print(f"✅ Fold {i+1} đã hoàn thành đủ {args.trials} trials từ trước. Bỏ qua chạy lại.")
        
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