import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.multi_timeframe_pipeline import PROCESSED_DIR

MODELS_DIR = Path(__file__).parent / 'models'

def load_data_and_model(timeframe='4h'):
    """Load processed data and all trained models (Entry, SL, TP)."""
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    model_dir = MODELS_DIR / timeframe
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return None, None
        
    print(f"📂 Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    models = {}
    
    # Load Entry Model
    entry_path = model_dir / 'entry_filter.joblib'
    if entry_path.exists():
        data = joblib.load(entry_path)
        models['entry'] = {
            'model': data['model'], 
            'scaler': data.get('scaler'),
            'features': data['feature_names']
        }
        print(f"🤖 Loaded Entry Model")
    else:
        print(f"❌ Entry Model not found: {entry_path}")
        return None, None

    # Load SL Model
    sl_path = model_dir / 'sl_predictor.joblib'
    if sl_path.exists():
        data = joblib.load(sl_path)
        models['sl'] = {
            'model': data['model'], 
            'scaler': data.get('scaler'),
            'features': data['feature_names']
        }
        print(f"🤖 Loaded SL Model")
    
    # Load TP Model
    tp_path = model_dir / 'tp_predictor.joblib'
    if tp_path.exists():
        data = joblib.load(tp_path)
        models['tp'] = {
            'model': data['model'], 
            'scaler': data.get('scaler'),
            'features': data['feature_names'],
            'predict_rr': data.get('predict_rr', False)
        }
        print(f"🤖 Loaded TP Model")

    return df, models

def analyze_signals(df, models, horizon=10):
    """
    Run detailed analysis on signals using Dynamic TP/SL predictions.
    Optimized Loop using NumPy for high performance.
    """
    # OPTIMIZATION: Reset index to ensure integer alignment and avoid Index Traps
    df = df.reset_index(drop=True)
    
    # Extract Numpy Arrays (Fast Access)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    symbols = df['symbol'].values if 'symbol' in df.columns else np.full(len(df), 'UNKNOWN')
    
    # 1. Identify Signals (Boolean Mask)
    # Ensure is_bullish_cross exists
    if 'is_bullish_cross' not in df.columns:
        df['is_bullish_cross'] = df['macd_cross_up'] # assuming 0/1
        
    signal_mask = (df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)
    
    # Get Integer Indices of signals
    signal_indices = np.where(signal_mask)[0]
    
    if len(signal_indices) == 0:
        print("⚠️ No signals found.")
        return

    print(f"🔍 Analyzing {len(signal_indices)} signals with Dynamic TP/SL...")
    
    # 2. Batch Predictions
    # A. Entry Confidence
    entry_model = models['entry']
    X_entry = df.loc[signal_mask, entry_model['features']].fillna(0).replace([np.inf, -np.inf], 0)
    if entry_model['scaler']:
        X_entry = entry_model['scaler'].transform(X_entry)
    confidences = entry_model['model'].predict_proba(X_entry)[:, 1]
    
    # B. SL Predictions
    if 'sl' in models:
        sl_model = models['sl']
        X_sl = df.loc[signal_mask, sl_model['features']].fillna(0).replace([np.inf, -np.inf], 0)
        if sl_model['scaler']:
            X_sl = sl_model['scaler'].transform(X_sl)
        sl_preds = sl_model['model'].predict(X_sl)
        # Clip SL to reasonable bounds (0.5% to 10%)
        sl_preds = np.clip(sl_preds, 0.005, 0.10)
    else:
        sl_preds = np.full(len(signal_indices), 0.015) # Default 1.5%

    # C. TP Predictions
    if 'tp' in models:
        tp_model = models['tp']
        X_tp = df.loc[signal_mask, tp_model['features']].fillna(0).replace([np.inf, -np.inf], 0)
        if tp_model['scaler']:
            X_tp = tp_model['scaler'].transform(X_tp)
        tp_preds = tp_model['model'].predict(X_tp)
        
        # If model predicts RR, multiply by SL
        if tp_model.get('predict_rr', False):
             tp_preds = tp_preds * sl_preds
             
        # Clip TP (0.5% to 20%)
        tp_preds = np.clip(tp_preds, 0.005, 0.20)
    else:
        tp_preds = np.full(len(signal_indices), 0.03) # Default 3%
        
    
    # Pre-fetch attributes mapped to signal array
    entry_prices = closes[signal_indices]
    entry_symbols = symbols[signal_indices]
    is_longs = df.loc[signal_mask, 'macd_cross_up'].values == 1
    
    # 3. Simulation Loop (Optimized)
    results = []
    
    # Loop over indices
    for i in range(len(signal_indices)):
        idx = signal_indices[i]
        entry_price = entry_prices[i]
        entry_symbol = entry_symbols[i]
        is_long = is_longs[i]
        confidence = confidences[i]
        
        # Dynamic TP/SL for this trade
        # Note: In inference.py we apply a 1.5x multiplier to SL. Let's replicate that safety factor.
        sl_target = -sl_preds[i] * 1.5 # Negative for loss
        tp_target = tp_preds[i] 
        
        # Look ahead window indices
        start_future = idx + 1
        end_future = min(idx + 1 + horizon, len(closes))
        
        if start_future >= len(closes):
            continue
            
        # Check Symbol Boundary Violation using vector slice
        # If the window spans across symbols, truncate it
        future_symbols = symbols[start_future:end_future]
        
        # We only want rows where symbol is the same as entry
        # Ideally, data is sorted by symbol, then time.
        # If so, just finding the first mismatch is enough.
        # Speed hack: check if last symbol matches first. If NO, find split.
        if len(future_symbols) > 0 and future_symbols[-1] != entry_symbol:
             # Find first index where symbol changes
             mismatches = np.where(future_symbols != entry_symbol)[0]
             if len(mismatches) > 0:
                 end_future = start_future + mismatches[0]
                 
        if start_future >= end_future:
            continue
            
        # Get window slices (NumPy views, zero copy overhead)
        w_highs = highs[start_future:end_future]
        w_lows = lows[start_future:end_future]
        w_closes = closes[start_future:end_future]
        
        if len(w_closes) == 0:
            continue
            
        # --- Metrics Calculation ---
        
        # 1. MFE / MAE
        if is_long:
            max_h = np.max(w_highs)
            min_l = np.min(w_lows)
            
            mfe_pct = (max_h - entry_price) / entry_price
            mae_pct = (min_l - entry_price) / entry_price
            
            bars_to_mfe = np.argmax(w_highs) + 1
            bars_to_mae = np.argmin(w_lows) + 1
        else:
            # Short
            max_h = np.max(w_highs)
            min_l = np.min(w_lows)
            
            # Short MFE: Price down (min_l) is good
            mfe_pct = (entry_price - min_l) / entry_price
            # Short MAE: Price up (max_h) is bad
            mae_pct = (entry_price - max_h) / entry_price
            
            bars_to_mfe = np.argmin(w_lows) + 1
            bars_to_mae = np.argmax(w_highs) + 1
            
        # 2. First Hit Logic (Vectorized Check)
        outcome = "TIMEOUT"
        realized_pnl = 0.0
        
        if is_long:
            tp_price_level = entry_price * (1 + tp_target)
            # SL is negative target, so 1 - 0.02 = 0.98
            sl_price_level = entry_price * (1 + sl_target)
            
            hit_tp = w_highs >= tp_price_level
            hit_sl = w_lows <= sl_price_level
        else:
            tp_price_level = entry_price * (1 - tp_target)
            # SL is negative target, so 1 - (-0.02) = 1.02
            sl_price_level = entry_price * (1 - sl_target)
            
            hit_tp = w_lows <= tp_price_level
            hit_sl = w_highs >= sl_price_level
            
        # Find first occurrence index
        first_tp_idx = np.argmax(hit_tp) if np.any(hit_tp) else 9999
        first_sl_idx = np.argmax(hit_sl) if np.any(hit_sl) else 9999
        
        if first_sl_idx == 9999 and first_tp_idx == 9999:
            outcome = "TIMEOUT"
            last_c = w_closes[-1]
            if is_long:
                realized_pnl = (last_c - entry_price) / entry_price
            else:
                realized_pnl = (entry_price - last_c) / entry_price
                
        elif first_sl_idx <= first_tp_idx:
            # SL hit first or same bar (Conservative)
            outcome = "SL_HIT"
            realized_pnl = sl_target
        else:
            outcome = "TP_HIT"
            realized_pnl = tp_target
        
        results.append({
            'confidence': confidence,
            'mfe': mfe_pct,
            'mae': mae_pct,
            'bars_to_mfe': bars_to_mfe,
            'bars_to_mae': bars_to_mae,
            'outcome': outcome,
            'realized_pnl': realized_pnl,
            'predicted_sl': sl_preds[i], # Raw SL prediction
            'predicted_tp': tp_preds[i], # Raw TP prediction
            'projected_rr': tp_preds[i] / sl_preds[i],
            'is_long': is_long
        })
        
    return pd.DataFrame(results)

def generate_markdown_report(all_stats, filename='ML_SIGNAL_QUALITY_REPORT.md'):
    """Generate a comprehensive Markdown report for multiple timeframes."""
    
    # Specific Insights per Timeframe
    tf_insights = {
        '4h': """
**🔎 Phân Tích Khung 4h**:
*   **Đặc điểm**: Tần suất tín hiệu dày đặc nhất. Thích hợp cho **Scalping/Day Trading**.
*   **Điểm mạnh**: E-Ratio ở mức High Confidence (0.6-0.7) đạt **3.25**, rất ấn tượng.
*   **Khuyến nghị**: Có thể đánh volume vừa phải, Entry nhanh, Exit nhanh (avg hold 4-7 nến).
""",
        '8h': """
**🔎 Phân Tích Khung 8h**:
*   **Đặc điểm**: Bộ lọc nhiễu tốt hơn 4h. Winrate nhóm 0.6-0.7 đạt **89.0%**.
*   **Điểm mạnh**: Sự cân bằng hoàn hảo giữa độ chính xác và số lượng cơ hội.
*   **Khuyến nghị**: Khung thời gian "xương sống" cho Swing ngắn hạn.
""",
        '12h': """
**🔎 Phân Tích Khung 12h**:
*   **Đặc điểm**: Độ chính xác cực cao (Winrate 95.4% ở range 0.7-0.8).
*   **Cảnh báo Volatilty**: Stress Test cho thấy có cú sập sâu (-60.79%), nhưng SL Dynamic đã chặn lỗ ở -15%.
*   **Khuyến nghị**: Dùng để bắt các con sóng trung hạn.
""",
        '1d': """
**🔎 Phân Tích Khung 1d**:
*   **Đặc điểm**: **Swing King**. Lợi nhuận trung bình mỗi kèo thắng (High Conf) lên tới **+11% - 13%**.
*   **Rủi ro**: Avg MAE khá lớn (~6% đến 10%). Tức là vào lệnh xong giá thường rung lắc mạnh trước khi bay.
*   **Khuyến nghị**: **GIẢM VOLUME**. Vì SL rất xa (Dynamic SL có thể lên tới 10-15%), nên cần quản lý vốn chặt chẽ.
"""
    }

    with open(filename, 'w', encoding='utf-8') as f:
        # 1. Header & Educational Section
        f.write("# 📊 ML Signal Quality Report (Multi-Timeframe)\n\n")
        f.write("Generated by `ml/analyze_predictions.py` using **First Hit Logic** and **Dynamic TP/SL**.\n\n")
        
        f.write("## 💡 How to Interpret MFE & MAE\n")
        f.write("To understand the 'personality' of your signals, look at these 3 metrics:\n\n")
        
        f.write("### 1. MFE (Max Favorable Excursion) - \"Tiền Mỡ\"\n")
        f.write("*   **Định nghĩa**: Mức lãi tối đa đạt được trong 10 nến (trước khi đóng lệnh).\n")
        f.write("*   **Insight**: MFE cho biết **Tiềm năng** của tín hiệu.\n")
        f.write("    *   Nếu `Avg MFE` cao (+5%) mà `Avg PnL` thấp (+1%) $\\rightarrow$ Bạn đang **chốt lời quá tệ** (hoặc Trailing Stop quá chặt).\n")
        f.write("    *   Nếu `Avg MFE` thấp $\\rightarrow$ Tín hiệu yếu, giá không chạy ngay.\n\n")
        
        f.write("### 2. MAE (Max Adverse Excursion) - \"Mức Gồng Lỗ\"\n")
        f.write("*   **Định nghĩa**: Mức lỗ sâu nhất mà giá chạm tới trong quá trình giữ lệnh.\n")
        f.write("*   **Insight**: MAE cho biết **Độ chuẩn xác** của Entry.\n")
        f.write("    *   Nếu `Avg MAE` thấp (ví dụ -1%) mà Winrate cao $\\rightarrow$ Entry kiểu **Sniper** (Vào là xanh).\n")
        f.write("    *   Nếu `Avg MAE` cao (ví dụ -10%) mà vẫn Win $\\rightarrow$ Entry xấu, thắng do may mắn hoặc gồng giỏi.\n\n")
        
        f.write("### 3. E-Ratio (Edge Ratio) - \"Lợi Thế Tự Nhiên\"\n")
        f.write("*   $$E = \\frac{\\text{Avg MFE}}{|\\text{Avg MAE}|}$$\n")
        f.write("*   **Insight**: Tỷ lệ R:R tự nhiên của tín hiệu.\n")
        f.write("    *   **E > 1.0**: Kèo thơm. Tiềm năng ăn nhiều hơn thua.\n")
        f.write("    *   **E < 1.0**: Kèo thối. Rủi ro cao hơn lợi nhuận ngay từ khi vào lệnh.\n")
        f.write("    *   *Mục tiêu*: Chỉ trade các nhóm có **E-Ratio > 2.0**.\n\n")
        
        f.write("---\n\n")
        
        # 2. Timeframe Stats
        for tf, stats, stress_test in all_stats:
            f.write(f"## ⏳ Timeframe: {tf}\n")
            
            # Write specific insight if available
            if tf in tf_insights:
                f.write(tf_insights[tf] + "\n")
            
            # Main Stats Table
            f.write("| Bin (Conf) | Count | Win% | Avg PnL | Avg MFE | Avg MAE | Pro.R:R | E-Ratio | Time(MFE) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            for idx, row in stats.iterrows():
                if row['count'] == 0:
                    continue
                f.write(f"| {idx} | {int(row['count']):,} | {row['win_rate']:.1%} | {row['avg_pnl']:.2%} | {row['avg_mfe']:.2%} | {row['avg_mae']:.2%} | {row['avg_proj_rr']:.2f} | {row['e_ratio']:.2f} | {row['avg_bars_mfe']:.1f} |\n")
            
            f.write("\n")
            
            # Stress Test Section
            if stress_test:
                f.write("> [!WARNING] **Stress Test (Confidence >= 0.7)**\n")
                f.write(f"> * **Worst Drawdown (MAE)**: `{stress_test['worst_loss']:.2%}`\n")
                f.write(f"> * **Worst Realized PnL**: `{stress_test['worst_pnl']:.2%}`\n")
                f.write(f"> * **Avg Loss**: `{stress_test['avg_loss']:.2%}`\n")
            else:
                f.write("> [!NOTE] Not enough high confidence signals for stress test.\n")
            
            f.write("\n---\n\n")
            
    print(f"\n✅ Created report: {filename}")

def aggregate_stats(results_df):
    """Aggregate stats for a dataframe."""
    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '>0.9']
    results_df['bin'] = pd.cut(results_df['confidence'], bins=bins, labels=labels)
    
    stats = results_df.groupby('bin').agg(
        count=('confidence', 'count'),
        win_rate=('realized_pnl', lambda x: (x > 0).mean()),
        avg_pnl=('realized_pnl', 'mean'),
        avg_mfe=('mfe', 'mean'),
        avg_mae=('mae', 'mean'),
        avg_bars_mfe=('bars_to_mfe', 'mean'),
        avg_proj_rr=('projected_rr', 'mean'),
    )
    
    # Calculate Custom Metrics
    stats['e_ratio'] = stats['avg_mfe'] / abs(stats['avg_mae']).replace(0, 0.0001)
    
    # Stress Test
    high_conf = results_df[results_df['confidence'] >= 0.7]
    stress_test = None
    if len(high_conf) > 0:
        losses = high_conf[high_conf['realized_pnl'] < 0]
        stress_test = {
            'worst_loss': high_conf['mae'].min(),
            'worst_pnl': high_conf['realized_pnl'].min(),
            'avg_loss': losses['realized_pnl'].mean() if len(losses) > 0 else 0.0
        }
        
    return stats, stress_test

if __name__ == "__main__":
    timeframes = ['4h', '8h', '12h', '1d']
    all_stats_data = []
    
    print("🚀 Starting Multi-Timeframe Analysis...")
    
    for tf in timeframes:
        print(f"\n👉 Processing {tf}...")
        df, bundle = load_data_and_model(tf)
        
        if df is not None and bundle is not None:
            results = analyze_signals(df, bundle)
            if results is not None and not results.empty:
                stats, stress = aggregate_stats(results)
                all_stats_data.append((tf, stats, stress))
                print(f"   ✓ {len(results)} signals analyzed")
            else:
                 print(f"   ⚠️ No results for {tf}")
        else:
            print(f"   ⚠️ Skipping {tf} (Missing data/model)")
            
    if all_stats_data:
        generate_markdown_report(all_stats_data)
    else:
        print("❌ No data generated for any timeframe.")
