#!/usr/bin/env python3
"""Generate model summary markdown file"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ML_DIR = Path(__file__).parent
MODELS_DIR = ML_DIR / 'models'
DATA_DIR = ML_DIR.parent / 'bitget-data' / 'processed'

def get_model_metrics():
    """Get metrics for all trained models"""
    timeframes = ['4h', '8h', '12h', '1d', '1w']
    results = []
    
    for tf in timeframes:
        tf_dir = MODELS_DIR / tf
        if not tf_dir.exists():
            continue
        
        row = {'timeframe': tf}
        
        # Entry Filter
        entry_path = tf_dir / 'entry_filter.joblib'
        if entry_path.exists():
            data = joblib.load(entry_path)
            row['entry_features'] = len(data['feature_names'])
            row['entry_trained'] = data.get('trained_at', 'N/A')
        
        # SL Predictor
        sl_path = tf_dir / 'sl_predictor.joblib'
        if sl_path.exists():
            data = joblib.load(sl_path)
            row['sl_features'] = len(data['feature_names'])
            row['sl_trained'] = data.get('trained_at', 'N/A')
        
        # TP Predictor
        tp_path = tf_dir / 'tp_predictor.joblib'
        if tp_path.exists():
            data = joblib.load(tp_path)
            row['tp_features'] = len(data['feature_names'])
            row['tp_predict_rr'] = data.get('predict_rr', False)
            row['tp_trained'] = data.get('trained_at', 'N/A')
        
        results.append(row)
    
    return results


def analyze_data_stats(tf: str):
    """Analyze training data statistics for a timeframe"""
    data_path = DATA_DIR / f'features_{tf}_full.parquet'
    if not data_path.exists():
        return None
    
    df = pd.read_parquet(data_path)
    
    # Crossover rows only
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    
    stats = {
        'total_rows': len(df),
        'crossover_signals': len(df_cross),
        'symbols': df['symbol'].nunique() if 'symbol' in df.columns else 0,
        'date_start': df['timestamp'].min() if 'timestamp' in df.columns else None,
        'date_end': df['timestamp'].max() if 'timestamp' in df.columns else None,
    }
    
    # Win rate
    if 'label' in df_cross.columns:
        stats['win_rate'] = df_cross['label'].mean() * 100
    
    # SL stats
    sl_col = 'sl_pct_used' if 'sl_pct_used' in df_cross.columns else 'actual_sl'
    if sl_col in df_cross.columns:
        sl_data = df_cross[sl_col].dropna()
        sl_data = sl_data[(sl_data > 0.005) & (sl_data < 0.15)]
        stats['sl_mean'] = sl_data.mean() * 100
        stats['sl_std'] = sl_data.std() * 100
        stats['sl_min'] = sl_data.min() * 100
        stats['sl_max'] = sl_data.max() * 100
    
    # TP stats
    tp_col = 'tp_pct_used' if 'tp_pct_used' in df_cross.columns else 'actual_tp'
    if tp_col in df_cross.columns:
        tp_data = df_cross[tp_col].dropna()
        tp_data = tp_data[(tp_data > 0.01) & (tp_data < 1.0)]
        stats['tp_mean'] = tp_data.mean() * 100
        stats['tp_std'] = tp_data.std() * 100
        stats['tp_min'] = tp_data.min() * 100
        stats['tp_max'] = tp_data.max() * 100
    
    return stats


def test_model_accuracy(tf: str):
    """Test model accuracy on test data"""
    from sklearn.metrics import mean_absolute_error, roc_auc_score
    
    tf_dir = MODELS_DIR / tf
    data_path = DATA_DIR / f'features_{tf}_full.parquet'
    
    if not tf_dir.exists() or not data_path.exists():
        return None
    
    df = pd.read_parquet(data_path)
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    
    # Use last 20% as test
    split_idx = int(len(df_cross) * 0.8)
    df_test = df_cross.iloc[split_idx:]
    
    results = {}
    
    # Entry Filter
    entry_path = tf_dir / 'entry_filter.joblib'
    if entry_path.exists() and 'label' in df_test.columns:
        data = joblib.load(entry_path)
        model = data['model']
        scaler = data.get('scaler')
        features = data['feature_names']
        
        # Add is_bullish_cross if needed
        if 'is_bullish_cross' in features and 'is_bullish_cross' not in df_test.columns:
            df_test = df_test.copy()
            df_test['is_bullish_cross'] = df_test['macd_cross_up'].values
        
        available_features = [f for f in features if f in df_test.columns]
        X_test = df_test[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Add missing columns with 0
        for f in features:
            if f not in X_test.columns:
                X_test[f] = 0
        X_test = X_test[features]
        
        if scaler:
            X_test = scaler.transform(X_test)
        
        y_test = df_test['label'].fillna(0).astype(int)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results['entry_auc'] = roc_auc_score(y_test, y_proba)
        
        # Accuracy at different thresholds
        for thresh in [0.5, 0.6, 0.65, 0.7, 0.75]:
            y_pred = (y_proba >= thresh).astype(int)
            acc = (y_pred == y_test).mean()
            precision = y_test[y_pred == 1].mean() if y_pred.sum() > 0 else 0
            results[f'entry_acc_{thresh}'] = acc * 100
            results[f'entry_precision_{thresh}'] = precision * 100
            results[f'entry_signals_{thresh}'] = y_pred.sum()
    
    # SL Predictor
    sl_path = tf_dir / 'sl_predictor.joblib'
    sl_col = 'sl_pct_used' if 'sl_pct_used' in df_test.columns else 'actual_sl'
    if sl_path.exists() and sl_col in df_test.columns:
        data = joblib.load(sl_path)
        model = data['model']
        scaler = data.get('scaler')
        features = data['feature_names']
        
        df_sl = df_test.dropna(subset=[sl_col]).copy()
        df_sl = df_sl[(df_sl[sl_col] > 0.005) & (df_sl[sl_col] < 0.15)]
        
        # Add is_bullish_cross if needed
        if 'is_bullish_cross' in features and 'is_bullish_cross' not in df_sl.columns:
            df_sl['is_bullish_cross'] = df_sl['macd_cross_up'].values
        
        available_features = [f for f in features if f in df_sl.columns]
        X_test = df_sl[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        for f in features:
            if f not in X_test.columns:
                X_test[f] = 0
        X_test = X_test[features]
        
        if scaler:
            X_test = scaler.transform(X_test)
        
        y_test = df_sl[sl_col]
        y_pred = model.predict(X_test)
        
        results['sl_mae'] = mean_absolute_error(y_test, y_pred) * 100
        results['sl_mae_pct'] = (np.abs(y_test - y_pred) / y_test).mean() * 100
    
    # TP Predictor
    tp_path = tf_dir / 'tp_predictor.joblib'
    tp_col = 'tp_pct_used' if 'tp_pct_used' in df_test.columns else 'actual_tp'
    if tp_path.exists() and tp_col in df_test.columns:
        data = joblib.load(tp_path)
        model = data['model']
        scaler = data.get('scaler')
        features = data['feature_names']
        
        df_tp = df_test.dropna(subset=[tp_col]).copy()
        df_tp = df_tp[(df_tp[tp_col] > 0.01) & (df_tp[tp_col] < 1.0)]
        
        # Add is_bullish_cross if needed
        if 'is_bullish_cross' in features and 'is_bullish_cross' not in df_tp.columns:
            df_tp['is_bullish_cross'] = df_tp['macd_cross_up'].values
        
        available_features = [f for f in features if f in df_tp.columns]
        X_test = df_tp[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        for f in features:
            if f not in X_test.columns:
                X_test[f] = 0
        X_test = X_test[features]
        
        if scaler:
            X_test = scaler.transform(X_test)
        
        y_test = df_tp[tp_col].clip(upper=0.30)
        y_pred = model.predict(X_test)
        
        results['tp_mae'] = mean_absolute_error(y_test, y_pred) * 100
        results['tp_mae_pct'] = (np.abs(y_test - y_pred) / y_test).mean() * 100
    
    return results


def generate_markdown():
    """Generate comprehensive markdown summary"""
    timeframes = ['4h', '8h', '12h', '1d', '1w']
    
    md = []
    md.append("# 📊 ML Model Summary - MACD Crossover Strategy")
    md.append(f"\n> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append("\n---\n")
    
    # Overview
    md.append("## 🎯 Tổng Quan\n")
    md.append("Hệ thống sử dụng 3-Stage ML để quyết định giao dịch:\n")
    md.append("1. **Entry Filter**: Quyết định có nên vào lệnh không (Classification)")
    md.append("2. **SL Predictor**: Dự đoán Stop Loss tối ưu (Regression)")
    md.append("3. **TP Predictor**: Dự đoán Take Profit tối ưu (Regression)\n")
    
    # Per timeframe details
    for tf in timeframes:
        tf_dir = MODELS_DIR / tf
        if not tf_dir.exists():
            continue
        
        md.append(f"\n---\n## ⏱️ Timeframe: {tf.upper()}\n")
        
        # Data stats
        stats = analyze_data_stats(tf)
        if stats:
            md.append("### 📈 Thống Kê Data\n")
            md.append(f"| Metric | Value |")
            md.append(f"|--------|-------|")
            md.append(f"| Tổng rows | {stats['total_rows']:,} |")
            md.append(f"| Crossover signals | {stats['crossover_signals']:,} |")
            md.append(f"| Số symbols | {stats['symbols']} |")
            if stats.get('date_start'):
                md.append(f"| Thời gian | {stats['date_start'].strftime('%Y-%m-%d')} → {stats['date_end'].strftime('%Y-%m-%d')} |")
            if stats.get('win_rate'):
                md.append(f"| Win rate (raw) | {stats['win_rate']:.1f}% |")
            md.append("")
        
        # Model accuracy
        accuracy = test_model_accuracy(tf)
        if accuracy:
            md.append("### 🎯 Entry Filter Accuracy\n")
            md.append(f"**AUC Score: {accuracy.get('entry_auc', 0):.3f}**\n")
            md.append("| Threshold | Accuracy | Precision | Signals |")
            md.append("|-----------|----------|-----------|---------|")
            for thresh in [0.5, 0.6, 0.65, 0.7, 0.75]:
                acc = accuracy.get(f'entry_acc_{thresh}', 0)
                prec = accuracy.get(f'entry_precision_{thresh}', 0)
                sigs = accuracy.get(f'entry_signals_{thresh}', 0)
                md.append(f"| {thresh:.2f} | {acc:.1f}% | {prec:.1f}% | {sigs:,} |")
            md.append("")
            
            md.append("### 📉 SL Predictor Accuracy\n")
            if stats:
                md.append(f"| Metric | Value |")
                md.append(f"|--------|-------|")
                md.append(f"| SL thực tế (trung bình) | {stats.get('sl_mean', 0):.2f}% |")
                md.append(f"| SL thực tế (std dev) | ±{stats.get('sl_std', 0):.2f}% |")
                md.append(f"| SL range | {stats.get('sl_min', 0):.2f}% - {stats.get('sl_max', 0):.2f}% |")
                md.append(f"| **MAE (sai số tuyệt đối)** | **{accuracy.get('sl_mae', 0):.2f}%** |")
                md.append(f"| Sai số tương đối | {accuracy.get('sl_mae_pct', 0):.1f}% |")
            md.append("")
            
            md.append("### 📈 TP Predictor Accuracy\n")
            if stats:
                md.append(f"| Metric | Value |")
                md.append(f"|--------|-------|")
                md.append(f"| TP thực tế (trung bình) | {stats.get('tp_mean', 0):.2f}% |")
                md.append(f"| TP thực tế (std dev) | ±{stats.get('tp_std', 0):.2f}% |")
                md.append(f"| TP range | {stats.get('tp_min', 0):.2f}% - {stats.get('tp_max', 0):.2f}% |")
                md.append(f"| **MAE (sai số tuyệt đối)** | **{accuracy.get('tp_mae', 0):.2f}%** |")
                md.append(f"| Sai số tương đối | {accuracy.get('tp_mae_pct', 0):.1f}% |")
            md.append("")
    
    # Recommendations
    md.append("\n---\n## 💡 Khuyến Nghị Khi Vào Lệnh\n")
    md.append("### Entry Filter")
    md.append("- Chỉ vào lệnh khi **confidence ≥ 0.65** (tối thiểu)")
    md.append("- Confidence **0.70-0.75** cho trades an toàn hơn")
    md.append("- Confidence càng cao → Win rate càng cao, nhưng ít signals hơn\n")
    
    md.append("### Stop Loss")
    md.append("- Model dự đoán SL với sai số khoảng **0.5-1%**")
    md.append("- Nên đặt SL = **SL dự đoán + buffer 0.3-0.5%** để an toàn")
    md.append("- Hoặc dùng trailing SL sau khi profit\n")
    
    md.append("### Take Profit")
    md.append("- Model dự đoán TP với sai số khoảng **1-2%**")
    md.append("- Có thể partial TP: **50% tại TP/2, 50% tại TP**")
    md.append("- Hoặc trailing TP nếu momentum mạnh\n")
    
    md.append("### Leverage")
    md.append("- **1x**: An toàn nhất, drawdown thấp")
    md.append("- **5x**: Cân bằng risk/reward, recommended")
    md.append("- **7x**: Aggressive, chỉ dùng với confidence cao (≥0.70)\n")
    
    md.append("### Position Sizing")
    md.append("- **Fixed $1000**: Đơn giản, MaxDD thấp hơn")
    md.append("- **% Equity**: Return cao hơn, nhưng DD cũng cao hơn")
    md.append("- Max **10 positions** cùng lúc")
    md.append("- Không mở 2 lệnh cùng 1 symbol\n")
    
    # Summary table
    md.append("\n---\n## 📋 Bảng Tổng Hợp Nhanh\n")
    md.append("| Timeframe | Entry AUC | SL MAE | TP MAE | Recommend Threshold |")
    md.append("|-----------|-----------|--------|--------|---------------------|")
    
    for tf in timeframes:
        accuracy = test_model_accuracy(tf)
        if accuracy:
            auc = accuracy.get('entry_auc', 0)
            sl_mae = accuracy.get('sl_mae', 0)
            tp_mae = accuracy.get('tp_mae', 0)
            recommend = "0.65" if auc > 0.60 else "0.60"
            md.append(f"| {tf} | {auc:.3f} | {sl_mae:.2f}% | {tp_mae:.2f}% | {recommend} |")
    
    md.append("")
    
    return "\n".join(md)


if __name__ == "__main__":
    print("Generating model summary...")
    md_content = generate_markdown()
    
    output_path = ML_DIR / 'MODEL_SUMMARY.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✓ Saved to: {output_path}")
    print("\n" + "="*60)
    print(md_content)
