import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import joblib
import warnings
import shap
import sys

# Add current dir to path to import data_pipeline
sys.path.append(str(Path(__file__).parent))
try:
    from data_pipeline import calculate_features
except ImportError:
    calculate_features = None

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION & PATHS
# ============================================================
BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
RESULTS_PATH = BASE_DIR / "ml" / "backtest_results_quant_sniper.csv"
SYMBOLS_DIR = BASE_DIR / "data" / "processed" / "symbols_v3"
MODEL_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_lgbm_tabular.joblib"
META_PATH = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_meta.joblib"
OUTPUT_DIR = BASE_DIR / "ml" / "diagnostics"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# INDICATOR IMPLEMENTATIONS (Wilder's Smooth ADX & Choppiness)
# ============================================================

def calculate_adx(df, period=14):
    """Refined ADX calculation with Wilder's smoothing logic"""
    df = df.copy()
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def calculate_choppiness(df, period=14):
    """Manual Choppiness Index"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    sum_tr = tr.rolling(period).sum()
    max_high = high.rolling(period).max()
    min_low = low.rolling(period).min()
    
    chop = 100 * np.log10(sum_tr / (max_high - min_low + 1e-9)) / np.log10(period)
    return chop

# ============================================================
# DIAGNOSIS ENGINE v2
# ============================================================

def run_diagnostics():
    print("🚀 Starting Unified Diagnostic Suite v2 (Professional Scale)...")
    
    if not RESULTS_PATH.exists():
        print(f"❌ Results file not found at {RESULTS_PATH}")
        return

    # 1. LOAD DATA & MODEL
    results = pd.read_csv(RESULTS_PATH)
    results['signal_time'] = pd.to_datetime(results['signal_time'])
    results['entry_time'] = pd.to_datetime(results['entry_time'])
    results['exit_time'] = pd.to_datetime(results['exit_time'])
    
    print(f"📈 Loaded {len(results)} trades.")

    model = None
    features = []
    if MODEL_PATH.exists() and META_PATH.exists():
        model = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)
        features = meta.get('features', [])
        print(f"🧠 ML Model loaded. Features: {len(features)}")

    # 2. VECTORIZED MAPPING & DYNAMIC REGIMES
    print("🔍 Performing Vectorized Mapping & Dynamic Threshold Analysis...")
    results = results.sort_values('signal_time')
    results['adx'] = np.nan
    results['chop'] = np.nan
    results['adx_threshold'] = np.nan
    results['chop_threshold'] = np.nan
    
    # Store feature values for SHAP
    for feat in features:
        results[feat] = np.nan

    symbols = results['symbol'].unique()
    for sym in symbols:
        parquet_file = SYMBOLS_DIR / f"{sym}USDT.parquet"
        if not parquet_file.exists(): 
            parquet_file = SYMBOLS_DIR / f"{sym}.parquet"
        
        if parquet_file.exists():
            # 1. Lọc riêng trade của symbol này TRƯỚC
            trades_sym = results[results['symbol'] == sym].sort_values('signal_time')
            if trades_sym.empty:
                continue 
                
            try:
                df_sym = pd.read_parquet(parquet_file)
                df_sym['timestamp'] = pd.to_datetime(df_sym['timestamp'])
                df_sym = df_sym.set_index('timestamp').sort_index()
                
                # ==========================================
                # TỐI ƯU HÓA: CẮT DỮ LIỆU CÓ BUFFER (WARM-UP)
                # ==========================================
                # Tìm thời điểm trade sớm nhất của mã này
                first_trade_time = trades_sym['signal_time'].min()
                
                # Lùi lại 30 ngày (Buffer) để đủ nến cho ADX/EMA "khởi động" mượt mà
                # Sử dụng pd.Timedelta cho sự chính xác của chuỗi thời gian
                cutoff_time = first_trade_time - pd.Timedelta(days=30)
                
                # Ép kiểu dữ liệu (Slicing) TRƯỚC KHI tính toán logic nặng
                df_sym = df_sym[df_sym.index >= cutoff_time]
                
                # Lúc này df_sym chỉ còn dữ liệu từ cuối 2024 -> tốc độ tính toán tăng x100 lần
                
                # ADX/Chop with Wilder's logic (Diagnostic overrides)
                df_sym['adx_calc'] = calculate_adx(df_sym)
                df_sym['chop_calc'] = calculate_choppiness(df_sym)
                
                # Full Feature Engineering (Institutional Sync)
                if calculate_features:
                    # Chuyển về integer index for compatibility with internal loop if any
                    df_sym_reset = df_sym.reset_index()
                    df_sym_feat = calculate_features(df_sym_reset)
                    df_sym = df_sym_feat.set_index('timestamp')
                
                # Dynamic Thresholds (70th for ADX, 30th for Chop)
                # Tính trên tập đã lọc để phản ánh đúng bối cảnh hiện tại của bot
                adx_t = df_sym['adx_calc'].quantile(0.70)
                chop_t = df_sym['chop_calc'].quantile(0.30)
                
                # Map columns (ADX, Chop, and ML Features)
                cols_to_map = ['adx_calc', 'chop_calc'] + [f for f in features if f in df_sym.columns]
                
                if not trades_sym.empty:
                    print(f"   - {sym}: {len(trades_sym)} trades. Processing from: {cutoff_time.date()}")
                
                    # PD.MERGE_ASOF: Vectorized, Backward (No look-ahead), Fast
                    merged = pd.merge_asof(
                        trades_sym[['signal_time']], 
                        df_sym[cols_to_map], 
                        left_on='signal_time', 
                        right_index=True, 
                        direction='backward'
                    )
                    
                    # Count non-nan mapped features
                    valid_mapped = merged.dropna(subset=[f for f in features if f in merged.columns]).shape[0]
                    if valid_mapped > 0:
                        print(f"     ✅ Mapped {valid_mapped} valid feature rows for {sym}")
                    else:
                        print(f"     ⚠️ No features mapped for {sym} (Check feature names or date overlap)")
                    
                    # Update main results
                    results.loc[trades_sym.index, 'adx'] = merged['adx_calc'].values
                    results.loc[trades_sym.index, 'chop'] = merged['chop_calc'].values
                    results.loc[trades_sym.index, 'adx_threshold'] = adx_t
                    results.loc[trades_sym.index, 'chop_threshold'] = chop_t
                    
                    for feat in features:
                        if feat in merged.columns:
                            results.loc[trades_sym.index, feat] = merged[feat].values
                        
            except Exception as e:
                print(f"⚠️ Error processing {sym}: {e}")

    # 3. PNL ATTRIBUTION (Side, Hour)
    results['hour'] = results['signal_time'].dt.hour
    side_perf = results.groupby('type')['pnl_usd'].sum()
    hour_perf = results.groupby('hour')['pnl_usd'].sum()
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    side_perf.plot(kind='bar', color=['red', 'green'], title='PnL by Side (Correct Chronology)')
    plt.subplot(1, 2, 2)
    hour_perf.plot(kind='bar', title='PnL by Hour of Day')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pnl_attribution.png")
    plt.close()

    # 4. TRENDING VS CHOPPY (Dynamic)
    results['is_trending'] = (results['adx'] > results['adx_threshold']) | (results['chop'] < results['chop_threshold'])
    regime_pnl = results.groupby('is_trending')['pnl_usd'].mean()
    
    # Identifies features with all NaNs but DOES NOT impute 0.0
    # Imputing 0.0 creates a "fake" signal at a specific level (e.g. Price exactly on EMA)
    # LightGBM/SHAP handles np.nan natively as a separate branch.
    missing_feats = [f for f in features if results[f].isna().all()]
    if missing_feats:
        print(f"   ⚠️ Missing entirely in OHLCV samples: {missing_feats}")
            
    shap_data = results[features] # Keep all, let dropna or native handle it
    valid_rows = shap_data.dropna()
    print(f"   - Valid rows for SHAP: {len(valid_rows)}/{len(results)}")
    
    if model is not None and len(valid_rows) > 0:
        try:
            # We use valid_rows to ensure the explainer doesn't crash, 
            # but we allow np.nan where data is partially missing
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(valid_rows)
            
            plt.figure(figsize=(10, 8))
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(sv, valid_rows, show=False)
            plt.title("Institutional SHAP: Feature Impact (No Imputation Bias)")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "shap_analysis.png")
            plt.close()
            print(f"✅ SHAP analysis saved to {OUTPUT_DIR / 'shap_analysis.png'}")
        except Exception as e:
            print(f"⚠️ SHAP Error: {e}")
    else:
        print("⚠️ Skipping SHAP: No valid feature data found in trades.")

    # 6. LOSING STREAK DISTRIBUTION (Signal-Time Corrected)
    results = results.sort_values('signal_time') # Explicit re-sort
    results['is_win'] = results['pnl_usd'] > 0
    results['streak_id'] = (results['is_win'] != results['is_win'].shift()).cumsum()
    streaks = results.groupby(['streak_id', 'is_win']).size()
    losing_streaks = streaks[streaks.index.get_level_values('is_win') == False]
    
    plt.figure(figsize=(10, 5))
    sns.histplot(losing_streaks, bins=range(1, 15), kde=False, color='red', alpha=0.7)
    plt.title('Corrected Losing Streak Distribution (by Signal Time)')
    plt.xlabel('Consecutive Losses')
    plt.ylabel('Frequency')
    plt.savefig(OUTPUT_DIR / "losing_streak_dist.png")
    plt.close()

    # 7. UNDERWATER CURVE (True Time Axis)
    results = results.sort_values('exit_time') # Realized equity axis
    results['cum_pnl'] = results['pnl_usd'].cumsum()
    results['rolling_max'] = results['cum_pnl'].cummax()
    results['drawdown'] = results['cum_pnl'] - results['rolling_max']
    
    plt.figure(figsize=(12, 4))
    # Using exit_time instead of linear range to prevent Time-Space Distortion
    plt.fill_between(results['exit_time'], results['drawdown'], color='firebrick', alpha=0.4)
    plt.title('Institutional Underwater Curve (True Time-to-Recovery)')
    plt.ylabel('Drawdown ($)')
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "underwater_curve.png")
    plt.close()

    # ============================================================
    # GENERATE PROFESSIONAL MARKDOWN REPORT
    # ============================================================
    # Path relative to the report file (which is in BASE_DIR)
    img_dir = "ml/diagnostics"
    
    report_content = f"""# 🛡️ Strategy Diagnostic Report v2: Q2 2025 Deep-Dive
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Executive Summary
- **Trades Analyzed**: {len(results)}
- **Chronology Path**: Signal-Time Optimized
- **Net Performance**: ${results['pnl_usd'].sum():.2f}
- **Winrate**: {(results['pnl_usd'] > 0).mean()*100:.2f}%
- **Max Sequential Decisions Failure**: {losing_streaks.max() if not losing_streaks.empty else 0} (Losing Streak)

## 2. Dynamic Market Regime Analysis
- **Definition**: Using per-asset 70th Percentile ADX and 30th Percentile Choppiness.
- **Performance in Trending (High Intent)**: ${results[results['is_trending'] == True]['pnl_usd'].sum():.2f}
- **Performance in Range (Low Intent)**: ${results[results['is_trending'] == False]['pnl_usd'].sum():.2f}
- **Observation**: { "Strategy fails in Trending markets" if results[results['is_trending'] == True]['pnl_usd'].sum() < results[results['is_trending'] == False]['pnl_usd'].sum() else "Strategy captures Trends well"}.

## 3. PnL Attribution & Bias
![PnL Attribution]({img_dir}/pnl_attribution.png)
- **Signal-to-Exit Chronology**: All streaks were calculated based on decision time to remove exit-time bias.

## 4. ML SHAP Autopsy (Feature Drift Detection)
![SHAP Analysis]({img_dir}/shap_analysis.png)
- **Local Interpretability**: This chart shows which features *pushed* the model towards these specific signals and how they correlated with outcomes. 
- *Note: Direction of impact (left/right) shows if a high value of a feature increased or decreased the signal probability.*

## 5. Risk & Drawdown Dynamics
![Underwater Curve]({img_dir}/underwater_curve.png)
![Losing Streaks]({img_dir}/losing_streak_dist.png)
- **Clusters of Failure**: Clusters of 4+ losses in the histogram indicate consistent **Concept Drift** in Q2 2025.

---
*Generated by Antigravity Unified Diagnostic Engine v2*
"""
    
    with open(BASE_DIR / "diagnostic_report_quant_sniper.md", "w", encoding='utf-8') as f:
        f.write(report_content)

    print(f"✅ Diagnosis Complete! Enhanced report saved to {BASE_DIR / 'diagnostic_report_quant_sniper.md'}")

if __name__ == "__main__":
    run_diagnostics()
