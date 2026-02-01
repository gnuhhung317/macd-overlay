
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fix paths
ML_DIR = Path(__file__).parent.parent
PROJECT_DIR = ML_DIR.parent
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
MODELS_DIR = ML_DIR / 'models'

def diagnose_4h_trades():
    print("🔍 Diagnosing 4h Low Trade Volume...")
    
    # 1. Load Data
    data_path = PROCESSED_DIR / 'features_4h_full.parquet'
    if not data_path.exists():
        print("❌ Data not found")
        return
        
    print(f"Loading {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Filter for 2024 (Test Period)
    df = df[df['timestamp'] >= '2024-01-01']
    print(f"Data rows (2024+): {len(df)}")
    
    # 2. Check "True" Opportunities (Ground Truth)
    # How many times did price actually move 20% in 10 bars after a cross?
    # We need to recreate the label logic roughly or check the 'label' column if it was generated with same params
    # But 'label' in file might be old. Let's recalculate simply.
    
    cross_mask = (df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)
    df_cross = df[cross_mask].copy()
    print(f"Total Crossovers: {len(df_cross)}")
    
    if 'max_profit' not in df_cross.columns:
        print("⚠️ 'max_profit' column missing. Cannot verify ground truth easily.")
    else:
        # Check how many hit > 20%
        high_yield = df_cross[df_cross['max_profit'] >= 0.20]
        print(f"Crossovers with >20% Profit potential: {len(high_yield)} ({len(high_yield)/len(df_cross):.2%})")
        
    # 3. Load Model and Check Predictions
    model_path = MODELS_DIR / '4h' / 'entry_filter.joblib'
    if not model_path.exists():
        print("❌ Model not found")
        return
        
    print(f"Loading Model: {model_path}")
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['feature_names']
    
    # Prepare X
    X = df_cross.copy()
    if 'is_bullish_cross' in features and 'is_bullish_cross' not in X.columns:
        X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    
    # Filter to required features
    X = X[features].fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
    
    print("Predicting probabilities...")
    probs = model.predict_proba(X_scaled)[:, 1]
    
    # 4. Analyze Distribution
    print("\n📊 Prediction Probability Stats:")
    print(pd.Series(probs).describe())
    
    thresholds = [0.50, 0.60, 0.65, 0.70, 0.80, 0.90]
    print("\nTrades at different thresholds:")
    for t in thresholds:
        count = (probs >= t).sum()
        print(f"Recall > {t:.2f}: {count} trades ({count/len(probs):.2%})")
        
    # 5. Plot Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(probs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(0.65, color='red', linestyle='--', label='Current Threshold (0.65)')
    plt.title('Distribution of Entry Model Probabilities (4h)')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ml/results/diagnosis_probs_4h.png')
    print("\nSaved histogram to ml/results/diagnosis_probs_4h.png")

if __name__ == "__main__":
    diagnose_4h_trades()
