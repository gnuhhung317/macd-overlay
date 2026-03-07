import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from training_utils import PROCESSED_DIR, get_feature_columns

def compare_xgboost(timeframe: str):
    data_path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
    df = pd.read_parquet(data_path)
    # Ensure time-based sorting before split to avoid symbol-based split
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Prepare data (exact same as Hybrid)
    mask = ((df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)) & df['label'].notnull()
    df_cross = df[mask].copy()
    
    # Filter indices where we have enough lookback (to match Hybrid's filtered size)
    window_size = 50
    df_cross = df_cross[df_cross.index >= window_size - 1]
    
    feature_cols = get_feature_columns(df_cross)
    X = df_cross[feature_cols].copy()
    X['is_bullish_cross'] = df_cross['macd_cross_up'].values
    feature_cols.append('is_bullish_cross')
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    y = df_cross['label'].astype(int)
    
    # 2. Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train XGBoost (Standard params)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)
    
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"XGBoost Baseline AUC for {timeframe}: {auc:.4f}")

if __name__ == "__main__":
    compare_xgboost('1d')
