import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

ML_DIR = Path(".")
MODELS_DIR = ML_DIR / "ml" / "models"
DATA_DIR = ML_DIR / "bitget-data" / "processed"

def calculate_ic(tf):
    model_path = MODELS_DIR / tf / "entry_filter.joblib"
    data_path = DATA_DIR / f"features_{tf}_full.parquet"
    
    if not model_path.exists() or not data_path.exists():
        return None
    
    data = joblib.load(model_path)
    model = data['model']
    scaler = data.get('scaler')
    features = data['feature_names']
    
    df = pd.read_parquet(data_path)
    df_cross = df[(df['macd_cross_up'] == 1) | (df['macd_cross_down'] == 1)].copy()
    
    # Last 20% test set
    split_idx = int(len(df_cross) * 0.8)
    df_test = df_cross.iloc[split_idx:]
    
    if 'is_bullish_cross' in features and 'is_bullish_cross' not in df_test.columns:
        df_test['is_bullish_cross'] = df_test['macd_cross_up'].values
        
    X_test = df_test[features].fillna(0).replace([np.inf, -np.inf], 0)
    if scaler:
        X_test = scaler.transform(X_test)
        
    y_test = df_test['label'].fillna(0).astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # IC is correlation
    ic = np.corrcoef(y_proba, y_test)[0, 1]
    auc = roc_auc_score(y_test, y_proba)
    
    return ic, auc

if __name__ == "__main__":
    ic, auc = calculate_ic("1w")
    print(f"1W Model Metrics:")
    print(f"  AUC: {auc:.4f}")
    print(f"  IC:  {ic:.4f}")
