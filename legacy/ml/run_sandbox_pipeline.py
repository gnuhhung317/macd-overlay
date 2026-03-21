#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

warnings.filterwarnings('ignore')

# Add root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

try:
    from sniper_bot.feature import calculate_features, generate_momentum_labels, apply_feature_shift
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

def process_single_symbol(symbol, btc_df, real_ohlcv_dir):
    # 1. 1H Data
    path_1h = None
    for suffix in ["USDT_USDT.parquet", "_USDT.parquet", "USDT.parquet", ".parquet"]:
        p = real_ohlcv_dir / f"{symbol}{suffix}"
        if p.exists():
            path_1h = p
            break
    
    if not path_1h and symbol.endswith("USDT"):
        s2 = symbol.replace("USDT", "")
        for suffix in ["USDT_USDT.parquet", "_USDT.parquet", "USDT.parquet", ".parquet"]:
            p = real_ohlcv_dir / f"{s2}{suffix}"
            if p.exists():
                path_1h = p
                break
    
    if not path_1h: return None
def generate_momentum_labels(df, horizon=12, min_pump=0.10):
    df = df.copy()
    df_rev = df.iloc[::-1].copy()
    if 'symbol' in df_rev.columns:
        future_max_high = df_rev.groupby('symbol', group_keys=False)['high'].apply(lambda x: x.rolling(horizon, min_periods=1).max())
    else:
        future_max_high = df_rev['high'].rolling(horizon, min_periods=1).max()
        
    df['future_max_high'] = future_max_high.sort_index()
    df['max_pump_pct'] = (df['future_max_high'] - df['close']) / df['close']
    df['label'] = (df['max_pump_pct'] >= min_pump).astype(int)
    
    if 'usd_vol_24h' in df.columns:
        df.loc[df['usd_vol_24h'] < 1000000, 'label'] = np.nan
    df['ignition'] = df['label']
    
    if 'symbol' in df.columns:
        df['future_return'] = df.groupby('symbol')['close'].shift(-horizon) / df['close'] - 1
    else:
        df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
        
    df['trade_result'] = np.where(df['label'] == 1, 'WIN', 'LOSS')
    return df.drop(columns=['future_max_high'])

def process_single_symbol(symbol, btc_df, ohlcv_dir):
    try:
        path_1h = None
        for suffix in ["USDT_USDT.parquet", "_USDT.parquet", "USDT.parquet", ".parquet"]:
            p = ohlcv_dir / f"{symbol}{suffix}"
            if p.exists():
                path_1h = p
                break
        if not path_1h: return None

        df_1h = pd.read_parquet(path_1h)
        if 'timestamp' not in df_1h.columns and 'open_time' in df_1h.columns: df_1h = df_1h.rename(columns={'open_time':'timestamp'})
        df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], unit='ms') if df_1h['timestamp'].dtype=='int64' else pd.to_datetime(df_1h['timestamp'])
        df_1h = df_1h.sort_values('timestamp')
        df_1h['symbol'] = symbol
        
        # 2. Resample 1D data from 1H for MTF
        df_1d_final = df_1h.set_index('timestamp').resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna().reset_index()
        
        # Calculate features
        # Order: Calculate -> Apply Shift -> Generate Labels
        df = calculate_features(df_1h, df_1d=df_1d_final, btc_df=btc_df)
        # Shift BEFORE labeling as per user's logic
        df = apply_feature_shift(df)
        df = generate_momentum_labels(df, horizon=12, min_pump=0.10)
        
        return df if not df.empty else None
    except Exception as e:
        return f"Error {symbol}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Parallel Sandbox Dataset Pipeline (Threaded)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--limit-symbols", type=int, default=0, help="Limit number of symbols (0=all)")
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel threads")
    args = parser.parse_args()

    SANDBOX_DIR = ROOT_DIR / "data" / "sandbox"
    (SANDBOX_DIR / "processed").mkdir(parents=True, exist_ok=True)
    
    REAL_OHLCV_DIR = ROOT_DIR / "data" / "ohlcv"
    
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        # Load all symbols...
        files = list(REAL_OHLCV_DIR.glob("*.parquet"))
        symbols = sorted(list(set([f.stem.replace("_USDT", "").replace("USDT", "") for f in files])))
            
        if args.limit_symbols > 0:
            symbols = symbols[:args.limit_symbols]
        
    print(f"🚀 Running Threaded Sandbox Pipeline for {len(symbols)} symbols with {args.workers} workers...")
    
    btc_path = None
    for suffix in ["USDT_USDT.parquet", "_USDT.parquet", "USDT.parquet", ".parquet"]:
        p = REAL_OHLCV_DIR / f"BTC{suffix}"
        if p.exists(): btc_path = p; break
            
    btc_df = None
    if btc_path:
        tmp_btc = pd.read_parquet(btc_path)
        if 'timestamp' not in tmp_btc.columns and 'open_time' in tmp_btc.columns: tmp_btc = tmp_btc.rename(columns={'open_time':'timestamp'})
        tmp_btc['timestamp'] = pd.to_datetime(tmp_btc['timestamp'], unit='ms') if tmp_btc['timestamp'].dtype=='int64' else pd.to_datetime(tmp_btc['timestamp'])
        tmp_btc = tmp_btc.sort_values('timestamp')
        btc_df = tmp_btc.copy()
        # Initial calculation for BTC features
        btc_df['ema_200'] = btc_df['close'].ewm(span=200).mean()
        tr = pd.concat([btc_df['high'] - btc_df['low'], abs(btc_df['high'] - btc_df['close'].shift(1)), abs(btc_df['low'] - btc_df['close'].shift(1))], axis=1).max(axis=1)
        dm_pos = btc_df['high'].diff(); dm_neg = -btc_df['low'].diff()
        dm_pos = dm_pos.where((dm_pos > dm_neg) & (dm_pos > 0), 0)
        dm_neg = dm_neg.where((dm_neg > dm_pos) & (dm_neg > 0), 0)
        di_pos = 100 * (dm_pos.rolling(14).mean() / tr.rolling(14).mean().replace(0, np.nan))
        di_neg = 100 * (dm_neg.rolling(14).mean() / tr.rolling(14).mean().replace(0, np.nan))
        btc_df['adx'] = (100 * abs(di_pos - di_neg) / (di_pos + di_neg).replace(0, np.nan)).rolling(14).mean()
        btc_df['log_returns'] = np.log(btc_df['close']/(btc_df['close'].shift(1)+1e-9))

    all_data = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single_symbol, s, btc_df, REAL_OHLCV_DIR): s for s in symbols}
        
        done_count = 0
        for future in as_completed(futures):
            res = future.result()
            if isinstance(res, pd.DataFrame):
                all_data.append(res)
            elif isinstance(res, str) and res.startswith("Error"):
                print(f"❌ {res}")
            
            done_count += 1
            if done_count % 100 == 0 or done_count == len(symbols):
                print(f"[{done_count}/{len(symbols)}] symbols completed...")

    if not all_data:
        print("❌ No data processed!")
        return

    print("=> Combining and saving...")
    df_combined = pd.concat(all_data, ignore_index=True).sort_values(['symbol', 'timestamp'])
    output_path = SANDBOX_DIR / "processed" / "features_1h_btc_context.parquet"
    df_combined.to_parquet(output_path)
    
    print(f"✅ Sandbox dataset created: {output_path}")
    print(f"📊 Rows: {len(df_combined):,}, Symbols: {df_combined['symbol'].nunique()}")

if __name__ == "__main__":
    main()
