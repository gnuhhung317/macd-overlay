import pandas as pd
from pathlib import Path
import numpy as np

BASE = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
import sys
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'sniper_bot'))

from ml.backtest_sniper import _prepare_research_selection_features as backtest_prep
try:
    from sniper_bot.sniper_scanner import _prepare_research_selection_features as scanner_prep
except Exception:
    scanner_prep = None


def load_parquet_for_symbol(sym):
    # sym like BTCUSDT or DEGOUSDT
    stem = sym.replace('USDT', '')
    folder = BASE / 'data' / 'ohlcv'
    # try preferred exact patterns first
    candidates = []
    p_exact = folder / f"{sym}.parquet"
    p_sym_usdt = folder / f"{stem}USDT.parquet"
    p_stem = folder / f"{stem}.parquet"
    if p_exact.exists():
        p = p_exact
    elif p_sym_usdt.exists():
        p = p_sym_usdt
    elif p_stem.exists():
        p = p_stem
    else:
        # fallback: glob any file containing the stem
        candidates = list(folder.glob(f"*{stem}*.parquet"))
        if not candidates:
            print('parquet not found for', sym, folder)
            return None
        # prefer filenames that start with the stem or contain the full symbol
        candidates_sorted = sorted(candidates, key=lambda x: (0 if x.name.startswith(stem) else 1, len(x.name)))
        p = candidates_sorted[0]

    df = pd.read_parquet(p)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    return df


def sample_setups_from_df(df, n=50):
    # crude: take last n rows and craft setup rows with required fields
    tail = df.tail(n).copy()
    # create fake setup columns needed by feature prep
    tail['entry_p'] = tail['close']
    tail['sl_p'] = tail['low'] * 0.995
    tail['tp_p'] = tail['high'] * 1.01
    tail['structure_size'] = 0.02
    tail['side'] = 1
    tail['z_trend_20_50'] = (tail['close'] - tail['close'].rolling(20).mean()) / (tail['close'].rolling(20).std()+1e-12)
    tail['z_price_to_ema200'] = (tail['close'] - tail['close'].ewm(span=200).mean())/(tail['close'].rolling(200).std()+1e-12)
    tail['z_volatility_atr'] = (tail['high'] - tail['low'])/ (tail['close']+1e-12)
    tail['pullback_depth'] = 0.0
    tail['dist_to_sl_pct'] = 0.0
    return tail


SYMS = ['BTCUSDT','ETHUSDT','DEGOUSDT','DRIFTUSDT','JELLYJELLYUSDT','FUNUSDT']

results = []
for s in SYMS:
    df = load_parquet_for_symbol(s)
    if df is None:
        continue
    setups = sample_setups_from_df(df, n=60)
    # keep only required cols
    cols = ['timestamp','entry_p','sl_p','tp_p','structure_size','side','z_trend_20_50','z_price_to_ema200','z_volatility_atr','pullback_depth','dist_to_sl_pct']
    setups = setups[cols].reset_index(drop=True)

    b = backtest_prep(setups.copy())
    s_p = None
    if scanner_prep is not None:
        s_p = scanner_prep(setups.copy())

    # compare numeric columns
    if s_p is None:
        print('Scanner prep not available, skipping scanner comparison')
        continue

    cmp_cols = [c for c in b.columns if c in s_p.columns and b[c].dtype.kind in 'fc']
    diffs = {}
    for c in cmp_cols:
        a = b[c].to_numpy(dtype=float)
        d = s_p[c].to_numpy(dtype=float)
        # compute max absolute and relative diff
        abs_max = float(np.nanmax(np.abs(a - d)))
        rel_max = float(np.nanmax(np.abs((a - d) / (np.where(np.abs(a) < 1e-12, 1e-12, a)))))
        diffs[c] = {'abs_max': abs_max, 'rel_max': rel_max}

    results.append({'symbol': s, 'diffs': diffs})

# print summary
for r in results:
    print('Symbol:', r['symbol'])
    for k,v in r['diffs'].items():
        if v['abs_max']>1e-8 or v['rel_max']>1e-6:
            print(f"  {k}: abs_max={v['abs_max']:.6g}, rel_max={v['rel_max']:.6g}")
    print('---')

# Save results
import json
outp = BASE / 'output' / 'quick_feature_parity.json'
outp.parent.mkdir(parents=True, exist_ok=True)
with open(outp, 'w') as f:
    json.dump(results, f, indent=2)
print('Saved parity to', outp)
