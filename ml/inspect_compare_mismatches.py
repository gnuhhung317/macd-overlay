#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output'

# Default filenames from recent compare run
SCANNER_CSV = OUT / 'scanner_logic_profit_compare_200sym_20260305_20260407_scanner_signals.csv'
BT_CSV = ROOT / 'ml' / 'backtest_results_quant_sniper_parity_strict_20260407.csv'
# fallback BT path (older naming)
if not BT_CSV.exists():
    BT_CSV = ROOT / 'ml' / 'backtest_results_quant_sniper_parity_strict_20260407.csv'

print(f"Scanner CSV: {SCANNER_CSV}")
print(f"Backtest CSV: {BT_CSV}")

if not SCANNER_CSV.exists():
    print('Scanner CSV not found; adjust SCANNER_CSV path in script.')
    raise SystemExit(1)
if not BT_CSV.exists():
    print('Backtest CSV not found; adjust BT_CSV path in script.')
    raise SystemExit(1)

scanner = pd.read_csv(SCANNER_CSV, parse_dates=['timestamp'])
backtest = pd.read_csv(BT_CSV, parse_dates=['signal_time','entry_time','exit_time'])

# Normalize symbol and timestamps
scanner['symbol'] = scanner['symbol'].astype(str).str.upper().str.strip()
backtest['symbol'] = backtest['symbol'].astype(str).str.upper().str.strip()
scanner['key'] = scanner['symbol'] + '|' + scanner['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Attempt to normalize backtest symbols to match scanner naming (common USDT suffix)
scanner_symbols = set(scanner['symbol'].unique())

def map_bt_symbol(s: str) -> str:
    s = (s or '').upper().strip()
    if s in scanner_symbols:
        return s
    # common case: backtest uses base ticker (e.g. DEGO) while scanner uses DEGOUSDT
    if not s.endswith('USDT') and (s + 'USDT') in scanner_symbols:
        return s + 'USDT'
    # opposite: backtest includes USDT but scanner used base
    if s.endswith('USDT') and s[:-4] in scanner_symbols:
        return s[:-4]
    return s

backtest['symbol_mapped'] = backtest['symbol'].apply(map_bt_symbol)
backtest['key'] = backtest['symbol_mapped'] + '|' + backtest['signal_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

scanner_keys = set(scanner['key'])
backtest_keys = set(backtest['key'])

matched_keys = scanner_keys & backtest_keys
scanner_only_keys = sorted(scanner_keys - backtest_keys)
backtest_only_keys = sorted(backtest_keys - scanner_keys)

print('--- Summary ---')
print(f'scanner signals: {len(scanner_keys)}')
print(f'backtest signals: {len(backtest_keys)}')
print(f'matched signals: {len(matched_keys)}')
print(f'scanner-only: {len(scanner_only_keys)}')
print(f'backtest-only: {len(backtest_only_keys)}')

# Compute PnL contributions from backtest
# backtest contains pnl_usd per executed trade (some signals may have no trade rows)
bt_pnl_total = backtest['pnl_usd'].sum()
matched_pnl = backtest[backtest['key'].isin(matched_keys)]['pnl_usd'].sum()
backtest_only_pnl = backtest[backtest['key'].isin(backtest_only_keys)]['pnl_usd'].sum()

print('\n--- PnL ---')
print(f'backtest total pnl_usd: {bt_pnl_total:.2f}')
print(f'matched keys pnl_usd: {matched_pnl:.2f}')
print(f'backtest-only pnl_usd: {backtest_only_pnl:.2f}')

# Save mismatch CSVs
scanner_only_df = scanner[scanner['key'].isin(scanner_only_keys)].copy()
backtest_only_df = backtest[backtest['key'].isin(backtest_only_keys)].copy()

scanner_only_out = OUT / 'inspect_mismatch_scanner_only.csv'
backtest_only_out = OUT / 'inspect_mismatch_backtest_only.csv'
scanner_only_df.to_csv(scanner_only_out, index=False)
backtest_only_df.to_csv(backtest_only_out, index=False)

print(f'Wrote scanner-only CSV: {scanner_only_out} ({len(scanner_only_df)})')
print(f'Wrote backtest-only CSV: {backtest_only_out} ({len(backtest_only_df)})')

# Top symbols by mismatch counts
from collections import Counter
sc_only_symbols = Counter(scanner_only_df['symbol'])
bt_only_symbols = Counter(backtest_only_df['symbol'])

print('\nTop scanner-only symbols:')
for sym, cnt in sc_only_symbols.most_common(10):
    print(f'  {sym}: {cnt}')

print('\nTop backtest-only symbols:')
for sym, cnt in bt_only_symbols.most_common(10):
    print(f'  {sym}: {cnt}')

# Top backtest-only symbols by pnl impact
bt_only_pnl_by_sym = backtest_only_df.groupby('symbol')['pnl_usd'].sum().sort_values()
print('\nBacktest-only PnL by symbol (worst first):')
print(bt_only_pnl_by_sym.head(10))

print('\nDone.')
