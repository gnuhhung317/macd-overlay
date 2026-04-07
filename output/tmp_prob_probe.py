import sys
sys.path.append('.')
import pandas as pd
from pathlib import Path
from sniper_bot.config import SniperBotConfig
from sniper_bot.sniper_scanner import SniperScanner
from scripts.compare_scan_backtest import LocalFileDataProcessor
from ml.p3 import RealDataQuantExtractor
from ml.backtest_sniper import _prepare_research_selection_features

cfg = SniperBotConfig.load(Path('sniper_bot/sniper_bot_config.json'))
proc = LocalFileDataProcessor(Path.cwd())
scanner = SniperScanner(cfg, data_processor=proc)

symbols = ['CATIUSDT','GUNUSDT','IDUSDT','INUSDT','IPUSDT','ONUSDT','PIPPINUSDT','RECALLUSDT','TREEUSDT']
sigs = scanner.scan(symbols, '1h')
print('scanner_threshold', scanner.threshold)
print('scanner_signals', len(sigs))

lookback_days = int(getattr(getattr(cfg, 'strategy', None), 'selector_lookback_days', 450))
fetch_start = str(max(lookback_days, 120)) + ' days ago UTC'

for s in sigs:
    sym = s['symbol']
    side_txt = str(s.get('type', '')).upper()
    side = 1 if side_txt == 'LONG' else -1
    entry = float(s.get('signal_price') or s.get('entry_p'))
    decision_ts = pd.to_datetime(s.get('timestamp'), errors='coerce')
    setup_ts = pd.to_datetime((s.get('meta') or {}).get('setup_timestamp'), errors='coerce')
    if pd.isna(setup_ts):
        setup_ts = decision_ts

    df = scanner._fetch_symbol_history(symbol=sym, timeframe='1h', full_fetch_start=fetch_start, required_bars=320)
    if df is None or df.empty:
        continue
    df_calc = df.iloc[:-1].copy()

    extractor = RealDataQuantExtractor(**scanner._build_extractor_kwargs())
    extractor.extract(df_calc, sym, include_future_labels=False)
    setup_df = pd.DataFrame(extractor.dataset)
    if setup_df.empty:
        continue

    setup_df['timestamp'] = pd.to_datetime(setup_df['timestamp'], errors='coerce').dt.tz_localize(None)
    setup_df = setup_df.dropna(subset=['timestamp']).copy()
    setup_df['symbol'] = sym
    setup_df['side'] = pd.to_numeric(setup_df.get('side', 0), errors='coerce')
    setup_df['entry_p'] = pd.to_numeric(setup_df.get('entry_p', 0), errors='coerce')

    target_setup_ts = pd.Timestamp(setup_ts).tz_localize(None)

    pick = setup_df[
        (setup_df['timestamp'] == target_setup_ts)
        & (setup_df['side'] == float(side))
        & ((setup_df['entry_p'] - entry).abs() < 1e-9)
    ].copy()

    if pick.empty:
        pick = setup_df[
            (setup_df['timestamp'] == target_setup_ts)
            & (setup_df['side'] == float(side))
        ].copy()
        if not pick.empty:
            pick = pick.iloc[[int((pick['entry_p'] - entry).abs().argmin())]].copy()

    if pick.empty:
        print(sym, 'no matching setup row found')
        continue

    sc_feat = _prepare_research_selection_features(pick)
    sc_frame = sc_feat.reindex(columns=scanner.features, fill_value=0.0)
    sc_prob = float(scanner.clf.predict_proba(sc_frame)[:, 1][0])

    full_feat = _prepare_research_selection_features(setup_df)
    bt_row = full_feat[
        (pd.to_datetime(full_feat['timestamp']) == target_setup_ts)
        & (pd.to_numeric(full_feat.get('side', 0), errors='coerce') == float(side))
        & ((pd.to_numeric(full_feat.get('entry_p', 0), errors='coerce') - entry).abs() < 1e-9)
    ].copy()

    if bt_row.empty:
        bt_row = full_feat[
            (pd.to_datetime(full_feat['timestamp']) == target_setup_ts)
            & (pd.to_numeric(full_feat.get('side', 0), errors='coerce') == float(side))
        ].copy()
        if not bt_row.empty:
            bt_row = bt_row.iloc[[int((pd.to_numeric(bt_row['entry_p'], errors='coerce') - entry).abs().argmin())]].copy()

    if bt_row.empty:
        print(sym, 'no matching full feature row')
        continue

    bt_frame = bt_row.reindex(columns=scanner.features, fill_value=0.0)
    bt_prob = float(scanner.clf.predict_proba(bt_frame)[:, 1][0])

    print(f"{sym} side={side_txt} decision={decision_ts} setup={target_setup_ts} entry={entry:.9f} prob_scanner_style={sc_prob:.6f} prob_backtest_style={bt_prob:.6f}")
