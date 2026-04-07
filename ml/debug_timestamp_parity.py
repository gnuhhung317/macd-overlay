import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "sniper_bot"))

from sniper_bot.config import SniperBotConfig
from sniper_bot.sniper_scanner import SniperScanner
from ml.backtest_sniper import BacktestConfig, backtest_symbol, _apply_profile_to_config

# Import comparator helpers
comp = importlib.import_module("ml.compare_scanner_backtest_window")


def _strip_symbol(sym: str) -> str:
    return str(sym).upper().replace("_USDT", "").replace("USDT", "")


def _to_usdt(sym: str) -> str:
    return f"{_strip_symbol(sym)}USDT"


def parse_ts(x):
    try:
        if x is None or x == "":
            return pd.NaT
        return pd.to_datetime(x, errors="coerce").tz_localize(None)
    except Exception:
        return pd.NaT


def run_debug(symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str, config_path: str, universe_mode: str, exchange: str):
    cfg = SniperBotConfig.load(Path(config_path))

    symbols_dir = comp._resolve_symbols_dir(universe_mode, exchange)
    core = _strip_symbol(symbol)
    usdt = _to_usdt(core)

    fp = comp._resolve_symbol_file(core, symbols_dir)
    if fp is None:
        print(f"Symbol parquet not found for: {symbol} in {symbols_dir}")
        return

    print(f"Using file: {fp}")

    # Collect scanner signals (per-hour scan over window)
    processor = comp.MockDataProcessor(symbols=[usdt], symbols_dir=symbols_dir, bars_tail=2000)
    scanner = SniperScanner(config=cfg, data_processor=processor)

    scanner_all = []
    for ts in pd.date_range(start, end, freq="h"):
        processor.current_time = ts
        sigs = scanner.scan([usdt], timeframe)
        for s in sigs:
            scanner_all.append(s)

    print(f"Collected {len(scanner_all)} scanner signals for {usdt}")

    # Collect backtest signals for the symbol file (fixed live_compatible extractor)
    bt_cfg = BacktestConfig(start_date=str(start), end_date=str(end), extractor_mode="live_compatible", universe_mode=universe_mode, exchange=exchange)

    # Align backtest profile/extractor params with the scanner's profile
    profile_path = Path(str(getattr(getattr(cfg, 'strategy', None), 'profile_path', '')))
    if profile_path:
        if not profile_path.is_absolute():
            profile_path = BASE_DIR / profile_path
        try:
            _apply_profile_to_config(bt_cfg, profile_path=profile_path, profile_name=str(getattr(getattr(cfg, 'strategy', None), 'profile_name', '') or None))
            print(f"Applied profile to backtest config: {profile_path}")
        except Exception as e:
            print(f"Warning: failed to apply profile to backtest config: {e}")
    pot, df, setup_rows = backtest_symbol(fp, bt_cfg)
    pot = pot or []
    print(f"Collected {len(pot)} backtest potential_signals from backtest_symbol")
    if df is not None and not df.empty:
        print("\nBacktest OHLCV tail (backtest_symbol df):")
        try:
            print(df.tail(12).astype(str).to_string(index=False))
        except Exception:
            print(df.tail(12))

    # Build DataFrames for easier comparison
    s_rows = []
    for sig in scanner_all:
        meta = sig.get("meta") if isinstance(sig.get("meta"), dict) else {}
        s_rows.append(
            {
                "scanner_timestamp": parse_ts(sig.get("timestamp")),
                "scanner_type": sig.get("type"),
                "scanner_price": sig.get("signal_price"),
                "scanner_conf": sig.get("confidence"),
                "meta_setup_ts": parse_ts(meta.get("setup_timestamp")),
                "meta_decision_ts": parse_ts(meta.get("decision_timestamp")),
                "raw_meta": meta,
            }
        )

    b_rows = []
    for sig in pot:
        b_rows.append(
            {
                "bt_timestamp": parse_ts(sig.get("timestamp")),
                "bt_signal_ts": parse_ts(sig.get("signal_timestamp")),
                "bt_entry_ts": parse_ts(sig.get("entry_timestamp")),
                "bt_type": sig.get("type"),
                "bt_entry_p": sig.get("entry_p"),
            }
        )

    s_df = pd.DataFrame(s_rows)
    b_df = pd.DataFrame(b_rows)

    # Show what scanner's MockDataProcessor loaded for visibility
    provider_df = None
    try:
        provider_df = processor.data.get(usdt)
    except Exception:
        provider_df = None
    if provider_df is not None and not provider_df.empty:
        print("\nScanner provider loaded price tail (processor.data):")
        try:
            print(provider_df.tail(12).astype(str).to_string(index=False))
        except Exception:
            print(provider_df.tail(12))

    print("\nScanner signals (sample):")
    if not s_df.empty:
        print(s_df.astype(str).head(20).to_string(index=False))
    else:
        print("  none")

    print("\nBacktest potential_signals (sample):")
    if not b_df.empty:
        print(b_df.astype(str).head(20).to_string(index=False))
    else:
        print("  none")

    # Matching by setup timestamp
    s_keys_setup = set()
    for _, r in s_df.iterrows():
        ts = r.get("meta_setup_ts")
        typ = r.get("scanner_type")
        if pd.isna(ts):
            continue
        s_keys_setup.add((pd.to_datetime(ts), str(typ).upper()))

    b_keys_setup = set()
    for _, r in b_df.iterrows():
        ts = r.get("bt_signal_ts")
        typ = r.get("bt_type")
        if pd.isna(ts):
            continue
        b_keys_setup.add((pd.to_datetime(ts), str(typ).upper()))

    both_setup = s_keys_setup & b_keys_setup
    only_scanner_setup = s_keys_setup - b_keys_setup
    only_backtest_setup = b_keys_setup - s_keys_setup

    # Matching by decision/entry timestamp
    s_keys_dec = set()
    for _, r in s_df.iterrows():
        ts = r.get("scanner_timestamp")
        typ = r.get("scanner_type")
        if pd.isna(ts):
            continue
        s_keys_dec.add((pd.to_datetime(ts), str(typ).upper()))

    b_keys_dec = set()
    for _, r in b_df.iterrows():
        ts = r.get("bt_timestamp")
        typ = r.get("bt_type")
        if pd.isna(ts):
            continue
        b_keys_dec.add((pd.to_datetime(ts), str(typ).upper()))

    both_dec = s_keys_dec & b_keys_dec
    only_scanner_dec = s_keys_dec - b_keys_dec
    only_backtest_dec = b_keys_dec - s_keys_dec

    print("\nMatch summary:")
    print(f"By setup_ts: scanner={len(s_keys_setup)} backtest={len(b_keys_setup)} matched={len(both_setup)} scanner_only={len(only_scanner_setup)} backtest_only={len(only_backtest_setup)}")
    print(f"By decision_ts: scanner={len(s_keys_dec)} backtest={len(b_keys_dec)} matched={len(both_dec)} scanner_only={len(only_scanner_dec)} backtest_only={len(only_backtest_dec)}")

    if only_scanner_dec:
        print("\nScanner-only (decision_ts) samples:")
        for k in list(only_scanner_dec)[:10]:
            print(f"  {k}")

    if only_backtest_dec:
        print("\nBacktest-only (decision_ts) samples:")
        for k in list(only_backtest_dec)[:10]:
            print(f"  {k}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--config-path", type=str, default=str(BASE_DIR / "sniper_bot" / "sniper_bot_config.json"))
    parser.add_argument("--universe-mode", type=str, default="research", choices=["research", "sniper"])
    parser.add_argument("--exchange", type=str, default="binance")
    # extractor mode removed: backtest debug always uses live_compatible to mirror scanner
    args = parser.parse_args()

    start = pd.to_datetime(args.start).tz_localize(None)
    end = pd.to_datetime(args.end).tz_localize(None)

    run_debug(symbol=args.symbol, start=start, end=end, timeframe=args.timeframe, config_path=args.config_path, universe_mode=args.universe_mode, exchange=args.exchange)
