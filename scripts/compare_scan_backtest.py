#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
import pandas as pd

# Ensure project root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sniper_bot.config import SniperBotConfig
from sniper_bot.sniper_scanner import SniperScanner
from ml.backtest_sniper import BacktestConfig, run_backtest_with_config, _apply_profile_to_config


class LocalFileDataProcessor:
    def __init__(self, base_dir: Path = None):
        self.base = (Path(base_dir) if base_dir else Path.cwd()).resolve()
        self.ohlcv = self.base / "data" / "ohlcv"
        self.processed = self.base / "data" / "processed" / "symbols_v3"

    def _find_file(self, symbol: str) -> Path | None:
        cand = [
            self.ohlcv / f"{symbol}_USDT.parquet",
            self.ohlcv / f"{symbol}.parquet",
            self.ohlcv / f"{symbol}_USDT_USDT.parquet",
        ]
        for p in cand:
            if p.exists():
                return p

        # fallback: any matching prefix
        if self.ohlcv.exists():
            for p in self.ohlcv.glob(f"{symbol}*.parquet"):
                return p
        if self.processed.exists():
            for p in self.processed.glob(f"{symbol}*.parquet"):
                return p
        return None

    def _parse_rel(self, v: str):
        if not v:
            return None
        s = str(v).strip().lower()
        if s == "now utc":
            return pd.Timestamp.utcnow().tz_localize(None)
        if "days ago" in s:
            try:
                n = int(s.split()[0])
                return (pd.Timestamp.utcnow() - pd.Timedelta(days=n)).tz_localize(None)
            except Exception:
                ts = pd.to_datetime(v, errors="coerce")
                if pd.isna(ts):
                    return ts
                if getattr(ts, "tzinfo", None) is not None:
                    return ts.tz_localize(None)
                return ts
        ts = pd.to_datetime(v, errors="coerce")
        if pd.isna(ts):
            return ts
        if getattr(ts, "tzinfo", None) is not None:
            return ts.tz_localize(None)
        return ts

    def get_historical_data(self, symbol: str, timeframe: str, start_date=None, end_date=None):
        p = self._find_file(symbol)
        if p is None:
            return pd.DataFrame()

        try:
            df = pd.read_parquet(p)
        except Exception:
            return pd.DataFrame()

        if "timestamp" not in df.columns:
            # try reset index
            if df.index.name in ("timestamp", "time"):
                df = df.reset_index()
            else:
                return pd.DataFrame()

        df = df.copy()
        # Normalize to timezone-naive UTC-like timestamps for consistent comparisons.
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
        if start_date:
            sd = self._parse_rel(start_date)
            if pd.notna(sd):
                df = df[df["timestamp"] >= sd]
        if end_date:
            ed = self._parse_rel(end_date)
            if pd.notna(ed):
                df = df[df["timestamp"] <= ed]

        return df.sort_values("timestamp").reset_index(drop=True)


def normalize_sig(sig: dict) -> dict:
    out = {}
    sym = str(sig.get("symbol", "")).upper().replace("_USDT", "").replace("USDT", "")
    out["symbol"] = f"{sym}USDT"
    ts = sig.get("timestamp") or sig.get("signal_timestamp") or sig.get("entry_timestamp")
    out["timestamp"] = pd.to_datetime(ts, errors="coerce", utc=True)
    if pd.notna(out["timestamp"]):
        out["timestamp"] = out["timestamp"].tz_localize(None)

    side = str(sig.get("type", "")).upper().strip()
    if side not in ("LONG", "SHORT"):
        side_val = sig.get("side")
        try:
            side_num = int(side_val)
            side = "LONG" if side_num == 1 else "SHORT"
        except Exception:
            side = ""
    out["side"] = side

    # entry price can be named differently in scanner vs backtest
    ep = sig.get("entry_p") if sig.get("entry_p") is not None else sig.get("signal_price")
    try:
        out["entry_p"] = float(ep) if ep is not None else None
    except Exception:
        out["entry_p"] = None
    out["raw"] = sig
    return out


def compare(scanner_sigs, backtest_sigs):
    s_map = {}
    for s in scanner_sigs:
        n = normalize_sig(s)
        if pd.isna(n["timestamp"]):
            continue
        key = (n["symbol"], int(pd.Timestamp(n["timestamp"]).value), n.get("side", ""))
        s_map[key] = n

    b_map = {}
    for b in backtest_sigs:
        n = normalize_sig(b)
        if pd.isna(n["timestamp"]):
            continue
        key = (n["symbol"], int(pd.Timestamp(n["timestamp"]).value), n.get("side", ""))
        b_map[key] = n

    matches = []
    s_only = []
    b_only = []

    for k, v in s_map.items():
        if k in b_map:
            # compare entry prices
            bp = b_map[k]["entry_p"]
            sp = v["entry_p"]
            close = False
            if bp is None or sp is None:
                close = True
            else:
                try:
                    close = abs(bp - sp) <= max(1e-6, abs(bp) * 1e-4)
                except Exception:
                    close = False

            matches.append({"key": k, "scanner": v, "backtest": b_map[k], "price_close": close})
        else:
            s_only.append(v)

    for k, v in b_map.items():
        if k not in s_map:
            b_only.append(v)

    return {"matches": matches, "scanner_only": s_only, "backtest_only": b_only}


def build_latest_closed_ts_map(proc: LocalFileDataProcessor, symbols, timeframe: str, lookback_days: int):
    out = {}
    fetch_start = f"{max(int(lookback_days), 120)} days ago UTC"
    for sym in symbols:
        df = proc.get_historical_data(sym, timeframe, fetch_start, "now UTC")
        if df is None or df.empty or len(df) < 2:
            continue
        df = df.sort_values("timestamp").reset_index(drop=True)
        ts = pd.to_datetime(df["timestamp"].iloc[-2], errors="coerce")
        if pd.isna(ts):
            continue
        out[str(sym).upper()] = pd.Timestamp(ts).tz_localize(None)
    return out


def filter_backtest_latest(backtest_sigs, latest_closed_ts_map):
    if not latest_closed_ts_map:
        return []
    out = []
    for sig in backtest_sigs:
        n = normalize_sig(sig)
        sym = n.get("symbol", "")
        ts = n.get("timestamp")
        if not sym or pd.isna(ts):
            continue
        latest_ts = latest_closed_ts_map.get(sym)
        if latest_ts is None:
            continue
        if pd.Timestamp(ts) == pd.Timestamp(latest_ts):
            out.append(sig)
    return out


def signal_brief(sig: dict) -> dict:
    n = normalize_sig(sig)
    ts_txt = ""
    if pd.notna(n.get("timestamp")):
        ts_txt = pd.Timestamp(n["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "symbol": n.get("symbol", ""),
        "timestamp": ts_txt,
        "side": n.get("side", ""),
        "entry_p": n.get("entry_p"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="sniper_bot/sniper_bot_config.json")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--backtest-window-days", type=int, default=14)
    parser.add_argument("--all-history", action="store_true", help="Compare against all backtest timestamps instead of latest closed candle only.")
    parser.add_argument("--out", type=str, default="output/compare_scan_backtest.json")
    args = parser.parse_args()

    cfg = SniperBotConfig.load(Path(args.config))
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = [s.upper() for s in getattr(cfg, "coins", [])]

    # Use local file processor so no network calls are made
    proc = LocalFileDataProcessor(ROOT)
    scanner = SniperScanner(cfg, data_processor=proc)
    lookback_days = int(getattr(getattr(cfg, "strategy", None), "selector_lookback_days", 450))
    latest_closed_ts_map = build_latest_closed_ts_map(proc, symbols, args.timeframe, lookback_days)
    latest_anchor_ts = None
    if latest_closed_ts_map:
        latest_anchor_ts = max(pd.to_datetime(v).tz_localize(None) for v in latest_closed_ts_map.values())
    if latest_anchor_ts is None:
        latest_anchor_ts = pd.Timestamp.utcnow().tz_localize(None)

    print(f"Running scanner on {len(symbols)} symbols: {symbols} (tf={args.timeframe})")
    scanner_results = scanner.scan(symbols, args.timeframe)
    print(f"Scanner produced {len(scanner_results)} signals")

    # Prepare backtest config
    bt = BacktestConfig()
    bt.top_coins = [s.upper() for s in symbols]
    bt.max_files = max(1, len(symbols))
    bt.extractor_mode = "causal"
    bt.selection_mode = "sniper"
    # Use raw ohlcv parquet (research universe) so backtest reads freshly fetched files
    bt.universe_mode = "research"
    bt.start_date = (latest_anchor_ts - pd.Timedelta(days=int(args.backtest_window_days))).strftime("%Y-%m-%d %H:%M:%S")
    bt.end_date = latest_anchor_ts.strftime("%Y-%m-%d %H:%M:%S")
    bt.use_research_model_selection = True
    bt.selector_artifact_path = str(getattr(getattr(cfg, "strategy", None), "selector_artifact_path", ""))
    bt.threshold = float(getattr(scanner, "threshold", 0.65))

    print(
        "Backtest window anchored to latest closed candle: "
        f"start={bt.start_date} end={bt.end_date}"
    )

    profile_path_raw = str(getattr(getattr(cfg, "strategy", None), "profile_path", "")).strip()
    profile_name = str(getattr(getattr(cfg, "strategy", None), "profile_name", "")).strip() or None
    if profile_path_raw:
        profile_path = Path(profile_path_raw)
        if not profile_path.is_absolute():
            profile_path = ROOT / profile_path
        if profile_path.exists():
            _apply_profile_to_config(bt, profile_path=profile_path, profile_name=profile_name)

    print("Running backtest_sniper in live_compatible extractor mode (local files)...")
    potential_signals, full_price_db, trades, curve = run_backtest_with_config(bt)
    print(f"Backtest produced {len(potential_signals)} potential signals and {len(trades)} trades")

    if args.all_history:
        backtest_compare_sigs = potential_signals
        print("Using all backtest timestamps for comparison.")
    else:
        backtest_compare_sigs = filter_backtest_latest(potential_signals, latest_closed_ts_map)
        print(f"Using latest-candle parity filter: backtest reduced to {len(backtest_compare_sigs)} signals")

    report = compare(scanner_results, backtest_compare_sigs)

    match_keys = [
        {
            "symbol": m["scanner"].get("symbol", ""),
            "timestamp": pd.Timestamp(m["scanner"]["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(m["scanner"].get("timestamp"))
            else "",
            "side": m["scanner"].get("side", ""),
            "price_close": bool(m.get("price_close", False)),
        }
        for m in report["matches"]
    ]
    scanner_only_keys = [signal_brief(s.get("raw", {})) for s in report["scanner_only"]]
    backtest_only_keys = [signal_brief(s.get("raw", {})) for s in report["backtest_only"]]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": "all_history" if args.all_history else "latest_only",
            "scanner_count": len(scanner_results),
            "backtest_count_total": len(potential_signals),
            "backtest_count_compare": len(backtest_compare_sigs),
            "matches": len(report["matches"]),
            "scanner_only": len(report["scanner_only"]),
            "backtest_only": len(report["backtest_only"]),
            "latest_closed_ts": {k: str(v) for k, v in latest_closed_ts_map.items()},
            "scanner_signals": [signal_brief(s) for s in scanner_results],
            "backtest_compare_signals": [signal_brief(s) for s in backtest_compare_sigs],
            "match_keys": match_keys,
            "scanner_only_keys": scanner_only_keys,
            "backtest_only_keys": backtest_only_keys,
        }, f, default=str, indent=2)

    print("Comparison summary:")
    print(
        f"  scanner={len(scanner_results)} | backtest_total={len(potential_signals)} "
        f"| backtest_compare={len(backtest_compare_sigs)} | matches={len(report['matches'])}"
    )
    print(f"  scanner_only={len(report['scanner_only'])} | backtest_only={len(report['backtest_only'])}")
    print(f"Wrote summary to: {out_path}")


if __name__ == "__main__":
    main()
