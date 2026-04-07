import argparse
import contextlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
sys.path.append(str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "sniper_bot"))

from ml.backtest_sniper import (  # noqa: E402
    BacktestConfig,
    _apply_profile_to_config,
    _apply_research_model_selection,
    _normalize_symbol,
    backtest_symbol,
)
from sniper_bot.config import SniperBotConfig  # noqa: E402
from sniper_bot.sniper_scanner import SniperScanner  # noqa: E402


def _strip_symbol(sym: str) -> str:
    return str(sym).upper().replace("_USDT", "").replace("USDT", "")


def _to_usdt(sym: str) -> str:
    core = _strip_symbol(sym)
    return f"{core}USDT"


def _resolve_symbols_dir(universe_mode: str, exchange: str) -> Path:
    if str(universe_mode).lower() == "research":
        return BASE_DIR / "data" / "ohlcv"
    if str(exchange).lower() == "bitget":
        return BASE_DIR / "bitget-data" / "symbols_v3"
    return BASE_DIR / "data" / "processed" / "symbols_v3"


def _discover_files(symbols_dir: Path) -> List[Path]:
    all_files = sorted(symbols_dir.glob("*.parquet"))
    deduped: Dict[str, Path] = {}

    for file_path in all_files:
        symbol = _normalize_symbol(file_path.stem).upper()
        prev = deduped.get(symbol)
        if prev is None:
            deduped[symbol] = file_path
            continue

        prev_name = prev.stem.upper()
        curr_name = file_path.stem.upper()
        prev_has_usdt = prev_name.endswith("USDT") or prev_name.endswith("_USDT")
        curr_has_usdt = curr_name.endswith("USDT") or curr_name.endswith("_USDT")
        if curr_has_usdt and not prev_has_usdt:
            deduped[symbol] = file_path

    return sorted(deduped.values())


def _select_files(
    symbols_dir: Path,
    symbols: Optional[Sequence[str]],
    max_files: int,
) -> List[Path]:
    files = _discover_files(symbols_dir)
    if symbols:
        wanted = {_strip_symbol(s) for s in symbols}
        files = [fp for fp in files if _strip_symbol(fp.stem) in wanted]
    if max_files > 0:
        files = files[: max_files]
    return files


class MockDataProcessor:
    def __init__(self, symbols: Sequence[str], symbols_dir: Path, bars_tail: int = 1200):
        self.current_time: Optional[pd.Timestamp] = None
        self.bars_tail = max(320, int(bars_tail))
        self.data: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            file_name_candidates = [
                f"{symbol}_USDT.parquet",
                f"{symbol}.parquet",
                f"{_strip_symbol(symbol)}USDT_USDT.parquet",
                f"{_strip_symbol(symbol)}USDT.parquet",
            ]
            path = None
            for candidate in file_name_candidates:
                candidate_path = symbols_dir / candidate
                if candidate_path.exists():
                    path = candidate_path
                    break
            if path is None:
                continue

            try:
                df = pd.read_parquet(path)
                if "timestamp" not in df.columns:
                    continue
                df = df.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                if not df.empty:
                    self.data[symbol] = df
            except Exception:
                continue

    def get_historical_data(self, symbol, timeframe, since, until):
        if self.current_time is None or symbol not in self.data:
            return pd.DataFrame()

        base = self.data[symbol]
        open_candle_ts = pd.to_datetime(self.current_time).floor("h")
        sliced = base[base["timestamp"] <= open_candle_ts].copy()

        # Scanner logic drops the last row as the currently-open candle.
        # In replay datasets, the true open bar at `open_candle_ts` may be absent,
        # which would shift `last_closed_ts` one bar too early and cause boundary
        # mismatches (especially near window end). Add a synthetic open bar when
        # needed so replay semantics match live scanning.
        if not sliced.empty:
            last_ts = pd.to_datetime(sliced["timestamp"].iloc[-1], errors="coerce")
            if pd.notna(last_ts):
                last_ts = last_ts.tz_localize(None)
                if last_ts < open_candle_ts:
                    prev = sliced.iloc[-1]
                    synthetic = {col: prev[col] for col in sliced.columns}
                    synthetic["timestamp"] = open_candle_ts
                    close_val = float(pd.to_numeric(prev.get("close"), errors="coerce"))
                    for px_col in ("open", "high", "low", "close"):
                        if px_col in synthetic:
                            synthetic[px_col] = close_val
                    if "volume" in synthetic:
                        synthetic["volume"] = 0.0
                    sliced = pd.concat([sliced, pd.DataFrame([synthetic])], ignore_index=True)

        if timeframe == "1d":
            sliced = (
                sliced.set_index("timestamp")
                .resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
            )

        return sliced.tail(self.bars_tail)


def _build_live_symbols(file_paths: Sequence[Path]) -> List[str]:
    # Scanner and data provider operate with *USDT symbols.
    out = []
    for fp in file_paths:
        out.append(_to_usdt(fp.stem))

    # BTC is useful as anchor symbol in many local datasets.
    if "BTCUSDT" not in out:
        out.append("BTCUSDT")
    return sorted(set(out))


def _collect_scanner_signals(
    cfg: SniperBotConfig,
    symbols_dir: Path,
    symbols: Sequence[str],
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    quiet_scanner: bool,
    parity_mode: str = "execution",
) -> Tuple[List[dict], float]:
    def _norm_ts(value) -> pd.Timestamp:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
        return ts

    bars_tail = max(1200, int(getattr(getattr(cfg, "strategy", None), "scan_history_bars", 1200)))
    processor = MockDataProcessor(symbols=symbols, symbols_dir=symbols_dir, bars_tail=bars_tail)
    scanner = SniperScanner(config=cfg, data_processor=processor)

    hours = pd.date_range(start, end, freq="h")
    all_rows: List[dict] = []

    for ts in hours:
        processor.current_time = ts
        if quiet_scanner:
            with contextlib.redirect_stdout(io.StringIO()):
                signals = scanner.scan(list(symbols), timeframe)
        else:
            signals = scanner.scan(list(symbols), timeframe)
        for sig in signals:
            meta = sig.get("meta") if isinstance(sig.get("meta"), dict) else {}
            decision_ts = _norm_ts(sig.get("timestamp"))
            setup_ts = _norm_ts(meta.get("setup_timestamp", sig.get("timestamp")))
            key_ts = setup_ts if str(parity_mode).lower() == "alert" else decision_ts
            if pd.isna(key_ts):
                continue
            all_rows.append(
                {
                    "timestamp": key_ts,
                    "setup_timestamp": setup_ts,
                    "decision_timestamp": decision_ts,
                    "symbol": _strip_symbol(sig.get("symbol", "")),
                    "side": str(sig.get("type", "")).upper(),
                    "confidence": float(sig.get("confidence", 0.0)),
                    "entry_price": float(sig.get("signal_price", float("nan"))),
                }
            )

    return all_rows, float(getattr(scanner, "threshold", float("nan")))


def _collect_backtest_signals(
    file_paths: Sequence[Path],
    cfg: BacktestConfig,
) -> List[dict]:
    potential_signals: List[dict] = []
    selection_rows: List[pd.DataFrame] = []

    for fp in file_paths:
        sigs, _ohlcv, setup_rows = backtest_symbol(fp, cfg)
        if sigs:
            potential_signals.extend(sigs)
        if setup_rows is not None and not setup_rows.empty:
            selection_rows.append(setup_rows)

    if cfg.use_research_model_selection:
        potential_signals = _apply_research_model_selection(
            potential_signals=potential_signals,
            selection_rows=selection_rows,
            config=cfg,
        )

    rows: List[dict] = []
    for sig in potential_signals:
        # Use setup timestamp for parity with live scanner signal emission.
        ts = sig.get("signal_timestamp", sig.get("timestamp"))
        rows.append(
            {
                "timestamp": pd.to_datetime(ts).tz_localize(None),
                "symbol": _strip_symbol(sig.get("symbol", "")),
                "side": str(sig.get("type", "")).upper(),
                "confidence": float(sig.get("prob", 0.0)),
            }
        )
    return rows


@dataclass
class CompareStats:
    scanner_signals: int
    scanner_raw_signals: int
    backtest_signals: int
    matched: int
    scanner_only: int
    backtest_only: int
    scanner_threshold: float


def _to_key(row: dict) -> Tuple[pd.Timestamp, str, str]:
    return (
        pd.to_datetime(row["timestamp"]).tz_localize(None),
        _strip_symbol(str(row["symbol"])),
        str(row["side"]).upper(),
    )


def _resolve_symbol_file(symbol: str, symbols_dir: Path) -> Optional[Path]:
    candidates = [
        f"{symbol}_USDT.parquet",
        f"{symbol}.parquet",
        f"{_strip_symbol(symbol)}USDT_USDT.parquet",
        f"{_strip_symbol(symbol)}USDT.parquet",
    ]
    for name in candidates:
        p = symbols_dir / name
        if p.exists():
            return p
    return None


def _filter_scanner_fillable(
    scanner_rows: Sequence[dict],
    symbols_dir: Path,
    max_hold_bars: int,
) -> List[dict]:
    """Keep only scanner alerts that would get filled under backtest limit-fill rules."""
    if max_hold_bars <= 0:
        return list(scanner_rows)

    data_cache: Dict[str, pd.DataFrame] = {}
    kept: List[dict] = []

    for row in scanner_rows:
        symbol = _to_usdt(str(row.get("symbol", "")))
        side = str(row.get("side", "")).upper()
        ts = pd.to_datetime(row.get("timestamp")).tz_localize(None)
        entry_p = float(row.get("entry_price", float("nan")))

        if not pd.notna(entry_p):
            continue

        if symbol not in data_cache:
            fp = _resolve_symbol_file(symbol, symbols_dir)
            if fp is None:
                data_cache[symbol] = pd.DataFrame()
            else:
                df = pd.read_parquet(fp)
                if "timestamp" not in df.columns:
                    data_cache[symbol] = pd.DataFrame()
                else:
                    dfx = df.copy()
                    dfx["timestamp"] = pd.to_datetime(dfx["timestamp"], errors="coerce").dt.tz_localize(None)
                    dfx = dfx.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                    data_cache[symbol] = dfx

        df = data_cache.get(symbol)
        if df is None or df.empty:
            continue

        ts_values = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
        hit = df.index[ts_values == ts]
        if len(hit) == 0:
            continue

        i = int(hit[0])
        fut = df.iloc[i + 1 : i + 1 + int(max_hold_bars)]
        if fut.empty:
            continue

        lows = pd.to_numeric(fut.get("low"), errors="coerce") if "low" in fut.columns else pd.Series(dtype=float)
        highs = pd.to_numeric(fut.get("high"), errors="coerce") if "high" in fut.columns else pd.Series(dtype=float)
        if lows.empty or highs.empty:
            continue

        if side == "LONG":
            fillable = bool((lows <= entry_p).any())
        elif side == "SHORT":
            fillable = bool((highs >= entry_p).any())
        else:
            fillable = False

        if fillable:
            kept.append(row)

    return kept


def _compare(scanner_rows: Sequence[dict], backtest_rows: Sequence[dict]):
    scanner_map = {_to_key(r): r for r in scanner_rows}
    backtest_map = {_to_key(r): r for r in backtest_rows}

    scanner_keys = set(scanner_map.keys())
    backtest_keys = set(backtest_map.keys())

    both = sorted(scanner_keys & backtest_keys)
    only_scanner = sorted(scanner_keys - backtest_keys)
    only_backtest = sorted(backtest_keys - scanner_keys)

    return scanner_map, backtest_map, both, only_scanner, only_backtest


def _top_symbol_counts(keys: Sequence[Tuple[pd.Timestamp, str, str]], top_n: int = 15) -> List[Tuple[str, int]]:
    if not keys:
        return []
    ser = pd.Series([k[1] for k in keys], dtype="object")
    vc = ser.value_counts().head(top_n)
    return [(str(sym), int(cnt)) for sym, cnt in vc.items()]


def main():
    parser = argparse.ArgumentParser(description="Compare scanner vs backtest signals on same window")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--config-path", type=str, default=str(BASE_DIR / "sniper_bot" / "sniper_bot_config.json"))
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument("--universe-mode", type=str, default="research", choices=["research", "sniper"])
    parser.add_argument("--exchange", type=str, default="binance")
    # extractor mode removed; backtests always use live_compatible to mirror scanner

    parser.add_argument("--profile-path", type=str, default="")
    parser.add_argument("--profile-name", type=str, default="")
    parser.add_argument("--selector-artifact-path", type=str, default="")
    parser.add_argument("--selector-threshold-override", type=float, default=None)
    parser.add_argument("--progress-log-path", type=str, default="output/scanner_backtest_compare_progress.log")
    parser.add_argument("--no-quiet-scanner", action="store_true")

    parser.add_argument("--output-prefix", type=str, default="output/scanner_backtest_compare")
    parser.add_argument(
        "--parity-mode",
        type=str,
        default="execution",
        choices=["alert", "execution"],
        help="alert=compare raw scanner alerts, execution=compare only scanner alerts that would get filled like backtest",
    )

    args = parser.parse_args()

    start = pd.to_datetime(args.start).tz_localize(None)
    end = pd.to_datetime(args.end).tz_localize(None)
    if end <= start:
        raise ValueError("--end must be after --start")

    cfg = SniperBotConfig.load(Path(args.config_path))
    if args.profile_path:
        cfg.strategy.profile_path = args.profile_path
    if args.profile_name:
        cfg.strategy.profile_name = args.profile_name
    if args.selector_artifact_path:
        cfg.strategy.selector_artifact_path = args.selector_artifact_path
    if args.selector_threshold_override is not None:
        cfg.strategy.selector_threshold_override = float(args.selector_threshold_override)
    # Backward/forward compatible with different config schema revisions.
    if hasattr(cfg.strategy, "progress_detail_log_path"):
        cfg.strategy.progress_detail_log_path = str(args.progress_log_path)

    symbols_dir = _resolve_symbols_dir(args.universe_mode, args.exchange)
    if not symbols_dir.exists():
        raise FileNotFoundError(f"symbols dir not found: {symbols_dir}")

    cli_symbols = [x.strip() for x in str(args.symbols).split(",") if x.strip()]
    file_paths = _select_files(
        symbols_dir=symbols_dir,
        symbols=cli_symbols if cli_symbols else None,
        max_files=max(0, int(args.max_files)),
    )
    if not file_paths:
        raise ValueError("No symbol parquet files selected for comparison")

    live_symbols = _build_live_symbols(file_paths)

    scanner_rows_raw, scanner_threshold = _collect_scanner_signals(
        cfg=cfg,
        symbols_dir=symbols_dir,
        symbols=live_symbols,
        timeframe=args.timeframe,
        start=start,
        end=end,
        quiet_scanner=not bool(args.no_quiet_scanner),
        parity_mode=str(args.parity_mode),
    )

    bt_cfg = BacktestConfig(
        start_date=str(start),
        end_date=str(end),
        exchange=str(args.exchange),
        leverage=float(cfg.exchange.leverage),
        risk_per_trade=float(cfg.risk.max_risk_per_trade),
        max_open_trades=int(cfg.risk.max_open_positions),
        universe_mode=str(args.universe_mode),
        extractor_mode="live_compatible",
        top_coins=[_to_usdt(fp.stem) for fp in file_paths],
        max_files=0,
        use_research_model_selection=True,
        selector_artifact_path=str(cfg.strategy.selector_artifact_path),
        threshold=None,
    )

    # Align extractor params with the same profile the scanner is using.
    profile_path = Path(str(cfg.strategy.profile_path))
    if not profile_path.is_absolute():
        profile_path = BASE_DIR / profile_path
    _apply_profile_to_config(
        bt_cfg,
        profile_path=profile_path,
        profile_name=str(cfg.strategy.profile_name) if cfg.strategy.profile_name else None,
    )

    # Match scanner threshold behavior: explicit override wins, otherwise use scanner's effective threshold.
    if args.selector_threshold_override is not None and args.selector_threshold_override >= 0:
        bt_cfg.threshold = float(args.selector_threshold_override)
    elif pd.notna(scanner_threshold) and float(scanner_threshold) >= 0:
        bt_cfg.threshold = float(scanner_threshold)

    backtest_rows = _collect_backtest_signals(file_paths=file_paths, cfg=bt_cfg)

    if str(args.parity_mode).lower() == "execution":
        scanner_rows = _filter_scanner_fillable(
            scanner_rows=scanner_rows_raw,
            symbols_dir=symbols_dir,
            max_hold_bars=int(bt_cfg.max_bars_hold),
        )
    else:
        scanner_rows = list(scanner_rows_raw)

    scanner_map, backtest_map, both, only_scanner, only_backtest = _compare(
        scanner_rows=scanner_rows,
        backtest_rows=backtest_rows,
    )

    stats = CompareStats(
        scanner_signals=len(scanner_map),
        scanner_raw_signals=len(scanner_rows_raw),
        backtest_signals=len(backtest_map),
        matched=len(both),
        scanner_only=len(only_scanner),
        backtest_only=len(only_backtest),
        scanner_threshold=float(scanner_threshold),
    )

    print("=== Scanner vs Backtest Signal Compare ===")
    print(f"window={start} -> {end} tf={args.timeframe}")
    print(f"symbols_dir={symbols_dir}")
    print(f"symbols_selected={len(file_paths)}")
    print(f"extractor_mode={bt_cfg.extractor_mode}")
    print(f"parity_mode={args.parity_mode}")
    print(f"scanner_threshold={stats.scanner_threshold:.4f}")
    print(f"scanner_raw_signals={stats.scanner_raw_signals}")
    print(f"scanner_signals={stats.scanner_signals}")
    print(f"backtest_signals={stats.backtest_signals}")
    print(f"matched={stats.matched}")
    print(f"scanner_only={stats.scanner_only}")
    print(f"backtest_only={stats.backtest_only}")

    if only_scanner:
        print("\nTop scanner-only symbols:")
        for sym, cnt in _top_symbol_counts(only_scanner):
            print(f"  {sym}: {cnt}")

    if only_backtest:
        print("\nTop backtest-only symbols:")
        for sym, cnt in _top_symbol_counts(only_backtest):
            print(f"  {sym}: {cnt}")

    output_base = BASE_DIR / args.output_prefix
    output_base.parent.mkdir(parents=True, exist_ok=True)

    scanner_df = pd.DataFrame(scanner_rows)
    backtest_df = pd.DataFrame(backtest_rows)

    only_scanner_rows = [
        {
            "timestamp": k[0],
            "symbol": k[1],
            "side": k[2],
            "source": "scanner_only",
            "confidence": float(scanner_map[k].get("confidence", 0.0)),
        }
        for k in only_scanner
    ]
    only_backtest_rows = [
        {
            "timestamp": k[0],
            "symbol": k[1],
            "side": k[2],
            "source": "backtest_only",
            "confidence": float(backtest_map[k].get("confidence", 0.0)),
        }
        for k in only_backtest
    ]
    matched_rows = [
        {
            "timestamp": k[0],
            "symbol": k[1],
            "side": k[2],
            "source": "matched",
            "scanner_confidence": float(scanner_map[k].get("confidence", 0.0)),
            "backtest_confidence": float(backtest_map[k].get("confidence", 0.0)),
        }
        for k in both
    ]
    diff_df = pd.DataFrame(only_scanner_rows + only_backtest_rows + matched_rows)

    scanner_csv = output_base.with_name(output_base.name + "_scanner.csv")
    backtest_csv = output_base.with_name(output_base.name + "_backtest.csv")
    diff_csv = output_base.with_name(output_base.name + "_diff.csv")
    summary_json = output_base.with_name(output_base.name + "_summary.json")

    scanner_df.to_csv(scanner_csv, index=False)
    backtest_df.to_csv(backtest_csv, index=False)
    diff_df.to_csv(diff_csv, index=False)

    summary_payload = {
        "stats": asdict(stats),
        "window": {"start": str(start), "end": str(end), "timeframe": args.timeframe},
        "symbols_dir": str(symbols_dir),
        "symbols_selected": len(file_paths),
        "extractor_mode": str(bt_cfg.extractor_mode),
        "profile_path": str(cfg.strategy.profile_path),
        "profile_name": str(cfg.strategy.profile_name),
        "selector_artifact_path": str(cfg.strategy.selector_artifact_path),
        "top_scanner_only": _top_symbol_counts(only_scanner),
        "top_backtest_only": _top_symbol_counts(only_backtest),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\nArtifacts:")
    print(f"  scanner: {scanner_csv}")
    print(f"  backtest: {backtest_csv}")
    print(f"  diff: {diff_csv}")
    print(f"  summary: {summary_json}")


if __name__ == "__main__":
    main()
