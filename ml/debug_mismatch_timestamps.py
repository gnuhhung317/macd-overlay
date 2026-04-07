import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")

import sys

sys.path.append(str(BASE_DIR))

from ml.backtest_sniper import (  # noqa: E402
    BacktestConfig,
    _apply_profile_to_config,
    _apply_research_model_selection,
    _normalize_symbol,
    backtest_symbol,
)
from ml.p3 import RealDataQuantExtractor  # noqa: E402


def _normalize_key_symbol(symbol: str) -> str:
    return str(symbol).upper().replace("_USDT", "").replace("USDT", "")


def _to_key(symbol: str, ts: pd.Timestamp, side: str) -> str:
    return f"{_normalize_key_symbol(symbol)}|{pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')}|{str(side).upper()}"


def _parse_key(key: str) -> Tuple[str, pd.Timestamp, str]:
    parts = str(key).split("|")
    if len(parts) != 3:
        raise ValueError(f"Invalid key format: {key}")
    symbol = _normalize_key_symbol(parts[0])
    ts = pd.to_datetime(parts[1]).tz_localize(None)
    side = str(parts[2]).upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid side in key: {key}")
    return symbol, ts, side


def _resolve_symbols_dir(universe_mode: str, exchange: str) -> Path:
    if str(universe_mode).lower() == "research":
        return BASE_DIR / "data" / "ohlcv"
    if str(exchange).lower() == "bitget":
        return BASE_DIR / "bitget-data" / "symbols_v3"
    return BASE_DIR / "data" / "processed" / "symbols_v3"


def _resolve_symbol_file(symbol: str, symbols_dir: Path) -> Optional[Path]:
    s = _normalize_key_symbol(symbol)
    candidates = [
        f"{s}_USDT.parquet",
        f"{s}.parquet",
        f"{s}USDT_USDT.parquet",
        f"{s}USDT.parquet",
    ]
    for name in candidates:
        p = symbols_dir / name
        if p.exists():
            return p
    return None


@dataclass
class ScanCycle:
    start_line: int
    end_line: int
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    symbols: int
    log_file: str


def _parse_scan_log(scan_log_path: Path):
    starts: List[Dict] = []
    ends: List[Dict] = []
    symbol_done_rows: List[Dict] = []

    lines = scan_log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for idx, line in enumerate(lines, start=1):
        m_start = pd.Series([line]).str.extract(
            r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| scan_start \| timeframe=(?P<tf>[^|]+) \| symbols=(?P<symbols>\d+)"
        )
        if not m_start.isna().all(axis=None):
            starts.append(
                {
                    "line": idx,
                    "ts": pd.to_datetime(m_start.iloc[0]["ts"]).tz_localize(None),
                    "symbols": int(m_start.iloc[0]["symbols"]),
                }
            )
            continue

        m_end = pd.Series([line]).str.extract(
            r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| scan_end \| timeframe=(?P<tf>[^|]+) \| symbols=(?P<symbols>\d+) \| total_signals=(?P<sig>\d+) \| elapsed_s=(?P<el>[^|]+) \| log_file=(?P<lf>.+)$"
        )
        if not m_end.isna().all(axis=None):
            ends.append(
                {
                    "line": idx,
                    "ts": pd.to_datetime(m_end.iloc[0]["ts"]).tz_localize(None),
                    "symbols": int(m_end.iloc[0]["symbols"]),
                    "log_file": str(m_end.iloc[0]["lf"]).strip(),
                }
            )
            continue

        m_done = pd.Series([line]).str.extract(
            r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| symbol_done \| idx=(?P<idx>\d+) \| total=(?P<total>\d+) \| symbol=(?P<symbol>[^|]+) \| status=(?P<status>[^|]+) \| reason=(?P<reason>[^|]+) \| signals=(?P<signals>[^|]+)"
        )
        if not m_done.isna().all(axis=None):
            symbol_done_rows.append(
                {
                    "line": idx,
                    "ts": pd.to_datetime(m_done.iloc[0]["ts"]).tz_localize(None),
                    "symbol": _normalize_key_symbol(str(m_done.iloc[0]["symbol"])),
                    "status": str(m_done.iloc[0]["status"]).strip(),
                    "reason": str(m_done.iloc[0]["reason"]).strip(),
                    "signals": int(float(str(m_done.iloc[0]["signals"]).strip() or 0.0)),
                }
            )

    cycles: List[ScanCycle] = []
    end_ptr = 0
    for s in starts:
        while end_ptr < len(ends) and ends[end_ptr]["line"] < s["line"]:
            end_ptr += 1
        if end_ptr >= len(ends):
            break
        e = ends[end_ptr]
        cycles.append(
            ScanCycle(
                start_line=int(s["line"]),
                end_line=int(e["line"]),
                start_ts=pd.to_datetime(s["ts"]),
                end_ts=pd.to_datetime(e["ts"]),
                symbols=int(e["symbols"]),
                log_file=str(e["log_file"]),
            )
        )
        end_ptr += 1

    def _cycle_for_line(line_no: int) -> Optional[ScanCycle]:
        for c in cycles:
            if c.start_line <= line_no <= c.end_line:
                return c
        return None

    for row in symbol_done_rows:
        c = _cycle_for_line(int(row["line"]))
        row["cycle_log_file"] = c.log_file if c else ""
        row["cycle_symbols"] = c.symbols if c else None
        row["cycle_start_ts"] = c.start_ts if c else None
        row["cycle_end_ts"] = c.end_ts if c else None

    return cycles, symbol_done_rows


def _pick_nearest_live_reason(
    symbol_done_rows: Sequence[Dict],
    symbol: str,
    ts: pd.Timestamp,
    source_filter: str,
) -> Optional[Dict]:
    rows = [r for r in symbol_done_rows if r.get("symbol") == _normalize_key_symbol(symbol)]
    if source_filter == "prod":
        rows = [r for r in rows if str(r.get("cycle_log_file", "")).startswith("/opt/")]
    elif source_filter == "local":
        rows = [r for r in rows if str(r.get("cycle_log_file", "")).startswith("D:\\")]

    if not rows:
        return None

    for r in rows:
        r["delta_min"] = abs((pd.to_datetime(r["ts"]) - pd.to_datetime(ts)).total_seconds()) / 60.0

    rows = sorted(rows, key=lambda x: float(x.get("delta_min", 1e12)))
    return rows[0]


def _build_backtest_config(args, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> BacktestConfig:
    cfg = BacktestConfig(
        start_date=str((start_ts - pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")),
        end_date=str((end_ts + pd.Timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")),
        exchange=str(args.exchange),
        universe_mode=str(args.universe_mode),
        extractor_mode=str(args.extractor_mode),
        use_research_model_selection=bool(args.use_research_model_selection),
        selector_artifact_path=str(args.selector_artifact_path) if args.selector_artifact_path else None,
        threshold=(float(args.threshold_override) if args.threshold_override is not None else None),
        selection_debug_checks=False,
    )

    if args.profile_path:
        p = Path(str(args.profile_path))
        if not p.is_absolute():
            p = BASE_DIR / p
        _apply_profile_to_config(
            cfg,
            profile_path=p,
            profile_name=str(args.profile_name) if args.profile_name else None,
        )
    return cfg


def _build_extractor_kwargs_from_config(config: BacktestConfig) -> Dict[str, float]:
    kwargs = {
        "tp_level": float(config.tp_level),
        "max_hold_bars": int(config.max_bars_hold),
        "min_mid_candles": int(config.min_mid_candles),
        "min_price_pct": float(config.min_price_pct),
        "entry_pullback": float(config.entry_pullback),
        "min_rr": float(config.min_rr),
        "rr_floor_to_tp": float(config.rr_floor_to_tp),
    }
    try:
        sig = inspect.signature(RealDataQuantExtractor.__init__)
        allowed = set(sig.parameters.keys())
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        kwargs.pop("rr_floor_to_tp", None)
    return kwargs


def _load_symbol_df(symbol: str, symbols_dir: Path, cache: Dict[str, Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    sym = _normalize_key_symbol(symbol)
    if sym in cache:
        return cache[sym]

    fp = _resolve_symbol_file(sym, symbols_dir)
    if fp is None:
        cache[sym] = None
        return None

    try:
        df = pd.read_parquet(fp)
    except Exception:
        cache[sym] = None
        return None

    if "timestamp" not in df.columns:
        cache[sym] = None
        return None

    dfx = df.copy()
    dfx.columns = [str(c).lower() for c in dfx.columns]
    dfx["timestamp"] = pd.to_datetime(dfx["timestamp"], errors="coerce").dt.tz_localize(None)
    dfx = dfx.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    if dfx.empty:
        cache[sym] = None
        return None

    cache[sym] = dfx.reset_index(drop=True)
    return cache[sym]


def _run_live_probe(
    symbol: str,
    target_ts: pd.Timestamp,
    side: str,
    nearest_scan_ts: Optional[str],
    symbols_dir: Path,
    extractor_kwargs: Dict[str, float],
    cache: Dict[str, Optional[pd.DataFrame]],
) -> Dict[str, Optional[object]]:
    out: Dict[str, Optional[object]] = {
        "probe_status": "disabled",
        "probe_scan_open_ts": None,
        "probe_last_data_ts": None,
        "probe_last_closed_ts": None,
        "probe_target_equals_last_closed": None,
        "probe_setup_total": None,
        "probe_latest_setup_count": None,
        "probe_latest_side_count": None,
        "probe_nearest_setup_ts": None,
        "probe_nearest_setup_delta_min": None,
    }

    if not nearest_scan_ts:
        out["probe_status"] = "missing_nearest_scan_ts"
        return out

    df = _load_symbol_df(symbol=symbol, symbols_dir=symbols_dir, cache=cache)
    if df is None or df.empty:
        out["probe_status"] = "symbol_data_missing"
        return out

    scan_ts = pd.to_datetime(nearest_scan_ts).tz_localize(None)
    scan_open_ts = pd.to_datetime(scan_ts).floor("h")
    out["probe_scan_open_ts"] = scan_open_ts.strftime("%Y-%m-%d %H:%M:%S")
    out["probe_last_data_ts"] = pd.to_datetime(df["timestamp"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S")

    live_slice = df[df["timestamp"] <= scan_open_ts].copy()
    if len(live_slice) < 2:
        out["probe_status"] = "insufficient_rows_before_scan"
        return out

    df_calc = live_slice.iloc[:-1].copy()
    if df_calc.empty:
        out["probe_status"] = "no_closed_candle"
        return out

    last_closed_ts = pd.to_datetime(df_calc["timestamp"].iloc[-1]).tz_localize(None)
    out["probe_last_closed_ts"] = last_closed_ts.strftime("%Y-%m-%d %H:%M:%S")
    out["probe_target_equals_last_closed"] = bool(pd.to_datetime(target_ts).tz_localize(None) == last_closed_ts)

    try:
        extractor = RealDataQuantExtractor(**extractor_kwargs)
        extractor.extract(df_calc, f"{_normalize_key_symbol(symbol)}USDT", include_future_labels=False)
        setup_df = pd.DataFrame(extractor.dataset)
    except Exception:
        out["probe_status"] = "extractor_error"
        return out

    if setup_df.empty or "timestamp" not in setup_df.columns:
        out["probe_status"] = "no_setup"
        out["probe_setup_total"] = 0
        out["probe_latest_setup_count"] = 0
        out["probe_latest_side_count"] = 0
        return out

    setup_df = setup_df.copy()
    setup_df["timestamp"] = pd.to_datetime(setup_df["timestamp"], errors="coerce").dt.tz_localize(None)
    setup_df = setup_df.dropna(subset=["timestamp"])
    if setup_df.empty:
        out["probe_status"] = "setup_timestamp_invalid"
        out["probe_setup_total"] = 0
        out["probe_latest_setup_count"] = 0
        out["probe_latest_side_count"] = 0
        return out

    out["probe_setup_total"] = int(len(setup_df))
    latest = setup_df[setup_df["timestamp"] == last_closed_ts]
    out["probe_latest_setup_count"] = int(len(latest))

    side_int = 1 if str(side).upper() == "LONG" else -1
    if "side" in latest.columns and not latest.empty:
        side_ser = pd.to_numeric(latest["side"], errors="coerce")
        out["probe_latest_side_count"] = int((side_ser == side_int).sum())
    else:
        out["probe_latest_side_count"] = 0

    setup_ts = pd.to_datetime(setup_df["timestamp"]).dt.tz_localize(None)
    deltas = (setup_ts - pd.to_datetime(target_ts).tz_localize(None)).abs()
    if len(deltas) > 0:
        min_idx = int(deltas.idxmin())
        near_ts = pd.to_datetime(setup_df.loc[min_idx, "timestamp"]).tz_localize(None)
        out["probe_nearest_setup_ts"] = near_ts.strftime("%Y-%m-%d %H:%M:%S")
        out["probe_nearest_setup_delta_min"] = float(deltas.loc[min_idx].total_seconds() / 60.0)

    if int(out.get("probe_latest_setup_count") or 0) <= 0:
        out["probe_status"] = "no_latest_setup"
    elif int(out.get("probe_latest_side_count") or 0) <= 0:
        out["probe_status"] = "latest_setup_side_mismatch"
    else:
        out["probe_status"] = "latest_setup_exists"

    return out


def _signal_side_to_text(side_val: int) -> str:
    return "LONG" if int(side_val) == 1 else "SHORT"


def _collect_backtest_debug(
    keys: Sequence[str],
    symbols_dir: Path,
    config: BacktestConfig,
) -> Dict[str, Dict]:
    parsed = [_parse_key(k) for k in keys]
    symbols = sorted({_normalize_key_symbol(s) for s, _t, _sd in parsed})

    symbol_files: Dict[str, Optional[Path]] = {s: _resolve_symbol_file(s, symbols_dir) for s in symbols}
    potentials_all: List[Dict] = []
    selection_rows: List[pd.DataFrame] = []
    symbol_state: Dict[str, Dict] = {}

    for sym in symbols:
        fp = symbol_files.get(sym)
        if fp is None:
            symbol_state[sym] = {"missing_file": True, "potentials": [], "setup_df": pd.DataFrame()}
            continue

        sigs, _ohlcv, setup_df = backtest_symbol(fp, config)
        sigs = sigs or []
        if setup_df is None:
            setup_df = pd.DataFrame()

        if not setup_df.empty:
            setup_df = setup_df.copy()
            setup_df["timestamp"] = pd.to_datetime(setup_df["timestamp"], errors="coerce").dt.tz_localize(None)
            selection_rows.append(setup_df)

        potentials_all.extend(sigs)
        symbol_state[sym] = {"missing_file": False, "potentials": sigs, "setup_df": setup_df}

    filtered_all = list(potentials_all)
    if bool(config.use_research_model_selection):
        filtered_all = _apply_research_model_selection(
            potential_signals=list(potentials_all),
            selection_rows=selection_rows,
            config=config,
        )

    pre_keys = set(
        _to_key(
            str(sig.get("symbol", "")),
            pd.to_datetime(sig.get("signal_timestamp", sig.get("timestamp"))).tz_localize(None),
            str(sig.get("type", "")).upper(),
        )
        for sig in potentials_all
    )
    post_keys = set(
        _to_key(
            str(sig.get("symbol", "")),
            pd.to_datetime(sig.get("signal_timestamp", sig.get("timestamp"))).tz_localize(None),
            str(sig.get("type", "")).upper(),
        )
        for sig in filtered_all
    )

    out: Dict[str, Dict] = {}
    for key in keys:
        sym, ts, side = _parse_key(key)
        state = symbol_state.get(sym, {"missing_file": True, "potentials": [], "setup_df": pd.DataFrame()})

        if state.get("missing_file", False):
            out[key] = {
                "bt_stage": "missing_symbol_file",
                "bt_setup_hits": 0,
                "bt_pre_selection": False,
                "bt_post_selection": False,
            }
            continue

        setup_df = state.get("setup_df", pd.DataFrame())
        setup_hits = 0
        if not setup_df.empty and "side" in setup_df.columns and "timestamp" in setup_df.columns:
            side_int = 1 if side == "LONG" else -1
            match_setup = setup_df[
                (pd.to_datetime(setup_df["timestamp"]).dt.tz_localize(None) == ts)
                & (pd.to_numeric(setup_df["side"], errors="coerce") == side_int)
            ]
            setup_hits = int(len(match_setup))

        in_pre = key in pre_keys
        in_post = key in post_keys

        if setup_hits <= 0:
            stage = "no_setup_at_timestamp"
        elif not in_pre:
            stage = "setup_exists_but_not_fillable"
        elif in_pre and not in_post:
            stage = "filtered_by_selection"
        else:
            stage = "signal_emitted"

        out[key] = {
            "bt_stage": stage,
            "bt_setup_hits": setup_hits,
            "bt_pre_selection": bool(in_pre),
            "bt_post_selection": bool(in_post),
        }

    return out


def _load_target_keys(args) -> List[str]:
    keys: List[str] = []

    if args.keys:
        keys.extend([x.strip() for x in str(args.keys).split(",") if x.strip()])

    if args.keys_file:
        p = Path(args.keys_file)
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.append(line)

    if keys:
        return sorted(set(keys))

    compare_csv = Path(args.compare_csv)
    if not compare_csv.exists():
        raise FileNotFoundError(f"compare csv not found: {compare_csv}")

    df = pd.read_csv(compare_csv)
    if not {"key", "in_live_unique", "in_backtest"}.issubset(set(df.columns)):
        raise ValueError("compare csv must contain columns: key,in_live_unique,in_backtest")

    mode = str(args.mismatch_mode).lower()
    if mode == "backtest_only":
        filt = (df["in_live_unique"] == 0) & (df["in_backtest"] == 1)
    elif mode == "live_only":
        filt = (df["in_live_unique"] == 1) & (df["in_backtest"] == 0)
    else:
        filt = df["in_live_unique"] != df["in_backtest"]

    keys = [str(k) for k in df.loc[filt, "key"].tolist()]
    return sorted(set(keys))


def _infer_divergence(live_reason: str, bt_stage: str) -> str:
    reason = str(live_reason or "")
    stage = str(bt_stage or "")

    if stage in {"signal_emitted", "signal_emitted_from_backtest_artifact"} and reason in {
        "no_latest_setup",
        "no_setup",
        "insufficient_history",
        "no_closed_candle",
        "setup_missing_timestamp",
        "setup_timestamp_invalid",
    }:
        return "scanner_setup_stage"
    if stage in {"signal_emitted", "signal_emitted_from_backtest_artifact"} and reason == "below_threshold":
        return "scanner_threshold_stage"
    if stage == "filtered_by_selection":
        return "backtest_selection_stage"
    if stage == "setup_exists_but_not_fillable":
        return "backtest_fill_stage"
    if stage == "no_setup_at_timestamp":
        return "backtest_setup_stage"
    if stage == "no_signal_in_backtest_artifact":
        return "backtest_output_stage"
    if reason == "symbol_not_found_in_scan_log":
        return "live_coverage_gap"
    return "undetermined"


def main():
    parser = argparse.ArgumentParser(description="Debug live-vs-backtest mismatches at exact timestamps")
    parser.add_argument("--compare-csv", type=str, default="output/live_backtest_dedup_compare_20260405.csv")
    parser.add_argument("--mismatch-mode", type=str, default="backtest_only", choices=["backtest_only", "live_only", "all"])
    parser.add_argument("--keys", type=str, default="")
    parser.add_argument("--keys-file", type=str, default="")
    parser.add_argument("--scan-log", type=str, default="output/prod_sniper_scan_progress_20260405.log")
    parser.add_argument("--scan-source-filter", type=str, default="prod", choices=["all", "prod", "local"])

    parser.add_argument("--universe-mode", type=str, default="research", choices=["research", "sniper"])
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--extractor-mode", type=str, default="causal", choices=["strict", "causal", "live_compatible"])
    parser.add_argument("--profile-path", type=str, default="")
    parser.add_argument("--profile-name", type=str, default="")
    parser.add_argument("--selector-artifact-path", type=str, default="")
    parser.add_argument("--threshold-override", type=float, default=None)
    parser.add_argument("--use-research-model-selection", action="store_true")
    parser.add_argument(
        "--backtest-debug-mode",
        type=str,
        default="artifact",
        choices=["artifact", "replay"],
        help="artifact=read in_backtest from compare csv; replay=rerun backtest extraction/fill/selection for stage debug",
    )
    parser.add_argument(
        "--inject-live-probe",
        action="store_true",
        help="Inject extra live-style extractor probe per key to explain no_latest_setup/below_threshold reasons",
    )

    parser.add_argument("--output-prefix", type=str, default="output/mismatch_timestamp_debug")

    args = parser.parse_args()

    keys = _load_target_keys(args)
    if not keys:
        raise ValueError("No keys selected")

    parsed = [_parse_key(k) for k in keys]
    min_ts = min(x[1] for x in parsed)
    max_ts = max(x[1] for x in parsed)

    symbols_dir = _resolve_symbols_dir(args.universe_mode, args.exchange)
    if not symbols_dir.exists():
        raise FileNotFoundError(f"symbols dir not found: {symbols_dir}")

    scan_log = Path(args.scan_log)
    if not scan_log.exists():
        raise FileNotFoundError(f"scan log not found: {scan_log}")

    cycles, symbol_done_rows = _parse_scan_log(scan_log)

    bt_cfg = None
    bt_debug: Dict[str, Dict] = {}
    probe_cfg = _build_backtest_config(args=args, start_ts=min_ts, end_ts=max_ts)
    probe_extractor_kwargs = _build_extractor_kwargs_from_config(probe_cfg)
    symbol_cache: Dict[str, Optional[pd.DataFrame]] = {}

    compare_lookup: Dict[str, Dict] = {}
    compare_csv = Path(args.compare_csv)
    if compare_csv.exists():
        cdf = pd.read_csv(compare_csv)
        if {"key", "in_live_unique", "in_backtest"}.issubset(set(cdf.columns)):
            compare_lookup = {
                str(r["key"]): {
                    "in_live_unique": int(r["in_live_unique"]),
                    "in_backtest": int(r["in_backtest"]),
                }
                for _, r in cdf.iterrows()
            }

    if str(args.backtest_debug_mode).lower() == "replay":
        bt_cfg = _build_backtest_config(args=args, start_ts=min_ts, end_ts=max_ts)
        if not args.selector_artifact_path:
            bt_cfg.selector_artifact_path = None
        bt_debug = _collect_backtest_debug(
            keys=keys,
            symbols_dir=symbols_dir,
            config=bt_cfg,
        )

    rows: List[Dict] = []
    for key in keys:
        sym, ts, side = _parse_key(key)
        near = _pick_nearest_live_reason(
            symbol_done_rows=symbol_done_rows,
            symbol=sym,
            ts=ts,
            source_filter=str(args.scan_source_filter),
        )

        live_reason = "symbol_not_found_in_scan_log"
        live_status = ""
        live_delta_min = None
        live_scan_ts = None
        live_scan_log_file = ""
        live_scan_symbols = None

        if near is not None:
            live_reason = str(near.get("reason", ""))
            live_status = str(near.get("status", ""))
            live_delta_min = float(near.get("delta_min", 0.0))
            live_scan_ts = pd.to_datetime(near.get("ts")).strftime("%Y-%m-%d %H:%M:%S")
            live_scan_log_file = str(near.get("cycle_log_file", ""))
            live_scan_symbols = near.get("cycle_symbols")

        probe = {
            "probe_status": None,
            "probe_scan_open_ts": None,
            "probe_last_data_ts": None,
            "probe_last_closed_ts": None,
            "probe_target_equals_last_closed": None,
            "probe_setup_total": None,
            "probe_latest_setup_count": None,
            "probe_latest_side_count": None,
            "probe_nearest_setup_ts": None,
            "probe_nearest_setup_delta_min": None,
        }
        if bool(args.inject_live_probe):
            probe = _run_live_probe(
                symbol=sym,
                target_ts=ts,
                side=side,
                nearest_scan_ts=live_scan_ts,
                symbols_dir=symbols_dir,
                extractor_kwargs=probe_extractor_kwargs,
                cache=symbol_cache,
            )

        bt = bt_debug.get(
            key,
            {
                "bt_stage": "unknown",
                "bt_setup_hits": 0,
                "bt_pre_selection": False,
                "bt_post_selection": False,
            },
        )

        compare = compare_lookup.get(key, {"in_live_unique": None, "in_backtest": None})

        if str(args.backtest_debug_mode).lower() == "artifact":
            in_bt = compare.get("in_backtest")
            if in_bt == 1:
                bt_stage = "signal_emitted_from_backtest_artifact"
                bt_setup_hits = None
                bt_pre_selection = True
                bt_post_selection = True
            elif in_bt == 0:
                bt_stage = "no_signal_in_backtest_artifact"
                bt_setup_hits = None
                bt_pre_selection = False
                bt_post_selection = False
            else:
                bt_stage = "unknown"
                bt_setup_hits = None
                bt_pre_selection = None
                bt_post_selection = None
        else:
            bt = bt_debug.get(
                key,
                {
                    "bt_stage": "unknown",
                    "bt_setup_hits": 0,
                    "bt_pre_selection": False,
                    "bt_post_selection": False,
                },
            )
            bt_stage = bt.get("bt_stage")
            bt_setup_hits = bt.get("bt_setup_hits")
            bt_pre_selection = bt.get("bt_pre_selection")
            bt_post_selection = bt.get("bt_post_selection")

        rows.append(
            {
                "key": key,
                "symbol": sym,
                "signal_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "side": side,
                "in_live_unique": compare.get("in_live_unique"),
                "in_backtest": compare.get("in_backtest"),
                "live_reason": live_reason,
                "live_status": live_status,
                "live_nearest_scan_ts": live_scan_ts,
                "live_nearest_delta_min": live_delta_min,
                "live_cycle_log_file": live_scan_log_file,
                "live_cycle_symbols": live_scan_symbols,
                "bt_stage": bt_stage,
                "bt_setup_hits": bt_setup_hits,
                "bt_pre_selection": bt_pre_selection,
                "bt_post_selection": bt_post_selection,
                "divergence_stage": _infer_divergence(live_reason=live_reason, bt_stage=str(bt_stage)),
                "probe_status": probe.get("probe_status"),
                "probe_scan_open_ts": probe.get("probe_scan_open_ts"),
                "probe_last_data_ts": probe.get("probe_last_data_ts"),
                "probe_last_closed_ts": probe.get("probe_last_closed_ts"),
                "probe_target_equals_last_closed": probe.get("probe_target_equals_last_closed"),
                "probe_setup_total": probe.get("probe_setup_total"),
                "probe_latest_setup_count": probe.get("probe_latest_setup_count"),
                "probe_latest_side_count": probe.get("probe_latest_side_count"),
                "probe_nearest_setup_ts": probe.get("probe_nearest_setup_ts"),
                "probe_nearest_setup_delta_min": probe.get("probe_nearest_setup_delta_min"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["signal_time", "symbol", "side"]).reset_index(drop=True)

    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = BASE_DIR / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")

    out_df.to_csv(csv_path, index=False)

    source_counts = (
        out_df["live_cycle_log_file"].fillna("").replace("", "(missing)").value_counts().to_dict()
        if not out_df.empty
        else {}
    )
    summary = {
        "keys_total": int(len(out_df)),
        "scan_cycles_total": int(len(cycles)),
        "scan_source_filter": str(args.scan_source_filter),
        "live_reason_counts": out_df["live_reason"].value_counts(dropna=False).to_dict() if not out_df.empty else {},
        "bt_stage_counts": out_df["bt_stage"].value_counts(dropna=False).to_dict() if not out_df.empty else {},
        "divergence_stage_counts": out_df["divergence_stage"].value_counts(dropna=False).to_dict() if not out_df.empty else {},
        "live_cycle_log_file_counts": source_counts,
        "paths": {
            "scan_log": str(scan_log),
            "compare_csv": str(compare_csv),
            "output_csv": str(csv_path),
            "output_json": str(json_path),
            "symbols_dir": str(symbols_dir),
        },
        "backtest_debug_mode": str(args.backtest_debug_mode),
        "inject_live_probe": bool(args.inject_live_probe),
        "backtest_config": (asdict(bt_cfg) if bt_cfg is not None else None),
        "probe_extractor_kwargs": probe_extractor_kwargs,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== mismatch timestamp debug ===")
    print(f"keys_total={len(out_df)}")
    print(f"scan_cycles_total={len(cycles)}")
    print(f"scan_source_filter={args.scan_source_filter}")
    print("live_reason_counts:")
    for k, v in summary["live_reason_counts"].items():
        print(f"  {k}: {v}")
    print("bt_stage_counts:")
    for k, v in summary["bt_stage_counts"].items():
        print(f"  {k}: {v}")
    print("divergence_stage_counts:")
    for k, v in summary["divergence_stage_counts"].items():
        print(f"  {k}: {v}")
    print("live_cycle_log_file_counts:")
    for k, v in summary["live_cycle_log_file_counts"].items():
        print(f"  {k}: {v}")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()
