import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
sys.path.append(str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "sniper_bot"))

from ml.backtest_sniper import (  # noqa: E402
    BacktestConfig,
    _apply_profile_to_config,
    _compute_curve_metrics,
    _compute_turnover_metrics,
    run_backtest_with_config,
    run_portfolio_simulation,
)
from sniper_bot.config import SniperBotConfig  # noqa: E402
from sniper_bot.sniper_scanner import SniperScanner  # noqa: E402


def _strip_symbol(sym: str) -> str:
    return str(sym).upper().replace("_USDT", "").replace("USDT", "")


def _to_usdt(sym: str) -> str:
    return f"{_strip_symbol(sym)}USDT"


def _resolve_symbols_dir(universe_mode: str, exchange: str) -> Path:
    if str(universe_mode).lower() == "research":
        return BASE_DIR / "data" / "ohlcv"
    if str(exchange).lower() == "bitget":
        return BASE_DIR / "bitget-data" / "symbols_v3"
    return BASE_DIR / "data" / "processed" / "symbols_v3"


def _discover_files(symbols_dir: Path) -> List[Path]:
    all_files = sorted(symbols_dir.glob("*.parquet"))
    deduped: Dict[str, Path] = {}
    for fp in all_files:
        core = _strip_symbol(fp.stem)
        prev = deduped.get(core)
        if prev is None:
            deduped[core] = fp
            continue
        prev_name = prev.stem.upper()
        curr_name = fp.stem.upper()
        prev_has_usdt = prev_name.endswith("USDT") or prev_name.endswith("_USDT")
        curr_has_usdt = curr_name.endswith("USDT") or curr_name.endswith("_USDT")
        if curr_has_usdt and not prev_has_usdt:
            deduped[core] = fp
    return sorted(deduped.values())


def _select_files(symbols_dir: Path, symbols: Optional[Sequence[str]], max_files: int) -> List[Path]:
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
        self.ts_ns: Dict[str, object] = {}

        for symbol in symbols:
            cands = [
                f"{symbol}_USDT.parquet",
                f"{symbol}.parquet",
                f"{_strip_symbol(symbol)}USDT_USDT.parquet",
                f"{_strip_symbol(symbol)}USDT.parquet",
            ]
            path = None
            for name in cands:
                p = symbols_dir / name
                if p.exists():
                    path = p
                    break
            if path is None:
                continue
            try:
                df = pd.read_parquet(path)
                if "timestamp" not in df.columns:
                    continue
                dfx = df.copy()
                dfx["timestamp"] = pd.to_datetime(dfx["timestamp"], errors="coerce").dt.tz_localize(None)
                dfx = dfx.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                if not dfx.empty:
                    self.data[symbol] = dfx
                    self.ts_ns[symbol] = pd.to_datetime(dfx["timestamp"], errors="coerce").astype("int64").to_numpy()
            except Exception:
                continue

    def get_historical_data(self, symbol, timeframe, since, until):
        if self.current_time is None or symbol not in self.data:
            return pd.DataFrame()
        base = self.data[symbol]
        open_candle_ts = pd.to_datetime(self.current_time).floor("h")
        ts_ns = self.ts_ns.get(symbol)
        if ts_ns is None or len(ts_ns) == 0:
            return pd.DataFrame()

        open_ns = int(pd.Timestamp(open_candle_ts).value)
        end_idx = int(ts_ns.searchsorted(open_ns, side="right"))
        if end_idx <= 0:
            return pd.DataFrame()

        # Using positional slicing avoids expensive boolean-mask filtering on large frames.
        sliced = base.iloc[:end_idx].copy()

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
    syms = [_to_usdt(fp.stem) for fp in file_paths]
    if "BTCUSDT" not in syms:
        syms.append("BTCUSDT")
    return sorted(set(syms))


def _collect_scanner_signals(
    scanner_cfg: SniperBotConfig,
    symbols_dir: Path,
    symbols: Sequence[str],
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[List[dict], float, Dict[str, pd.DataFrame]]:
    bars_tail = max(1200, int(getattr(getattr(scanner_cfg, "strategy", None), "scan_history_bars", 1200)))
    processor = MockDataProcessor(symbols=symbols, symbols_dir=symbols_dir, bars_tail=bars_tail)
    scanner = SniperScanner(config=scanner_cfg, data_processor=processor)

    out: Dict[Tuple[pd.Timestamp, str, str, float], dict] = {}
    for ts in pd.date_range(start, end, freq="h"):
        processor.current_time = ts
        signals = scanner.scan(list(symbols), timeframe)
        for sig in signals:
            decision_ts = pd.to_datetime(sig.get("timestamp"), errors="coerce")
            if pd.isna(decision_ts):
                continue
            decision_ts = decision_ts.tz_localize(None)
            symbol = _to_usdt(sig.get("symbol", ""))
            side_txt = str(sig.get("type", "")).upper()
            if side_txt not in {"LONG", "SHORT"}:
                continue
            entry_p = float(pd.to_numeric(sig.get("signal_price", float("nan")), errors="coerce"))
            if not pd.notna(entry_p):
                continue
            key = (decision_ts, symbol, side_txt, entry_p)
            conf = float(pd.to_numeric(sig.get("confidence", 0.0), errors="coerce"))
            prev = out.get(key)
            if prev is None or conf > float(prev.get("confidence", 0.0)):
                out[key] = {
                    "timestamp": decision_ts,
                    "symbol": symbol,
                    "side": side_txt,
                    "entry_p": entry_p,
                    "confidence": conf,
                    "sl_pct": float(pd.to_numeric(sig.get("sl_pct", float("nan")), errors="coerce")),
                    "tp_pct": float(pd.to_numeric(sig.get("tp_pct", float("nan")), errors="coerce")),
                }

    rows = sorted(out.values(), key=lambda x: (x["timestamp"], x["symbol"], x["side"]))
    return rows, float(getattr(scanner, "threshold", float("nan"))), processor.data


def _scanner_rows_to_potential_signals(
    scanner_rows: Sequence[dict],
    full_price_db: Dict[str, pd.DataFrame],
    max_hold_bars: int,
) -> List[dict]:
    max_hold = max(1, int(max_hold_bars))
    out: List[dict] = []

    for idx, row in enumerate(scanner_rows):
        symbol = _to_usdt(str(row.get("symbol", "")))
        df = full_price_db.get(symbol)
        if df is None or df.empty:
            continue

        ts = pd.to_datetime(row.get("timestamp"), errors="coerce")
        if pd.isna(ts):
            continue
        ts = ts.tz_localize(None)

        ts_series = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
        hit = df.index[ts_series == ts]
        if len(hit) == 0:
            continue
        i = int(hit[0])

        fut = df.iloc[i + 1 : i + 1 + max_hold]
        if fut.empty:
            continue

        lows = pd.to_numeric(fut.get("low"), errors="coerce").dropna().astype(float).tolist()
        highs = pd.to_numeric(fut.get("high"), errors="coerce").dropna().astype(float).tolist()
        closes = pd.to_numeric(fut.get("close"), errors="coerce").dropna().astype(float).tolist()
        n = min(len(lows), len(highs), len(closes))
        if n <= 0:
            continue
        lows = lows[:n]
        highs = highs[:n]
        closes = closes[:n]

        entry_p = float(row["entry_p"])
        sl_pct = float(row.get("sl_pct", float("nan")))
        tp_pct = float(row.get("tp_pct", float("nan")))
        if not pd.notna(sl_pct) or not pd.notna(tp_pct):
            continue

        side = 1 if str(row.get("side", "")).upper() == "LONG" else -1
        if side == 1:
            sl_p = entry_p * (1.0 - sl_pct)
            tp_p = entry_p * (1.0 + tp_pct)
        else:
            sl_p = entry_p * (1.0 + sl_pct)
            tp_p = entry_p * (1.0 - tp_pct)

        end_idx = min(len(df) - 1, i + max_hold)
        end_time = pd.to_datetime(df.iloc[end_idx]["timestamp"], errors="coerce")
        if pd.isna(end_time):
            end_time = ts
        else:
            end_time = end_time.tz_localize(None)

        out.append(
            {
                "signal_id": f"{_strip_symbol(symbol)}|{int(pd.Timestamp(ts).value)}|{idx}",
                "timestamp": ts,
                "signal_timestamp": ts,
                "entry_timestamp": ts,
                "end_time": end_time,
                "symbol": _strip_symbol(symbol),
                "type": "LONG" if side == 1 else "SHORT",
                "side": side,
                "prob": float(row.get("confidence", 0.0)),
                "prob_long": 1.0 if side == 1 else 0.0,
                "prob_short": 1.0 if side == -1 else 0.0,
                "entry_p": float(entry_p),
                "sl_p": float(sl_p),
                "tp_p": float(tp_p),
                "future_lows": lows,
                "future_highs": highs,
                "future_closes": closes,
                "atr_val": 0.0,
            }
        )

    return out


def _trade_metrics(trades: List) -> Dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "pnl_usd_total": 0.0,
            "avg_pnl_usd": 0.0,
        }
    pnls = [float(getattr(t, "pnl_usd", 0.0)) for t in trades]
    wins = [p for p in pnls if p > 0]
    return {
        "trades": int(len(trades)),
        "win_rate_pct": float((len(wins) / max(1, len(trades))) * 100.0),
        "pnl_usd_total": float(sum(pnls)),
        "avg_pnl_usd": float(sum(pnls) / max(1, len(pnls))),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare profitability: scanner-logic backtest vs standard backtest")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--config-path", type=str, default=str(BASE_DIR / "sniper_bot" / "sniper_bot_config.json"))
    parser.add_argument("--profile-path", type=str, default="")
    parser.add_argument("--profile-name", type=str, default="")
    parser.add_argument("--selector-artifact-path", type=str, default="")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--universe-mode", type=str, default="research", choices=["research", "sniper"])
    parser.add_argument("--extractor-mode", type=str, default="causal", choices=["strict", "causal", "live_compatible"])
    parser.add_argument("--output-prefix", type=str, default="output/scanner_logic_profit_compare")
    args = parser.parse_args()

    start = pd.to_datetime(args.start).tz_localize(None)
    end = pd.to_datetime(args.end).tz_localize(None)
    if end <= start:
        raise ValueError("--end must be after --start")

    scanner_cfg = SniperBotConfig.load(Path(args.config_path))
    if args.profile_path:
        scanner_cfg.strategy.profile_path = str(args.profile_path)
    if args.profile_name:
        scanner_cfg.strategy.profile_name = str(args.profile_name)
    if args.selector_artifact_path:
        scanner_cfg.strategy.selector_artifact_path = str(args.selector_artifact_path)

    symbols_dir = _resolve_symbols_dir(args.universe_mode, args.exchange)
    cli_symbols = [x.strip() for x in str(args.symbols).split(",") if x.strip()]
    file_paths = _select_files(symbols_dir, cli_symbols if cli_symbols else None, int(args.max_files))
    if not file_paths:
        raise ValueError("No symbols selected")

    live_symbols = _build_live_symbols(file_paths)
    scanner_rows, scanner_threshold, full_price_db = _collect_scanner_signals(
        scanner_cfg=scanner_cfg,
        symbols_dir=symbols_dir,
        symbols=live_symbols,
        timeframe=args.timeframe,
        start=start,
        end=end,
    )

    bt_cfg = BacktestConfig(
        start_date=str(start),
        end_date=str(end),
        exchange=str(args.exchange),
        leverage=float(scanner_cfg.exchange.leverage),
        risk_per_trade=float(scanner_cfg.risk.max_risk_per_trade),
        max_open_trades=int(scanner_cfg.risk.max_open_positions),
        universe_mode=str(args.universe_mode),
        extractor_mode=str(args.extractor_mode),
        top_coins=[_to_usdt(fp.stem) for fp in file_paths],
        max_files=0,
        use_research_model_selection=True,
        selector_artifact_path=str(scanner_cfg.strategy.selector_artifact_path),
        threshold=float(scanner_threshold) if pd.notna(scanner_threshold) else None,
        selection_debug_checks=False,
    )

    profile_path = Path(str(scanner_cfg.strategy.profile_path))
    if not profile_path.is_absolute():
        profile_path = BASE_DIR / profile_path
    _apply_profile_to_config(
        bt_cfg,
        profile_path=profile_path,
        profile_name=str(scanner_cfg.strategy.profile_name) if scanner_cfg.strategy.profile_name else None,
    )

    scanner_potential_signals = _scanner_rows_to_potential_signals(
        scanner_rows=scanner_rows,
        full_price_db=full_price_db,
        max_hold_bars=int(bt_cfg.max_bars_hold),
    )

    scanner_trades, scanner_curve, _ = run_portfolio_simulation(scanner_potential_signals, full_price_db, bt_cfg)
    std_signals, std_price_db, std_trades, std_curve = run_backtest_with_config(bt_cfg)

    scanner_curve_metrics = _compute_curve_metrics(scanner_curve)
    std_curve_metrics = _compute_curve_metrics(std_curve)
    scanner_turnover = _compute_turnover_metrics(scanner_trades, scanner_curve)
    std_turnover = _compute_turnover_metrics(std_trades, std_curve)

    summary = {
        "window": {"start": str(start), "end": str(end), "timeframe": args.timeframe},
        "symbols_selected": len(file_paths),
        "scanner_threshold": float(scanner_threshold) if pd.notna(scanner_threshold) else None,
        "config": {
            "profile_path": str(scanner_cfg.strategy.profile_path),
            "profile_name": str(scanner_cfg.strategy.profile_name),
            "selector_artifact_path": str(scanner_cfg.strategy.selector_artifact_path),
            "extractor_mode": str(bt_cfg.extractor_mode),
            "risk_per_trade": float(bt_cfg.risk_per_trade),
            "max_open_trades": int(bt_cfg.max_open_trades),
            "leverage": float(bt_cfg.leverage),
            "fee_rate": float(bt_cfg.fee_rate),
            "slippage": float(bt_cfg.slippage),
            "max_bars_hold": int(bt_cfg.max_bars_hold),
        },
        "scanner_logic_backtest": {
            "signal_count": int(len(scanner_rows)),
            "portfolio_signal_count": int(len(scanner_potential_signals)),
            "trade_metrics": _trade_metrics(scanner_trades),
            "curve_metrics": scanner_curve_metrics,
            "turnover_metrics": scanner_turnover,
        },
        "standard_backtest": {
            "signal_count": int(len(std_signals)),
            "trade_metrics": _trade_metrics(std_trades),
            "curve_metrics": std_curve_metrics,
            "turnover_metrics": std_turnover,
        },
        "delta": {
            "trade_count": int(len(scanner_trades) - len(std_trades)),
            "pnl_usd_total": float(_trade_metrics(scanner_trades)["pnl_usd_total"] - _trade_metrics(std_trades)["pnl_usd_total"]),
            "sharpe_daily": float(scanner_curve_metrics.get("sharpe_daily", float("nan")) - std_curve_metrics.get("sharpe_daily", float("nan"))),
            "max_drawdown_pct": float(scanner_curve_metrics.get("max_drawdown_pct", float("nan")) - std_curve_metrics.get("max_drawdown_pct", float("nan"))),
        },
    }

    output_base = BASE_DIR / str(args.output_prefix)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_base.with_name(output_base.name + "_summary.json")
    scanner_sig_path = output_base.with_name(output_base.name + "_scanner_signals.csv")

    pd.DataFrame(scanner_rows).to_csv(scanner_sig_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Scanner-Logic vs Standard Backtest Profit Compare ===")
    print(f"window={start} -> {end} tf={args.timeframe}")
    print(f"symbols_selected={len(file_paths)}")
    print(
        f"scanner_logic: signals={len(scanner_rows)} portfolio_signals={len(scanner_potential_signals)} "
        f"trades={len(scanner_trades)} pnl_usd={_trade_metrics(scanner_trades)['pnl_usd_total']:.2f} "
        f"sharpe={scanner_curve_metrics.get('sharpe_daily', float('nan')):.4f} "
        f"mdd={scanner_curve_metrics.get('max_drawdown_pct', float('nan')):.2f}%"
    )
    print(
        f"standard_bt: signals={len(std_signals)} trades={len(std_trades)} "
        f"pnl_usd={_trade_metrics(std_trades)['pnl_usd_total']:.2f} "
        f"sharpe={std_curve_metrics.get('sharpe_daily', float('nan')):.4f} "
        f"mdd={std_curve_metrics.get('max_drawdown_pct', float('nan')):.2f}%"
    )
    print(f"summary: {summary_path}")
    print(f"scanner_signals: {scanner_sig_path}")


if __name__ == "__main__":
    main()
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
    _compute_curve_metrics,
    _compute_turnover_metrics,
    run_backtest_with_config,
    run_portfolio_simulation,
)
from sniper_bot.config import SniperBotConfig  # noqa: E402
from sniper_bot.sniper_scanner import SniperScanner  # noqa: E402


def _strip_symbol(sym: str) -> str:
    return str(sym).upper().replace("_USDT", "").replace("USDT", "")


def _to_usdt(sym: str) -> str:
    return f"{_strip_symbol(sym)}USDT"


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
        symbol = _strip_symbol(file_path.stem)
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


def _select_files(symbols_dir: Path, symbols: Optional[Sequence[str]], max_files: int) -> List[Path]:
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
            candidates = [
                f"{symbol}_USDT.parquet",
                f"{symbol}.parquet",
                f"{_strip_symbol(symbol)}USDT_USDT.parquet",
                f"{_strip_symbol(symbol)}USDT.parquet",
            ]
            path = None
            for candidate in candidates:
                p = symbols_dir / candidate
                if p.exists():
                    path = p
                    break
            if path is None:
                continue

            df = pd.read_parquet(path)
            if "timestamp" not in df.columns:
                continue
            dfx = df.copy()
            dfx["timestamp"] = pd.to_datetime(dfx["timestamp"], errors="coerce").dt.tz_localize(None)
            dfx = dfx.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            if not dfx.empty:
                self.data[symbol] = dfx

    def get_historical_data(self, symbol, timeframe, since, until):
        if self.current_time is None or symbol not in self.data:
            return pd.DataFrame()

        base = self.data[symbol]
        open_candle_ts = pd.to_datetime(self.current_time).floor("h")
        sliced = base[base["timestamp"] <= open_candle_ts].copy()

        # Keep replay behavior aligned with live scanner where current open bar exists.
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


def _collect_scanner_raw_signals(
    cfg: SniperBotConfig,
    symbols_dir: Path,
    symbols: Sequence[str],
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    quiet_scanner: bool,
) -> Tuple[List[dict], Dict[str, pd.DataFrame], float]:
    bars_tail = max(1200, int(getattr(getattr(cfg, "strategy", None), "scan_history_bars", 1200)))
    processor = MockDataProcessor(symbols=symbols, symbols_dir=symbols_dir, bars_tail=bars_tail)
    scanner = SniperScanner(config=cfg, data_processor=processor)

    all_signals: List[dict] = []
    for ts in pd.date_range(start, end, freq="h"):
        processor.current_time = ts
        if quiet_scanner:
            with contextlib.redirect_stdout(io.StringIO()):
                sigs = scanner.scan(list(symbols), timeframe)
        else:
            sigs = scanner.scan(list(symbols), timeframe)
        all_signals.extend(sigs)

    return all_signals, processor.data, float(getattr(scanner, "threshold", float("nan")))


def _to_side_value(side_text: str) -> int:
    return 1 if str(side_text).upper() == "LONG" else -1


def _build_scanner_sim_signals(
    scanner_raw_signals: Sequence[dict],
    price_db_by_usdt: Dict[str, pd.DataFrame],
    timeframe: str,
    max_hold_bars: int,
) -> List[dict]:
    bar_delta = pd.Timedelta(hours=1)
    if str(timeframe).lower().endswith("m"):
        bar_delta = pd.Timedelta(minutes=int(str(timeframe).lower()[:-1]))
    elif str(timeframe).lower().endswith("h"):
        bar_delta = pd.Timedelta(hours=int(str(timeframe).lower()[:-1]))

    out: List[dict] = []
    seen = set()

    for sig in scanner_raw_signals:
        symbol_usdt = _to_usdt(sig.get("symbol", ""))
        symbol = _strip_symbol(symbol_usdt)
        side_text = str(sig.get("type", "")).upper()
        if side_text not in {"LONG", "SHORT"}:
            continue

        entry_p = pd.to_numeric(sig.get("signal_price"), errors="coerce")
        sl_p = pd.to_numeric(sig.get("stop_loss"), errors="coerce")
        tp_p = pd.to_numeric(sig.get("take_profit"), errors="coerce")
        confidence = pd.to_numeric(sig.get("confidence", 0.0), errors="coerce")
        decision_ts = pd.to_datetime(sig.get("timestamp"), errors="coerce")
        meta = sig.get("meta") if isinstance(sig.get("meta"), dict) else {}
        setup_ts = pd.to_datetime(meta.get("setup_timestamp", decision_ts), errors="coerce")

        if pd.isna(entry_p) or pd.isna(sl_p) or pd.isna(tp_p) or pd.isna(decision_ts):
            continue

        decision_ts = decision_ts.tz_localize(None)
        if not pd.isna(setup_ts):
            setup_ts = setup_ts.tz_localize(None)

        df = price_db_by_usdt.get(symbol_usdt)
        if df is None or df.empty:
            continue

        ts_series = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
        hit = df.index[ts_series == decision_ts]
        if len(hit) == 0:
            continue

        i = int(hit[0])
        fut = df.iloc[i + 1 : i + 1 + int(max_hold_bars)]
        if fut.empty:
            continue

        lows = pd.to_numeric(fut["low"], errors="coerce").dropna().astype(float).tolist()
        highs = pd.to_numeric(fut["high"], errors="coerce").dropna().astype(float).tolist()
        closes = pd.to_numeric(fut["close"], errors="coerce").dropna().astype(float).tolist()
        max_len = min(len(lows), len(highs), len(closes))
        if max_len <= 0:
            continue

        lows = lows[:max_len]
        highs = highs[:max_len]
        closes = closes[:max_len]

        end_time = decision_ts + bar_delta * int(max_hold_bars)

        key = (symbol, decision_ts, side_text, float(entry_p))
        if key in seen:
            continue
        seen.add(key)

        side = _to_side_value(side_text)
        out.append(
            {
                "signal_id": f"{symbol}|{int(pd.Timestamp(decision_ts).value)}|{len(out)}",
                "timestamp": decision_ts,
                "signal_timestamp": setup_ts if pd.notna(setup_ts) else decision_ts,
                "entry_timestamp": decision_ts,
                "end_time": end_time,
                "symbol": symbol,
                "type": side_text,
                "side": side,
                "prob": float(confidence) if pd.notna(confidence) else 0.0,
                "prob_long": 1.0 if side == 1 else 0.0,
                "prob_short": 1.0 if side == -1 else 0.0,
                "entry_p": float(entry_p),
                "sl_p": float(sl_p),
                "tp_p": float(tp_p),
                "future_lows": lows,
                "future_highs": highs,
                "future_closes": closes,
                "atr_val": float(pd.to_numeric(meta.get("atr", 0.0), errors="coerce") or 0.0),
            }
        )

    return out


@dataclass
class PerfSummary:
    trades: int
    wins: int
    win_rate_pct: float
    total_pnl_usd: float
    ending_equity: float
    max_drawdown_pct: float
    sharpe_annualized: float
    turnover_raw: float
    turnover_annualized: float


def _summarize_performance(trades, curve, initial_capital: float) -> PerfSummary:
    wins = sum(1 for t in trades if str(t.result).upper() == "WIN")
    win_rate = (wins / len(trades) * 100.0) if trades else 0.0
    total_pnl = float(sum(float(t.pnl_usd) for t in trades)) if trades else 0.0
    ending_equity = float(curve[-1][1]) if curve else float(initial_capital)
    metrics = _compute_curve_metrics(curve)
    turnover = _compute_turnover_metrics(trades, curve)
    return PerfSummary(
        trades=int(len(trades)),
        wins=int(wins),
        win_rate_pct=float(win_rate),
        total_pnl_usd=float(total_pnl),
        ending_equity=float(ending_equity),
        max_drawdown_pct=float(metrics.get("max_drawdown_pct", float("nan"))),
        sharpe_annualized=float(metrics.get("sharpe_annualized", float("nan"))),
        turnover_raw=float(turnover.get("turnover_raw", float("nan"))),
        turnover_annualized=float(turnover.get("turnover_annualized", float("nan")),),
    )


def main():
    parser = argparse.ArgumentParser(description="Backtest using scanner-generated signals and compare with standard backtest")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--config-path", type=str, default=str(BASE_DIR / "sniper_bot" / "sniper_bot_config.json"))
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--universe-mode", type=str, default="research", choices=["research", "sniper"])
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--extractor-mode", type=str, default="strict", choices=["strict", "causal", "live_compatible"])
    parser.add_argument("--profile-path", type=str, default="")
    parser.add_argument("--profile-name", type=str, default="")
    parser.add_argument("--selector-artifact-path", type=str, default="")
    parser.add_argument("--selector-threshold-override", type=float, default=None)
    parser.add_argument("--no-quiet-scanner", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="output/scanner_logic_backtest_compare")

    args = parser.parse_args()

    start = pd.to_datetime(args.start).tz_localize(None)
    end = pd.to_datetime(args.end).tz_localize(None)
    if end <= start:
        raise ValueError("--end must be after --start")

    cfg = SniperBotConfig.load(Path(args.config_path))
    if args.profile_path:
        cfg.strategy.profile_path = str(args.profile_path)
    if args.profile_name:
        cfg.strategy.profile_name = str(args.profile_name)
    if args.selector_artifact_path:
        cfg.strategy.selector_artifact_path = str(args.selector_artifact_path)
    if args.selector_threshold_override is not None:
        cfg.strategy.selector_threshold_override = float(args.selector_threshold_override)

    symbols_dir = _resolve_symbols_dir(args.universe_mode, args.exchange)
    if not symbols_dir.exists():
        raise FileNotFoundError(f"symbols dir not found: {symbols_dir}")

    cli_symbols = [x.strip() for x in str(args.symbols).split(",") if x.strip()]
    file_paths = _select_files(symbols_dir, cli_symbols if cli_symbols else None, max(0, int(args.max_files)))
    if not file_paths:
        raise ValueError("No symbol parquet files selected")

    live_symbols = sorted({_to_usdt(fp.stem) for fp in file_paths})
    if "BTCUSDT" not in live_symbols:
        live_symbols.append("BTCUSDT")

    scanner_raw, scanner_price_db, scanner_threshold = _collect_scanner_raw_signals(
        cfg=cfg,
        symbols_dir=symbols_dir,
        symbols=live_symbols,
        timeframe=args.timeframe,
        start=start,
        end=end,
        quiet_scanner=not bool(args.no_quiet_scanner),
    )

    bt_cfg = BacktestConfig(
        start_date=str(start),
        end_date=str(end),
        exchange=str(args.exchange),
        leverage=float(cfg.exchange.leverage),
        risk_per_trade=float(cfg.risk.max_risk_per_trade),
        max_open_trades=int(cfg.risk.max_open_positions),
        universe_mode=str(args.universe_mode),
        extractor_mode=str(args.extractor_mode),
        top_coins=[_to_usdt(fp.stem) for fp in file_paths],
        max_files=0,
        use_research_model_selection=True,
        selector_artifact_path=str(cfg.strategy.selector_artifact_path),
        threshold=None,
    )

    profile_path = Path(str(cfg.strategy.profile_path))
    if not profile_path.is_absolute():
        profile_path = BASE_DIR / profile_path
    _apply_profile_to_config(
        bt_cfg,
        profile_path=profile_path,
        profile_name=str(cfg.strategy.profile_name) if cfg.strategy.profile_name else None,
    )

    if args.selector_threshold_override is not None and args.selector_threshold_override >= 0:
        bt_cfg.threshold = float(args.selector_threshold_override)
    elif pd.notna(scanner_threshold) and float(scanner_threshold) >= 0:
        bt_cfg.threshold = float(scanner_threshold)

    scanner_sim_signals = _build_scanner_sim_signals(
        scanner_raw_signals=scanner_raw,
        price_db_by_usdt=scanner_price_db,
        timeframe=args.timeframe,
        max_hold_bars=int(bt_cfg.max_bars_hold),
    )

    scanner_price_db_norm = {_strip_symbol(k): v for k, v in scanner_price_db.items()}
    scanner_trades, scanner_curve, _ = run_portfolio_simulation(scanner_sim_signals, scanner_price_db_norm, bt_cfg)

    bt_signals, bt_price_db, bt_trades, bt_curve = run_backtest_with_config(bt_cfg)

    scanner_summary = _summarize_performance(scanner_trades, scanner_curve, bt_cfg.initial_capital)
    backtest_summary = _summarize_performance(bt_trades, bt_curve, bt_cfg.initial_capital)

    print("=== Scanner-Logic Backtest vs Standard Backtest ===")
    print(f"window={start} -> {end} tf={args.timeframe}")
    print(f"symbols_selected={len(file_paths)}")
    print(f"extractor_mode={bt_cfg.extractor_mode}")
    print(f"scanner_threshold={float(scanner_threshold):.4f}")
    print("\n[Scanner-Logic]")
    print(f"raw_scanner_signals={len(scanner_raw)}")
    print(f"sim_scanner_signals={len(scanner_sim_signals)}")
    print(f"trades={scanner_summary.trades} | win_rate={scanner_summary.win_rate_pct:.2f}%")
    print(f"pnl_usd={scanner_summary.total_pnl_usd:.2f} | ending_equity={scanner_summary.ending_equity:.2f}")
    print(
        f"max_dd={scanner_summary.max_drawdown_pct:.2f}% | sharpe_ann={scanner_summary.sharpe_annualized:.4f} "
        f"| turnover_raw={scanner_summary.turnover_raw:.4f}"
    )

    print("\n[Standard Backtest]")
    print(f"signals={len(bt_signals)}")
    print(f"trades={backtest_summary.trades} | win_rate={backtest_summary.win_rate_pct:.2f}%")
    print(f"pnl_usd={backtest_summary.total_pnl_usd:.2f} | ending_equity={backtest_summary.ending_equity:.2f}")
    print(
        f"max_dd={backtest_summary.max_drawdown_pct:.2f}% | sharpe_ann={backtest_summary.sharpe_annualized:.4f} "
        f"| turnover_raw={backtest_summary.turnover_raw:.4f}"
    )

    out_base = BASE_DIR / str(args.output_prefix)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_base.with_name(out_base.name + "_summary.json")
    scanner_trades_path = out_base.with_name(out_base.name + "_scanner_trades.csv")
    backtest_trades_path = out_base.with_name(out_base.name + "_backtest_trades.csv")

    pd.DataFrame([t.__dict__ for t in scanner_trades]).to_csv(scanner_trades_path, index=False)
    pd.DataFrame([t.__dict__ for t in bt_trades]).to_csv(backtest_trades_path, index=False)

    payload = {
        "window": {"start": str(start), "end": str(end), "timeframe": args.timeframe},
        "symbols_selected": len(file_paths),
        "extractor_mode": str(bt_cfg.extractor_mode),
        "profile_path": str(cfg.strategy.profile_path),
        "profile_name": str(cfg.strategy.profile_name),
        "selector_artifact_path": str(cfg.strategy.selector_artifact_path),
        "scanner_threshold": float(scanner_threshold),
        "scanner_logic": {
            "raw_scanner_signals": int(len(scanner_raw)),
            "sim_scanner_signals": int(len(scanner_sim_signals)),
            "performance": asdict(scanner_summary),
        },
        "standard_backtest": {
            "signals": int(len(bt_signals)),
            "performance": asdict(backtest_summary),
        },
        "artifacts": {
            "summary": str(summary_path),
            "scanner_trades": str(scanner_trades_path),
            "backtest_trades": str(backtest_trades_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nArtifacts:")
    print(f"  summary: {summary_path}")
    print(f"  scanner_trades: {scanner_trades_path}")
    print(f"  backtest_trades: {backtest_trades_path}")


if __name__ == "__main__":
    main()
