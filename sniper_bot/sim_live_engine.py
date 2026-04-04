import argparse
import glob
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
sys.path.append(str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "sniper_bot"))

from sniper_bot.config import SniperBotConfig
from sniper_bot.sniper_scanner import SniperScanner
from telegram_notifier import TelegramNotifier


class MockDataProcessor:
    def __init__(self, symbols):
        self.current_time = None
        self.data = {}
        for symbol in symbols:
            pq_path = BASE_DIR / "bitget-data" / "ohlcv" / f"{symbol}_USDT.parquet"
            if not pq_path.exists():
                continue
            try:
                df = pd.read_parquet(pq_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
                df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
                if not df.empty:
                    self.data[symbol] = df
            except Exception as e:
                print(f"[sim] Warning: failed to load {symbol}: {e}")

    def get_historical_data(self, symbol, timeframe, since, until):
        if symbol not in self.data:
            return pd.DataFrame()
        df = self.data[symbol]
        open_candle_ts = self.current_time.floor("h")
        sliced = df[df["timestamp"] <= open_candle_ts].copy()
        if timeframe == "1d":
            sliced = (
                sliced.set_index("timestamp")
                .resample("1D")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .dropna()
                .reset_index()
            )
        # auto_038 selector scanner needs >=320 bars; keep larger tail for stable extraction.
        return sliced.tail(1200)


def _discover_symbols(max_symbols: int) -> list:
    files = glob.glob(str(BASE_DIR / "bitget-data" / "ohlcv" / "*_USDT.parquet"))
    symbols = [Path(f).stem.replace("_USDT", "") for f in files]
    symbols = sorted([s for s in symbols if s != "BTCUSDT"])
    if max_symbols > 0:
        symbols = symbols[:max_symbols]
    return symbols


def _resolve_window(args, processor: MockDataProcessor):
    if args.start and args.end:
        start_dt = pd.Timestamp(args.start)
        end_dt = pd.Timestamp(args.end)
        return start_dt, end_dt

    if not processor.data:
        raise ValueError("No market data loaded for simulation window inference.")

    anchor_symbol = "BTCUSDT" if "BTCUSDT" in processor.data else next(iter(processor.data.keys()))
    end_dt = pd.to_datetime(processor.data[anchor_symbol]["timestamp"].max()).floor("h")
    start_dt = end_dt - pd.Timedelta(hours=args.lookback_hours)
    return start_dt, end_dt


def _build_telegram_notifier(cfg: SniperBotConfig, force_send: bool):
    if not force_send:
        return None
    if not cfg.telegram.enabled:
        print("[sim] Telegram disabled in config, skip notify.")
        return None
    if not cfg.telegram.token or not cfg.telegram.chat_id:
        print("[sim] Telegram token/chat_id missing, skip notify.")
        return None
    return TelegramNotifier(token=cfg.telegram.token, chat_id=cfg.telegram.chat_id)


def _send_telegram_summary(notifier: TelegramNotifier, summary: dict):
    if notifier is None:
        return

    message = (
        "🧪 <b>Sniper Dry-Run Summary</b>\n"
        f"Timeframe: <b>{summary['timeframe']}</b>\n"
        f"Window: <code>{summary['start']}</code> → <code>{summary['end']}</code>\n"
        f"Hours: <b>{summary['hours']}</b>\n"
        f"Symbols: <b>{summary['symbols']}</b>\n"
        f"Artifact: <code>{summary['artifact_name']}</code>\n"
        f"Threshold: <b>{summary['threshold']:.3f}</b>\n\n"
        f"Signals: <b>{summary['signals_total']}</b>\n"
        f"Long/Short: <b>{summary['signals_long']}/{summary['signals_short']}</b>\n"
        f"Active Hours: <b>{summary['active_hours']}</b>\n"
        f"Unique Symbols: <b>{summary['unique_signal_symbols']}</b>\n"
        f"Avg/Max Conf: <b>{summary['avg_confidence']:.4f}/{summary['max_confidence']:.4f}</b>\n"
        f"Scan Errors: <b>{summary['scan_errors']}</b>"
    )
    notifier.send_message(message)


def simulate_dry_run(args):
    cfg = SniperBotConfig.load(Path(args.config_path))
    cfg.exchange.dry_run = True

    if args.profile_path:
        cfg.strategy.profile_path = args.profile_path
    if args.profile_name:
        cfg.strategy.profile_name = args.profile_name
    if args.selector_artifact_path:
        cfg.strategy.selector_artifact_path = args.selector_artifact_path
    if args.selector_threshold_override is not None:
        cfg.strategy.selector_threshold_override = args.selector_threshold_override
    if args.selector_lookback_days is not None:
        cfg.strategy.selector_lookback_days = args.selector_lookback_days

    symbols = _discover_symbols(args.max_symbols)
    if not symbols:
        raise ValueError("No symbols discovered from bitget-data/ohlcv.")

    # Always preload BTC for robust window anchoring.
    preload_symbols = list(dict.fromkeys(symbols + ["BTCUSDT"]))
    print(f"[sim] Preloading parquet data for {len(preload_symbols)} symbols...")
    processor = MockDataProcessor(preload_symbols)

    # Keep only symbols that have loaded data.
    symbols = [s for s in symbols if s in processor.data]
    if not symbols:
        raise ValueError("No symbols with usable parquet data.")

    start_dt, end_dt = _resolve_window(args, processor)
    if end_dt <= start_dt:
        raise ValueError("end must be greater than start.")

    scanner = SniperScanner(config=cfg, data_processor=processor)

    print(
        f"[sim] Starting dry-run simulation: tf={args.timeframe} "
        f"window={start_dt} -> {end_dt} symbols={len(symbols)}"
    )

    current = start_dt
    total_hours = len(pd.date_range(start_dt, end_dt, freq="h"))
    progress_step = max(1, args.progress_every_hours)

    signals_total = 0
    signals_long = 0
    signals_short = 0
    active_hours = 0
    scan_errors = 0
    confidences = []
    symbol_counter = Counter()

    while current <= end_dt:
        processor.current_time = current
        hour_index = len(pd.date_range(start_dt, current, freq="h"))

        if hour_index % progress_step == 1:
            progress = (hour_index / max(total_hours, 1)) * 100
            print(
                f"[sim] Progress {progress:.1f}% ({hour_index}/{total_hours}h) "
                f"at {current.strftime('%Y-%m-%d %H:%M')}"
            )

        original_stdout = sys.stdout
        try:
            if args.quiet_scanner:
                with open(os.devnull, "w", encoding="utf-8") as sink:
                    sys.stdout = sink
                    signals = scanner.scan(symbols, args.timeframe)
            else:
                signals = scanner.scan(symbols, args.timeframe)
        except Exception as e:
            scan_errors += 1
            print(f"[sim] Scanner error at {current}: {e}")
            signals = []
        finally:
            sys.stdout = original_stdout

        if signals:
            active_hours += 1

        for sig in signals:
            conf = float(sig.get("confidence", 0.0))
            side = str(sig.get("type", "")).upper()
            symbol = str(sig.get("symbol", ""))

            signals_total += 1
            confidences.append(conf)
            symbol_counter[symbol] += 1

            if side == "LONG":
                signals_long += 1
            elif side == "SHORT":
                signals_short += 1

            print(
                f"[signal] {current.strftime('%Y-%m-%d %H:%M')} "
                f"{symbol} {side} conf={conf:.4f}"
            )

        if args.stop_after_signals > 0 and signals_total >= args.stop_after_signals:
            print(f"[sim] Early stop reached: {signals_total} signals.")
            break

        current += pd.Timedelta(hours=1)

    summary = {
        "timeframe": args.timeframe,
        "start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end": min(current, end_dt).strftime("%Y-%m-%d %H:%M:%S"),
        "hours": int(len(pd.date_range(start_dt, min(current, end_dt), freq="h"))),
        "symbols": int(len(symbols)),
        "artifact_name": Path(cfg.strategy.selector_artifact_path).name,
        "threshold": float(scanner.threshold),
        "signals_total": int(signals_total),
        "signals_long": int(signals_long),
        "signals_short": int(signals_short),
        "active_hours": int(active_hours),
        "unique_signal_symbols": int(len(symbol_counter)),
        "avg_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
        "max_confidence": float(max(confidences)) if confidences else 0.0,
        "scan_errors": int(scan_errors),
    }

    print("[sim] ===== DRY-RUN SUMMARY =====")
    for key in [
        "timeframe",
        "start",
        "end",
        "hours",
        "symbols",
        "artifact_name",
        "threshold",
        "signals_total",
        "signals_long",
        "signals_short",
        "active_hours",
        "unique_signal_symbols",
        "avg_confidence",
        "max_confidence",
        "scan_errors",
    ]:
        print(f"{key}={summary[key]}")

    notifier = _build_telegram_notifier(cfg, args.send_telegram)
    _send_telegram_summary(notifier, summary)


def build_parser():
    parser = argparse.ArgumentParser(description="Sniper dry-run live-engine simulation")
    parser.add_argument("--config-path", type=str, default=str(BASE_DIR / "sniper_bot" / "sniper_bot_config.json"))
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--lookback-hours", type=int, default=168)
    parser.add_argument("--max-symbols", type=int, default=40)
    parser.add_argument("--progress-every-hours", type=int, default=12)
    parser.add_argument("--stop-after-signals", type=int, default=0)
    parser.add_argument("--send-telegram", action="store_true")
    parser.add_argument("--quiet-scanner", action="store_true")
    parser.add_argument("--no-quiet-scanner", action="store_true")
    parser.add_argument("--profile-path", type=str, default="")
    parser.add_argument("--profile-name", type=str, default="")
    parser.add_argument("--selector-artifact-path", type=str, default="")
    parser.add_argument("--selector-threshold-override", type=float, default=None)
    parser.add_argument("--selector-lookback-days", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if not args.quiet_scanner and not args.no_quiet_scanner:
        args.quiet_scanner = True
    if args.no_quiet_scanner:
        args.quiet_scanner = False
    try:
        simulate_dry_run(args)
    except Exception as e:
        print(f"[sim] Failed: {e}")
        raise
