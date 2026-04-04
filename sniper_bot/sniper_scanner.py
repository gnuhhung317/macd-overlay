import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from data_processor import BinanceDataProcessor
from ml.p3 import RealDataQuantExtractor

try:
    from ml.backtest_sniper import _prepare_research_selection_features
except Exception:
    def _prepare_research_selection_features(frame: pd.DataFrame) -> pd.DataFrame:
        """Fallback feature builder used when ml.backtest_sniper export is unavailable."""
        out = frame.copy()

        if "timestamp" in out.columns:
            out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

        for c in [
            "entry_p",
            "sl_p",
            "tp_p",
            "structure_size",
            "side",
            "z_trend_20_50",
            "z_price_to_ema200",
            "z_volatility_atr",
            "pullback_depth",
            "dist_to_sl_pct",
        ]:
            if c not in out.columns:
                out[c] = 0.0
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        entry_abs = np.abs(out["entry_p"]) + 1e-12
        sl_dist_abs = np.abs(out["entry_p"] - out["sl_p"])
        tp_dist_abs = np.abs(out["tp_p"] - out["entry_p"])

        out["risk_reward_ratio"] = tp_dist_abs / (sl_dist_abs + 1e-12)
        out["tp_distance_pct"] = tp_dist_abs / entry_abs
        out["sl_distance_pct"] = sl_dist_abs / entry_abs
        out["entry_to_sl_over_structure"] = sl_dist_abs / (np.abs(out["structure_size"]) + 1e-12)
        out["side_z_trend"] = out["side"] * out["z_trend_20_50"]
        out["side_z_price"] = out["side"] * out["z_price_to_ema200"]

        # Conservative default for regime-shift features in fallback mode.
        out["vol_regime_shift_6"] = 0.0
        out["trend_regime_shift_6"] = 0.0
        out["price_regime_shift_6"] = 0.0

        return out


class SniperScanner:
    def __init__(self, config=None, data_processor=None):
        self.config = config
        self.processor = data_processor if data_processor else BinanceDataProcessor(use_futures=True)

        self.base_dir = Path(__file__).resolve().parent.parent

        # Auto-038 selector assets.
        self.profile_path = self._resolve_path(
            str(getattr(getattr(self.config, "strategy", None), "profile_path", "ml/p3_edge_research/experiments/auto_038_live_test.json")),
            [self.base_dir, self.base_dir / "ml" / "p3_edge_research" / "experiments"],
        )
        self.profile_name = str(getattr(getattr(self.config, "strategy", None), "profile_name", "auto_038_live"))
        self.selector_artifact_path = self._resolve_path(
            str(getattr(getattr(self.config, "strategy", None), "selector_artifact_path", "output/selector_artifacts/auto_038_selector_fullasset.joblib")),
            [self.base_dir, self.base_dir / "output" / "selector_artifacts"],
        )

        self.clf = None
        self.features: List[str] = []
        self.threshold: float = 0.65
        self.extractor_params: Dict[str, Any] = {}
        self.profile_info: Dict[str, Any] = {}
        self.progress_log_path = self._resolve_progress_log_path()
        self._history_cache: Dict[str, pd.DataFrame] = {}

        self._load_auto038_selector_model()

    def _resolve_progress_log_path(self) -> Path:
        raw = str(
            getattr(
                getattr(self.config, "strategy", None),
                "progress_detail_log_path",
                "logs/sniper_scan_progress.log",
            )
        )
        path = Path(raw)
        if not path.is_absolute():
            path = self.base_dir / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _append_progress_log(self, line: str) -> None:
        try:
            with open(self.progress_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            # Never fail scanning because file logging failed.
            pass

    @staticmethod
    def _sanitize_log_value(value: Any) -> str:
        return str(value).replace("\n", " ").replace("\r", " ").replace("|", "/")

    def _log_scan_event(self, event: str, **fields: Any) -> None:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        payload = " | ".join([f"{k}={self._sanitize_log_value(v)}" for k, v in fields.items()])
        line = f"{ts} | {event}" + (f" | {payload}" if payload else "")
        self._append_progress_log(line)

    def _normalize_symbol_frame(self, raw_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if raw_df is None or raw_df.empty:
            return pd.DataFrame()

        df = raw_df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        if "timestamp" not in df.columns:
            return pd.DataFrame()

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return pd.DataFrame()

        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        required_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if required_cols:
            df = df.dropna(subset=required_cols)
        if df.empty:
            return pd.DataFrame()

        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        return df.reset_index(drop=True)

    def _fetch_symbol_history(
        self,
        symbol: str,
        timeframe: str,
        full_fetch_start: str,
        required_bars: int,
    ) -> pd.DataFrame:
        cache_key = f"{symbol}|{timeframe}"
        strategy_cfg = getattr(self.config, "strategy", None)
        incremental_scan = bool(getattr(strategy_cfg, "incremental_scan", True))
        incremental_refresh_days = max(1, int(getattr(strategy_cfg, "incremental_refresh_days", 7)))
        scan_history_bars = max(required_bars, int(getattr(strategy_cfg, "scan_history_bars", 1200)))

        def _tf_minutes(tf: str) -> int:
            tf = str(tf).strip().lower()
            if tf.endswith("m"):
                return max(1, int(tf[:-1]))
            if tf.endswith("h"):
                return max(1, int(tf[:-1])) * 60
            if tf.endswith("d"):
                return max(1, int(tf[:-1])) * 24 * 60
            if tf.endswith("w"):
                return max(1, int(tf[:-1])) * 7 * 24 * 60
            return 60

        def _initial_fetch_start() -> str:
            # Warm cache with a bounded window first; fallback below keeps correctness.
            tf_min = _tf_minutes(timeframe)
            bars_per_day = max(1.0, (24.0 * 60.0) / float(tf_min))
            bars_target = max(float(scan_history_bars), float(required_bars))
            days = int((bars_target / bars_per_day) * 1.35) + 2
            return f"{max(7, days)} days ago UTC"

        # Fresh full fetch (first run or when incremental mode disabled).
        if (not incremental_scan) or (cache_key not in self._history_cache):
            first_fetch_start = _initial_fetch_start() if incremental_scan else full_fetch_start
            raw = self.processor.get_historical_data(symbol, timeframe, first_fetch_start, "now UTC")
            df = self._normalize_symbol_frame(raw)

            # Keep correctness if bounded warm-up was too short for this symbol/timeframe.
            if incremental_scan and len(df) < required_bars:
                raw = self.processor.get_historical_data(symbol, timeframe, full_fetch_start, "now UTC")
                df = self._normalize_symbol_frame(raw)

            if not df.empty:
                self._history_cache[cache_key] = df.tail(scan_history_bars).reset_index(drop=True)
            return self._history_cache.get(cache_key, pd.DataFrame())

        cached = self._history_cache.get(cache_key, pd.DataFrame())
        delta_start = f"{incremental_refresh_days} days ago UTC"
        delta_raw = self.processor.get_historical_data(symbol, timeframe, delta_start, "now UTC")
        delta_df = self._normalize_symbol_frame(delta_raw)

        if delta_df.empty:
            merged = cached.copy()
        else:
            merged = self._normalize_symbol_frame(pd.concat([cached, delta_df], ignore_index=True))

        # Fallback to full fetch if cache+delta is still insufficient.
        if len(merged) < required_bars:
            full_raw = self.processor.get_historical_data(symbol, timeframe, full_fetch_start, "now UTC")
            full_df = self._normalize_symbol_frame(full_raw)
            if not full_df.empty:
                merged = full_df

        if not merged.empty:
            merged = merged.tail(scan_history_bars).reset_index(drop=True)
            self._history_cache[cache_key] = merged

        return merged

    def _resolve_path(self, raw_value: str, search_roots: List[Path]) -> Path:
        raw = Path(raw_value)
        if raw.exists():
            return raw
        for root in search_roots:
            candidate = root / raw
            if candidate.exists():
                return candidate
        return raw

    def _load_profile_experiment(self) -> Optional[Dict[str, Any]]:
        if not self.profile_path.exists():
            print(f"[SniperScanner] Profile not found: {self.profile_path}")
            return None

        with open(self.profile_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        experiments = payload.get("experiments", [])
        if not experiments:
            print(f"[SniperScanner] No experiments in profile: {self.profile_path}")
            return None

        chosen = None
        if self.profile_name:
            for exp in experiments:
                if str(exp.get("name", "")).lower() == self.profile_name.lower():
                    chosen = exp
                    break
        if chosen is None:
            chosen = experiments[0]

        self.profile_info = {
            "profile_name": chosen.get("name", "unknown"),
            "profile_path": str(self.profile_path),
        }
        return chosen

    def _load_auto038_selector_model(self) -> None:
        try:
            chosen = self._load_profile_experiment()
            if chosen is None:
                return

            if not self.selector_artifact_path.exists():
                print(f"[SniperScanner] Selector artifact not found: {self.selector_artifact_path}")
                return

            payload = joblib.load(self.selector_artifact_path)
            if not isinstance(payload, dict) or "model" not in payload:
                print(f"[SniperScanner] Invalid selector artifact format: {self.selector_artifact_path}")
                return

            self.clf = payload["model"]
            self.features = list(payload.get("features", []))

            artifact_threshold = float(payload.get("threshold", 0.65))
            selector_override = float(getattr(getattr(self.config, "strategy", None), "selector_threshold_override", -1.0))
            self.threshold = selector_override if selector_override >= 0 else artifact_threshold

            self.extractor_params = {
                "tp_level": float(chosen.get("tp_level", 1.6)),
                "max_hold_bars": int(chosen.get("max_hold_bars", 24)),
                "min_mid_candles": int(chosen.get("min_mid_candles", 6)),
                "min_price_pct": float(chosen.get("min_price_pct", 3.0)),
                "entry_pullback": float(chosen.get("entry_pullback", 0.0)),
                "min_rr": float(chosen.get("min_rr", 1.0)),
            }

            print(
                "[SniperScanner] auto038 selector loaded. "
                f"profile={self.profile_info.get('profile_name')} "
                f"features={len(self.features)} threshold={self.threshold:.3f}"
            )
        except Exception as e:
            print(f"[SniperScanner] Error loading auto038 selector: {e}")

    def scan(self, symbols: List[str], timeframe: str, lookback_days: int = 4) -> List[Dict[str, Any]]:
        if not self.clf:
            print("[SniperScanner] Model missing, cannot scan.")
            return []
        return self._scan_auto038_selector(symbols, timeframe)

    def _build_live_extractor_frame(self, df_calc: pd.DataFrame) -> pd.DataFrame:
        """
        RealDataQuantExtractor was designed for labeled research and needs future bars
        up to max_hold_bars. For live inference, append flat synthetic bars so extraction
        can still evaluate setups at the latest closed candle without lookahead data.
        """
        hold = int(self.extractor_params.get("max_hold_bars", 24))
        if hold <= 0 or df_calc.empty:
            return df_calc

        last_ts = pd.to_datetime(df_calc["timestamp"].iloc[-1])
        deltas = df_calc["timestamp"].diff().dropna()
        step = deltas.mode().iloc[0] if not deltas.empty else pd.Timedelta(hours=1)
        if pd.isna(step) or step <= pd.Timedelta(0):
            step = pd.Timedelta(hours=1)

        base = df_calc.iloc[-1].copy()
        future_rows = []
        for k in range(1, hold + 1):
            row = base.copy()
            row["timestamp"] = last_ts + step * k
            for col in ["open", "high", "low", "close"]:
                row[col] = float(base["close"])
            if "volume" in row.index:
                row["volume"] = 0.0
            future_rows.append(row)

        if not future_rows:
            return df_calc
        future_df = pd.DataFrame(future_rows, columns=df_calc.columns)
        return pd.concat([df_calc, future_df], ignore_index=True)

    def _scan_auto038_selector(self, symbols: List[str], timeframe: str) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        if not symbols:
            return signals

        lookback_days = int(getattr(getattr(self.config, "strategy", None), "selector_lookback_days", 450))
        fetch_start = f"{max(lookback_days, 120)} days ago UTC"
        required_bars = 320
        batch_predict = bool(getattr(getattr(self.config, "strategy", None), "selector_batch_predict", True))

        total_symbols = len(symbols)
        start_time = time.time()
        last_heartbeat_ts = start_time
        heartbeat_interval_sec = 20.0

        # Print more frequent milestones for large universes so scans do not look frozen.
        progress_step_pct = 5 if total_symbols >= 40 else 10
        progress_marks = {
            max(1, int(round(total_symbols * pct / 100.0)))
            for pct in range(progress_step_pct, 101, progress_step_pct)
        }

        print(
            f"[Scanner:auto038] Start scan: symbols={total_symbols} tf={timeframe} "
            f"lookback_days={lookback_days}"
        )
        self._log_scan_event(
            "scan_start",
            timeframe=timeframe,
            symbols=total_symbols,
            lookback_days=lookback_days,
            batch_predict=batch_predict,
        )

        candidate_frames: List[pd.DataFrame] = []
        pending_symbol_records: Dict[int, Dict[str, Any]] = {}

        for i, symbol in enumerate(symbols):
            symbol_start = time.time()
            symbol_status = "ok"
            symbol_reason = "processed"
            symbol_signals = 0
            try:
                idx = i + 1

                if idx in progress_marks:
                    percent = ((i + 1) / total_symbols) * 100
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    rem_time = avg_time * (total_symbols - (i + 1))
                    print(
                        f"[Scanner:auto038] Progress: {percent:.0f}% ({i + 1}/{total_symbols}) "
                        f"Elapsed: {elapsed:.1f}s Est.Rem: {rem_time:.1f}s"
                    )

                now_ts = time.time()
                if now_ts - last_heartbeat_ts >= heartbeat_interval_sec:
                    elapsed = now_ts - start_time
                    print(
                        f"[Scanner:auto038] Heartbeat: {idx}/{total_symbols} "
                        f"current={symbol} elapsed={elapsed:.1f}s"
                    )
                    last_heartbeat_ts = now_ts

                df = self._fetch_symbol_history(
                    symbol=symbol,
                    timeframe=timeframe,
                    full_fetch_start=fetch_start,
                    required_bars=required_bars,
                )
                if df is None or df.empty or len(df) < required_bars:
                    symbol_status = "skip"
                    symbol_reason = "insufficient_history"
                    continue
                live_price = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])

                # Only completed candles are used for setup extraction.
                df_calc = df.iloc[:-1].copy()
                if df_calc.empty:
                    symbol_status = "skip"
                    symbol_reason = "no_closed_candle"
                    continue

                extractor = RealDataQuantExtractor(**self.extractor_params)
                df_for_extract = self._build_live_extractor_frame(df_calc)
                extractor.extract(df_for_extract, symbol)
                setup_df = pd.DataFrame(extractor.dataset)
                if setup_df.empty:
                    symbol_status = "skip"
                    symbol_reason = "no_setup"
                    continue

                # Some extractor payloads may carry duplicated column names.
                if setup_df.columns.duplicated().any():
                    setup_df = setup_df.loc[:, ~setup_df.columns.duplicated()].copy()
                if "timestamp" not in setup_df.columns:
                    symbol_status = "skip"
                    symbol_reason = "setup_missing_timestamp"
                    continue

                setup_df["timestamp"] = pd.to_datetime(setup_df["timestamp"], errors="coerce")
                setup_df = setup_df.dropna(subset=["timestamp"])
                if setup_df.empty:
                    symbol_status = "skip"
                    symbol_reason = "setup_timestamp_invalid"
                    continue
                setup_df["timestamp"] = setup_df["timestamp"].dt.tz_localize(None)
                last_closed_ts = pd.to_datetime(df_calc["timestamp"].iloc[-1]).tz_localize(None)

                latest_setups = setup_df[setup_df["timestamp"] == last_closed_ts].copy()
                if latest_setups.empty:
                    symbol_status = "skip"
                    symbol_reason = "no_latest_setup"
                    continue

                latest_setups["symbol"] = symbol
                latest_setups = latest_setups.reset_index(drop=True)
                latest_setups["signal_id"] = [
                    f"{symbol}|{int(pd.Timestamp(ts).value)}|{signal_ord}"
                    for signal_ord, ts in enumerate(latest_setups["timestamp"])
                ]
                latest_setups["__scan_symbol"] = symbol
                latest_setups["__scan_idx"] = idx
                latest_setups["__live_price"] = live_price

                if batch_predict:
                    candidate_frames.append(latest_setups)
                    symbol_status = "pending_batch"
                    symbol_reason = "candidate_ready"
                else:
                    selection_frame = _prepare_research_selection_features(latest_setups)
                    feature_frame = selection_frame.reindex(columns=self.features, fill_value=0.0)
                    probs = self.clf.predict_proba(feature_frame)[:, 1]

                    for row_idx, row in latest_setups.iterrows():
                        confidence = float(probs[row_idx])
                        if confidence < self.threshold:
                            continue

                        side = int(row.get("side", 1))
                        trade_type = "LONG" if side == 1 else "SHORT"

                        entry_p = float(row["entry_p"])
                        sl_p = float(row["sl_p"])
                        tp_p = float(row["tp_p"])

                        sl_pct = abs(entry_p - sl_p) / max(abs(entry_p), 1e-12)
                        tp_pct = abs(tp_p - entry_p) / max(abs(entry_p), 1e-12)
                        risk_reward = tp_pct / max(sl_pct, 1e-12)

                        signals.append(
                            {
                                "symbol": symbol,
                                "type": trade_type,
                                "timestamp": pd.to_datetime(row["timestamp"]),
                                "confidence": confidence,
                                "status": "SNIPER_AUTO038",
                                "signal_price": float(entry_p),
                                "limit_price": float(entry_p),
                                "current_price": float(live_price),
                                "refined_score": 1.0,
                                "sl_pct": float(sl_pct),
                                "tp_pct": float(tp_pct),
                                "risk_reward": float(risk_reward),
                                "meta": {
                                    "origin": "sniper_scanner_auto038_selector",
                                    "profile_name": self.profile_info.get("profile_name", self.profile_name),
                                    "selector_artifact": str(self.selector_artifact_path),
                                    "selector_threshold": float(self.threshold),
                                    "side": int(side),
                                },
                            }
                        )
                        symbol_signals += 1

                    if symbol_signals == 0 and symbol_status == "ok":
                        symbol_reason = "below_threshold"
                    elif symbol_signals > 0:
                        symbol_reason = "signal_emitted"

            except Exception as e:
                err_msg = str(e)
                symbol_status = "error"
                symbol_reason = err_msg[:160]
                if "-1003" in err_msg:
                    print(f"[SniperScanner] Rate limit hit, sleeping 60s ({err_msg})")
                    time.sleep(60)
                else:
                    print(f"[SniperScanner] Error scanning {symbol} in auto038 mode: {e}")
            finally:
                if symbol_status == "pending_batch" and batch_predict:
                    pending_symbol_records[i + 1] = {
                        "idx": i + 1,
                        "total": total_symbols,
                        "symbol": symbol,
                        "status": symbol_status,
                        "reason": symbol_reason,
                        "signals": 0,
                        "start_ts": symbol_start,
                    }
                else:
                    self._log_scan_event(
                        "symbol_done",
                        idx=i + 1,
                        total=total_symbols,
                        symbol=symbol,
                        status=symbol_status,
                        reason=symbol_reason,
                        signals=symbol_signals,
                        elapsed_s=f"{(time.time() - symbol_start):.3f}",
                    )

        if batch_predict and candidate_frames:
            all_candidates = pd.concat(candidate_frames, ignore_index=True)
            prepared = _prepare_research_selection_features(all_candidates)
            candidate_ids = all_candidates["signal_id"].astype(str).tolist()

            if prepared.empty or "signal_id" not in prepared.columns:
                for rec in pending_symbol_records.values():
                    rec["status"] = "skip"
                    rec["reason"] = "feature_prepare_empty"
            else:
                prepared = prepared.copy()
                prepared["signal_id"] = prepared["signal_id"].astype(str)
                prepared = prepared.drop_duplicates(subset=["signal_id"], keep="last").set_index("signal_id")
                feature_frame = prepared.reindex(candidate_ids, fill_value=0.0).reindex(columns=self.features, fill_value=0.0)
                probs = self.clf.predict_proba(feature_frame)[:, 1]

                for row_idx, row in all_candidates.iterrows():
                    confidence = float(probs[row_idx])
                    symbol = str(row["__scan_symbol"])
                    idx = int(row["__scan_idx"])
                    if confidence < self.threshold:
                        continue

                    side = int(row.get("side", 1))
                    trade_type = "LONG" if side == 1 else "SHORT"

                    entry_p = float(row["entry_p"])
                    sl_p = float(row["sl_p"])
                    tp_p = float(row["tp_p"])

                    sl_pct = abs(entry_p - sl_p) / max(abs(entry_p), 1e-12)
                    tp_pct = abs(tp_p - entry_p) / max(abs(entry_p), 1e-12)
                    risk_reward = tp_pct / max(sl_pct, 1e-12)

                    signals.append(
                        {
                            "symbol": symbol,
                            "type": trade_type,
                            "timestamp": pd.to_datetime(row["timestamp"]),
                            "confidence": confidence,
                            "status": "SNIPER_AUTO038",
                            "signal_price": float(entry_p),
                            "limit_price": float(entry_p),
                            "current_price": float(row["__live_price"]),
                            "refined_score": 1.0,
                            "sl_pct": float(sl_pct),
                            "tp_pct": float(tp_pct),
                            "risk_reward": float(risk_reward),
                            "meta": {
                                "origin": "sniper_scanner_auto038_selector",
                                "profile_name": self.profile_info.get("profile_name", self.profile_name),
                                "selector_artifact": str(self.selector_artifact_path),
                                "selector_threshold": float(self.threshold),
                                "side": int(side),
                            },
                        }
                    )

                    rec = pending_symbol_records.get(idx)
                    if rec is not None:
                        rec["signals"] = int(rec.get("signals", 0)) + 1

                for rec in pending_symbol_records.values():
                    if int(rec.get("signals", 0)) > 0:
                        rec["status"] = "ok"
                        rec["reason"] = "signal_emitted"
                    else:
                        rec["status"] = "ok"
                        rec["reason"] = "below_threshold"

            now_ts = time.time()
            for rec in sorted(pending_symbol_records.values(), key=lambda x: int(x.get("idx", 0))):
                elapsed_s = now_ts - float(rec.get("start_ts", now_ts))
                self._log_scan_event(
                    "symbol_done",
                    idx=rec.get("idx"),
                    total=rec.get("total"),
                    symbol=rec.get("symbol"),
                    status=rec.get("status"),
                    reason=rec.get("reason"),
                    signals=rec.get("signals", 0),
                    elapsed_s=f"{elapsed_s:.3f}",
                )

        self._log_scan_event(
            "scan_end",
            timeframe=timeframe,
            symbols=total_symbols,
            total_signals=len(signals),
            elapsed_s=f"{(time.time() - start_time):.3f}",
            log_file=str(self.progress_log_path),
        )

        return signals
