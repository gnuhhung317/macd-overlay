import argparse
import importlib
import importlib.util
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from ml.p3_edge_research.quant_metrics import PortfolioAssumptions, evaluate_trades
except Exception:
    from p3_edge_research.quant_metrics import PortfolioAssumptions, evaluate_trades

warnings.filterwarnings("ignore")


RESEARCH_MODEL_PROFILES = {
    "baseline": {
        "n_estimators": 300,
        "learning_rate": 0.01,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 50,
        "min_gain_to_split": 0.0,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "class_weight": "balanced",
    },
    "capacity_regularized": {
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "min_gain_to_split": 0.01,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "class_weight": "balanced",
    },
}

RESEARCH_SELECTION_FEATURES = [
    "side",
    "z_trend_20_50",
    "z_price_to_ema200",
    "z_volatility_atr",
    "structure_size",
    "pullback_depth",
    "dist_to_sl_pct",
    "risk_reward_ratio",
    "tp_distance_pct",
    "sl_distance_pct",
    "entry_to_sl_over_structure",
    "side_z_trend",
    "side_z_price",
    "vol_regime_shift_6",
    "trend_regime_shift_6",
    "price_regime_shift_6",
]


class TradeState(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"


@dataclass
class BacktestConfig:
    initial_capital: float = 100.0
    risk_per_trade: float = 0.05
    fee_rate: float = 0.0003
    slippage: float = 0.0005
    max_open_trades: int = 5
    max_bars_hold: int = 24
    start_date: str = "2025-01-01"
    end_date: Optional[str] = None
    leverage: float = 1.0
    exchange: str = "binance"
    long_atr_offset: float = 0.0
    short_atr_offset: float = 0.0
    limit_wait_bars: int = 2
    tp_mult_long: float = 3.0
    sl_mult_long: float = 3.0
    tp_mult_short: float = 3.0
    sl_mult_short: float = 3.0
    threshold: Optional[float] = None
    max_signals_per_timestamp: int = 0

    # Raw p3 setup params
    tp_level: float = 1.6
    entry_pullback: float = 0.0
    min_rr: float = 1.0
    rr_floor_to_tp: float = 0.0
    min_mid_candles: int = 6
    min_price_pct: float = 3.0

    # Optional symbol filter (e.g. ["BTCUSDT", "ETHUSDT"])
    top_coins: List[str] = field(default_factory=list)
    max_files: int = 0

    # Optional ML filter on top of raw p3 setups
    use_ml_filter: bool = False
    ml_model_path: Optional[str] = None
    ml_meta_path: Optional[str] = None
    ml_threshold: Optional[float] = None

    # Simulation/reporting mode
    equity_mode: str = "event"
    universe_mode: str = "sniper"
    selection_mode: str = "sniper"
    extractor_mode: str = "strict"
    enforce_symbol_lock: bool = True
    min_stop_distance: float = 0.0
    output_tag: str = ""
    use_research_model_selection: bool = False
    selection_train_end: str = "2025-01-01"
    selection_val_end: str = "2025-05-01"
    selection_min_val_trades: int = 25
    selection_model_profile: str = "baseline"
    selector_artifact_path: Optional[str] = None
    selector_train_only: bool = False
    selector_force_retrain: bool = False
    selection_debug_checks: bool = True
    selection_debug_shift_zscore: bool = False
    selection_debug_permutation_runs: int = 3
    selection_debug_fail_on_suspect: bool = True
    selection_debug_real_auc_suspect: float = 0.70
    selection_debug_perm_auc_suspect: float = 0.58


@dataclass
class Trade:
    symbol: str
    signal_time: datetime
    type: str
    side: int
    limit_price: float
    tp_price: float
    sl_price: float
    atr_val: float
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    state: TradeState = TradeState.PENDING
    result: str = "PENDING"
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration: int = 0
    wait_bars: int = 0
    pos_size_usd: float = 0.0
    mfe_atr: float = 0.0
    mae_atr: float = 0.0


BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
_P3_EXTRACTOR_CLASS = None
_ML_FILTER_CACHE = {}


def load_assets():
    # Backward-compatible API: this module now runs raw-only and does not load ML assets.
    return None, [], None


def _normalize_symbol(stem: str) -> str:
    return stem.replace("_USDT", "").replace("USDT", "")


def _sanitize_output_tag(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    cleaned = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in str(raw).strip())
    cleaned = cleaned.strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _resolve_config_from_args(args: Tuple) -> BacktestConfig:
    for arg in reversed(args):
        if isinstance(arg, BacktestConfig):
            return arg
    return BacktestConfig()


def _resolve_selector_artifact_path(config: BacktestConfig) -> Optional[Path]:
    if not config.selector_artifact_path:
        return None
    return Path(str(config.selector_artifact_path))


def _save_selector_artifact(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def _load_selector_artifact(path: Path) -> Dict[str, Any]:
    raw = joblib.load(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid selector artifact format: {path}")
    if "model" not in raw:
        raise ValueError(f"Selector artifact missing model: {path}")
    return raw


def _prepare_research_selection_features(frame: pd.DataFrame, config: Optional[BacktestConfig] = None) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).copy()

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

    if "symbol" not in out.columns:
        if "coin" in out.columns:
            out["symbol"] = out["coin"].astype(str).str.upper()
        else:
            out["symbol"] = "UNKNOWN"

    if bool(getattr(config, "selection_debug_shift_zscore", False)):
        out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        grp_dbg = out.groupby("symbol", group_keys=False)
        for zc in ["z_trend_20_50", "z_price_to_ema200", "z_volatility_atr"]:
            out[zc] = grp_dbg[zc].shift(1)
        out[["z_trend_20_50", "z_price_to_ema200", "z_volatility_atr"]] = (
            out[["z_trend_20_50", "z_price_to_ema200", "z_volatility_atr"]]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    entry_abs = np.abs(out["entry_p"]) + 1e-12
    sl_dist_abs = np.abs(out["entry_p"] - out["sl_p"])
    tp_dist_abs = np.abs(out["tp_p"] - out["entry_p"])

    out["risk_reward_ratio"] = tp_dist_abs / (sl_dist_abs + 1e-12)
    out["tp_distance_pct"] = tp_dist_abs / entry_abs
    out["sl_distance_pct"] = sl_dist_abs / entry_abs
    out["entry_to_sl_over_structure"] = out["sl_distance_pct"] / (np.abs(out["structure_size"]) + 1e-12)
    out["side_z_trend"] = out["side"] * out["z_trend_20_50"]
    out["side_z_price"] = out["side"] * out["z_price_to_ema200"]

    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    grp = out.groupby("symbol", group_keys=False)
    out["vol_regime_shift_6"] = grp["z_volatility_atr"].shift(1) - grp["z_volatility_atr"].shift(7)
    out["trend_regime_shift_6"] = grp["z_trend_20_50"].shift(1) - grp["z_trend_20_50"].shift(7)
    out["price_regime_shift_6"] = grp["z_price_to_ema200"].shift(1) - grp["z_price_to_ema200"].shift(7)

    for c in RESEARCH_SELECTION_FEATURES:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "target_win" not in out.columns:
        out["target_win"] = 0
    out["target_win"] = pd.to_numeric(out["target_win"], errors="coerce").fillna(0).astype(int)
    return out


def _research_time_split(df: pd.DataFrame, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["timestamp"] < pd.to_datetime(train_end)].copy()
    val = df[(df["timestamp"] >= pd.to_datetime(train_end)) & (df["timestamp"] < pd.to_datetime(val_end))].copy()
    test = df[df["timestamp"] >= pd.to_datetime(val_end)].copy()
    return train, val, test


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy(dtype=float)
    auc = (ranks[pos].sum() - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)


def _binary_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 1e-9, 1.0 - 1e-9)
    loss = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(np.mean(loss))


def _assert_split_integrity(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Invalid split integrity: one of train/val/test is empty")

    train_max = pd.to_datetime(train["timestamp"]).max()
    val_min = pd.to_datetime(val["timestamp"]).min()
    val_max = pd.to_datetime(val["timestamp"]).max()
    test_min = pd.to_datetime(test["timestamp"]).min()

    if not (train_max < val_min and val_max < test_min):
        raise ValueError(
            "Split integrity violation: expected train_max < val_min and val_max < test_min, "
            f"got train_max={train_max}, val_min={val_min}, val_max={val_max}, test_min={test_min}"
        )


def _run_permutation_debug(
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_profile: str,
    runs: int,
) -> Dict[str, float]:
    n_runs = max(1, int(runs))
    model_params = dict(RESEARCH_MODEL_PROFILES.get(str(model_profile), RESEARCH_MODEL_PROFILES["baseline"]))
    rng = np.random.default_rng(42)

    auc_vals: List[float] = []
    ll_vals: List[float] = []

    for _ in range(n_runs):
        y_perm = rng.permutation(train["target_win"].to_numpy(dtype=int))
        if np.unique(y_perm).size < 2:
            continue

        model = lgb.LGBMClassifier(**model_params, verbose=-1)
        model.fit(
            train[RESEARCH_SELECTION_FEATURES],
            y_perm,
            eval_set=[(val[RESEARCH_SELECTION_FEATURES], val["target_win"])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )
        val_prob = model.predict_proba(val[RESEARCH_SELECTION_FEATURES])[:, 1]
        y_val = val["target_win"].to_numpy(dtype=int)
        auc_vals.append(_binary_auc(y_val, val_prob))
        ll_vals.append(_binary_logloss(y_val, val_prob))

    return {
        "perm_auc_mean": float(np.nanmean(auc_vals)) if auc_vals else float("nan"),
        "perm_logloss_mean": float(np.nanmean(ll_vals)) if ll_vals else float("nan"),
        "perm_runs": float(len(auc_vals)),
    }


def _print_selection_debug_report(
    model: Any,
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: BacktestConfig,
) -> None:
    y_val = val["target_win"].to_numpy(dtype=int)
    val_prob = model.predict_proba(val[RESEARCH_SELECTION_FEATURES])[:, 1]
    real_auc = _binary_auc(y_val, val_prob)
    real_ll = _binary_logloss(y_val, val_prob)

    perm_stats = _run_permutation_debug(
        train=train,
        val=val,
        model_profile=str(config.selection_model_profile),
        runs=int(config.selection_debug_permutation_runs),
    )

    print(
        "[SELECTION DEBUG] "
        f"val_auc_real={real_auc:.4f} | val_logloss_real={real_ll:.5f} | "
        f"val_auc_perm={perm_stats['perm_auc_mean']:.4f} | "
        f"val_logloss_perm={perm_stats['perm_logloss_mean']:.5f} | "
        f"perm_runs={int(perm_stats['perm_runs'])}"
    )

    perm_auc = perm_stats["perm_auc_mean"]
    suspect_high_perm = False
    suspect_close_gap = False
    if np.isfinite(perm_auc):
        if perm_auc >= float(config.selection_debug_perm_auc_suspect):
            print(
                "[SELECTION DEBUG][WARN] Permuted-label AUC is suspiciously high; "
                f"val_auc_perm={perm_auc:.4f} >= {float(config.selection_debug_perm_auc_suspect):.2f}"
            )
        if np.isfinite(real_auc) and abs(real_auc - perm_auc) <= 0.03:
            print("[SELECTION DEBUG][WARN] Real AUC is too close to permuted AUC; signal may be unstable/leaky.")

    if np.isfinite(real_auc) and np.isfinite(perm_auc):
        suspect_high_perm = (
            float(real_auc) >= float(config.selection_debug_real_auc_suspect)
            and float(perm_auc) >= float(config.selection_debug_perm_auc_suspect)
        )
        suspect_close_gap = abs(float(real_auc) - float(perm_auc)) <= 0.03

    if suspect_high_perm:
        print(
            "[SELECTION DEBUG][WARN] Leakage suspicion gate triggered: "
            f"val_auc_real={real_auc:.4f} >= {float(config.selection_debug_real_auc_suspect):.2f} and "
            f"val_auc_perm={perm_auc:.4f} >= {float(config.selection_debug_perm_auc_suspect):.2f}"
        )

    if bool(config.selection_debug_fail_on_suspect) and suspect_high_perm:
        raise ValueError(
            "Leakage suspicion gate failed in selector debug checks. "
            "Disable with --no-selection-debug-fail-on-suspect only after manual review."
        )


def _build_threshold_grid(config: BacktestConfig) -> List[float]:
    if config.threshold is not None:
        return [float(config.threshold)]
    return [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def _portfolio_assumptions_from_config(config: BacktestConfig) -> PortfolioAssumptions:
    return PortfolioAssumptions(
        initial_capital=float(config.initial_capital),
        leverage=float(config.leverage),
        risk_per_trade=float(config.risk_per_trade),
        max_concurrent_positions=int(config.max_open_trades),
        min_stop_distance=float(config.min_stop_distance),
        fee_bps_per_side=float(config.fee_rate * 10000.0),
        slippage_bps_per_side=float(config.slippage * 10000.0),
        panic_extra_slippage_bps=10.0,
    )


def _pick_threshold_from_val(
    val_df: pd.DataFrame,
    val_probs: np.ndarray,
    threshold_grid: List[float],
    assumptions: PortfolioAssumptions,
    min_val_trades: int,
) -> float:
    if len(val_df) == 0:
        return 0.82

    best_threshold = float(threshold_grid[0]) if threshold_grid else 0.82
    best_score = -np.inf
    for th in sorted(set(float(x) for x in threshold_grid)):
        picked = val_df.loc[val_probs >= th].copy()
        if len(picked) < int(min_val_trades):
            continue
        summary, _ = evaluate_trades(picked, assumptions=assumptions, periods_per_year=8760)
        sharpe = summary.get("sharpe_annualized", np.nan)
        net_return = summary.get("net_return_pct", np.nan)
        score = (float(sharpe) if np.isfinite(sharpe) else -999.0) * 1000.0 + (
            float(net_return) if np.isfinite(net_return) else -999.0
        )
        if score > best_score:
            best_score = score
            best_threshold = float(th)

    return best_threshold


def _apply_research_model_selection(
    potential_signals: List[Dict],
    selection_rows: List[pd.DataFrame],
    config: BacktestConfig,
) -> List[Dict]:
    if not potential_signals or not selection_rows:
        return potential_signals

    data = pd.concat(selection_rows, ignore_index=True)
    data = _prepare_research_selection_features(data, config=config)
    if data.empty or "signal_id" not in data.columns:
        return potential_signals

    model = None
    chosen_th: float
    artifact_path = _resolve_selector_artifact_path(config)
    artifact_loaded = False
    inference_features: List[str] = list(RESEARCH_SELECTION_FEATURES)
    train = pd.DataFrame()
    val = pd.DataFrame()

    if artifact_path is not None and artifact_path.exists() and not bool(config.selector_force_retrain):
        payload = _load_selector_artifact(artifact_path)
        model = payload["model"]
        chosen_th = float(payload.get("threshold", 0.7))
        artifact_loaded = True
        payload_features = payload.get("features", None)
        if isinstance(payload_features, list) and payload_features:
            inference_features = [str(x) for x in payload_features]
        elif hasattr(model, "feature_name_") and getattr(model, "feature_name_"):
            inference_features = [str(x) for x in list(getattr(model, "feature_name_"))]
        print(f"[SELECTION] loaded artifact: {artifact_path}")

        if bool(config.selection_debug_checks):
            train, val, test_dbg = _research_time_split(
                data,
                train_end=str(config.selection_train_end),
                val_end=str(config.selection_val_end),
            )
            if min(len(train), len(val), len(test_dbg)) > 0:
                _assert_split_integrity(train, val, test_dbg)
                print(
                    "[SELECTION DEBUG] split_ranges | "
                    f"train=[{train['timestamp'].min()} -> {train['timestamp'].max()}] n={len(train)} | "
                    f"val=[{val['timestamp'].min()} -> {val['timestamp'].max()}] n={len(val)} | "
                    f"test=[{test_dbg['timestamp'].min()} -> {test_dbg['timestamp'].max()}] n={len(test_dbg)}"
                )
                if inference_features == list(RESEARCH_SELECTION_FEATURES):
                    _print_selection_debug_report(model=model, train=train, val=val, config=config)
                else:
                    print(
                        "[SELECTION DEBUG] Skipped metric debug for external artifact "
                        "with custom feature schema."
                    )
            else:
                print(
                    "[SELECTION DEBUG] Skipped split/permutation checks during artifact inference "
                    "because current date window does not contain full train/val/test ranges."
                )
    else:
        train, val, test_dbg = _research_time_split(
            data,
            train_end=str(config.selection_train_end),
            val_end=str(config.selection_val_end),
        )

        if min(len(train), len(val)) == 0:
            raise ValueError(
                "Calendar split produced empty train/val. "
                "Set --start earlier than --selection-train-end or adjust split dates. "
                "Fallback split is disabled to avoid hidden leakage."
            )

        if bool(config.selection_debug_checks):
            _assert_split_integrity(train, val, test_dbg)
            print(
                "[SELECTION DEBUG] split_ranges | "
                f"train=[{train['timestamp'].min()} -> {train['timestamp'].max()}] n={len(train)} | "
                f"val=[{val['timestamp'].min()} -> {val['timestamp'].max()}] n={len(val)} | "
                f"test=[{test_dbg['timestamp'].min()} -> {test_dbg['timestamp'].max()}] n={len(test_dbg)}"
            )

        if min(len(train), len(val)) <= 10 or train["target_win"].nunique() < 2:
            print("[SELECTION] insufficient train/val data, fallback to raw scoring")
            return potential_signals

        model_params = dict(RESEARCH_MODEL_PROFILES.get(str(config.selection_model_profile), RESEARCH_MODEL_PROFILES["baseline"]))
        model = lgb.LGBMClassifier(**model_params, verbose=-1)
        model.fit(
            train[RESEARCH_SELECTION_FEATURES],
            train["target_win"],
            eval_set=[(val[RESEARCH_SELECTION_FEATURES], val["target_win"])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        val_probs = model.predict_proba(val[RESEARCH_SELECTION_FEATURES])[:, 1]
        threshold_grid = _build_threshold_grid(config)
        assumptions = _portfolio_assumptions_from_config(config)
        chosen_th = _pick_threshold_from_val(
            val_df=val,
            val_probs=val_probs,
            threshold_grid=threshold_grid,
            assumptions=assumptions,
            min_val_trades=int(config.selection_min_val_trades),
        )

        if artifact_path is not None:
            payload = {
                "model": model,
                "threshold": float(chosen_th),
                "selection_model_profile": str(config.selection_model_profile),
                "selection_train_end": str(config.selection_train_end),
                "selection_val_end": str(config.selection_val_end),
                "features": list(RESEARCH_SELECTION_FEATURES),
            }
            _save_selector_artifact(artifact_path, payload)
            print(f"[SELECTION] saved artifact: {artifact_path}")

    if bool(config.selection_debug_checks) and not artifact_loaded:
        _print_selection_debug_report(model=model, train=train, val=val, config=config)

    for c in inference_features:
        if c not in data.columns:
            data[c] = 0.0
        data[c] = pd.to_numeric(data[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    config.threshold = float(chosen_th)

    if bool(config.selector_train_only):
        print(
            f"[SELECTION] train-only complete | "
            f"model_profile={config.selection_model_profile} | threshold={chosen_th:.4f}"
        )
        return []

    all_probs = model.predict_proba(data[inference_features])[:, 1]

    data = data.copy()
    data["ai_prob"] = all_probs
    keep_ids = set(data.loc[data["ai_prob"] >= float(chosen_th), "signal_id"].astype(str).tolist())
    score_map = dict(zip(data["signal_id"].astype(str), data["ai_prob"].astype(float)))

    filtered: List[Dict] = []
    for sig in potential_signals:
        sid = str(sig.get("signal_id", ""))
        score = float(score_map.get(sid, 0.0))
        sig["prob"] = score
        sig["ai_prob"] = score
        if sid in keep_ids:
            filtered.append(sig)

    print(
        f"[SELECTION] model_profile={config.selection_model_profile} | "
        f"threshold={chosen_th:.4f} | loaded_artifact={artifact_loaded} | "
        f"{len(potential_signals)} -> {len(filtered)}"
    )
    return filtered


def _default_auto018_profile_path() -> Path:
    return BASE_DIR / "ml" / "p3_edge_research" / "experiments" / "auto_018_live_test.json"


def _normalize_extractor_mode(raw_mode: Optional[str]) -> str:
    mode = str(raw_mode or "strict").strip().lower()
    aliases = {
        "live": "causal",
        "live-compatible": "causal",
        "live_compat": "causal",
        "live_compatible": "causal",
        "compat": "causal",
        "causal": "causal",
        "strict": "strict",
    }
    normalized = aliases.get(mode, mode)
    if normalized not in {"strict", "causal"}:
        raise ValueError(f"Invalid extractor_mode={raw_mode!r}. Expected strict or causal.")
    return normalized


def _build_live_compatible_extractor_frame(df_calc: pd.DataFrame, max_hold_bars: int) -> pd.DataFrame:
    """
    Mirror scanner behavior by appending flat synthetic future bars.
    This lets extraction evaluate setups at the newest candles consistently.
    """
    hold = max(0, int(max_hold_bars))
    if hold <= 0 or df_calc.empty or "timestamp" not in df_calc.columns or "close" not in df_calc.columns:
        return df_calc

    frame = df_calc.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    if frame.empty:
        return df_calc

    last_ts = pd.to_datetime(frame["timestamp"].iloc[-1])
    deltas = frame["timestamp"].diff().dropna()
    step = deltas.mode().iloc[0] if not deltas.empty else pd.Timedelta(hours=1)
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(hours=1)

    base = frame.iloc[-1].copy()
    base_close = float(pd.to_numeric(pd.Series([base.get("close")]), errors="coerce").fillna(0.0).iloc[0])
    future_rows = []
    for k in range(1, hold + 1):
        row = base.copy()
        row["timestamp"] = last_ts + (step * k)
        for col in ("open", "high", "low", "close"):
            if col in frame.columns:
                row[col] = base_close
        if "volume" in frame.columns:
            row["volume"] = 0.0
        future_rows.append(row)

    if not future_rows:
        return frame
    future_df = pd.DataFrame(future_rows, columns=frame.columns)
    return pd.concat([frame, future_df], ignore_index=True)


def _apply_profile_to_config(config: BacktestConfig, profile_path: Path, profile_name: Optional[str]) -> Dict[str, object]:
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    with open(profile_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    experiments = payload.get("experiments", [])
    if not experiments:
        raise ValueError(f"No experiments found in profile: {profile_path}")

    chosen = None
    if profile_name:
        for exp in experiments:
            if str(exp.get("name", "")).lower() == str(profile_name).lower():
                chosen = exp
                break
        if chosen is None:
            raise ValueError(f"Profile '{profile_name}' not found in {profile_path}")
    else:
        chosen = experiments[0]

    mapping = {
        "tp_level": ("tp_level", float),
        "entry_pullback": ("entry_pullback", float),
        "min_rr": ("min_rr", float),
        "rr_floor_to_tp": ("rr_floor_to_tp", float),
        "max_hold_bars": ("max_bars_hold", int),
        "min_mid_candles": ("min_mid_candles", int),
        "min_price_pct": ("min_price_pct", float),
    }
    for src_key, (dst_attr, cast_fn) in mapping.items():
        if src_key in chosen and chosen.get(src_key) is not None:
            setattr(config, dst_attr, cast_fn(chosen[src_key]))

    threshold_grid = chosen.get("threshold_grid", [])
    if config.threshold is None and isinstance(threshold_grid, list) and threshold_grid:
        config.threshold = float(threshold_grid[0])

    assumptions = payload.get("assumptions", {}) or {}
    if assumptions.get("fee_bps_per_side") is not None:
        config.fee_rate = float(assumptions["fee_bps_per_side"]) / 10000.0
    if assumptions.get("slippage_bps_per_side") is not None:
        config.slippage = float(assumptions["slippage_bps_per_side"]) / 10000.0

    return {
        "profile_name": chosen.get("name", "<unknown>"),
        "profile_path": str(profile_path),
    }


def _load_p3_extractor_class():
    global _P3_EXTRACTOR_CLASS
    if _P3_EXTRACTOR_CLASS is not None:
        return _P3_EXTRACTOR_CLASS

    module_candidates = ["ml.p3", "p3"]
    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
            _P3_EXTRACTOR_CLASS = getattr(mod, "RealDataQuantExtractor")
            return _P3_EXTRACTOR_CLASS
        except Exception:
            continue

    p3_path = Path(__file__).with_name("p3.py")
    spec = importlib.util.spec_from_file_location("p3_raw_setup", p3_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load RealDataQuantExtractor from p3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _P3_EXTRACTOR_CLASS = getattr(mod, "RealDataQuantExtractor")
    return _P3_EXTRACTOR_CLASS


def _build_extractor(config: BacktestConfig):
    extractor_cls = _load_p3_extractor_class()
    return extractor_cls(
        tp_level=config.tp_level,
        max_hold_bars=config.max_bars_hold,
        min_mid_candles=config.min_mid_candles,
        min_price_pct=config.min_price_pct,
        entry_pullback=config.entry_pullback,
        min_rr=config.min_rr,
        rr_floor_to_tp=config.rr_floor_to_tp,
    )


def _default_ml_model_path() -> Path:
    return BASE_DIR / "ml" / "models" / "p3_meta_edge_model.joblib"


def _default_ml_meta_path() -> Path:
    return BASE_DIR / "ml" / "models" / "p3_meta_edge_meta.json"


def _resolve_ml_paths(config: BacktestConfig) -> Tuple[Path, Path]:
    model_path = Path(config.ml_model_path) if config.ml_model_path else _default_ml_model_path()
    meta_path = Path(config.ml_meta_path) if config.ml_meta_path else _default_ml_meta_path()
    return model_path, meta_path


def _load_ml_filter_assets(config: BacktestConfig):
    if not config.use_ml_filter:
        return None, {}

    model_path, meta_path = _resolve_ml_paths(config)
    cache_key = (str(model_path.resolve()), str(meta_path.resolve()))
    if cache_key in _ML_FILTER_CACHE:
        return _ML_FILTER_CACHE[cache_key]

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    filter_type = str(meta.get("filter_type", "classification")).lower()
    model = None
    if filter_type == "classification":
        if not model_path.exists():
            raise FileNotFoundError(f"ML model not found: {model_path}")
        model = joblib.load(model_path)
    elif filter_type == "regime":
        if not meta:
            raise ValueError("Regime filter meta is missing or empty")

    _ML_FILTER_CACHE[cache_key] = (model, meta)
    return model, meta


def _assign_regime(series: pd.Series, edges: List[float]) -> pd.Series:
    bins = list(edges)
    if len(bins) != 4:
        bins = [-np.inf, 0.0, 1e-9, np.inf]
    out = pd.cut(pd.to_numeric(series, errors="coerce"), bins=bins, labels=["low", "mid", "high"], include_lowest=True)
    out = out.astype("object").where(out.notna(), "mid")
    return out.astype(str)


def _apply_regime_filter(setup_df: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    regime_def = meta.get("regime_definition", {})
    edges = regime_def.get("edges", {})

    vol_edges = edges.get("volatility", [-np.inf, 0.0, 1e-9, np.inf])
    volume_edges = edges.get("volume", [-np.inf, 0.0, 1e-9, np.inf])
    reaction_edges = edges.get("reaction", [-np.inf, 0.0, 1e-9, np.inf])

    out = setup_df.copy()
    side_series = out["side"] if "side" in out.columns else pd.Series(0.0, index=out.index)
    mom1_series = out["momentum_1"] if "momentum_1" in out.columns else pd.Series(0.0, index=out.index)
    side_series = pd.to_numeric(side_series, errors="coerce").fillna(0.0)
    mom1_series = pd.to_numeric(mom1_series, errors="coerce").fillna(0.0)
    vol_series = out["z_volatility_atr"] if "z_volatility_atr" in out.columns else pd.Series(0.0, index=out.index)
    volume_series = out["volume_ratio_20"] if "volume_ratio_20" in out.columns else pd.Series(0.0, index=out.index)
    vol_series = pd.to_numeric(vol_series, errors="coerce").fillna(0.0)
    volume_series = pd.to_numeric(volume_series, errors="coerce").fillna(0.0)

    out["reaction_score"] = side_series * mom1_series
    out["volatility_regime"] = _assign_regime(vol_series, vol_edges)
    out["volume_regime"] = _assign_regime(volume_series, volume_edges)
    out["reaction_regime"] = _assign_regime(out["reaction_score"], reaction_edges)
    out["regime_key"] = out["volatility_regime"] + "|" + out["volume_regime"] + "|" + out["reaction_regime"]

    allowed_list = meta.get("allowed_regimes", [])
    if "allowed_regimes" in meta and len(allowed_list) == 0:
        return out.iloc[0:0].copy()

    allowed = set(allowed_list)
    if allowed:
        out = out[out["regime_key"].isin(allowed)].copy()

    score_map = meta.get("regime_score_map", {})
    raw_score = pd.to_numeric(out["regime_key"].map(score_map), errors="coerce")
    raw_score = raw_score.fillna(raw_score.min() if raw_score.notna().any() else 0.0)
    out["ml_prob_raw"] = raw_score
    out["ml_prob"] = raw_score.rank(method="average", pct=True).fillna(0.0)
    return out


def _default_ml_feature_columns() -> List[str]:
    return [
        "side",
        "z_trend_20_50",
        "z_price_to_ema200",
        "z_volatility_atr",
        "structure_size",
        "pullback_depth",
        "volume_spike_z",
        "volume_ratio_20",
        "momentum_1",
        "momentum_3",
        "momentum_6",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "wick_imbalance",
        "micro_efficiency",
        "close_in_range",
        "side_z_trend",
        "side_z_price",
        "side_momentum_3",
        "side_wick_imbalance",
    ]


def _prepare_ml_filter_frame(setup_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    mdf = setup_df.copy()
    side_series = mdf["side"] if "side" in mdf.columns else pd.Series(0.0, index=mdf.index)
    z_trend_series = mdf["z_trend_20_50"] if "z_trend_20_50" in mdf.columns else pd.Series(0.0, index=mdf.index)
    z_price_series = mdf["z_price_to_ema200"] if "z_price_to_ema200" in mdf.columns else pd.Series(0.0, index=mdf.index)
    mom3_series = mdf["momentum_3"] if "momentum_3" in mdf.columns else pd.Series(0.0, index=mdf.index)
    wick_series = mdf["wick_imbalance"] if "wick_imbalance" in mdf.columns else pd.Series(0.0, index=mdf.index)

    side_series = pd.to_numeric(side_series, errors="coerce").fillna(0.0)
    z_trend_series = pd.to_numeric(z_trend_series, errors="coerce").fillna(0.0)
    z_price_series = pd.to_numeric(z_price_series, errors="coerce").fillna(0.0)
    mom3_series = pd.to_numeric(mom3_series, errors="coerce").fillna(0.0)
    wick_series = pd.to_numeric(wick_series, errors="coerce").fillna(0.0)

    mdf["side_z_trend"] = side_series * z_trend_series
    mdf["side_z_price"] = side_series * z_price_series
    mdf["side_momentum_3"] = side_series * mom3_series
    mdf["side_wick_imbalance"] = side_series * wick_series

    for col in feature_columns:
        if col not in mdf.columns:
            mdf[col] = 0.0

    return mdf[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _simulate_trade_path(signal: Dict, leverage: float, sl_panic_slippage: float = 0.002):
    side = int(signal.get("side", 1))
    entry_p = float(signal["entry_p"])
    sl_p = float(signal["sl_p"])
    tp_p = float(signal["tp_p"])

    lows = list(signal.get("future_lows", []))
    highs = list(signal.get("future_highs", []))
    closes = list(signal.get("future_closes", []))

    max_len = min(len(lows), len(highs))
    for i in range(max_len):
        low_t = lows[i]
        high_t = highs[i]

        if side == 1:
            if (low_t / entry_p - 1.0) * leverage <= -0.85:
                return "LIQUIDATED", entry_p * (1.0 - 0.85 / max(leverage, 1e-8)), i
            if low_t <= sl_p:
                return "LOSS", sl_p * (1.0 - sl_panic_slippage), i
            if high_t >= tp_p:
                return "WIN", tp_p, i
        else:
            if (high_t / entry_p - 1.0) * leverage >= 0.85:
                return "LIQUIDATED", entry_p * (1.0 + 0.85 / max(leverage, 1e-8)), i
            if high_t >= sl_p:
                return "LOSS", sl_p * (1.0 + sl_panic_slippage), i
            if low_t <= tp_p:
                return "WIN", tp_p, i

    if closes:
        return "TIMEOUT", float(closes[-1]), max(0, len(closes) - 1)
    if side == 1 and lows:
        return "TIMEOUT", float(lows[-1]), max(0, len(lows) - 1)
    if side == -1 and highs:
        return "TIMEOUT", float(highs[-1]), max(0, len(highs) - 1)
    return "TIMEOUT", entry_p, 0


def _compute_target_win_from_future(side: int, entry_p: float, sl_p: float, tp_p: float, lows: List[float], highs: List[float]) -> int:
    target_win = 0
    for low_t, high_t in zip(lows, highs):
        if side == 1:
            if (low_t / max(entry_p, 1e-12) - 1.0) * 10.0 <= -0.85:
                target_win = 0
                break
            if low_t <= sl_p:
                target_win = 0
                break
            if high_t >= tp_p:
                target_win = 1
                break
        else:
            if (high_t / max(entry_p, 1e-12) - 1.0) * 10.0 >= 0.85:
                target_win = 0
                break
            if high_t >= sl_p:
                target_win = 0
                break
            if low_t <= tp_p:
                target_win = 1
                break
    return int(target_win)


def _estimate_exit_time(symbol: str, signal_time: pd.Timestamp, bar_offset: int, fallback_time, full_price_db):
    df = full_price_db.get(symbol)
    if df is None or "timestamp" not in df.columns or df.empty:
        return pd.to_datetime(fallback_time) if fallback_time is not None else signal_time

    ts_series = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    if ts_series.empty:
        return pd.to_datetime(fallback_time) if fallback_time is not None else signal_time

    entry_idx = int(ts_series.searchsorted(pd.to_datetime(signal_time), side="left"))
    exit_idx = min(len(ts_series) - 1, max(0, entry_idx + 1 + int(bar_offset)))
    return ts_series.iloc[exit_idx]


def _prepare_price_lookup(full_price_db: Dict[str, pd.DataFrame]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    lookup = {}
    for symbol, df in full_price_db.items():
        if df is None or df.empty or "timestamp" not in df.columns or "close" not in df.columns:
            continue

        tmp = df[["timestamp", "close"]].copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"]).dt.tz_localize(None)
        tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
        tmp = tmp.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        tmp = tmp.drop_duplicates(subset=["timestamp"], keep="last")
        if tmp.empty:
            continue

        ts_ns = tmp["timestamp"].astype("int64").to_numpy()
        px = tmp["close"].to_numpy(dtype=float)
        lookup[symbol] = (ts_ns, px)

    return lookup


def _lookup_price_at_or_before(symbol: str, ts: pd.Timestamp, price_lookup: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Optional[float]:
    item = price_lookup.get(symbol)
    if item is None:
        return None

    ts_ns, px = item
    if len(ts_ns) == 0:
        return None

    key = np.int64(pd.Timestamp(ts).value)
    idx = int(np.searchsorted(ts_ns, key, side="right") - 1)
    if idx < 0:
        return None
    return float(px[idx])


def _build_mtm_equity_curve(trades: List[Trade], full_price_db, config: BacktestConfig) -> List[Tuple[pd.Timestamp, float]]:
    if not trades:
        return []

    entry_events: Dict[pd.Timestamp, List[Trade]] = {}
    exit_events: Dict[pd.Timestamp, List[Trade]] = {}
    timeline = set()
    traded_symbols = set()

    for tr in trades:
        if tr.entry_time is None:
            continue

        entry_ts = pd.to_datetime(tr.entry_time).tz_localize(None)
        exit_ts = pd.to_datetime(tr.exit_time).tz_localize(None) if tr.exit_time is not None else entry_ts

        entry_events.setdefault(entry_ts, []).append(tr)
        exit_events.setdefault(exit_ts, []).append(tr)
        timeline.add(entry_ts)
        timeline.add(exit_ts)
        traded_symbols.add(tr.symbol)

    if not timeline:
        return []

    t_min = min(timeline)
    t_max = max(timeline)
    for symbol in traded_symbols:
        df = full_price_db.get(symbol)
        if df is None or df.empty or "timestamp" not in df.columns:
            continue
        ts_series = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        mask = (ts_series >= t_min) & (ts_series <= t_max)
        if mask.any():
            timeline.update(ts_series.loc[mask].tolist())

    ordered_ts = sorted(timeline)
    price_lookup = _prepare_price_lookup(full_price_db)

    realized_capital = float(config.initial_capital)
    open_positions: List[Trade] = []
    mtm_curve: List[Tuple[pd.Timestamp, float]] = []

    for ts in ordered_ts:
        for tr in exit_events.get(ts, []):
            if tr in open_positions:
                open_positions.remove(tr)
            realized_capital += float(tr.pnl_usd)

        for tr in entry_events.get(ts, []):
            open_positions.append(tr)

        floating_pnl = 0.0
        for tr in open_positions:
            current_price = _lookup_price_at_or_before(tr.symbol, ts, price_lookup)
            if current_price is None:
                current_price = float(tr.entry_price)

            if tr.side == 1:
                pnl_pct = (current_price - tr.entry_price) / max(tr.entry_price, 1e-12)
            else:
                pnl_pct = (tr.entry_price - current_price) / max(tr.entry_price, 1e-12)

            fee_est = tr.pos_size_usd * config.fee_rate * 2.0
            floating_pnl += (tr.pos_size_usd * pnl_pct) - fee_est

        mtm_curve.append((pd.to_datetime(ts), float(realized_capital + floating_pnl)))

    return mtm_curve


def _compute_curve_metrics(equity_curve: List[Tuple[pd.Timestamp, float]]) -> Dict[str, float]:
    if not equity_curve:
        return {
            "max_drawdown_pct": float("nan"),
            "min_balance": float("nan"),
            "sharpe_annualized": float("nan"),
            "sharpe_not_annualized": float("nan"),
            "sample_size": 0,
            "periods_per_year": 365,
        }

    eq_df = pd.DataFrame(equity_curve, columns=["time", "val"])
    eq_df["time"] = pd.to_datetime(eq_df["time"]).dt.tz_localize(None)
    eq_df["val"] = pd.to_numeric(eq_df["val"], errors="coerce")
    eq_df = eq_df.dropna(subset=["time", "val"]).sort_values("time")
    eq_df = eq_df.drop_duplicates(subset=["time"], keep="last")
    if eq_df.empty:
        return {
            "max_drawdown_pct": float("nan"),
            "min_balance": float("nan"),
            "sharpe_annualized": float("nan"),
            "sharpe_not_annualized": float("nan"),
            "sample_size": 0,
            "periods_per_year": 365,
        }

    equity_series = eq_df.set_index("time")["val"]
    daily_equity = equity_series.resample("D").last().ffill()
    daily_returns = daily_equity.pct_change().dropna()

    periods_per_year = 365
    sample_size = int(len(daily_returns))
    sharpe_not_annualized = float("nan")
    sharpe_annualized = float("nan")
    if sample_size > 1:
        mean_ret = float(daily_returns.mean())
        std_ret = float(daily_returns.std())
        sharpe_not_annualized = mean_ret / (std_ret + 1e-12)
        if sample_size >= 30:
            sharpe_annualized = np.sqrt(periods_per_year) * sharpe_not_annualized

    roll_max = equity_series.cummax()
    max_dd = float(((equity_series / (roll_max + 1e-12)) - 1.0).min() * 100.0)

    return {
        "max_drawdown_pct": max_dd,
        "min_balance": float(equity_series.min()),
        "sharpe_annualized": float(sharpe_annualized),
        "sharpe_not_annualized": float(sharpe_not_annualized),
        "sample_size": sample_size,
        "periods_per_year": periods_per_year,
    }


def _compute_turnover_metrics(trades: List[Trade], equity_curve: List[Tuple[pd.Timestamp, float]]) -> Dict[str, float]:
    if not trades or not equity_curve:
        return {
            "turnover_raw": float("nan"),
            "turnover_annualized": float("nan"),
            "avg_notional_per_trade": float("nan"),
        }

    traded_notional = float(sum(max(0.0, tr.pos_size_usd) * 2.0 for tr in trades))
    eq_vals = [float(v) for _, v in equity_curve]
    avg_equity = float(np.mean(eq_vals)) if eq_vals else float("nan")

    times = [pd.to_datetime(t) for t, _ in equity_curve]
    span_days = max((max(times) - min(times)).total_seconds() / 86400.0, 1e-9)

    turnover_raw = traded_notional / max(avg_equity, 1e-12)
    turnover_annualized = turnover_raw * (365.0 / span_days)
    avg_notional_per_trade = traded_notional / max(len(trades), 1)

    return {
        "turnover_raw": float(turnover_raw),
        "turnover_annualized": float(turnover_annualized),
        "avg_notional_per_trade": float(avg_notional_per_trade),
    }


def backtest_symbol(file_path, *args):
    config = _resolve_config_from_args(args)

    try:
        df = pd.read_parquet(file_path)
        if df.empty:
            return None, None, None

        df.columns = [c.lower() for c in df.columns]
        if "timestamp" not in df.columns:
            return None, None, None

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)

        start_ts = pd.to_datetime(config.start_date) if config.start_date else df["timestamp"].min()
        end_ts = pd.to_datetime(config.end_date) if config.end_date else df["timestamp"].max()

        if config.start_date:
            padding_bars = 1000
            start_idx = df[df["timestamp"] >= start_ts].index
            if len(start_idx) > 0:
                crop_start = max(0, start_idx[0] - padding_bars)
                end_idx = df[df["timestamp"] <= end_ts].index
                crop_end = end_idx[-1] + padding_bars if len(end_idx) > 0 else len(df)
                df = df.iloc[crop_start:crop_end].reset_index(drop=True)

        symbol = _normalize_symbol(Path(file_path).stem)
        extractor_mode = _normalize_extractor_mode(config.extractor_mode)

        extractor = _build_extractor(config)
        extractor.extract(df, f"{symbol}USDT", include_future_labels=(extractor_mode == "strict"))

        setup_df = pd.DataFrame(extractor.dataset)
        if setup_df.empty:
            return None, df, None

        setup_df["timestamp"] = pd.to_datetime(setup_df["timestamp"]).dt.tz_localize(None)
        if "end_time" in setup_df.columns:
            setup_df["end_time"] = pd.to_datetime(setup_df["end_time"], errors="coerce").dt.tz_localize(None)
        else:
            setup_df["end_time"] = setup_df["timestamp"]
        setup_df = setup_df[(setup_df["timestamp"] >= start_ts) & (setup_df["timestamp"] <= end_ts)].copy()
        if setup_df.empty:
            return None, df, None

        if config.use_ml_filter:
            model, meta = _load_ml_filter_assets(config)
            before_filter = len(setup_df)
            filter_type = str(meta.get("filter_type", "classification")).lower()

            if filter_type == "regime":
                setup_df = _apply_regime_filter(setup_df, meta)
                print(f"  Regime filter {symbol}: {before_filter} -> {len(setup_df)}")
            else:
                feature_columns = meta.get("feature_columns", _default_ml_feature_columns())
                threshold = config.ml_threshold
                if threshold is None:
                    threshold = float(meta.get("optimal_threshold", 0.5))

                if model is None:
                    raise ValueError("Classification filter selected but model is not loaded")

                Xf = _prepare_ml_filter_frame(setup_df, feature_columns)
                probs = model.predict_proba(Xf)[:, 1]
                setup_df["ml_prob"] = probs
                setup_df = setup_df[setup_df["ml_prob"] >= float(threshold)].copy()
                print(
                    f"  ML filter {symbol}: {before_filter} -> {len(setup_df)} "
                    f"(threshold={float(threshold):.3f})"
                )

            if setup_df.empty:
                return None, df, None

        setup_for_selection = setup_df.copy()
        setup_for_selection["symbol"] = symbol

        ts_series = pd.to_datetime(df["timestamp"], errors="coerce").dt.tz_localize(None)
        ts_ns = ts_series.astype("int64").to_numpy()
        ts_idx_map = {int(v): i for i, v in enumerate(ts_ns)}
        max_hold = max(1, int(config.max_bars_hold))

        potential_signals = []
        for row_idx, row in setup_df.sort_values("timestamp").iterrows():
            side = int(row["side"])
            trade_type = "LONG" if side == 1 else "SHORT"

            setup_signal_ts = pd.to_datetime(row["timestamp"]).tz_localize(None)
            signal_ts = setup_signal_ts
            signal_end_time = pd.to_datetime(row.get("end_time", setup_signal_ts)).tz_localize(None)

            entry_p = float(row["entry_p"])
            sl_p = float(row["sl_p"])
            tp_p = float(row["tp_p"])

            future_lows = list(row.get("future_lows", []) or [])
            future_highs = list(row.get("future_highs", []) or [])
            future_closes = list(row.get("future_closes", []) or [])
            target_win = row.get("target_win", np.nan)

            if extractor_mode != "strict":
                signal_idx = None
                raw_signal_idx = row.get("signal_idx", np.nan)
                if pd.notna(raw_signal_idx):
                    signal_idx = int(raw_signal_idx)
                else:
                    signal_idx = ts_idx_map.get(int(pd.Timestamp(signal_ts).value))

                if signal_idx is None or signal_idx < 0 or signal_idx >= len(df):
                    continue

                fut_start = signal_idx + 1
                fut_stop = min(len(df), fut_start + max_hold)
                if fut_stop <= fut_start:
                    continue

                fut = df.iloc[fut_start:fut_stop]
                future_lows = pd.to_numeric(fut["low"], errors="coerce").dropna().astype(float).tolist()
                future_highs = pd.to_numeric(fut["high"], errors="coerce").dropna().astype(float).tolist()
                future_closes = pd.to_numeric(fut["close"], errors="coerce").dropna().astype(float).tolist()
                max_len = min(len(future_lows), len(future_highs), len(future_closes))
                future_lows = future_lows[:max_len]
                future_highs = future_highs[:max_len]
                future_closes = future_closes[:max_len]
                if max_len == 0:
                    continue

                fill_idx = extractor.get_fill_index(side, entry_p, future_lows, future_highs)
                if fill_idx is None:
                    continue

                future_lows = future_lows[fill_idx:]
                future_highs = future_highs[fill_idx:]
                future_closes = future_closes[fill_idx:]
                if len(future_lows) == 0:
                    continue

                if config.entry_pullback <= 0:
                    signal_ts = pd.to_datetime(ts_series.iloc[signal_idx])
                else:
                    entry_idx = min(len(ts_series) - 1, fut_start + int(fill_idx))
                    signal_ts = pd.to_datetime(ts_series.iloc[entry_idx])

                end_idx = min(len(ts_series) - 1, signal_idx + max_hold)
                signal_end_time = pd.to_datetime(ts_series.iloc[end_idx])
                target_win = _compute_target_win_from_future(side, entry_p, sl_p, tp_p, future_lows, future_highs)

                # Keep setup timestamp for selector parity with live scanner.
                setup_for_selection.loc[row_idx, "timestamp"] = setup_signal_ts
                setup_for_selection.loc[row_idx, "end_time"] = signal_end_time
                setup_for_selection.loc[row_idx, "target_win"] = target_win

            signal_id = f"{symbol}|{int(pd.Timestamp(setup_signal_ts).value)}|{int(row_idx)}"
            potential_signals.append({
                "signal_id": signal_id,
                "timestamp": signal_ts,
                "signal_timestamp": setup_signal_ts,
                "entry_timestamp": signal_ts,
                "end_time": signal_end_time,
                "symbol": symbol,
                "type": trade_type,
                "side": side,
                "prob": float(row.get("ml_prob", 1.0)),
                "prob_long": 1.0 if side == 1 else 0.0,
                "prob_short": 1.0 if side == -1 else 0.0,
                "entry_p": entry_p,
                "sl_p": sl_p,
                "tp_p": tp_p,
                "future_lows": future_lows,
                "future_highs": future_highs,
                "future_closes": future_closes,
                "atr_val": 0.0,
            })

            setup_for_selection.loc[row_idx, "target_win"] = target_win

            setup_for_selection.loc[row_idx, "signal_id"] = signal_id

        return potential_signals, df, setup_for_selection

    except Exception as exc:
        print(f"Error processing {Path(file_path).name}: {exc}")
        return None, None, None


def run_portfolio_simulation(all_signals, full_price_db, config: BacktestConfig):
    if not all_signals:
        return [], [], 0

    print(
        f"Processing {len(all_signals)} raw p3 setup signals... "
        f"(selection_mode={config.selection_mode}, symbol_lock={config.enforce_symbol_lock})"
    )

    if config.selection_mode == "research":
        signals_sorted = sorted(
            all_signals,
            key=lambda x: (
                pd.to_datetime(x.get("timestamp")),
                -float(x.get("prob", 1.0)),
                x.get("symbol", ""),
            ),
        )
    else:
        signals_sorted = sorted(
            all_signals,
            key=lambda x: (
                pd.to_datetime(x.get("timestamp")),
                -float(x.get("prob", 1.0)),
                x.get("symbol", ""),
            ),
        )

    pre_filter_count = len(signals_sorted)
    if config.threshold is not None:
        th = float(config.threshold)
        signals_sorted = [s for s in signals_sorted if float(s.get("prob", 1.0)) >= th]
        print(f"Selection threshold applied: prob >= {th:.4f} | {pre_filter_count} -> {len(signals_sorted)}")

    if int(config.max_signals_per_timestamp) > 0 and signals_sorted:
        k = int(config.max_signals_per_timestamp)
        trimmed: List[Dict] = []
        ts_bucket = None
        used = 0
        for sig in signals_sorted:
            ts_key = pd.to_datetime(sig.get("timestamp")).value
            if ts_bucket != ts_key:
                ts_bucket = ts_key
                used = 0
            if used >= k:
                continue
            trimmed.append(sig)
            used += 1
        print(f"Per-timestamp cap applied: top-{k} | {len(signals_sorted)} -> {len(trimmed)}")
        signals_sorted = trimmed

    if not signals_sorted:
        return [], [], 0

    balance = config.initial_capital
    closed_trades: List[Trade] = []
    equity_curve: List[Tuple[pd.Timestamp, float]] = []
    active_positions: List[Dict] = []

    for sig in signals_sorted:
        ts = pd.to_datetime(sig["timestamp"])

        active_positions = [p for p in active_positions if p["end_time"] > ts]
        if config.enforce_symbol_lock and any(p["symbol"] == sig["symbol"] for p in active_positions):
            continue
        if len(active_positions) >= config.max_open_trades:
            continue

        side = int(sig.get("side", 1 if sig.get("type") == "LONG" else -1))
        entry_p = float(sig["entry_p"])
        sl_p = float(sig["sl_p"])
        tp_p = float(sig["tp_p"])

        dist_to_sl = (entry_p - sl_p) / entry_p if side == 1 else (sl_p - entry_p) / entry_p
        if dist_to_sl <= max(float(config.min_stop_distance), 0.0):
            continue

        notional = (balance * config.risk_per_trade) / max(dist_to_sl, 1e-8)
        margin_req = notional / max(config.leverage, 1e-8)

        cap_margin = balance * 0.10
        if margin_req > cap_margin:
            margin_req = cap_margin
            notional = margin_req * config.leverage

        if notional < 10:
            continue

        result, exit_price, hit_idx = _simulate_trade_path(sig, config.leverage)
        fallback_end = sig.get("end_time", ts)
        exit_time = _estimate_exit_time(sig["symbol"], ts, hit_idx, fallback_end, full_price_db)

        entry_exec = entry_p * (1.0 + config.slippage) if side == 1 else entry_p * (1.0 - config.slippage)
        exit_exec = float(exit_price) * (1.0 - config.slippage) if side == 1 else float(exit_price) * (1.0 + config.slippage)
        raw_pnl = (exit_exec - entry_exec) / max(entry_exec, 1e-12) if side == 1 else (entry_exec - exit_exec) / max(entry_exec, 1e-12)
        fees = (notional * config.fee_rate) * 2.0
        pnl_usd = (notional * raw_pnl) - fees
        pnl_pct = (pnl_usd / notional) * 100.0
        balance += pnl_usd

        trade = Trade(
            symbol=sig["symbol"],
            signal_time=ts,
            type="LONG" if side == 1 else "SHORT",
            side=side,
            limit_price=entry_p,
            tp_price=tp_p,
            sl_price=sl_p,
            atr_val=float(sig.get("atr_val", 0.0)),
            entry_time=ts,
            exit_time=exit_time,
            entry_price=float(entry_exec),
            exit_price=float(exit_exec),
            state=TradeState.CLOSED,
            result=result,
            pnl_usd=float(pnl_usd),
            pnl_pct=float(pnl_pct),
            fees=float(fees),
            duration=int(hit_idx) + 1,
            pos_size_usd=float(notional),
        )
        closed_trades.append(trade)
        equity_curve.append((exit_time, balance))

        active_positions.append({
            "symbol": sig["symbol"],
            "end_time": exit_time,
        })

    if not equity_curve:
        first_ts = pd.to_datetime(signals_sorted[0]["timestamp"])
        equity_curve = [(first_ts, balance)]

    equity_curve.sort(key=lambda x: x[0])
    return closed_trades, equity_curve, 0


def run_backtest_with_config(config: BacktestConfig):
    config.extractor_mode = _normalize_extractor_mode(config.extractor_mode)

    if config.exchange == "bitget":
        symbols_dir = BASE_DIR / "bitget-data" / "symbols_v3"
    elif str(config.universe_mode).lower() == "research":
        symbols_dir = BASE_DIR / "data" / "ohlcv"
    elif bool(config.use_research_model_selection):
        # Keep historical behavior for selector artifacts/training that were built on research-universe files.
        symbols_dir = BASE_DIR / "data" / "ohlcv"
        print(
            "[WARN] universe_mode=sniper with --use-research-model-selection: "
            "using data/ohlcv for backward-compatible selector behavior."
        )
    else:
        symbols_dir = BASE_DIR / "data" / "processed" / "symbols_v3"

    all_files = sorted(symbols_dir.glob("*.parquet"))

    # Keep only one file per normalized symbol (prefer *_USDT or *USDT naming).
    deduped = {}
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

    all_files = sorted(deduped.values())
    if config.top_coins:
        coin_set = {c.upper() for c in config.top_coins}
        all_files = [
            f for f in all_files if f"{_normalize_symbol(f.stem).upper()}USDT" in coin_set
        ]
    if int(config.max_files) > 0:
        all_files = all_files[: int(config.max_files)]

    if not config.top_coins and len(all_files) >= 100 and not config.use_ml_filter:
        print(
            "[WARN] Broad universe without ML filter can degrade performance: "
            f"{len(all_files)} symbols selected. Consider --max-files 60 and/or --top-coins for liquid names."
        )

    mode_txt = "raw p3 setup + ML filter" if config.use_ml_filter else "raw p3 setup"
    print(
        f"Scanning {len(all_files)} symbols with {mode_txt} "
        f"(universe_mode={config.universe_mode}, selection_mode={config.selection_mode}, extractor_mode={config.extractor_mode})..."
    )

    potential_signals = []
    full_price_db = {}
    selection_rows: List[pd.DataFrame] = []
    latest_data_ts: Optional[pd.Timestamp] = None

    for i, file_path in enumerate(all_files):
        print(f"Progress: {i + 1}/{len(all_files)}... ({file_path.name})")
        sigs, ohlcv, setup_rows = backtest_symbol(file_path, config)
        if ohlcv is not None and not ohlcv.empty and "timestamp" in ohlcv.columns:
            file_latest_ts = pd.to_datetime(ohlcv["timestamp"], errors="coerce").max()
            if pd.notna(file_latest_ts):
                if latest_data_ts is None or file_latest_ts > latest_data_ts:
                    latest_data_ts = file_latest_ts
        if sigs:
            potential_signals.extend(sigs)
            full_price_db[_normalize_symbol(Path(file_path).stem)] = ohlcv
        if setup_rows is not None and not setup_rows.empty:
            selection_rows.append(setup_rows)

    if config.end_date and latest_data_ts is not None:
        requested_end = pd.to_datetime(config.end_date, errors="coerce")
        if pd.notna(requested_end):
            lag_hours = (requested_end - latest_data_ts).total_seconds() / 3600.0
            if lag_hours > 24:
                print(
                    "[WARN] Data appears stale for requested backtest window: "
                    f"latest_data_ts={latest_data_ts} < requested_end={requested_end} "
                    f"(gap={lag_hours:.1f}h)."
                )

    if config.use_research_model_selection:
        potential_signals = _apply_research_model_selection(
            potential_signals=potential_signals,
            selection_rows=selection_rows,
            config=config,
        )

    trades, event_equity_curve, _ = run_portfolio_simulation(potential_signals, full_price_db, config)
    if not trades:
        print("No trades executed.")
        return potential_signals, full_price_db, trades, event_equity_curve

    mtm_equity_curve: List[Tuple[pd.Timestamp, float]] = []
    if config.equity_mode in {"mtm", "both"}:
        mtm_equity_curve = _build_mtm_equity_curve(trades, full_price_db, config)

    selected_curve = event_equity_curve
    if config.equity_mode == "mtm" and mtm_equity_curve:
        selected_curve = mtm_equity_curve

    report_df = pd.DataFrame([vars(t) for t in trades]).sort_values("entry_time")

    print(f"\nPORTFOLIO RESULTS (Raw p3 setup, equity_mode={config.equity_mode})")
    final_cap = config.initial_capital + report_df["pnl_usd"].sum()
    print(
        f"Initial: ${config.initial_capital:.2f} | Final: ${final_cap:.2f} | "
        f"Return: {((final_cap / config.initial_capital) - 1) * 100:.2f}%"
    )
    print(f"Trades: {len(report_df)}")

    curves = {"event": event_equity_curve}
    if mtm_equity_curve:
        curves["mtm"] = mtm_equity_curve

    for curve_name, curve_data in curves.items():
        metrics = _compute_curve_metrics(curve_data)
        sharpe_ann = metrics["sharpe_annualized"]
        sharpe_na = metrics["sharpe_not_annualized"]
        sharpe_ann_txt = "n/a (<30 daily samples)" if not np.isfinite(sharpe_ann) else f"{sharpe_ann:.2f}"
        sharpe_na_txt = "n/a" if not np.isfinite(sharpe_na) else f"{sharpe_na:.4f}"
        print(
            f"[{curve_name}] MaxDrawdown: {metrics['max_drawdown_pct']:.2f}% | "
            f"Sharpe(ann): {sharpe_ann_txt} | Sharpe(raw): {sharpe_na_txt} | "
            f"Min Balance: ${metrics['min_balance']:.2f} | "
            f"samples={metrics['sample_size']} periods_per_year={metrics['periods_per_year']}"
        )

        out_suffix = f"_{config.output_tag}" if config.output_tag else ""
        eq_file = BASE_DIR / "ml" / f"backtest_equity_{curve_name}_sniper{out_suffix}.csv"
        pd.DataFrame(curve_data, columns=["timestamp", "equity"]).to_csv(eq_file, index=False)
        print(f"[{curve_name}] Equity curve saved: {eq_file}")

    turnover = _compute_turnover_metrics(trades, selected_curve)
    print(
        "Turnover(raw): "
        f"{turnover['turnover_raw']:.4f} | Turnover(annualized): {turnover['turnover_annualized']:.4f} | "
        f"Avg traded notional/trade: ${turnover['avg_notional_per_trade']:.2f}"
    )
    round_trip_bps = (config.fee_rate + config.slippage) * 2.0 * 10000.0
    print(
        "Assumptions: "
        f"fee={config.fee_rate * 10000.0:.2f} bps/side, slippage={config.slippage * 10000.0:.2f} bps/side, "
        f"round-trip all-in={round_trip_bps:.2f} bps"
    )
    print(
        "Selection controls: "
        f"threshold={config.threshold if config.threshold is not None else 'none'}, "
        f"max_signals_per_timestamp={int(config.max_signals_per_timestamp)}"
    )

    out_suffix = f"_{config.output_tag}" if config.output_tag else ""
    output_file = BASE_DIR / "ml" / f"backtest_results_quant_sniper{out_suffix}.csv"
    report_df.to_csv(output_file, index=False)
    print(f"Report saved: {output_file}")

    return potential_signals, full_price_db, trades, selected_curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2026-03-01")
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--capital", type=float, default=100.0)
    parser.add_argument("--risk", type=float, default=0.05)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-bars-hold", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=None, help="Minimum signal score/probability used for selection.")
    parser.add_argument(
        "--max-signals-per-timestamp",
        type=int,
        default=0,
        help="Keep only top-K signals per timestamp after score sorting (0 disables).",
    )
    parser.add_argument(
        "--equity-mode",
        type=str,
        default="event",
        choices=["event", "mtm", "both"],
        help="Event-driven realized equity, mark-to-market equity, or both.",
    )

    parser.add_argument("--tp-level", type=float, default=1.6)
    parser.add_argument("--entry-pullback", type=float, default=0.0)
    parser.add_argument("--min-rr", type=float, default=1.0)
    parser.add_argument("--rr-floor-to-tp", type=float, default=0.0, help="If >0, enforce TP reward >= rr_floor_to_tp * risk by shifting TP outward")
    parser.add_argument("--min-mid-candles", type=int, default=6)
    parser.add_argument("--min-price-pct", type=float, default=3.0)
    parser.add_argument(
        "--top-coins",
        type=str,
        default="",
        help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT. Empty means all symbols.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of symbol files scanned after filtering (0 = all).",
    )
    parser.add_argument("--use-ml-filter", action="store_true", help="Apply ML meta filter on top of raw p3 setups")
    parser.add_argument("--ml-model-path", type=str, default=None, help="Path to ML model .joblib")
    parser.add_argument("--ml-meta-path", type=str, default=None, help="Path to ML meta .json")
    parser.add_argument("--ml-threshold", type=float, default=None, help="Override ML decision threshold")
    parser.add_argument("--profile-path", type=str, default=None, help="Path to profile JSON (p3_edge_research experiment format)")
    parser.add_argument("--profile-name", type=str, default=None, help="Experiment name in profile JSON")
    parser.add_argument("--use-auto018-profile", action="store_true", help="Load auto_018_live profile defaults")
    parser.add_argument("--fee-bps-per-side", type=float, default=None, help="Override fee in bps per side")
    parser.add_argument("--slippage-bps-per-side", type=float, default=None, help="Override slippage in bps per side")
    parser.add_argument(
        "--universe-mode",
        type=str,
        default="sniper",
        choices=["sniper", "research"],
        help="sniper=data/processed/symbols_v3, research=data/ohlcv",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="sniper",
        choices=["sniper", "research"],
        help="Signal selection policy; research matches quant_metrics-style portfolio gating.",
    )
    parser.add_argument(
        "--extractor-mode",
        type=str,
        default="strict",
        choices=["strict", "causal", "live_compatible"],
        help="strict=research extraction with future labels; causal=live-style no-lookahead setup extraction.",
    )
    parser.add_argument("--min-stop-distance", type=float, default=0.0, help="Minimum distance to stop as fraction of entry.")
    parser.add_argument("--no-symbol-lock", action="store_true", help="Allow concurrent positions on same symbol.")
    parser.add_argument("--research-compatible", action="store_true", help="Shortcut for fair comparison against run_research.")
    parser.add_argument("--output-tag", type=str, default=None, help="Tag output filenames to avoid overwrite.")
    parser.add_argument("--use-research-model-selection", action="store_true", help="Train in-sniper model score and tune threshold on val split.")
    parser.add_argument("--selection-train-end", type=str, default="2025-01-01")
    parser.add_argument("--selection-val-end", type=str, default="2025-05-01")
    parser.add_argument("--selection-min-val-trades", type=int, default=25)
    parser.add_argument(
        "--selection-model-profile",
        type=str,
        default="baseline",
        choices=sorted(RESEARCH_MODEL_PROFILES.keys()),
    )
    parser.add_argument("--selector-artifact-path", type=str, default=None, help="Path to save/load selection model artifact.")
    parser.add_argument("--selector-train-only", action="store_true", help="Train selector and exit without running portfolio backtest.")
    parser.add_argument("--selector-force-retrain", action="store_true", help="Force retraining even if selector artifact already exists.")
    parser.add_argument(
        "--selection-debug-checks",
        dest="selection_debug_checks",
        action="store_true",
        help="Run split integrity and permutation diagnostics for leakage debugging.",
    )
    parser.add_argument(
        "--no-selection-debug-checks",
        dest="selection_debug_checks",
        action="store_false",
        help="Disable leakage debug checks (not recommended).",
    )
    parser.add_argument(
        "--selection-debug-shift-zscore",
        action="store_true",
        help="Shift zscore-like features by one extra bar (debug-only leak sensitivity test).",
    )
    parser.add_argument(
        "--selection-debug-permutation-runs",
        type=int,
        default=3,
        help="Number of label-permutation runs for debug checks.",
    )
    parser.add_argument(
        "--selection-debug-fail-on-suspect",
        dest="selection_debug_fail_on_suspect",
        action="store_true",
        help="Fail fast when leakage suspicion gate triggers.",
    )
    parser.add_argument(
        "--no-selection-debug-fail-on-suspect",
        dest="selection_debug_fail_on_suspect",
        action="store_false",
        help="Do not fail on leakage suspicion gate (debug output only).",
    )
    parser.add_argument(
        "--selection-debug-real-auc-suspect",
        type=float,
        default=0.70,
        help="Real validation AUC threshold used by leakage suspicion gate.",
    )
    parser.add_argument(
        "--selection-debug-perm-auc-suspect",
        type=float,
        default=0.58,
        help="Permutation validation AUC threshold used by leakage suspicion gate.",
    )
    parser.set_defaults(selection_debug_checks=True, selection_debug_fail_on_suspect=True)

    args = parser.parse_args()
    top_coins = [x.strip().upper() for x in args.top_coins.split(",") if x.strip()]

    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        leverage=args.leverage,
        exchange=args.exchange,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        max_open_trades=args.max_positions,
        max_bars_hold=args.max_bars_hold,
        threshold=args.threshold,
        max_signals_per_timestamp=max(0, int(args.max_signals_per_timestamp)),
        tp_level=args.tp_level,
        entry_pullback=args.entry_pullback,
        min_rr=args.min_rr,
        rr_floor_to_tp=args.rr_floor_to_tp,
        min_mid_candles=args.min_mid_candles,
        min_price_pct=args.min_price_pct,
        top_coins=top_coins,
        max_files=int(args.max_files) if args.max_files is not None else 0,
        use_ml_filter=args.use_ml_filter,
        ml_model_path=args.ml_model_path,
        ml_meta_path=args.ml_meta_path,
        ml_threshold=args.ml_threshold,
        equity_mode=args.equity_mode,
        universe_mode=args.universe_mode,
        selection_mode=args.selection_mode,
        extractor_mode=_normalize_extractor_mode(args.extractor_mode),
        enforce_symbol_lock=not bool(args.no_symbol_lock),
        min_stop_distance=float(args.min_stop_distance),
        output_tag=_sanitize_output_tag(args.output_tag),
        use_research_model_selection=bool(args.use_research_model_selection),
        selection_train_end=str(args.selection_train_end),
        selection_val_end=str(args.selection_val_end),
        selection_min_val_trades=int(args.selection_min_val_trades),
        selection_model_profile=str(args.selection_model_profile),
        selector_artifact_path=args.selector_artifact_path,
        selector_train_only=bool(args.selector_train_only),
        selector_force_retrain=bool(args.selector_force_retrain),
        selection_debug_checks=bool(args.selection_debug_checks),
        selection_debug_shift_zscore=bool(args.selection_debug_shift_zscore),
        selection_debug_permutation_runs=max(1, int(args.selection_debug_permutation_runs)),
        selection_debug_fail_on_suspect=bool(args.selection_debug_fail_on_suspect),
        selection_debug_real_auc_suspect=float(args.selection_debug_real_auc_suspect),
        selection_debug_perm_auc_suspect=float(args.selection_debug_perm_auc_suspect),
    )

    profile_path = Path(args.profile_path) if args.profile_path else None
    profile_name = args.profile_name

    if args.use_auto018_profile:
        if profile_path is None:
            profile_path = _default_auto018_profile_path()
        if profile_name is None:
            profile_name = "auto_018_live"
        if args.max_files is None:
            config.max_files = 60

    if args.research_compatible:
        config.universe_mode = "research"
        config.selection_mode = "research"
        config.enforce_symbol_lock = False
        if float(args.min_stop_distance) <= 0.0:
            config.min_stop_distance = 0.005

    if profile_path is not None:
        profile_info = _apply_profile_to_config(config, profile_path, profile_name)
        print(
            "Loaded profile "
            f"{profile_info['profile_name']} from {profile_info['profile_path']}"
        )
        if not config.output_tag:
            config.output_tag = _sanitize_output_tag(profile_info["profile_name"])

    if args.fee_bps_per_side is not None:
        config.fee_rate = float(args.fee_bps_per_side) / 10000.0
    if args.slippage_bps_per_side is not None:
        config.slippage = float(args.slippage_bps_per_side) / 10000.0

    run_backtest_with_config(config)
