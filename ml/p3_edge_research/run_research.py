from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .quant_metrics import PERIODS_PER_YEAR_MAP, PortfolioAssumptions, evaluate_trades
except ImportError:
    from quant_metrics import PERIODS_PER_YEAR_MAP, PortfolioAssumptions, evaluate_trades


BASE_FEATURES = [
    "side",
    "z_trend_20_50",
    "z_price_to_ema200",
    "z_volatility_atr",
    "structure_size",
    "pullback_depth",
    "dist_to_sl_pct",
]

SETUP_INPUT_FEATURES = [
    "risk_reward_ratio",
    "tp_distance_pct",
    "sl_distance_pct",
    "entry_to_sl_over_structure",
    "side_z_trend",
    "side_z_price",
]

REGIME_SHIFT_FEATURES = [
    "vol_regime_shift_6",
    "trend_regime_shift_6",
    "price_regime_shift_6",
]

DERIVATIVES_FEATURES = [
    "drv_log_oi",
    "drv_oi_change_1h",
    "drv_oi_change_24h",
    "drv_top_ls_ratio",
    "drv_global_ls_ratio",
    "drv_top_vs_global_ls",
    "drv_oi_accel",
]

CROSS_ASSET_FEATURES = [
    "btc_oi_change_1h",
    "btc_top_vs_global_ls",
    "eth_oi_change_1h",
    "eth_top_vs_global_ls",
    "cross_oi_beta_btc",
    "cross_oi_beta_eth",
]

FEATURES = (
    BASE_FEATURES
    + SETUP_INPUT_FEATURES
    + REGIME_SHIFT_FEATURES
    + DERIVATIVES_FEATURES
    + CROSS_ASSET_FEATURES
)

DEFAULT_THRESHOLD_GRID = [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
CACHE_SCHEMA_VERSION = "p3_edge_research_dataset_cache_v2"

_DERIV_SYMBOL_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}

MODEL_PROFILES = {
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


def load_extractor_class():
    p3_path = Path(__file__).resolve().parents[1] / "p3.py"
    spec = importlib.util.spec_from_file_location("p3_module", p3_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load extractor from ml/p3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "RealDataQuantExtractor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3 edge research loop")
    parser.add_argument("--data-glob", default=r"data/ohlcv/*.parquet")
    parser.add_argument("--config", default=r"ml/p3_edge_research/experiments/baseline_grid.json")
    parser.add_argument("--output-dir", default=r"output/p3_edge_research")
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--val-end", default="2025-05-01")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--periods-per-year", type=int, default=0)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--min-val-trades", type=int, default=30)
    parser.add_argument("--search-iters", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--threshold-min", type=float, default=0.45)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--wf-max-folds", type=int, default=1)
    parser.add_argument("--embargo-bars", type=int, default=0)
    parser.add_argument("--round-trip-cost-bps", type=float, default=-1.0)

    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.02)
    parser.add_argument("--max-concurrent-positions", type=int, default=5)
    parser.add_argument("--fee-bps-per-side", type=float, default=5.0)
    parser.add_argument("--slippage-bps-per-side", type=float, default=5.0)
    parser.add_argument("--panic-extra-slippage-bps", type=float, default=10.0)

    parser.add_argument("--target-oos-sharpe", type=float, default=1.5)
    parser.add_argument("--max-oos-drawdown-pct", type=float, default=15.0)
    parser.add_argument("--min-oos-trades", type=int, default=60)
    parser.add_argument("--derivatives-dir", default=r"data/derivatives")
    parser.add_argument("--disable-derivatives-features", action="store_true")
    parser.add_argument("--dataset-cache-dir", default=r"output/p3_edge_research/_dataset_cache")
    parser.add_argument("--disable-dataset-cache", action="store_true")
    parser.add_argument("--refresh-dataset-cache", action="store_true")
    parser.add_argument(
        "--model-profile",
        choices=sorted(MODEL_PROFILES.keys()),
        default="baseline",
        help="Model hyperparameter profile for LightGBM.",
    )
    return parser


def _resolve_model_params(args) -> Dict:
    return dict(MODEL_PROFILES.get(str(args.model_profile), MODEL_PROFILES["baseline"]))


def _build_files_signature(files: List[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for f in files:
        p = Path(f)
        try:
            st = p.stat()
            out.append(
                {
                    "path": p.as_posix(),
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                }
            )
        except Exception:
            out.append({"path": p.as_posix(), "size": -1, "mtime_ns": -1})
    return out


def _build_dataset_cache_key(
    files: List[str],
    extractor_params: Dict,
    derivatives_dir: Path,
    use_derivatives: bool,
) -> str:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "extractor_params": extractor_params,
        "features": FEATURES,
        "use_derivatives": bool(use_derivatives),
        "derivatives_dir": str(derivatives_dir.resolve()),
        "files": _build_files_signature(files),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load_dataset_cache(cache_path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(cache_path)
    except Exception:
        return pd.DataFrame()


def _save_dataset_cache(cache_path: Path, data: pd.DataFrame) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(cache_path, index=False)


def _normalize_coin_to_symbol(coin_value: str) -> str:
    stem = Path(str(coin_value)).stem.upper()
    stem = stem.split("-")[0]
    if stem.endswith("_USDT"):
        stem = stem[: -len("_USDT")]
    stem = stem.replace("_", "")
    if not stem.endswith("USDT"):
        stem = f"{stem}USDT"
    return stem


def _load_symbol_derivatives(symbol: str, derivatives_dir: Path) -> pd.DataFrame:
    cache_key = (str(derivatives_dir.resolve()), symbol)
    if cache_key in _DERIV_SYMBOL_CACHE:
        return _DERIV_SYMBOL_CACHE[cache_key]

    path = derivatives_dir / f"{symbol}.parquet"
    if not path.exists():
        _DERIV_SYMBOL_CACHE[cache_key] = pd.DataFrame(columns=["timestamp", "symbol"])
        return _DERIV_SYMBOL_CACHE[cache_key]

    try:
        d = pd.read_parquet(path)
    except Exception:
        _DERIV_SYMBOL_CACHE[cache_key] = pd.DataFrame(columns=["timestamp", "symbol"])
        return _DERIV_SYMBOL_CACHE[cache_key]

    if d.empty or "timestamp" not in d.columns:
        _DERIV_SYMBOL_CACHE[cache_key] = pd.DataFrame(columns=["timestamp", "symbol"])
        return _DERIV_SYMBOL_CACHE[cache_key]

    d = d.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    d = d.drop_duplicates(subset=["timestamp"], keep="last")

    raw_cols = [
        "sum_open_interest",
        "top_ls_ratio",
        "global_ls_ratio",
        "oi_change_1h",
        "oi_change_24h",
    ]
    for col in raw_cols:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Strict causality: only use information available before signal timestamp.
    d[raw_cols] = d[raw_cols].shift(1)

    d["drv_log_oi"] = np.log1p(np.maximum(d["sum_open_interest"].fillna(0.0), 0.0))
    d["drv_oi_change_1h"] = d["oi_change_1h"].fillna(0.0)
    d["drv_oi_change_24h"] = d["oi_change_24h"].fillna(0.0)
    d["drv_top_ls_ratio"] = d["top_ls_ratio"].fillna(0.0)
    d["drv_global_ls_ratio"] = d["global_ls_ratio"].fillna(0.0)
    d["drv_top_vs_global_ls"] = d["drv_top_ls_ratio"] - d["drv_global_ls_ratio"]
    d["drv_oi_accel"] = d["drv_oi_change_1h"] - (d["drv_oi_change_24h"] / 24.0)

    out_cols = ["timestamp", "symbol"] + DERIVATIVES_FEATURES
    out = d.assign(symbol=symbol)[out_cols].copy()
    _DERIV_SYMBOL_CACHE[cache_key] = out
    return out


def _merge_derivatives_features(data: pd.DataFrame, derivatives_dir: Path, use_derivatives: bool) -> pd.DataFrame:
    out = data.copy()

    for col in DERIVATIVES_FEATURES + CROSS_ASSET_FEATURES:
        if col not in out.columns:
            out[col] = 0.0

    if not use_derivatives:
        return out

    if not derivatives_dir.exists():
        return out

    symbols = sorted(set(out.get("symbol", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()))
    frames = []
    for sym in symbols:
        d = _load_symbol_derivatives(sym, derivatives_dir)
        if not d.empty:
            frames.append(d)

    if frames:
        deriv_map = {
            sym: grp.drop(columns=["symbol"]).sort_values("timestamp").reset_index(drop=True)
            for sym, grp in pd.concat(frames, ignore_index=True).groupby("symbol", sort=False)
        }

        merged_chunks = []
        for sym, chunk in out.groupby("symbol", sort=False):
            c = chunk.sort_values("timestamp").copy()
            d = deriv_map.get(sym)
            if d is None or d.empty:
                for col in DERIVATIVES_FEATURES:
                    c[col] = 0.0
                merged_chunks.append(c)
                continue

            cm = pd.merge_asof(
                c,
                d,
                on="timestamp",
                direction="backward",
                allow_exact_matches=True,
            )
            merged_chunks.append(cm)

        out = pd.concat(merged_chunks, ignore_index=True)

    btc = _load_symbol_derivatives("BTCUSDT", derivatives_dir)
    eth = _load_symbol_derivatives("ETHUSDT", derivatives_dir)

    if not btc.empty:
        b = btc[["timestamp", "drv_oi_change_1h", "drv_top_vs_global_ls"]].rename(
            columns={
                "drv_oi_change_1h": "btc_oi_change_1h",
                "drv_top_vs_global_ls": "btc_top_vs_global_ls",
            }
        )
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            b.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

    if not eth.empty:
        e = eth[["timestamp", "drv_oi_change_1h", "drv_top_vs_global_ls"]].rename(
            columns={
                "drv_oi_change_1h": "eth_oi_change_1h",
                "drv_top_vs_global_ls": "eth_top_vs_global_ls",
            }
        )
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            e.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

    drv_oi = pd.to_numeric(
        out["drv_oi_change_1h"] if "drv_oi_change_1h" in out.columns else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0)
    btc_oi = pd.to_numeric(
        out["btc_oi_change_1h"] if "btc_oi_change_1h" in out.columns else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0)
    eth_oi = pd.to_numeric(
        out["eth_oi_change_1h"] if "eth_oi_change_1h" in out.columns else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0)

    out["cross_oi_beta_btc"] = drv_oi - btc_oi
    out["cross_oi_beta_eth"] = drv_oi - eth_oi

    for col in DERIVATIVES_FEATURES + CROSS_ASSET_FEATURES:
        base_series = out[col] if col in out.columns else pd.Series(0.0, index=out.index)
        out[col] = pd.to_numeric(base_series, errors="coerce").fillna(0.0)

    return out


def _add_setup_and_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    for c in ["entry_p", "sl_p", "tp_p", "structure_size", "side", "z_trend_20_50", "z_price_to_ema200", "z_volatility_atr"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    entry_abs = np.abs(out["entry_p"]) + 1e-12
    sl_dist_abs = np.abs(out["entry_p"] - out["sl_p"])
    tp_dist_abs = np.abs(out["tp_p"] - out["entry_p"])

    out["risk_reward_ratio"] = tp_dist_abs / (sl_dist_abs + 1e-12)
    out["tp_distance_pct"] = tp_dist_abs / entry_abs
    out["sl_distance_pct"] = sl_dist_abs / entry_abs
    out["entry_to_sl_over_structure"] = out["sl_distance_pct"] / (np.abs(out["structure_size"]) + 1e-12)
    out["side_z_trend"] = out["side"] * out["z_trend_20_50"]
    out["side_z_price"] = out["side"] * out["z_price_to_ema200"]

    if "symbol" not in out.columns:
        out["symbol"] = "UNKNOWN"

    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    grp = out.groupby("symbol", group_keys=False)
    out["vol_regime_shift_6"] = grp["z_volatility_atr"].shift(1) - grp["z_volatility_atr"].shift(7)
    out["trend_regime_shift_6"] = grp["z_trend_20_50"].shift(1) - grp["z_trend_20_50"].shift(7)
    out["price_regime_shift_6"] = grp["z_price_to_ema200"].shift(1) - grp["z_price_to_ema200"].shift(7)

    for c in SETUP_INPUT_FEATURES + REGIME_SHIFT_FEATURES:
        out[c] = pd.to_numeric(out.get(c, 0.0), errors="coerce").fillna(0.0)

    return out


def _prepare_model_features(data: pd.DataFrame, derivatives_dir: Path, use_derivatives: bool) -> pd.DataFrame:
    out = data.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    coin_series = out["coin"] if "coin" in out.columns else pd.Series(["UNKNOWN"] * len(out), index=out.index)
    out["symbol"] = coin_series.astype(str).apply(_normalize_coin_to_symbol)
    out = _add_setup_and_regime_features(out)
    out = _merge_derivatives_features(out, derivatives_dir=derivatives_dir, use_derivatives=use_derivatives)

    for col in FEATURES:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def _build_threshold_grid(exp: Dict, args) -> List[float]:
    raw = exp.get("threshold_grid", None)
    if isinstance(raw, list) and raw:
        return sorted(set(float(x) for x in raw))

    step = max(float(args.threshold_step), 0.001)
    th_min = float(args.threshold_min)
    th_max = float(args.threshold_max)
    if th_max < th_min:
        th_min, th_max = th_max, th_min

    grid = np.arange(th_min, th_max + step * 0.5, step, dtype=float)
    if len(grid) == 0:
        return list(DEFAULT_THRESHOLD_GRID)
    return [float(x) for x in grid]


def _perturb(value: float, rng: np.random.Generator, pct: float, minimum: float) -> float:
    scale = max(abs(value) * pct, minimum)
    return float(max(minimum, value + rng.normal(0.0, scale)))


def expand_experiments(experiments: List[Dict], search_iters: int, seed: int, args) -> List[Dict]:
    if search_iters <= 0 or search_iters <= len(experiments):
        return experiments if search_iters <= 0 else experiments[:search_iters]

    rng = np.random.default_rng(seed)
    expanded = [dict(exp) for exp in experiments]
    base = [dict(exp) for exp in experiments]

    while len(expanded) < search_iters:
        parent = dict(base[len(expanded) % len(base)])
        child = dict(parent)
        child["name"] = f"auto_{len(expanded) + 1:03d}"

        child["tp_level"] = round(_perturb(float(parent.get("tp_level", 1.6)), rng, 0.25, 0.8), 3)
        child["entry_pullback"] = round(_perturb(float(parent.get("entry_pullback", 0.0)), rng, 0.35, 0.0), 4)
        child["min_rr"] = round(_perturb(float(parent.get("min_rr", 0.7)), rng, 0.3, 0.1), 3)
        child["max_hold_bars"] = int(np.clip(round(_perturb(float(parent.get("max_hold_bars", 24)), rng, 0.4, 4.0)), 4, 240))
        child["min_mid_candles"] = int(np.clip(round(_perturb(float(parent.get("min_mid_candles", 6)), rng, 0.5, 2.0)), 2, 40))
        child["min_price_pct"] = round(_perturb(float(parent.get("min_price_pct", 3.0)), rng, 0.4, 0.5), 3)
        child["threshold_grid"] = _build_threshold_grid(child, args)
        expanded.append(child)

    return expanded


def load_experiments(config_path: Path) -> Tuple[Dict, List[Dict]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assumptions_block = payload.get("assumptions", {})
    experiments = payload.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in config")
    return assumptions_block, experiments


def extract_dataset(
    files: List[str],
    extractor_params: Dict,
    extractor_cls,
    derivatives_dir: Path,
    use_derivatives: bool,
    cache_dir: Path,
    use_cache: bool,
    refresh_cache: bool,
) -> pd.DataFrame:
    cache_path = None
    if use_cache:
        cache_key = _build_dataset_cache_key(
            files=files,
            extractor_params=extractor_params,
            derivatives_dir=derivatives_dir,
            use_derivatives=use_derivatives,
        )
        cache_path = cache_dir / f"dataset_{cache_key}.parquet"
        if (not refresh_cache) and cache_path.exists():
            cached = _load_dataset_cache(cache_path)
            if not cached.empty:
                print(f"[CACHE HIT] {cache_path.name} rows={len(cached)}")
                return cached

    extractor = extractor_cls(**extractor_params)

    for file_path in tqdm(files, desc="Extract"):
        try:
            raw = pd.read_parquet(file_path)
            raw.columns = [c.lower() for c in raw.columns]
            extractor.extract(raw, Path(file_path).name)
        except Exception:
            continue

    data = pd.DataFrame(extractor.dataset)
    if data.empty:
        return data

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["end_time"] = pd.to_datetime(data["end_time"])
    data = data.sort_values("timestamp").reset_index(drop=True)
    data = _prepare_model_features(data, derivatives_dir=derivatives_dir, use_derivatives=use_derivatives)

    if use_cache and cache_path is not None:
        try:
            _save_dataset_cache(cache_path, data)
            print(f"[CACHE SAVE] {cache_path.name} rows={len(data)}")
        except Exception:
            pass

    return data


def time_split(df: pd.DataFrame, train_end: str, val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    train = df[df["timestamp"] < train_end].copy()
    val = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)].copy()
    test = df[df["timestamp"] >= val_end].copy()
    mode = "calendar"

    if min(len(train), len(val), len(test)) == 0:
        n = len(df)
        if n < 3:
            raise ValueError("Not enough samples for split")
        cut1 = max(1, int(n * 0.70))
        cut2 = max(cut1 + 1, int(n * 0.85))
        cut2 = min(cut2, n - 1)
        train = df.iloc[:cut1].copy()
        val = df.iloc[cut1:cut2].copy()
        test = df.iloc[cut2:].copy()
        mode = "time-quantile-fallback"

    return train, val, test, mode


def train_model(train: pd.DataFrame, val: pd.DataFrame, args) -> lgb.LGBMClassifier:
    model_params = _resolve_model_params(args)
    model = lgb.LGBMClassifier(**model_params, verbose=-1)
    model.fit(
        train[FEATURES],
        train["target_win"],
        eval_set=[(val[FEATURES], val["target_win"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def pick_threshold(
    val_df: pd.DataFrame,
    val_probs: np.ndarray,
    threshold_grid: List[float],
    assumptions: PortfolioAssumptions,
    periods_per_year: int,
    min_val_trades: int,
) -> Tuple[float, dict]:
    if len(val_df) == 0:
        return 0.82, {"trades": 0, "sharpe_annualized": np.nan, "net_return_pct": np.nan}

    threshold_grid = sorted(set(float(x) for x in threshold_grid))
    best_threshold = float(threshold_grid[0]) if threshold_grid else 0.82
    best_summary = None
    best_score = -np.inf

    for th in threshold_grid:
        mask = val_probs >= th
        selected = val_df.loc[mask].copy()
        if len(selected) < min_val_trades:
            continue

        summary, _ = evaluate_trades(selected, assumptions=assumptions, periods_per_year=periods_per_year)
        sharpe = summary.get("sharpe_annualized", np.nan)
        net_return = summary.get("net_return_pct", np.nan)

        score = float(sharpe) if np.isfinite(sharpe) else -999.0
        tie_break = float(net_return) if np.isfinite(net_return) else -999.0
        composite = score * 1000.0 + tie_break

        if composite > best_score:
            best_score = composite
            best_threshold = float(th)
            best_summary = summary

    if best_summary is None:
        fallback_min = max(1, min(min_val_trades, len(val_probs)))
        sorted_probs = np.sort(val_probs.astype(float))
        threshold_idx = max(0, len(sorted_probs) - fallback_min)
        best_threshold = float(sorted_probs[threshold_idx]) - 1e-12
        fallback_mask = val_probs >= best_threshold
        fallback_selected = val_df.loc[fallback_mask].copy()
        best_summary, _ = evaluate_trades(
            fallback_selected,
            assumptions=assumptions,
            periods_per_year=periods_per_year,
        )
        best_summary["fallback_selected_trades"] = int(len(fallback_selected))

    return best_threshold, best_summary


def walk_forward_splits(n: int, folds: int, embargo_bars: int) -> List[Tuple[slice, slice, slice]]:
    if folds <= 1 or n < 80:
        return []

    val_size = max(10, int(n * 0.12))
    test_size = max(10, int(n * 0.12))
    min_train = max(30, int(n * 0.4))
    max_train_anchor = n - (val_size + test_size + 2 * max(embargo_bars, 0))
    if max_train_anchor <= min_train:
        return []

    anchors = np.linspace(min_train, max_train_anchor, num=folds, dtype=int)
    uniq_anchors = sorted(set(int(x) for x in anchors))

    out: List[Tuple[slice, slice, slice]] = []
    for train_end in uniq_anchors:
        val_start = train_end + max(embargo_bars, 0)
        val_end = val_start + val_size
        test_start = val_end + max(embargo_bars, 0)
        test_end = test_start + test_size

        if test_end > n:
            continue
        if min(train_end, val_end - val_start, test_end - test_start) <= 0:
            continue

        out.append((slice(0, train_end), slice(val_start, val_end), slice(test_start, test_end)))

    return out


def evaluate_walk_forward(
    data: pd.DataFrame,
    threshold_grid: List[float],
    assumptions: PortfolioAssumptions,
    periods_per_year: int,
    args,
) -> Tuple[dict, List[Dict], pd.DataFrame]:
    splits = walk_forward_splits(len(data), folds=int(args.wf_max_folds), embargo_bars=int(args.embargo_bars))
    if not splits:
        return {
            "trades": 0,
            "net_return_pct": np.nan,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "sharpe_annualized": np.nan,
            "sample_size": 0,
            "periods_per_year": periods_per_year,
            "max_drawdown_pct": np.nan,
            "mdd_peak_time": None,
            "mdd_trough_time": None,
            "turnover_raw": np.nan,
            "turnover_annualized": np.nan,
            "avg_traded_notional": np.nan,
        }, [], pd.DataFrame()

    fold_rows: List[Dict] = []
    selected_test_chunks: List[pd.DataFrame] = []

    for fold_idx, (train_sl, val_sl, test_sl) in enumerate(splits, start=1):
        train = data.iloc[train_sl].copy()
        val = data.iloc[val_sl].copy()
        test = data.iloc[test_sl].copy()

        if min(len(train), len(val), len(test)) <= 2:
            continue
        if train["target_win"].nunique() < 2:
            continue

        model = train_model(train=train, val=val, args=args)
        val_probs = model.predict_proba(val[FEATURES])[:, 1]
        test_probs = model.predict_proba(test[FEATURES])[:, 1]

        threshold, _ = pick_threshold(
            val_df=val,
            val_probs=val_probs,
            threshold_grid=threshold_grid,
            assumptions=assumptions,
            periods_per_year=periods_per_year,
            min_val_trades=max(1, int(args.min_val_trades)),
        )

        fold_selected_test = test.loc[test_probs >= threshold].copy()
        fold_summary, _ = evaluate_trades(
            fold_selected_test,
            assumptions=assumptions,
            periods_per_year=periods_per_year,
        )

        fold_rows.append(
            {
                "fold": fold_idx,
                "train_samples": int(len(train)),
                "val_samples": int(len(val)),
                "test_samples": int(len(test)),
                "selected_test_trades": int(len(fold_selected_test)),
                "threshold": float(threshold),
                "oos_sharpe": fold_summary.get("sharpe_annualized", np.nan),
                "oos_mdd_pct": fold_summary.get("max_drawdown_pct", np.nan),
                "oos_net_return_pct": fold_summary.get("net_return_pct", np.nan),
            }
        )

        if not fold_selected_test.empty:
            fold_selected_test = fold_selected_test.copy()
            fold_selected_test["wf_fold"] = fold_idx
            selected_test_chunks.append(fold_selected_test)

    if not selected_test_chunks:
        return {
            "trades": 0,
            "net_return_pct": np.nan,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "sharpe_annualized": np.nan,
            "sample_size": 0,
            "periods_per_year": periods_per_year,
            "max_drawdown_pct": np.nan,
            "mdd_peak_time": None,
            "mdd_trough_time": None,
            "turnover_raw": np.nan,
            "turnover_annualized": np.nan,
            "avg_traded_notional": np.nan,
        }, fold_rows, pd.DataFrame()

    combined = pd.concat(selected_test_chunks, ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    wf_summary, wf_ledger = evaluate_trades(
        combined,
        assumptions=assumptions,
        periods_per_year=periods_per_year,
    )
    wf_summary["folds_executed"] = int(len(fold_rows))
    wf_summary["selected_trades_total"] = int(len(combined))
    return wf_summary, fold_rows, wf_ledger


def evaluate_split(
    split_df: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    assumptions: PortfolioAssumptions,
    periods_per_year: int,
) -> Tuple[dict, pd.DataFrame]:
    selected = split_df.loc[probs >= threshold].copy()
    summary, ledger = evaluate_trades(selected, assumptions=assumptions, periods_per_year=periods_per_year)
    summary["selected_trades"] = int(len(selected))
    return summary, ledger


def run_experiment(
    exp: Dict,
    files: List[str],
    extractor_cls,
    args,
    periods_per_year: int,
) -> Dict:
    extractor_params = {
        "tp_level": float(exp.get("tp_level", 1.6)),
        "max_hold_bars": int(exp.get("max_hold_bars", 24)),
        "min_mid_candles": int(exp.get("min_mid_candles", 6)),
        "min_price_pct": float(exp.get("min_price_pct", 3.0)),
        "entry_pullback": float(exp.get("entry_pullback", 0.0)),
        "min_rr": float(exp.get("min_rr", 0.5)),
    }

    data = extract_dataset(
        files=files,
        extractor_params=extractor_params,
        extractor_cls=extractor_cls,
        derivatives_dir=Path(args.derivatives_dir),
        use_derivatives=not bool(args.disable_derivatives_features),
        cache_dir=Path(args.dataset_cache_dir),
        use_cache=not bool(args.disable_dataset_cache),
        refresh_cache=bool(args.refresh_dataset_cache),
    )
    if data.empty:
        return {
            "experiment": exp.get("name", "unnamed"),
            "error": "no_data",
        }

    train, val, test, split_mode = time_split(data, train_end=args.train_end, val_end=args.val_end)
    if min(len(train), len(val), len(test)) == 0:
        return {
            "experiment": exp.get("name", "unnamed"),
            "error": "invalid_split",
        }

    assumptions = PortfolioAssumptions(
        initial_capital=args.initial_capital,
        leverage=args.leverage,
        risk_per_trade=args.risk_per_trade,
        max_concurrent_positions=args.max_concurrent_positions,
        fee_bps_per_side=args.fee_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
        panic_extra_slippage_bps=args.panic_extra_slippage_bps,
    )

    model = train_model(train=train, val=val, args=args)

    train_probs = model.predict_proba(train[FEATURES])[:, 1]
    val_probs = model.predict_proba(val[FEATURES])[:, 1]
    test_probs = model.predict_proba(test[FEATURES])[:, 1]

    threshold_grid = _build_threshold_grid(exp, args)
    best_th, val_tuned_summary = pick_threshold(
        val_df=val,
        val_probs=val_probs,
        threshold_grid=threshold_grid,
        assumptions=assumptions,
        periods_per_year=periods_per_year,
        min_val_trades=args.min_val_trades,
    )

    train_summary, _ = evaluate_split(train, train_probs, best_th, assumptions, periods_per_year)
    val_summary, _ = evaluate_split(val, val_probs, best_th, assumptions, periods_per_year)
    test_summary, test_ledger = evaluate_split(test, test_probs, best_th, assumptions, periods_per_year)

    wf_summary, wf_folds, wf_ledger = evaluate_walk_forward(
        data=data,
        threshold_grid=threshold_grid,
        assumptions=assumptions,
        periods_per_year=periods_per_year,
        args=args,
    )

    if int(args.wf_max_folds) > 1:
        gate_summary = wf_summary
        gate_source = "walk_forward_oos"
    else:
        gate_summary = test_summary
        gate_source = "single_split_test"

    gate_trades = int(gate_summary.get("trades", 0) or 0)
    gate_sharpe = float(gate_summary.get("sharpe_annualized", np.nan))
    gate_mdd = float(gate_summary.get("max_drawdown_pct", np.nan))

    trades_ok = gate_trades >= int(args.min_oos_trades)
    sharpe_ok = np.isfinite(gate_sharpe) and gate_sharpe >= float(args.target_oos_sharpe)
    mdd_ok = np.isfinite(gate_mdd) and abs(gate_mdd) <= float(args.max_oos_drawdown_pct)
    accepted = bool(trades_ok and sharpe_ok and mdd_ok)

    gate_fail_reasons: List[str] = []
    if not trades_ok:
        gate_fail_reasons.append("min_oos_trades")
    if not sharpe_ok:
        gate_fail_reasons.append("oos_sharpe")
    if not mdd_ok:
        gate_fail_reasons.append("oos_max_drawdown")

    return {
        "experiment": exp.get("name", "unnamed"),
        "split_mode": split_mode,
        "extractor_params": extractor_params,
        "threshold": best_th,
        "threshold_tune_val": val_tuned_summary,
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
        "walk_forward": wf_summary,
        "walk_forward_folds": wf_folds,
        "gate_source": gate_source,
        "gate_trades": gate_trades,
        "gate_sharpe_annualized": gate_sharpe,
        "gate_max_drawdown_pct": gate_mdd,
        "gate_fail_reasons": gate_fail_reasons,
        "accepted_oos_gate": bool(accepted),
        "counts": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
        "test_ledger": test_ledger,
        "walk_forward_oos_ledger": wf_ledger,
    }


def flatten_result(result: Dict) -> Dict:
    if "error" in result:
        return {
            "experiment": result.get("experiment", "unnamed"),
            "error": result["error"],
        }

    test = result["test"]
    val = result["val"]
    wf = result.get("walk_forward", {})

    return {
        "experiment": result["experiment"],
        "split_mode": result["split_mode"],
        "threshold": result["threshold"],
        "gate_source": result.get("gate_source", "single_split_test"),
        "gate_trades": result.get("gate_trades", np.nan),
        "gate_sharpe_annualized": result.get("gate_sharpe_annualized", np.nan),
        "gate_max_drawdown_pct": result.get("gate_max_drawdown_pct", np.nan),
        "gate_fail_reasons": "|".join(result.get("gate_fail_reasons", [])),
        "accepted_oos_gate": result["accepted_oos_gate"],
        "test_trades": test.get("trades", 0),
        "test_net_return_pct": test.get("net_return_pct", np.nan),
        "test_win_rate_pct": test.get("win_rate_pct", np.nan),
        "test_profit_factor": test.get("profit_factor", np.nan),
        "test_sharpe_annualized": test.get("sharpe_annualized", np.nan),
        "test_max_drawdown_pct": test.get("max_drawdown_pct", np.nan),
        "test_turnover_raw": test.get("turnover_raw", np.nan),
        "test_turnover_annualized": test.get("turnover_annualized", np.nan),
        "val_sharpe_annualized": val.get("sharpe_annualized", np.nan),
        "val_net_return_pct": val.get("net_return_pct", np.nan),
        "wf_oos_trades": wf.get("trades", np.nan),
        "wf_oos_sharpe_annualized": wf.get("sharpe_annualized", np.nan),
        "wf_oos_max_drawdown_pct": wf.get("max_drawdown_pct", np.nan),
        "wf_oos_net_return_pct": wf.get("net_return_pct", np.nan),
        "wf_oos_turnover_raw": wf.get("turnover_raw", np.nan),
        "wf_oos_turnover_annualized": wf.get("turnover_annualized", np.nan),
        "train_samples": result["counts"]["train"],
        "val_samples": result["counts"]["val"],
        "test_samples": result["counts"]["test"],
    }


def write_report(output_dir: Path, assumptions_block: Dict, args, periods_per_year: int, summary_df: pd.DataFrame) -> None:
    report_path = output_dir / "reports" / "latest_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# P3 Edge Research Report",
        "",
        "## Assumptions",
        f"- Timeframe and bar frequency: {assumptions_block.get('timeframe', args.timeframe)}",
        f"- Capital base and leverage model: start={args.initial_capital}, leverage={args.leverage}",
        f"- Fee model: {args.fee_bps_per_side} bps per side",
        f"- Slippage model: {args.slippage_bps_per_side} bps per side",
        f"- Round-trip all-in baseline (fee+slippage): {2.0 * (args.fee_bps_per_side + args.slippage_bps_per_side):.2f} bps",
        f"- Position sizing and rebalancing logic: risk_per_trade={args.risk_per_trade}, event-driven",
        "- Turnover definition used: sum(abs(entry_notional)+abs(exit_notional))/average_equity",
        "",
        "## Summary",
        f"- periods_per_year: {periods_per_year}",
        f"- experiments: {len(summary_df)}",
        f"- acceptance gate: sharpe >= {args.target_oos_sharpe}, |max_drawdown_pct| <= {args.max_oos_drawdown_pct}, trades >= {args.min_oos_trades}",
        f"- walk_forward_folds: {args.wf_max_folds}",
        f"- model_profile: {args.model_profile}",
        f"- model_params: {json.dumps(_resolve_model_params(args), ensure_ascii=True)}",
        f"- dataset_cache: {'disabled' if args.disable_dataset_cache else str(Path(args.dataset_cache_dir))}",
        "",
    ]

    if not summary_df.empty:
        ordered = summary_df.copy()
        if "accepted_oos_gate" in ordered.columns:
            ordered = ordered.sort_values("accepted_oos_gate", ascending=False)
        if "gate_sharpe_annualized" in ordered.columns:
            ordered = ordered.sort_values(
                ["accepted_oos_gate", "gate_sharpe_annualized", "gate_max_drawdown_pct", "test_net_return_pct"],
                ascending=[False, False, True, False],
                na_position="last",
            )
        try:
            lines.append(ordered.to_markdown(index=False))
        except Exception:
            lines.append("```text")
            lines.append(ordered.to_string(index=False))
            lines.append("```")
    else:
        lines.append("No valid experiment result.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.round_trip_cost_bps >= 0:
        per_component = max(float(args.round_trip_cost_bps) / 4.0, 0.0)
        args.fee_bps_per_side = per_component
        args.slippage_bps_per_side = per_component

    assumptions_block, experiments = load_experiments(Path(args.config))
    experiments = expand_experiments(
        experiments=experiments,
        search_iters=int(args.search_iters),
        seed=int(args.random_seed),
        args=args,
    )

    files = sorted(glob.glob(args.data_glob))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No data files found with pattern: {args.data_glob}")

    if args.periods_per_year > 0:
        periods_per_year = int(args.periods_per_year)
    else:
        periods_per_year = PERIODS_PER_YEAR_MAP.get(args.timeframe, 8760)

    extractor_cls = load_extractor_class()

    results = []
    flat_rows = []

    for exp in experiments:
        print(f"\n[RUN] {exp.get('name', 'unnamed')}")
        result = run_experiment(
            exp=exp,
            files=files,
            extractor_cls=extractor_cls,
            args=args,
            periods_per_year=periods_per_year,
        )
        results.append(result)
        flat_rows.append(flatten_result(result))

        if "test_ledger" in result and isinstance(result["test_ledger"], pd.DataFrame):
            ledger_path = output_dir / f"ledger_{result['experiment']}.csv"
            result["test_ledger"].to_csv(ledger_path, index=False)
        if "walk_forward_oos_ledger" in result and isinstance(result["walk_forward_oos_ledger"], pd.DataFrame):
            wf_ledger_path = output_dir / f"ledger_wf_oos_{result['experiment']}.csv"
            result["walk_forward_oos_ledger"].to_csv(wf_ledger_path, index=False)

    summary_df = pd.DataFrame(flat_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    serializable_results = []
    for item in results:
        out = dict(item)
        if isinstance(out.get("test_ledger"), pd.DataFrame):
            out["test_ledger"] = f"ledger_{out.get('experiment', 'unnamed')}.csv"
        if isinstance(out.get("walk_forward_oos_ledger"), pd.DataFrame):
            out["walk_forward_oos_ledger"] = f"ledger_wf_oos_{out.get('experiment', 'unnamed')}.csv"
        serializable_results.append(out)

    details_path = output_dir / "details.json"
    details_path.write_text(json.dumps(serializable_results, default=str, indent=2), encoding="utf-8")

    write_report(output_dir, assumptions_block, args, periods_per_year, summary_df)

    print("\n[DONE]")
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")


if __name__ == "__main__":
    main()
