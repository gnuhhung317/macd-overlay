import argparse
import glob
import json
import os
from pathlib import Path

import importlib.util
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(r"d:\Code\Projects\self-projects\macd-overlay - Copy")
REGIME_LABELS = ["low", "mid", "high"]


def _load_extractor_class():
    p3_path = Path(__file__).with_name("p3.py")
    spec = importlib.util.spec_from_file_location("p3_train_module", p3_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import RealDataQuantExtractor from p3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "RealDataQuantExtractor")


def _simulate_trade_return(row, leverage, fee_rate, sl_panic_slippage=0.002):
    side = int(row["side"])
    entry_p = float(row["entry_p"])
    sl_p = float(row["sl_p"])
    tp_p = float(row["tp_p"])

    lows = list(row.get("future_lows", []))
    highs = list(row.get("future_highs", []))
    closes = list(row.get("future_closes", []))

    max_len = min(len(lows), len(highs))
    result = "TIMEOUT"
    exit_price = entry_p

    for i in range(max_len):
        low_t = float(lows[i])
        high_t = float(highs[i])

        if side == 1:
            if (low_t / entry_p - 1.0) * leverage <= -0.85:
                result = "LIQUIDATED"
                exit_price = entry_p * (1.0 - 0.85 / max(leverage, 1e-8))
                break
            if low_t <= sl_p:
                result = "LOSS"
                exit_price = sl_p * (1.0 - sl_panic_slippage)
                break
            if high_t >= tp_p:
                result = "WIN"
                exit_price = tp_p
                break
        else:
            if (high_t / entry_p - 1.0) * leverage >= 0.85:
                result = "LIQUIDATED"
                exit_price = entry_p * (1.0 + 0.85 / max(leverage, 1e-8))
                break
            if high_t >= sl_p:
                result = "LOSS"
                exit_price = sl_p * (1.0 + sl_panic_slippage)
                break
            if low_t <= tp_p:
                result = "WIN"
                exit_price = tp_p
                break
    else:
        if closes:
            exit_price = float(closes[-1])
        elif side == 1 and lows:
            exit_price = float(lows[-1])
        elif side == -1 and highs:
            exit_price = float(highs[-1])

    raw_ret = (exit_price - entry_p) / entry_p if side == 1 else (entry_p - exit_price) / entry_p
    net_ret = raw_ret - (2.0 * fee_rate)
    return result, float(raw_ret), float(net_ret)


def _add_training_targets(df, leverage, fee_rate):
    results = []
    raw_rets = []
    net_rets = []

    for _, row in df.iterrows():
        result, raw_ret, net_ret = _simulate_trade_return(row, leverage=leverage, fee_rate=fee_rate)
        results.append(result)
        raw_rets.append(raw_ret)
        net_rets.append(net_ret)

    out = df.copy()
    out["sim_result"] = results
    out["raw_ret"] = raw_rets
    out["net_ret"] = net_rets
    out["target_positive_net"] = (out["net_ret"] > 0).astype(int)
    return out


def _build_features(df):
    feat_df = df.copy()
    feat_df["side_z_trend"] = feat_df["side"] * feat_df["z_trend_20_50"]
    feat_df["side_z_price"] = feat_df["side"] * feat_df["z_price_to_ema200"]
    feat_df["side_momentum_3"] = feat_df["side"] * feat_df["momentum_3"]
    feat_df["side_wick_imbalance"] = feat_df["side"] * feat_df["wick_imbalance"]

    feature_columns = [
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

    for col in feature_columns:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    X = feat_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return feat_df, X, feature_columns


def _split_data(df, train_end, val_end):
    train = df[df["timestamp"] < train_end].copy()
    val = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)].copy()
    test = df[df["timestamp"] >= val_end].copy()
    mode = "calendar"

    if min(len(train), len(val), len(test)) == 0:
        n = len(df)
        cut1 = max(1, int(n * 0.70))
        cut2 = max(cut1 + 1, int(n * 0.85))
        cut2 = min(cut2, n - 1)
        train = df.iloc[:cut1].copy()
        val = df.iloc[cut1:cut2].copy()
        test = df.iloc[cut2:].copy()
        mode = "time-quantile-fallback"

    return train, val, test, mode


def _optimize_threshold(val_probs, val_net_ret, min_trades=30):
    best_th = 0.5
    best_ev = -np.inf
    best_count = 0

    for th in np.arange(0.50, 0.96, 0.01):
        m = val_probs >= th
        cnt = int(m.sum())
        if cnt < min_trades:
            continue
        ev = float(val_net_ret[m].mean())
        if ev > best_ev:
            best_ev = ev
            best_th = float(th)
            best_count = cnt

    if not np.isfinite(best_ev):
        return 0.5, float(val_net_ret.mean()), int(len(val_net_ret))

    return best_th, best_ev, best_count


def _evaluate_split(name, probs, net_ret, threshold):
    selected = probs >= threshold
    count = int(selected.sum())
    if count == 0:
        return {
            "split": name,
            "selected_trades": 0,
            "win_rate_pct": np.nan,
            "avg_net_ret_pct": np.nan,
            "expectancy_pct": np.nan,
            "total_net_ret_pct": np.nan,
        }

    sr = net_ret[selected]
    return {
        "split": name,
        "selected_trades": count,
        "win_rate_pct": float((sr > 0).mean() * 100),
        "avg_net_ret_pct": float(sr.mean() * 100),
        "expectancy_pct": float(sr.mean() * 100),
        "total_net_ret_pct": float(sr.sum() * 100),
    }


def _compute_tercile_edges(series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return [-np.inf, 0.0, 0.0 + 1e-9, np.inf]

    q1, q2 = s.quantile([1 / 3, 2 / 3]).values
    q1 = float(q1)
    q2 = float(q2)
    if not np.isfinite(q1):
        q1 = 0.0
    if not np.isfinite(q2):
        q2 = q1 + 1e-9
    if q2 <= q1:
        q2 = q1 + 1e-9

    return [-np.inf, q1, q2, np.inf]


def _assign_regime(series, edges):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.cut(s, bins=edges, labels=REGIME_LABELS, include_lowest=True)
    out = out.astype("object").where(out.notna(), "mid")
    return out.astype(str)


def _build_regime_columns(df, edges=None):
    out = df.copy()
    out["reaction_score"] = out["side"] * pd.to_numeric(out.get("momentum_1", 0.0), errors="coerce").fillna(0.0)

    if edges is None:
        edges = {
            "volatility": _compute_tercile_edges(out["z_volatility_atr"]),
            "volume": _compute_tercile_edges(out["volume_ratio_20"]),
            "reaction": _compute_tercile_edges(out["reaction_score"]),
        }

    out["volatility_regime"] = _assign_regime(out["z_volatility_atr"], edges["volatility"])
    out["volume_regime"] = _assign_regime(out["volume_ratio_20"], edges["volume"])
    out["reaction_regime"] = _assign_regime(out["reaction_score"], edges["reaction"])
    out["regime_key"] = (
        out["volatility_regime"] + "|" + out["volume_regime"] + "|" + out["reaction_regime"]
    )

    return out, edges


def _profit_factor_from_series(s):
    pos = s[s > 0].sum()
    neg = -s[s < 0].sum()
    if neg <= 0:
        return np.nan if pos <= 0 else np.inf
    return float(pos / neg)


def _build_regime_stats(df):
    grouped = df.groupby("regime_key", dropna=False)
    stats = grouped.agg(
        samples=("net_ret", "size"),
        win_rate_pct=("target_positive_net", lambda x: float(x.mean() * 100)),
        expectancy_pct=("net_ret", lambda x: float(x.mean() * 100)),
        total_net_ret_pct=("net_ret", lambda x: float(x.sum() * 100)),
        avg_raw_ret_pct=("raw_ret", lambda x: float(x.mean() * 100)),
    ).reset_index()
    stats["profit_factor"] = grouped["net_ret"].apply(_profit_factor_from_series).values
    return stats.sort_values(["expectancy_pct", "samples"], ascending=[False, False]).reset_index(drop=True)


def _merge_regime_stats(train_stats, val_stats):
    merged = train_stats.merge(
        val_stats[["regime_key", "samples", "expectancy_pct", "profit_factor"]],
        on="regime_key",
        how="left",
        suffixes=("_train", "_val"),
    )
    merged["samples_val"] = merged["samples_val"].fillna(0).astype(int)
    merged["expectancy_pct_val"] = merged["expectancy_pct_val"].fillna(np.nan)
    merged["profit_factor_val"] = merged["profit_factor_val"].fillna(np.nan)
    return merged


def _select_allowed_regimes(
    train_stats,
    val_stats,
    min_samples_train,
    min_samples_val,
    min_expectancy_pct,
    max_regimes,
):
    merged = _merge_regime_stats(train_stats, val_stats)

    filt = merged[
        (merged["samples_train"] >= min_samples_train)
        & (merged["samples_val"] >= min_samples_val)
        & (merged["expectancy_pct_train"] > min_expectancy_pct)
        & (merged["expectancy_pct_val"] > 0.0)
        & ((merged["profit_factor_train"].isna()) | (merged["profit_factor_train"] > 1.0))
        & ((merged["profit_factor_val"].isna()) | (merged["profit_factor_val"] > 1.0))
    ].copy()

    if filt.empty:
        return [], filt

    # Regime score emphasizes positive expectancy with enough support in both train and val.
    filt["regime_score"] = (
        filt["expectancy_pct_train"] * np.log1p(filt["samples_train"])
        + filt["expectancy_pct_val"] * np.log1p(filt["samples_val"])
    )
    filt = filt.sort_values(["regime_score", "expectancy_pct_val"], ascending=[False, False]).head(max_regimes)
    return filt["regime_key"].tolist(), filt


def _evaluate_regime_subset(name, df, allowed_regimes):
    m = df["regime_key"].isin(set(allowed_regimes))
    sr = df.loc[m, "net_ret"]
    if sr.empty:
        return {
            "split": name,
            "selected_trades": 0,
            "coverage_pct": 0.0,
            "win_rate_pct": np.nan,
            "expectancy_pct": np.nan,
            "total_net_ret_pct": np.nan,
        }

    return {
        "split": name,
        "selected_trades": int(sr.shape[0]),
        "coverage_pct": float(m.mean() * 100),
        "win_rate_pct": float((sr > 0).mean() * 100),
        "expectancy_pct": float(sr.mean() * 100),
        "total_net_ret_pct": float(sr.sum() * 100),
    }


def _run_classification_mode(args, full_df, feature_columns, X_all, train_df, val_df, test_df, split_mode):
    X_train = X_all.loc[train_df.index]
    X_val = X_all.loc[val_df.index]
    X_test = X_all.loc[test_df.index]

    y_train = train_df["target_positive_net"]
    y_val = val_df["target_positive_net"]

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=80,
        class_weight="balanced",
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    optimal_th, val_ev, val_count = _optimize_threshold(
        val_probs=val_probs,
        val_net_ret=val_df["net_ret"].values,
        min_trades=args.min_val_trades,
    )

    val_metrics = _evaluate_split("val", val_probs, val_df["net_ret"].values, optimal_th)
    test_metrics = _evaluate_split("test", test_probs, test_df["net_ret"].values, optimal_th)

    out_model_path = Path(args.out_model)
    out_meta_path = Path(args.out_meta)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_model_path)

    meta = {
        "filter_type": "classification",
        "model_type": "lgbm_classifier",
        "feature_columns": feature_columns,
        "optimal_threshold": float(optimal_th),
        "params": {
            "tp_level": args.tp_level,
            "entry_pullback": args.entry_pullback,
            "min_rr": args.min_rr,
            "max_hold_bars": args.max_hold_bars,
            "leverage": args.leverage,
            "fee_rate": args.fee_rate,
        },
        "split_mode": split_mode,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "validation": {
            "threshold_expectancy": float(val_ev * 100),
            "threshold_trades": int(val_count),
            **val_metrics,
        },
        "test": test_metrics,
    }

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "mode": "classification",
        "model_path": str(out_model_path),
        "meta_path": str(out_meta_path),
        "threshold": float(optimal_th),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def _run_regime_mode(args, train_df, val_df, test_df, split_mode):
    train_reg, edges = _build_regime_columns(train_df, edges=None)
    val_reg, _ = _build_regime_columns(val_df, edges=edges)
    test_reg, _ = _build_regime_columns(test_df, edges=edges)

    stats_train = _build_regime_stats(train_reg)
    stats_val = _build_regime_stats(val_reg)
    allowed_regimes, selected_stats = _select_allowed_regimes(
        train_stats=stats_train,
        val_stats=stats_val,
        min_samples_train=args.min_regime_samples,
        min_samples_val=args.min_regime_samples_val,
        min_expectancy_pct=args.min_regime_expectancy_pct,
        max_regimes=args.max_regimes,
    )

    val_metrics = _evaluate_regime_subset("val", val_reg, allowed_regimes)
    test_metrics = _evaluate_regime_subset("test", test_reg, allowed_regimes)

    score_map = {
        str(row["regime_key"]): float(row["expectancy_pct"]) / 100.0
        for _, row in stats_train.iterrows()
    }

    out_meta_path = Path(args.out_meta)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "filter_type": "regime",
        "model_type": None,
        "feature_columns": [],
        "optimal_threshold": None,
        "regime_definition": {
            "volatility_source": "z_volatility_atr",
            "volume_source": "volume_ratio_20",
            "reaction_source": "side*momentum_1",
            "labels": REGIME_LABELS,
            "edges": edges,
        },
        "allowed_regimes": allowed_regimes,
        "regime_score_map": score_map,
        "selection_rules": {
            "min_regime_samples": args.min_regime_samples,
            "min_regime_samples_val": args.min_regime_samples_val,
            "min_regime_expectancy_pct": args.min_regime_expectancy_pct,
            "max_regimes": args.max_regimes,
        },
        "split_mode": split_mode,
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "validation": val_metrics,
        "test": test_metrics,
    }

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.regime_stats_csv:
        out_csv_path = Path(args.regime_stats_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        stats_train.to_csv(out_csv_path, index=False)

    return {
        "mode": "regime",
        "meta_path": str(out_meta_path),
        "allowed_regimes": allowed_regimes,
        "selected_stats": selected_stats,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train p3 meta edge filter (regime-first)")
    parser.add_argument("--data-glob", default=r"data\ohlcv\*.parquet")
    parser.add_argument("--tp-level", type=float, default=1.6)
    parser.add_argument("--entry-pullback", type=float, default=0.0)
    parser.add_argument("--min-rr", type=float, default=1.0)
    parser.add_argument("--max-hold-bars", type=int, default=24)
    parser.add_argument("--min-mid-candles", type=int, default=6)
    parser.add_argument("--min-price-pct", type=float, default=3.0)
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--val-end", default="2025-05-01")
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--fee-rate", type=float, default=0.0003)

    parser.add_argument("--selection-mode", choices=["regime", "classification"], default="regime")
    parser.add_argument("--min-val-trades", type=int, default=30)
    parser.add_argument("--min-regime-samples", type=int, default=40)
    parser.add_argument("--min-regime-samples-val", type=int, default=20)
    parser.add_argument("--min-regime-expectancy-pct", type=float, default=0.0)
    parser.add_argument("--max-regimes", type=int, default=12)
    parser.add_argument("--regime-stats-csv", default=str(BASE_DIR / "output" / "p3_regime_stats.csv"))

    parser.add_argument("--out-model", default=str(BASE_DIR / "ml" / "models" / "p3_meta_edge_model.joblib"))
    parser.add_argument("--out-meta", default=str(BASE_DIR / "ml" / "models" / "p3_meta_edge_meta.json"))
    args = parser.parse_args()

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise FileNotFoundError(f"No files found with pattern: {args.data_glob}")

    extractor_cls = _load_extractor_class()
    extractor = extractor_cls(
        tp_level=args.tp_level,
        max_hold_bars=args.max_hold_bars,
        min_mid_candles=args.min_mid_candles,
        min_price_pct=args.min_price_pct,
        entry_pullback=args.entry_pullback,
        min_rr=args.min_rr,
    )

    print(f"Extracting setups from {len(files)} files...")
    for f in tqdm(files):
        try:
            raw = pd.read_parquet(f)
            raw.columns = [c.lower() for c in raw.columns]
            coin_name = os.path.basename(f)
            extractor.extract(raw, coin_name)
        except Exception:
            continue

    full_df = pd.DataFrame(extractor.dataset)
    if full_df.empty:
        raise ValueError("No setup extracted for training")

    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"]).dt.tz_localize(None)
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    full_df = _add_training_targets(full_df, leverage=args.leverage, fee_rate=args.fee_rate)
    full_df, X_all, feature_columns = _build_features(full_df)

    train_df, val_df, test_df, split_mode = _split_data(full_df, args.train_end, args.val_end)
    print(f"Split mode: {split_mode}")
    print(f"Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Insufficient split sizes after fallback")

    if args.selection_mode == "classification":
        summary = _run_classification_mode(args, full_df, feature_columns, X_all, train_df, val_df, test_df, split_mode)
        print("\nTraining complete (classification mode)")
        print(f"Model saved: {summary['model_path']}")
        print(f"Meta saved : {summary['meta_path']}")
        print(f"Optimal threshold: {summary['threshold']:.3f}")
        print("\nValidation metrics:")
        for k, v in summary["val_metrics"].items():
            print(f"  {k}: {v}")
        print("\nTest metrics:")
        for k, v in summary["test_metrics"].items():
            print(f"  {k}: {v}")
        return

    summary = _run_regime_mode(args, train_df, val_df, test_df, split_mode)
    print("\nTraining complete (regime mode)")
    print(f"Meta saved : {summary['meta_path']}")
    print(f"Allowed regimes: {len(summary['allowed_regimes'])}")

    if not summary["selected_stats"].empty:
        print("\nTop selected regimes:")
        print(summary["selected_stats"].head(12).to_string(index=False))

    print("\nValidation metrics:")
    for k, v in summary["val_metrics"].items():
        print(f"  {k}: {v}")

    print("\nTest metrics:")
    for k, v in summary["test_metrics"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
