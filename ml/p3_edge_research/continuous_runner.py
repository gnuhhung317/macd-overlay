from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuous overnight P3 edge research runner")
    parser.add_argument("--data-glob", default=r"data/ohlcv/*.parquet")
    parser.add_argument("--config", default=r"ml/p3_edge_research/experiments/baseline_grid.json")
    parser.add_argument("--output-dir", default=r"output/p3_edge_research/continuous")
    parser.add_argument("--python-exe", default=sys.executable)

    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--append", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search-iters", type=int, default=40)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)

    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--val-end", default="2025-05-01")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--min-val-trades", type=int, default=10)

    parser.add_argument("--initial-capital", type=float, default=100.0)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.02)
    parser.add_argument("--max-concurrent-positions", type=int, default=5)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)

    parser.add_argument("--threshold-min", type=float, default=0.45)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--wf-max-folds", type=int, default=4)
    parser.add_argument("--embargo-bars", type=int, default=24)

    parser.add_argument("--target-oos-sharpe", type=float, default=1.5)
    parser.add_argument("--max-oos-drawdown-pct", type=float, default=15.0)
    parser.add_argument("--min-oos-trades", type=int, default=60)
    return parser


def _run_batch(args, batch_idx: int, seed: int, batch_out: Path) -> subprocess.CompletedProcess:
    cmd: List[str] = [
        args.python_exe,
        "ml/p3_edge_research/run_research.py",
        "--data-glob",
        args.data_glob,
        "--config",
        args.config,
        "--output-dir",
        str(batch_out),
        "--train-end",
        args.train_end,
        "--val-end",
        args.val_end,
        "--timeframe",
        args.timeframe,
        "--min-val-trades",
        str(args.min_val_trades),
        "--initial-capital",
        str(args.initial_capital),
        "--leverage",
        str(args.leverage),
        "--risk-per-trade",
        str(args.risk_per_trade),
        "--max-concurrent-positions",
        str(args.max_concurrent_positions),
        "--round-trip-cost-bps",
        str(args.round_trip_cost_bps),
        "--threshold-min",
        str(args.threshold_min),
        "--threshold-max",
        str(args.threshold_max),
        "--threshold-step",
        str(args.threshold_step),
        "--wf-max-folds",
        str(args.wf_max_folds),
        "--embargo-bars",
        str(args.embargo_bars),
        "--target-oos-sharpe",
        str(args.target_oos_sharpe),
        "--max-oos-drawdown-pct",
        str(args.max_oos_drawdown_pct),
        "--min-oos-trades",
        str(args.min_oos_trades),
        "--search-iters",
        str(args.search_iters),
        "--random-seed",
        str(seed),
    ]

    if args.max_files > 0:
        cmd.extend(["--max-files", str(args.max_files)])

    print(f"\n[BATCH {batch_idx:03d}] seed={seed}")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _safe_float(value):
    try:
        out = float(value)
        if np.isnan(out):
            return float("-inf")
        return out
    except Exception:
        return float("-inf")


def _safe_mdd_abs(value):
    try:
        out = float(value)
        if np.isnan(out):
            return float("inf")
        return abs(out)
    except Exception:
        return float("inf")


def _safe_trades(value):
    try:
        out = int(float(value))
        return max(out, 0)
    except Exception:
        return 0


def _existing_batch_ids(root_out: Path) -> List[int]:
    ids: List[int] = []
    for p in root_out.glob("batch_*"):
        if not p.is_dir():
            continue
        suffix = p.name.replace("batch_", "")
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(set(ids))


def main() -> None:
    args = build_parser().parse_args()
    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    all_rows = []

    existing_ids = _existing_batch_ids(root_out)
    start_batch = (max(existing_ids) + 1) if (args.append and existing_ids) else 1

    if args.append and existing_ids:
        print(f"[APPEND] Found existing batches up to {max(existing_ids):03d}. New run starts at {start_batch:03d}.")

    for offset in range(int(args.batches)):
        batch_id = start_batch + offset
        seed = 1000 + batch_id
        batch_out = root_out / f"batch_{batch_id:03d}"
        batch_out.mkdir(parents=True, exist_ok=True)

        proc = _run_batch(args=args, batch_idx=batch_id, seed=seed, batch_out=batch_out)

        log_path = batch_out / "run.log"
        log_path.write_text(
            "STDOUT\n" + proc.stdout + "\n\nSTDERR\n" + proc.stderr,
            encoding="utf-8",
        )

        summary_path = batch_out / "summary.csv"
        if proc.returncode == 0 and summary_path.exists():
            df = pd.read_csv(summary_path)
            if not df.empty:
                df["batch"] = batch_id
                df["seed"] = seed
                all_rows.append(df)
            print(f"[BATCH {batch_id:03d}] done: {summary_path}")
        else:
            print(f"[BATCH {batch_id:03d}] failed, see: {log_path}")

        if offset < int(args.batches) - 1 and float(args.sleep_seconds) > 0:
            time.sleep(float(args.sleep_seconds))

    history_path = root_out / "batch_history.csv"
    if all_rows:
        new_history = pd.concat(all_rows, ignore_index=True)

        if args.append and history_path.exists():
            old_history = pd.read_csv(history_path)
            history = pd.concat([old_history, new_history], ignore_index=True)
            history = history.drop_duplicates(subset=["batch", "experiment"], keep="last")
            history = history.sort_values(["batch", "experiment"]).reset_index(drop=True)
        else:
            history = new_history

        history.to_csv(history_path, index=False)

        if "wf_oos_sharpe_annualized" in history.columns:
            score_col = "wf_oos_sharpe_annualized"
        else:
            score_col = "test_sharpe_annualized"

        if "accepted_oos_gate" in history.columns:
            accepted_mask = history["accepted_oos_gate"].fillna(False).astype(bool)
            candidate = history.loc[accepted_mask].copy()
            ranking_scope = "accepted_only" if not candidate.empty else "all_rows_fallback"
        else:
            candidate = history.copy()
            ranking_scope = "all_rows_no_gate_column"

        if "gate_max_drawdown_pct" in candidate.columns:
            mdd_col = "gate_max_drawdown_pct"
        elif "wf_oos_max_drawdown_pct" in candidate.columns:
            mdd_col = "wf_oos_max_drawdown_pct"
        else:
            mdd_col = "test_max_drawdown_pct"

        if "wf_oos_net_return_pct" in candidate.columns:
            ret_col = "wf_oos_net_return_pct"
        else:
            ret_col = "test_net_return_pct"

        if "gate_trades" in candidate.columns:
            trades_col = "gate_trades"
        elif "wf_oos_trades" in candidate.columns:
            trades_col = "wf_oos_trades"
        else:
            trades_col = "test_trades"

        candidate = candidate.copy()
        candidate["_score"] = candidate[score_col].apply(_safe_float)
        candidate["_mdd_abs"] = candidate[mdd_col].apply(_safe_mdd_abs)
        candidate["_ret"] = candidate[ret_col].apply(_safe_float)
        candidate["_trades"] = candidate[trades_col].apply(_safe_trades)

        candidate = candidate.sort_values(
            ["_score", "_mdd_abs", "_ret", "_trades"],
            ascending=[False, True, False, False],
            na_position="last",
        )
        best_row = candidate.iloc[0]

        best_path = root_out / "best_so_far.md"
        try:
            best_table = best_row.to_frame().to_markdown()
        except Exception:
            best_table = "```text\n" + best_row.to_frame().to_string() + "\n```"

        best_lines = [
            "# Continuous P3 Edge Research Best Result",
            "",
            f"- batches completed: {int(history['batch'].nunique()) if 'batch' in history.columns else int(args.batches)}",
            f"- search iters per batch: {int(args.search_iters)}",
            f"- target oos sharpe: {float(args.target_oos_sharpe)}",
            f"- max oos drawdown pct: {float(args.max_oos_drawdown_pct)}",
            f"- min oos trades: {int(args.min_oos_trades)}",
            f"- score column: {score_col}",
            f"- ranking scope: {ranking_scope}",
            "",
            "## Best Row",
            best_table,
        ]
        best_path.write_text("\n".join(best_lines), encoding="utf-8")

        print("\n[DONE] Continuous research completed.")
        print(f"History: {history_path}")
        print(f"Best: {best_path}")
    else:
        print("\n[DONE] No successful batch output to aggregate.")


if __name__ == "__main__":
    main()
