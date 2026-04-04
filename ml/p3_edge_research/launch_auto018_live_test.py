from __future__ import annotations

import argparse
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Launch periodic auto_018 live-test cycles")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--data-glob", default=r"data/ohlcv/*.parquet")
    p.add_argument("--config", default=r"ml/p3_edge_research/experiments/auto_018_live_test.json")
    p.add_argument("--output-dir", default=r"output/p3_edge_research/live_test_auto018")
    p.add_argument("--batches", type=int, default=168)
    p.add_argument("--sleep-seconds", type=float, default=3600.0)
    p.add_argument("--max-files", type=int, default=60)
    p.add_argument("--search-iters", type=int, default=1)
    p.add_argument("--wf-max-folds", type=int, default=4)
    p.add_argument("--embargo-bars", type=int, default=24)
    p.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    p.add_argument("--target-oos-sharpe", type=float, default=1.5)
    p.add_argument("--max-oos-drawdown-pct", type=float, default=20.0)
    p.add_argument("--min-oos-trades", type=int, default=60)
    p.add_argument("--min-val-trades", type=int, default=25)
    p.add_argument("--initial-capital", type=float, default=100.0)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--risk-per-trade", type=float, default=0.005)
    p.add_argument("--max-concurrent-positions", type=int, default=3)
    return p


def main() -> None:
    args = build_parser().parse_args()

    cmd = [
        args.python_exe,
        "ml/p3_edge_research/continuous_runner.py",
        "--data-glob",
        args.data_glob,
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
        "--batches",
        str(args.batches),
        "--append",
        "--search-iters",
        str(args.search_iters),
        "--sleep-seconds",
        str(args.sleep_seconds),
        "--max-files",
        str(args.max_files),
        "--wf-max-folds",
        str(args.wf_max_folds),
        "--embargo-bars",
        str(args.embargo_bars),
        "--round-trip-cost-bps",
        str(args.round_trip_cost_bps),
        "--target-oos-sharpe",
        str(args.target_oos_sharpe),
        "--max-oos-drawdown-pct",
        str(args.max_oos_drawdown_pct),
        "--min-oos-trades",
        str(args.min_oos_trades),
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
    ]

    print("Launching auto_018 live-test cycles:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()