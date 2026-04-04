import argparse
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and persist sniper selector artifact")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--profile-path", required=True)
    p.add_argument("--profile-name", required=True)
    p.add_argument("--selector-artifact-path", required=True)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--max-files", type=int, default=60)
    p.add_argument("--selection-train-end", default="2025-01-01")
    p.add_argument("--selection-val-end", default="2025-05-01")
    p.add_argument("--selection-min-val-trades", type=int, default=25)
    p.add_argument("--selection-model-profile", choices=["baseline", "capacity_regularized"], default="baseline")
    p.add_argument("--risk", type=float, default=0.005)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--force-retrain", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    cmd = [
        args.python_exe,
        "ml/backtest_sniper.py",
        "--research-compatible",
        "--use-research-model-selection",
        "--selector-train-only",
        "--profile-path",
        args.profile_path,
        "--profile-name",
        args.profile_name,
        "--selector-artifact-path",
        args.selector_artifact_path,
        "--start",
        args.start,
        "--end",
        args.end,
        "--selection-train-end",
        args.selection_train_end,
        "--selection-val-end",
        args.selection_val_end,
        "--selection-min-val-trades",
        str(args.selection_min_val_trades),
        "--selection-model-profile",
        args.selection_model_profile,
        "--risk",
        str(args.risk),
        "--leverage",
        str(args.leverage),
        "--max-positions",
        str(args.max_positions),
        "--max-files",
        str(args.max_files),
        "--output-tag",
        f"selector_train_{args.profile_name}",
    ]

    if args.force_retrain:
        cmd.append("--selector-force-retrain")

    print("Training selector artifact:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
