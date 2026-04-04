import argparse
from pathlib import Path
import subprocess
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run sniper backtest using a pre-trained selector artifact")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--profile-path", required=True)
    p.add_argument("--profile-name", required=True)
    p.add_argument("--selector-artifact-path", required=True)
    p.add_argument("--start", default="2025-05-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--max-files", type=int, default=60)
    p.add_argument("--risk", type=float, default=0.005)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--equity-mode", choices=["event", "mtm", "both"], default="both")
    p.add_argument("--output-tag", default="")
    return p


def _resolve_input_path(raw_value: str, search_roots: list[Path], label: str) -> str:
    raw = Path(raw_value)
    if raw.exists():
        return str(raw)

    for root in search_roots:
        candidate = root / raw
        if candidate.exists():
            print(f"Resolved {label}: {raw_value} -> {candidate}")
            return str(candidate)

    raise FileNotFoundError(
        f"{label} not found: {raw_value}. "
        f"Provide full path or place file under one of: {', '.join(str(x) for x in search_roots)}"
    )


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent.parent

    output_tag = args.output_tag.strip() or f"{args.profile_name}_selector_run"
    profile_path = _resolve_input_path(
        args.profile_path,
        search_roots=[
            base_dir,
            base_dir / "ml" / "p3_edge_research" / "experiments",
        ],
        label="Profile",
    )
    selector_artifact_path = _resolve_input_path(
        args.selector_artifact_path,
        search_roots=[
            base_dir,
            base_dir / "output" / "selector_artifacts",
        ],
        label="Selector artifact",
    )

    cmd = [
        args.python_exe,
        "ml/backtest_sniper.py",
        "--research-compatible",
        "--use-research-model-selection",
        "--profile-path",
        profile_path,
        "--profile-name",
        args.profile_name,
        "--selector-artifact-path",
        selector_artifact_path,
        "--start",
        args.start,
        "--end",
        args.end,
        "--risk",
        str(args.risk),
        "--leverage",
        str(args.leverage),
        "--max-positions",
        str(args.max_positions),
        "--max-files",
        str(args.max_files),
        "--equity-mode",
        args.equity_mode,
        "--output-tag",
        output_tag,
    ]

    print("Running sniper with selector artifact:")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
