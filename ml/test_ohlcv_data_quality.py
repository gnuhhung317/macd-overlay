#!/usr/bin/env python3
"""Quick data-quality test for raw OHLCV parquet files in data/ohlcv.

Checks per file:
- Required schema: timestamp, open, high, low, close, volume
- Timestamp monotonicity and duplicate timestamps
- Price/volume sanity (non-positive prices, negative volume)
- OHLC consistency (high/low bounds)
- Recency (last bar staleness)

Exit code:
- 0 if no errors found
- 1 if any errors found
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class FileCheckResult:
    file_name: str
    rows: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _check_single_file(file_path: Path, stale_hours: int) -> FileCheckResult:
    result = FileCheckResult(file_name=file_path.name)

    try:
        df = pd.read_parquet(file_path, columns=REQUIRED_COLUMNS)
    except Exception as exc:
        result.errors.append(f"cannot_read_or_missing_columns: {exc}")
        return result

    result.rows = len(df)
    if result.rows == 0:
        result.errors.append("empty_file")
        return result

    # Normalize timestamp first to avoid mixed dtypes.
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception as exc:
        result.errors.append(f"timestamp_parse_failed: {exc}")
        return result

    if df["timestamp"].isna().any():
        result.errors.append(f"invalid_timestamp_rows={int(df['timestamp'].isna().sum())}")

    dup_count = int(df["timestamp"].duplicated().sum())
    if dup_count > 0:
        result.errors.append(f"duplicate_timestamps={dup_count}")

    if not df["timestamp"].is_monotonic_increasing:
        result.errors.append("timestamp_not_sorted")

    non_positive_prices = int((df[["open", "high", "low", "close"]] <= 0).sum().sum())
    if non_positive_prices > 0:
        result.errors.append(f"non_positive_prices={non_positive_prices}")

    negative_volume = int((df["volume"] < 0).sum())
    if negative_volume > 0:
        result.errors.append(f"negative_volume_rows={negative_volume}")

    bad_high = int((df["high"] < df[["open", "close", "low"]].max(axis=1)).sum())
    bad_low = int((df["low"] > df[["open", "close", "high"]].min(axis=1)).sum())
    if bad_high > 0:
        result.errors.append(f"high_bound_violations={bad_high}")
    if bad_low > 0:
        result.errors.append(f"low_bound_violations={bad_low}")

    # Gap warning (not strict error because listing starts and exchange halts can create gaps).
    if result.rows > 1:
        deltas = df["timestamp"].diff().dropna().dt.total_seconds().div(3600)
        large_gaps = int((deltas > 3).sum())
        if large_gaps > 0:
            result.warnings.append(f"gaps_gt_3h={large_gaps}")

    last_ts = df["timestamp"].iloc[-1]
    if pd.notna(last_ts):
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        age_h = (now_utc - last_ts).total_seconds() / 3600
        if age_h > stale_hours:
            result.warnings.append(f"stale_last_bar_hours={age_h:.1f}")

    return result


def run_quality_test(ohlcv_dir: Path, max_files: int, stale_hours: int) -> int:
    files = sorted(ohlcv_dir.glob("*.parquet"))
    if max_files > 0:
        files = files[:max_files]

    if not files:
        print(f"No parquet files found in: {ohlcv_dir}")
        return 1

    results: List[FileCheckResult] = []
    for file_path in files:
        results.append(_check_single_file(file_path, stale_hours=stale_hours))

    total_files = len(results)
    total_rows = sum(r.rows for r in results)
    files_with_errors = [r for r in results if r.errors]
    files_with_warnings = [r for r in results if r.warnings]

    print("\n" + "=" * 78)
    print("OHLCV Data Quality Report")
    print("=" * 78)
    print(f"Directory: {ohlcv_dir}")
    print(f"Files tested: {total_files}")
    print(f"Total rows: {total_rows}")
    print(f"Files with errors: {len(files_with_errors)}")
    print(f"Files with warnings: {len(files_with_warnings)}")

    if files_with_errors:
        print("\nTop files with errors:")
        for r in files_with_errors[:30]:
            print(f"- {r.file_name}: {'; '.join(r.errors)}")
        if len(files_with_errors) > 30:
            print(f"... and {len(files_with_errors) - 30} more")

    if files_with_warnings:
        print("\nTop files with warnings:")
        for r in files_with_warnings[:30]:
            print(f"- {r.file_name}: {'; '.join(r.warnings)}")
        if len(files_with_warnings) > 30:
            print(f"... and {len(files_with_warnings) - 30} more")

    print("\nSample clean files:")
    clean = [r for r in results if not r.errors and not r.warnings][:10]
    for r in clean:
        print(f"- {r.file_name}: rows={r.rows}")

    print("=" * 78)

    return 1 if files_with_errors else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Test raw OHLCV parquet integrity")
    parser.add_argument(
        "--ohlcv-dir",
        type=Path,
        default=Path("data") / "ohlcv",
        help="Path to OHLCV parquet directory",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of files to test (0 = all)",
    )
    parser.add_argument(
        "--stale-hours",
        type=int,
        default=12,
        help="Warn if last bar age exceeds this threshold",
    )
    args = parser.parse_args()

    exit_code = run_quality_test(
        ohlcv_dir=args.ohlcv_dir,
        max_files=args.max_files,
        stale_hours=args.stale_hours,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
