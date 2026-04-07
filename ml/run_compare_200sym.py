import sys
from pathlib import Path

# Ensure project base is importable
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

sys.argv = [
    "run_compare",
    "--start", "2026-03-05",
    "--end", "2026-04-07 23:59:59",
    "--timeframe", "1h",
    "--max-files", "200",
    "--universe-mode", "research",
    "--exchange", "binance",
    "--profile-path", "ml/p3_edge_research/experiments/auto_038_risk_probe_20260405.json",
    "--profile-name", "tp120_rr000_floor0",
    "--selector-artifact-path", "output/selector_artifacts/hrf_selector_20260405.joblib",
    "--output-prefix", "output/scanner_backtest_compare_200sym_20260305_20260407",
]

import ml.compare_scanner_backtest_window as cs

if __name__ == '__main__':
    cs.main()
