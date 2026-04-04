# P3 Edge Research Folder

This folder is a structured research lab for discovering robust edge from setup p3.

## Goals
- Run repeatable parameter experiments for p3 extraction logic.
- Evaluate performance on net returns after transaction costs.
- Report Sharpe, max drawdown, turnover, and assumptions in one place.
- Keep train/validation/test outputs clearly separated.

## Structure
- `run_research.py`: main experiment runner.
- `continuous_runner.py`: multi-batch overnight research loop.
- `quant_metrics.py`: cost-aware portfolio simulation and mandatory metrics.
- `experiments/baseline_grid.json`: parameter sets to test.
- `reports/latest_report.md`: generated report after each run.

## Metric Standard Applied
- Fee per side default: 5 bps.
- Slippage per side default: 5 bps.
- Round-trip all-in default baseline: 20 bps.
- Sharpe uses annualized formula on net returns:
  - sharpe = sqrt(periods_per_year) * mean(net_ret) / std(net_ret)
- `periods_per_year` for annualization is inferred from realized trade frequency.
- Sharpe annualization is reported only when sample size is sufficient (>= 30 trades).
- Max drawdown is computed from cumulative net equity.
- Turnover definition:
  - turnover = sum(abs(entry_notional) + abs(exit_notional)) / average_equity

## Quick Start
Run from workspace root:

python ml/p3_edge_research/run_research.py --data-glob "data/ohlcv/*.parquet" --config "ml/p3_edge_research/experiments/baseline_grid.json" --output-dir "output/p3_edge_research"

Example with broader search and walk-forward OOS:

python ml/p3_edge_research/run_research.py --data-glob "data/ohlcv/*.parquet" --config "ml/p3_edge_research/experiments/baseline_grid.json" --output-dir "output/p3_edge_research" --search-iters 30 --wf-max-folds 4 --embargo-bars 24 --round-trip-cost-bps 20 --min-val-trades 10

Example with strict OOS gate (minimum trade count included):

python ml/p3_edge_research/run_research.py --data-glob "data/ohlcv/*.parquet" --config "ml/p3_edge_research/experiments/baseline_grid.json" --output-dir "output/p3_edge_research" --search-iters 30 --wf-max-folds 4 --embargo-bars 24 --round-trip-cost-bps 20 --min-val-trades 10 --min-oos-trades 60 --target-oos-sharpe 1.5 --max-oos-drawdown-pct 15

Continuous overnight mode:

python ml/p3_edge_research/continuous_runner.py --data-glob "data/ohlcv/*.parquet" --config "ml/p3_edge_research/experiments/baseline_grid.json" --output-dir "output/p3_edge_research/continuous" --batches 12 --search-iters 40 --wf-max-folds 4 --embargo-bars 24 --round-trip-cost-bps 20

Auto_018 paper/live-test profile (periodic append cycles):

python ml/p3_edge_research/launch_auto018_live_test.py --batches 168 --sleep-seconds 3600

Purpose of this profile/launcher:

- It is a paper/live-test monitor for one strategy profile only (`auto_018_live`), not the full-grid overnight research.
- It runs periodic append cycles to detect performance drift before real capital deployment.
- Defaults are reduced-load (`--max-files 60`, `--wf-max-folds 4`) to avoid overheating local machines.

Auto_018 single smoke cycle:

python ml/p3_edge_research/continuous_runner.py --data-glob "data/ohlcv/*.parquet" --config "ml/p3_edge_research/experiments/auto_018_live_test.json" --output-dir "output/p3_edge_research/live_test_auto018" --batches 1 --search-iters 1 --wf-max-folds 6 --embargo-bars 24 --min-val-trades 25 --min-oos-trades 60 --round-trip-cost-bps 20 --target-oos-sharpe 1.5 --max-oos-drawdown-pct 20 --risk-per-trade 0.005 --max-concurrent-positions 3

Move to 24/7 server (Linux) with one command (prompts for SSH auth):

powershell -ExecutionPolicy Bypass -File ml/p3_edge_research/deploy_auto018_to_server.ps1 -Server "root@200.200.201.4" -RemoteDir "/root/macd-overlay"

## Main Outputs
- `output/p3_edge_research/summary.csv`
- `output/p3_edge_research/details.json`
- `output/p3_edge_research/ledger_<experiment>.csv`
- `output/p3_edge_research/ledger_wf_oos_<experiment>.csv`
- `output/p3_edge_research/reports/latest_report.md`

Continuous mode outputs:

- `output/p3_edge_research/continuous/batch_*/summary.csv`
- `output/p3_edge_research/continuous/batch_*/run.log`
- `output/p3_edge_research/continuous/batch_history.csv`
- `output/p3_edge_research/continuous/best_so_far.md`

## Notes
- Gate defaults for acceptance:
  - OOS Sharpe >= 1.5
  - OOS Max Drawdown <= 15%
  - OOS Trades >= 60
- If walk-forward is enabled (`--wf-max-folds > 1`), acceptance gate is checked on combined walk-forward OOS metrics (no fallback to single-split test).
- You can tighten or loosen gates via CLI arguments.
- You can add more experiments in `experiments/baseline_grid.json`, or use `--search-iters` for automatic parameter expansion.
