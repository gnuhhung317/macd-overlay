# Live Test Report - auto_038 - Batch 001

## Run Metadata
- Config: ml/p3_edge_research/experiments/auto_038_live_test.json
- Output root: output/p3_edge_research/live_test_auto038
- Batch: 001
- Seed: 1001
- Timeframe: 1h
- Search iterations: 1
- Walk-forward folds: 4
- Embargo bars: 24

## Cost and Risk Assumptions
- Fee per side: 5 bps
- Slippage per side: 5 bps
- Round-trip all-in: 20 bps
- Risk per trade: 0.005
- Max concurrent positions: 3
- Gate: OOS Sharpe >= 1.5, max drawdown <= 15%, OOS trades >= 60

## Batch 001 Result
- Experiment: auto_038_live
- Threshold selected: 0.65
- Gate source: walk_forward_oos
- Accepted OOS gate: True

Walk-forward OOS:
- Trades: 870
- Sharpe annualized: 7.5394
- Max drawdown: -3.5835%
- Net return: 130.0777%
- Turnover raw: 429.6416
- Turnover annualized: 358.5463

Hold-out test split:
- Trades: 675
- Net return: 90.1310%
- Win rate: 74.2222%
- Profit factor: 1.6742
- Sharpe annualized: 7.0924
- Max drawdown: -10.9852%
- Turnover raw: 338.2940

## Decision
Continue forward validation with periodic live-test batches and monitor metric drift versus this baseline.

## Source Files
- output/p3_edge_research/live_test_auto038/batch_001/summary.csv
- output/p3_edge_research/live_test_auto038/batch_history.csv
- output/p3_edge_research/live_test_auto038/best_so_far.md
