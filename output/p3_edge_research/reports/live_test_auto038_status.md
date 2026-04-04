# Live Test Status - auto_038

## Current State
- Continuous live-test process is running in background.
- Terminal id: 5d2e9aa1-dd2f-42b4-92f6-5d8028d3bc69
- Output root: output/p3_edge_research/live_test_auto038
- Config: ml/p3_edge_research/experiments/auto_038_live_test.json

## Completed Batches Observed
- Batch 001: completed
- Batch 002: completed

## Latest Observed Metrics (auto_038_live)
- Threshold: 0.65
- OOS gate accepted: True
- Walk-forward trades: 870
- Walk-forward Sharpe annualized: 7.5394
- Walk-forward max drawdown: -3.5835%
- Walk-forward net return: 130.0777%
- Test trades: 675
- Test Sharpe annualized: 7.0924
- Test max drawdown: -10.9852%
- Test net return: 90.1310%

## Monitoring Notes
- continuous_runner updates batch_history.csv at the end of the full configured run.
- Per-batch outputs are available immediately under output/p3_edge_research/live_test_auto038/batch_XXX/.
- Use terminal output checks to monitor progress while the process is running.
