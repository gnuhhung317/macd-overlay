# Champion-Challenger Live Plan - 2026-04-03

## Context
Second-tier candidates auto_008 and auto_023 were valid accepted setups in full-market research and should be run as shadow challengers, not discarded.

## Immediate Batch-001 Forward Results
### auto_038_live (champion)
- OOS gate accepted: True
- WF Sharpe: 7.5394
- WF max drawdown: -3.5835%
- WF net return: 130.0777%
- Test Sharpe: 7.0924
- Test net return: 90.1310%

### auto_008_live (challenger)
- OOS gate accepted: True
- WF Sharpe: 2.5994
- WF max drawdown: -7.0398%
- WF net return: 27.1082%
- Test Sharpe: 2.1466
- Test net return: 17.9686%

### auto_023_live (challenger)
- OOS gate accepted: True
- WF Sharpe: 3.5602
- WF max drawdown: -6.7918%
- WF net return: 48.6112%
- Test Sharpe: 0.9460
- Test net return: 9.9009%

## Decision Now
- Keep auto_038 as production candidate.
- Keep auto_008 and auto_023 as shadow challengers in parallel forward validation.
- Do not drop second-tier setups yet; they are positive and accepted under the same cost assumptions.

## Running Processes
- auto_038 continuous runner is active.
- auto_008 shadow runner started for 24 hourly batches.
- auto_023 shadow runner started for 24 hourly batches.

## Promotion and Demotion Rules (for next 24-batch checkpoint)
Promote challenger if all are true:
1. Acceptance ratio >= 70% over completed batches.
2. Median WF Sharpe >= 2.0.
3. Worst WF max drawdown >= -12%.
4. Net return remains positive after costs.

Demote challenger if any are true:
1. Acceptance ratio < 40%.
2. Median WF Sharpe < 1.0.
3. Any sustained drawdown regime worse than -15% threshold in repeated batches.

## Artifact Paths
- output/p3_edge_research/live_test_auto038/
- output/p3_edge_research/live_test_auto008/
- output/p3_edge_research/live_test_auto023/
