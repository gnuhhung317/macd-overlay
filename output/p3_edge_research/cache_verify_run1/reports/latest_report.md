# P3 Edge Research Report

## Assumptions
- Timeframe and bar frequency: 1h
- Capital base and leverage model: start=100.0, leverage=10.0
- Fee model: 5.0 bps per side
- Slippage model: 5.0 bps per side
- Round-trip all-in baseline (fee+slippage): 20.00 bps
- Position sizing and rebalancing logic: risk_per_trade=0.005, event-driven
- Turnover definition used: sum(abs(entry_notional)+abs(exit_notional))/average_equity

## Summary
- periods_per_year: 8760
- experiments: 1
- acceptance gate: sharpe >= 0.5, |max_drawdown_pct| <= 25.0, trades >= 20
- walk_forward_folds: 2
- model_profile: capacity_regularized
- model_params: {"n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_child_samples": 100, "min_gain_to_split": 0.01, "subsample": 0.7, "colsample_bytree": 0.7, "lambda_l1": 1.0, "lambda_l2": 1.0, "class_weight": "balanced"}
- dataset_cache: output\p3_edge_research\_dataset_cache_test

```text
   experiment split_mode  threshold      gate_source  gate_trades  gate_sharpe_annualized  gate_max_drawdown_pct gate_fail_reasons  accepted_oos_gate  test_trades  test_net_return_pct  test_win_rate_pct  test_profit_factor  test_sharpe_annualized  test_max_drawdown_pct  test_turnover_raw  test_turnover_annualized  val_sharpe_annualized  val_net_return_pct  wf_oos_trades  wf_oos_sharpe_annualized  wf_oos_max_drawdown_pct  wf_oos_net_return_pct  wf_oos_turnover_raw  wf_oos_turnover_annualized  train_samples  val_samples  test_samples
auto_018_live   calendar        0.8 walk_forward_oos           86                1.870095              -2.082595                                 True          114            11.489798           82.45614            2.674303                5.577866              -1.181848          43.704049                 53.128985               1.944485            1.667195             86                  1.870095                -2.082595               3.414514            35.865627                    32.15463           1842          494          1187
```