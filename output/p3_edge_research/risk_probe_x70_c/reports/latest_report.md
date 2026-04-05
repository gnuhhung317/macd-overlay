# P3 Edge Research Report

## Assumptions
- Timeframe and bar frequency: 1h
- Capital base and leverage model: start=100.0, leverage=70.0
- Fee model: 5.0 bps per side
- Slippage model: 5.0 bps per side
- Round-trip all-in baseline (fee+slippage): 20.00 bps
- Position sizing and rebalancing logic: risk_per_trade=0.0015, event-driven
- Turnover definition used: sum(abs(entry_notional)+abs(exit_notional))/average_equity

## Summary
- periods_per_year: 8760
- experiments: 1
- acceptance gate: sharpe >= 1.5, |max_drawdown_pct| <= 15.0, trades >= 60
- walk_forward_folds: 4
- model_profile: baseline
- model_params: {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 31, "min_child_samples": 50, "min_gain_to_split": 0.0, "subsample": 0.7, "colsample_bytree": 0.7, "lambda_l1": 0.0, "lambda_l2": 0.0, "class_weight": "balanced"}
- dataset_cache: output\p3_edge_research\_dataset_cache

```text
        experiment split_mode  threshold      gate_source  gate_trades  gate_sharpe_annualized  gate_max_drawdown_pct gate_fail_reasons  accepted_oos_gate  test_trades  test_net_return_pct  test_win_rate_pct  test_profit_factor  test_sharpe_annualized  test_max_drawdown_pct  test_turnover_raw  test_turnover_annualized  val_sharpe_annualized  val_net_return_pct  wf_oos_trades  wf_oos_sharpe_annualized  wf_oos_max_drawdown_pct  wf_oos_net_return_pct  wf_oos_turnover_raw  wf_oos_turnover_annualized  train_samples  val_samples  test_samples
tp120_rr000_floor0   calendar       0.65 walk_forward_oos         3953                 3.33423              -2.433706                                 True         2809            17.261672          57.173371            1.164364                3.558918              -2.562814         393.547496                471.417485               4.481222            8.103317           3953                   3.33423                -2.433706              22.847522           545.767926                  451.883462          61334        13197         39866
```