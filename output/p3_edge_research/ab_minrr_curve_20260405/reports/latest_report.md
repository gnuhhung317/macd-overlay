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
- experiments: 3
- acceptance gate: sharpe >= 1.5, |max_drawdown_pct| <= 15.0, trades >= 60
- walk_forward_folds: 4
- model_profile: baseline
- model_params: {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 31, "min_child_samples": 50, "min_gain_to_split": 0.0, "subsample": 0.7, "colsample_bytree": 0.7, "lambda_l1": 0.0, "lambda_l2": 0.0, "class_weight": "balanced"}
- dataset_cache: output\p3_edge_research\_dataset_cache

```text
         experiment split_mode  threshold      gate_source  gate_trades  gate_sharpe_annualized  gate_max_drawdown_pct gate_fail_reasons  accepted_oos_gate  test_trades  test_net_return_pct  test_win_rate_pct  test_profit_factor  test_sharpe_annualized  test_max_drawdown_pct  test_turnover_raw  test_turnover_annualized  val_sharpe_annualized  val_net_return_pct  wf_oos_trades  wf_oos_sharpe_annualized  wf_oos_max_drawdown_pct  wf_oos_net_return_pct  wf_oos_turnover_raw  wf_oos_turnover_annualized  train_samples  val_samples  test_samples
 tp120_rr000_floor0   calendar       0.65 walk_forward_oos         2449                7.946155              -7.378549                                 True         1725           147.171782          76.347826            1.484111                6.544266             -11.117308         805.445342                964.816244               8.075869           48.419428           2449                  7.946155                -7.378549             313.711967          1131.345937                  936.728772          61334        13197         39866
tp120_rr1000_floor0   calendar       0.75 walk_forward_oos         1684                5.589364             -10.177617                                 True         1024           180.805759          66.210938            1.420056                6.489524              -9.917335         804.584034                965.500841               5.305387           37.956250           1684                  5.589364               -10.177617             333.957430          1324.889333                 1113.715628          43525         9639         29087
tp120_rr0668_floor0   calendar       0.75 walk_forward_oos         2106                6.156231             -16.844541  oos_max_drawdown              False         1435           300.806633          71.846690            1.581665                9.068596              -5.410122         836.271547               1003.113618               6.623872           47.613966           2106                  6.156231               -16.844541             365.167929          1277.121323                 1071.299702          49177        10797         32691
```