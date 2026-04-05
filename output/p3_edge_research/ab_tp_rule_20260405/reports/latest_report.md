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
auto_038_tp120_rr0668_floor1000   calendar       0.75 walk_forward_oos         1781                6.769171             -13.182447                                 True         1049           118.200533          66.825548            1.385232                5.466474              -6.385605         756.704640                909.166458               8.973654           47.634529           1781                  6.769171               -13.182447             360.752847          1227.146685                 1017.107102          61334        13197         39866
          auto_038_tp120_rr0668   calendar       0.75 walk_forward_oos         2106                6.156231             -16.844541  oos_max_drawdown              False         1435           300.806633          71.846690            1.581665                9.068596              -5.410122         836.271547               1003.113618               6.623872           47.613966           2106                  6.156231               -16.844541             365.167929          1277.121323                 1071.299702          49177        10797         32691
  auto_038_baseline_1293_rr0668   calendar       0.75 walk_forward_oos         2051                5.286728             -15.711472  oos_max_drawdown              False         1416           119.991044          66.807910            1.287469                4.837366             -15.429250         761.650136                913.354578               6.404765           41.225892           2051                  5.286728               -15.711472             246.020151          1120.068709                  932.326291          52561        11395         34686
```