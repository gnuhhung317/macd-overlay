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
        auto_038_tp120_rr0668   calendar       0.65 walk_forward_oos         2279                8.958347              -5.638776                                 True         1677           340.223505          74.299344            1.623624                9.630756              -4.007829         884.695613               1060.617705               7.203334           57.905289           2279                  8.958347                -5.638776             426.089447          1132.083778                  940.272484          54698        11837         35926
auto_038_baseline_1293_rr0668   calendar       0.75 walk_forward_oos         2270                8.227996              -5.557140                                 True         1520           187.519541          77.368421            1.669151                7.754118              -4.578014         693.904257                832.001271              10.757157           64.459719           2270                  8.227996                -5.557140             407.910550          1073.794401                  890.508280          57128        12285         37376
        auto_038_tp120_rr1000   calendar       0.70 walk_forward_oos         2097                8.102716              -9.006793                                 True         1518           221.855260          77.009223            1.668626                8.240381              -6.030850         756.842434                908.459814               8.422925           56.124294           2097                  8.102716                -9.006793             364.605047          1013.387727                  841.208802          51869        11225         33920
```