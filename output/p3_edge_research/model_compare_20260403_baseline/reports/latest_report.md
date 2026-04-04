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
- experiments: 5
- acceptance gate: sharpe >= 1.5, |max_drawdown_pct| <= 15.0, trades >= 60
- walk_forward_folds: 6
- model_profile: baseline
- model_params: {"n_estimators": 300, "learning_rate": 0.01, "max_depth": 5, "num_leaves": 31, "min_child_samples": 50, "min_gain_to_split": 0.0, "subsample": 0.7, "colsample_bytree": 0.7, "lambda_l1": 0.0, "lambda_l2": 0.0, "class_weight": "balanced"}

```text
           experiment split_mode  threshold      gate_source  gate_trades  gate_sharpe_annualized  gate_max_drawdown_pct           gate_fail_reasons  accepted_oos_gate  test_trades  test_net_return_pct  test_win_rate_pct  test_profit_factor  test_sharpe_annualized  test_max_drawdown_pct  test_turnover_raw  test_turnover_annualized  val_sharpe_annualized  val_net_return_pct  wf_oos_trades  wf_oos_sharpe_annualized  wf_oos_max_drawdown_pct  wf_oos_net_return_pct  wf_oos_turnover_raw  wf_oos_turnover_annualized  train_samples  val_samples  test_samples
pullback_conservative   calendar       0.75 walk_forward_oos         1282                0.943277             -19.177816 oos_sharpe|oos_max_drawdown              False          814            35.072397          76.289926            1.335667                3.837386              -7.245531         237.026673                283.965215               1.694749            5.594569           1282                  0.943277               -19.177816              16.248189           429.691613                  357.634064          61752        13617         40785
      tighter_quality   calendar       0.80 walk_forward_oos          497                0.549661              -4.636878                  oos_sharpe              False          361            10.891969          80.332410            1.312835                2.601271              -2.652546          56.941292                 68.395135               1.249860            2.158156            497                  0.549661                -4.636878               3.643150            79.012457                   65.675028          34034         7941         21813
  higher_quality_wave   calendar       0.65 walk_forward_oos          834               -2.778076             -29.062771 oos_sharpe|oos_max_drawdown              False          775           -23.518226          62.193548            0.822627               -2.606995             -27.624343         176.299618                211.617519              -0.452725           -1.959636            834                 -2.778076               -29.062771             -27.605219           178.693395                  149.280387          43302        10074         28219
             baseline   calendar       0.45 walk_forward_oos         1248               -3.127275             -46.996251 oos_sharpe|oos_max_drawdown              False          904           -42.647669          53.982301            0.771870               -4.575653             -44.719881         340.800302                409.128498              -2.299265          -11.124644           1248                 -3.127275               -46.996251             -43.566810           428.486785                  360.363310          70609        15688         47367
      faster_rotation   calendar       0.40 walk_forward_oos         1905               -6.794797             -78.930517 oos_sharpe|oos_max_drawdown              False         1362           -65.519729          57.342144            0.655035               -6.426683             -67.145242         689.756226                827.027723              -3.345068          -17.324621           1905                 -6.794797               -78.930517             -77.284562           895.586063                  761.756861         102719        22356         71001
```