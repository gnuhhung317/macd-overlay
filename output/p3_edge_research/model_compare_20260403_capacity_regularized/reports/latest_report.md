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
- model_profile: capacity_regularized
- model_params: {"n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_child_samples": 100, "min_gain_to_split": 0.01, "subsample": 0.7, "colsample_bytree": 0.7, "lambda_l1": 1.0, "lambda_l2": 1.0, "class_weight": "balanced"}

```text
           experiment split_mode  threshold      gate_source  gate_trades  gate_sharpe_annualized  gate_max_drawdown_pct           gate_fail_reasons  accepted_oos_gate  test_trades  test_net_return_pct  test_win_rate_pct  test_profit_factor  test_sharpe_annualized  test_max_drawdown_pct  test_turnover_raw  test_turnover_annualized  val_sharpe_annualized  val_net_return_pct  wf_oos_trades  wf_oos_sharpe_annualized  wf_oos_max_drawdown_pct  wf_oos_net_return_pct  wf_oos_turnover_raw  wf_oos_turnover_annualized  train_samples  val_samples  test_samples
pullback_conservative   calendar       0.60 walk_forward_oos         1280                2.593525             -10.093218                                           True          997            -6.954682          56.469408            0.967933               -0.394459             -22.000024         386.580241                463.325067               2.698434           15.231693           1280                  2.593525               -10.093218              49.697352           415.794197                  346.034312          61752        13617         40785
      tighter_quality   calendar       0.85 walk_forward_oos          653               -1.035958             -15.869863 oos_sharpe|oos_max_drawdown              False          330             6.248042          80.000000            1.196954                1.621723              -4.510880          53.706299                 64.509417              -0.060497           -0.135759            653                 -1.035958               -15.869863             -11.452665           132.722699                  110.026577          34034         7941         21813
  higher_quality_wave   calendar       0.85 walk_forward_oos         1023               -1.404132             -30.353635 oos_sharpe|oos_max_drawdown              False          400            -8.598075          76.500000            0.704590               -2.609907             -10.282462          70.800397                 84.983759              -0.263842           -0.377171           1023                 -1.404132               -30.353635             -19.819661           249.712793                  208.570182          43302        10074         28219
             baseline   calendar       0.45 walk_forward_oos         1233               -3.279483             -49.467875 oos_sharpe|oos_max_drawdown              False          904           -44.143468          53.650442            0.758520               -4.860651             -45.727226         339.053110                407.031006              -2.112617          -10.346667           1233                 -3.279483               -49.467875             -48.265804           436.067193                  367.055694          70609        15688         47367
      faster_rotation   calendar       0.40 walk_forward_oos         1891               -7.650700             -78.991293 oos_sharpe|oos_max_drawdown              False         1362           -68.210881          56.681351            0.631507               -6.913003             -69.709524         692.616671                830.457437              -4.954020          -24.278119           1891                 -7.650700               -78.991293             -78.664974           850.023789                  723.003048         102719        22356         71001
```