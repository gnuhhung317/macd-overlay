# P3 Edge Research Report

## Assumptions
- Timeframe and bar frequency: 1h
- Capital base and leverage model: start=100.0, leverage=10.0
- Fee model: 5.0 bps per side
- Slippage model: 5.0 bps per side
- Round-trip all-in baseline (fee+slippage): 20.00 bps
- Position sizing and rebalancing logic: risk_per_trade=0.02, event-driven
- Turnover definition used: sum(abs(entry_notional)+abs(exit_notional))/average_equity

## Summary
- periods_per_year: 8760
- experiments: 5
- acceptance gate: sharpe >= 1.5, |max_drawdown_pct| <= 15.0, trades >= 60
- walk_forward_folds: 3

```text
           experiment split_mode  threshold      gate_source  gate_trades  gate_sharpe_annualized  gate_max_drawdown_pct                          gate_fail_reasons  accepted_oos_gate  test_trades  test_net_return_pct  test_win_rate_pct  test_profit_factor  test_sharpe_annualized  test_max_drawdown_pct  test_turnover_raw  test_turnover_annualized  val_sharpe_annualized  val_net_return_pct  wf_oos_trades  wf_oos_sharpe_annualized  wf_oos_max_drawdown_pct  wf_oos_net_return_pct  wf_oos_turnover_raw  wf_oos_turnover_annualized  train_samples  val_samples  test_samples
  higher_quality_wave   calendar       0.50 walk_forward_oos           69                2.177841              -7.499820                                                          True          129            18.794200          55.038760            1.166113                1.256350             -24.415965         135.326961                164.533544              -0.710341           -6.732781             69                  2.177841                -7.499820              24.452159            81.267168                  125.378724            210          136           386
pullback_conservative   calendar       0.50 walk_forward_oos          153               -0.748854             -32.379206                oos_sharpe|oos_max_drawdown              False          147            30.194039          52.380952            1.229258                1.670115             -26.761589         221.735075                282.161426              -1.253910          -11.925753            153                 -0.748854               -32.379206             -15.854495           249.069944                  362.373810            283          181           514
      tighter_quality   calendar       0.50 walk_forward_oos           47               -0.803166             -17.281845 min_oos_trades|oos_sharpe|oos_max_drawdown              False           86            -1.188601          44.186047            0.987140                0.124307             -24.367844          87.537879                115.382459               1.957433           15.557263             47                 -0.803166               -17.281845              -7.825663            53.537936                   87.140899            184          112           294
             baseline   calendar       0.45 walk_forward_oos          242               -2.151188             -44.889720                oos_sharpe|oos_max_drawdown              False          554           -66.711701          46.570397            0.801057               -2.412769             -71.356881         838.677096               1011.957488              -2.772098          -40.016348            242                 -2.151188               -44.889720             -33.725162           358.597028                  554.511910            297          199           601
      faster_rotation   calendar       0.40 walk_forward_oos          230               -3.373000             -44.543099                oos_sharpe|oos_max_drawdown              False          764           -86.357401          53.272251            0.683689               -5.052956             -87.374226        1228.992615               1476.409121              -4.353637          -47.053814            230                 -3.373000               -44.543099             -40.692385           390.274167                  645.422258            329          234           802
```