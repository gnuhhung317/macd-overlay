# Continuous P3 Edge Research Best Result

- batches completed: 1
- search iters per batch: 1
- target oos sharpe: 1.5
- max oos drawdown pct: 15.0
- min oos trades: 60
- score column: wf_oos_sharpe_annualized
- ranking scope: accepted_only

## Best Row
```text
                                           0
experiment                     auto_023_live
split_mode                          calendar
threshold                               0.65
gate_source                 walk_forward_oos
gate_trades                              783
gate_sharpe_annualized               3.56018
gate_max_drawdown_pct              -6.791807
gate_fail_reasons                        NaN
accepted_oos_gate                       True
test_trades                              669
test_net_return_pct                 9.900856
test_win_rate_pct                  67.414051
test_profit_factor                  1.070873
test_sharpe_annualized              0.946021
test_max_drawdown_pct             -10.609463
test_turnover_raw                 357.759894
test_turnover_annualized          428.665938
val_sharpe_annualized               2.317145
val_net_return_pct                  8.494428
wf_oos_trades                            783
wf_oos_sharpe_annualized             3.56018
wf_oos_max_drawdown_pct            -6.791807
wf_oos_net_return_pct              48.611178
wf_oos_turnover_raw               394.768191
wf_oos_turnover_annualized        334.122643
train_samples                           7534
val_samples                             1848
test_samples                            4827
batch                                      1
seed                                    1001
_score                               3.56018
_mdd_abs                            6.791807
_ret                               48.611178
_trades                                  783
```