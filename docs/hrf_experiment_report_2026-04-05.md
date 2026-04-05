# HRF Experiment Report (2026-04-05)

## Work status
- Local research/backtest pipeline: OK.
- HRF selector artifact training + reuse: OK.
- GUI backtest with trained artifact: OK.
- Existing-model-only mode with unrelated model (`p3_meta_edge_model`) is not equivalent to HRF and can produce zero trades.

## Setup used for HRF
- Profile source: `ml/p3_edge_research/experiments/auto_038_risk_probe_20260405.json`
- Experiment name: `tp120_rr000_floor0`
- Risk: `risk_per_trade=0.012`, `max_positions=8`, `leverage=20`
- Baseline costs for high-risk scan: `fee/slippage = 4/6 bps per side`

## 1) Best high-risk candidate scan (WFO)

| Case | WFO Sharpe | WFO MDD % | WFO Net % | WFO Trades |
|---|---:|---:|---:|---:|
| hr_a | 8.547 | -8.262 | 724.13 | 3220 |
| hr_b | 8.105 | -10.591 | 2023.96 | 3955 |
| hr_c | 9.377 | -14.900 | 5202.36 | 4628 |
| hr_d | 8.161 | -16.683 | 6883.55 | 5211 |
| hr_e | 8.876 | -12.725 | 4698.52 | 5272 |
| hr_f | 8.235 | -22.531 | 29168.88 | 5795 |

Conclusion:
- Highest return with still high Sharpe: **hr_f**.
- Best Sharpe under this scan: **hr_c**.

## 2) OOS-only realism check (artifact fixed, no retrain)

Window: `2025-08-29 -> 2026-03-01`

| Cost scenario (bps/side) | Trades | Return % | Event MDD % | Event Sharpe ann |
|---|---:|---:|---:|---:|
| 4/6 | 6650 | 39965.06 | -30.46 | 10.30 |
| 5/7 | 6650 | 9287.43 | -39.85 | 8.04 |
| 6/8 | 6648 | 2108.46 | -49.15 | 5.70 |

Conclusion:
- Edge remains positive under cost stress, but degrades sharply as costs rise.
- This profile is highly execution-sensitive and drawdown expands quickly at higher cost.

## 3) Impact of min_rr and RR floor

### min_rr curve (WFO)

| Config | WFO Sharpe | WFO MDD % | WFO Net % | Gate |
|---|---:|---:|---:|---|
| tp120_rr000_floor0 | 7.946 | -7.379 | 313.71 | Pass |
| tp120_rr0668_floor0 | 6.156 | -16.845 | 365.17 | Fail |
| tp120_rr1000_floor0 | 5.589 | -10.178 | 333.96 | Pass |

Takeaway:
- Adding `min_rr=0.668` increased net return but worsened Sharpe and drawdown.
- `min_rr=0.0` gave better risk-adjusted profile in this setup.

### RR floor TP rule A/B (post-fix)

| Config | WFO Sharpe | WFO MDD % | WFO Net % | Gate |
|---|---:|---:|---:|---|
| auto_038_tp120_rr0668 | 6.156 | -16.845 | 365.17 | Fail |
| auto_038_tp120_rr0668_floor1000 | 6.769 | -13.182 | 360.75 | Pass |
| auto_038_baseline_1293_rr0668 | 5.287 | -15.711 | 246.02 | Fail |

Takeaway:
- `rr_floor_to_tp=1.0` improved drawdown materially with small net-return tradeoff.

## 4) Impact of max positions vs risk scaling

### Capacity/risk probe (WFO)

| Case | Params | WFO Sharpe | WFO MDD % | WFO Net % |
|---|---|---:|---:|---:|
| r1 | rpt=0.005, pos=3, lev=10 | 7.946 | -7.379 | 313.71 |
| r2 | rpt=0.0075, pos=5, lev=15 | 8.116 | -10.613 | 2031.26 |
| r3 | rpt=0.01, pos=7, lev=20 | 8.194 | -16.701 | 7006.46 |

### x70 isolation

| Case | Params | WFO Sharpe | WFO MDD % | WFO Net % |
|---|---|---:|---:|---:|
| x70_a | rpt=0.0015, pos=3 | 2.771 | -2.257 | 13.30 |
| x70_b | rpt=0.0020, pos=3 | 2.771 | -3.002 | 18.06 |
| x70_c | rpt=0.0015, pos=5 | 3.334 | -2.434 | 22.85 |

Takeaway:
- Increasing `max_positions` often improved Sharpe/Calmar-like efficiency better than only increasing `risk_per_trade`.

## 5) Final recommendation by objective

- Objective A: Max return, still high Sharpe
  - Use **hr_f**.
- Objective B: Better Sharpe with lower drawdown
  - Consider **hr_c** or **hr_e**.

## 6) Deployment note

- If cloud pipeline is pure `git pull`: must `commit + push` before deploy.
- Current ansible sniper deploy playbook also copies local folders/artifacts, so local changes can still be deployed when running playbook from local machine.

## 7) Key output files

- `output/p3_edge_research/highrisk_scan_20260405_hr_f/summary.csv`
- `ml/backtest_results_quant_sniper_hrf_gui_20260405.csv`
- `ml/backtest_results_quant_sniper_hrf_oos_only_20260405.csv`
- `ml/backtest_results_quant_sniper_hrf_oos_cost57_20260405.csv`
- `ml/backtest_results_quant_sniper_hrf_oos_cost68_20260405.csv`