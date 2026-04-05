# Deployment Note - 2026-04-05 (High Risk / High Return)

## 1) Muc tieu va boi canh
- Muc tieu: toi uu profile theo huong Sharpe cao + high risk high return.
- Gia dinh phi giao dich de sat thuc te:
- Fee: 0.04% moi side (4 bps).
- Slippage: 0.06% moi side (6 bps).
- Tong cost: 0.10% moi side, round-trip 0.20%.
- Yeu cau van hanh: uu tien limit order (maker) de giu phi thap.

## 2) Cap nhat logic/code trong ngay
- Da bo sung tham so rr_floor_to_tp va day xuyen qua pipeline research/backtest/scanner.
- Da sua logic min_rr doi xung cho ca long va short (tranh bias 1 chieu).
- Cac file lien quan:
- ml/p3.py
- ml/p3_edge_research/run_research.py
- ml/backtest_sniper.py
- sniper_bot/sniper_scanner.py

## 3) Protocol danh gia (split va walk-forward)
- Calendar split:
- Train: 2020-01-15 03:00 -> 2024-12-31 23:00
- Val: 2025-01-01 00:00 -> 2025-04-30 23:00
- Test: 2025-05-01 00:00 -> 2026-03-01 19:00
- WFO: 4 folds, embargo-bars = 24.
- Trong tung fold: train/val/test tach thoi gian, khong overlap timestamp.
- Giua cac fold: test windows co overlap nho theo row index:
- Fold1-Fold2: 15 rows
- Fold2-Fold3: 15 rows
- Fold3-Fold4: 14 rows
- Ghi chu: wf_oos co the > 5k trades la binh thuong vi la tong selected trades cua nhieu fold.

## 4) Ket qua thi nghiem theo batch

### 4.1 TP/RR A-B va rule
| Batch | Experiment | WFO Sharpe | WFO MDD % | WFO Net % | Gate |
|---|---|---:|---:|---:|---|
| ab_tp_20260404_all | auto_038_tp120_rr0668 | 8.958 | -5.639 | 426.09 | Pass |
| ab_tp_20260404_all | auto_038_baseline_1293_rr0668 | 8.228 | -5.557 | 407.91 | Pass |
| ab_tp_20260404_all | auto_038_tp120_rr1000 | 8.103 | -9.007 | 364.61 | Pass |
| ab_tp_rule_20260405 | auto_038_tp120_rr0668_floor1000 | 6.769 | -13.182 | 360.75 | Pass |
| ab_tp_rule_20260405 | auto_038_tp120_rr0668 | 6.156 | -16.845 | 365.17 | Fail (MDD) |
| ab_tp_rule_20260405 | auto_038_baseline_1293_rr0668 | 5.287 | -15.711 | 246.02 | Fail (MDD) |

### 4.2 Min RR isolation/curve
| Batch | Experiment | WFO Sharpe | WFO MDD % | WFO Net % | Gate |
|---|---|---:|---:|---:|---|
| ab_minrr_curve_20260405 | tp120_rr000_floor0 | 7.946 | -7.379 | 313.71 | Pass |
| ab_minrr_curve_20260405 | tp120_rr0668_floor0 | 6.156 | -16.845 | 365.17 | Fail (MDD) |
| ab_minrr_curve_20260405 | tp120_rr1000_floor0 | 5.589 | -10.178 | 333.96 | Pass |

### 4.3 Risk probe (5+5 bps cu)
| Batch | WFO Sharpe | WFO MDD % | WFO Net % | Gate |
|---|---:|---:|---:|---|
| risk_probe_r1 | 7.946 | -7.379 | 313.71 | Pass |
| risk_probe_r2 | 8.116 | -10.613 | 2031.26 | Pass |
| risk_probe_r3 | 8.194 | -16.701 | 7006.46 | Fail (MDD) |

### 4.4 Cost stress (10+10 bps)
| Batch | WFO Sharpe | WFO MDD % | WFO Net % | Gate |
|---|---:|---:|---:|---|
| risk_probe_r2_coststress | 1.006 | -19.921 | 38.34 | Fail (Sharpe + MDD) |
| risk_probe_r3_coststress | -0.188 | -45.710 | -20.70 | Fail (Sharpe + MDD) |

Nhan xet: edge rat nhay voi cost khi volume/risk tang manh.

### 4.5 X70 isolation
| Batch | WFO Sharpe | WFO MDD % | WFO Net % |
|---|---:|---:|---:|
| risk_probe_x70_a | 2.771 | -2.257 | 13.30 |
| risk_probe_x70_b | 2.771 | -3.002 | 18.06 |
| risk_probe_x70_c | 3.334 | -2.434 | 22.85 |

Nhan xet: tang max positions cho hieu qua Sharpe/Calmar tot hon viec chi tang risk per trade trong setup x70.

### 4.6 High-risk scan theo cost thuc te user (4+6 bps)
| Case | risk_per_trade | max_positions | leverage | threshold | WFO Sharpe | WFO MDD % | WFO Net % |
|---|---:|---:|---:|---:|---:|---:|---:|
| hr_a | 0.006 | 4 | 12 | 0.65 | 8.547 | -8.262 | 724.13 |
| hr_b | 0.0075 | 5 | 15 | 0.65 | 8.105 | -10.591 | 2023.96 |
| hr_c | 0.009 | 6 | 18 | 0.65 | 9.377 | -14.900 | 5202.36 |
| hr_d | 0.010 | 7 | 20 | 0.65 | 8.161 | -16.683 | 6883.55 |
| hr_e | 0.008 | 7 | 15 | 0.75 | 8.876 | -12.725 | 4698.52 |
| hr_f | 0.012 | 8 | 20 | 0.75 | 8.235 | -22.531 | 29168.88 |

## 5) Ket luan va lua chon profile de trien khai
- Neu uu tien Sharpe cao nhat: hr_c.
- Neu uu tien high risk high return hung han nhat: hr_f.
- Neu muon can bang giua do hung han va do gay drawdown: hr_d.

De xuat chot de vao live ngay mai:
- Option 1 (uu tien on dinh Sharpe): hr_c.
- Option 2 (uu tien return cao hon): hr_d.
- Option 3 (rat aggressive): hr_f.

## 6) Checklist trien khai ngay mai
1. Chon 1 profile giua hr_c/hr_d/hr_f theo muc drawdown chap nhan.
2. Dung maker-first cho entry (post-only neu san ho tro).
3. Ghi log day du phi va slippage thuc te theo trade (de doi chieu voi 4+6 bps assumption).
4. Chay pilot quy mo nho trong phien dau, sau do nang quy mo neu metric thuc te khop voi backtest.
5. Theo yeu cau hien tai: khong bat buoc kill-switch, nhung van theo doi drawdown/equity drift de tranh regime break.

## 7) Duong dan output chinh de doi chieu
- output/p3_edge_research/highrisk_scan_20260405_hr_a/summary.csv
- output/p3_edge_research/highrisk_scan_20260405_hr_b/summary.csv
- output/p3_edge_research/highrisk_scan_20260405_hr_c/summary.csv
- output/p3_edge_research/highrisk_scan_20260405_hr_d/summary.csv
- output/p3_edge_research/highrisk_scan_20260405_hr_e/summary.csv
- output/p3_edge_research/highrisk_scan_20260405_hr_f/summary.csv
- output/p3_edge_research/risk_probe_r2_coststress/summary.csv
- output/p3_edge_research/risk_probe_r3_coststress/summary.csv
- output/p3_edge_research/ab_minrr_curve_20260405/summary.csv
- output/p3_edge_research/ab_tp_rule_20260405/summary.csv
