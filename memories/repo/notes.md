# Handoff 2026-04-03 (sleep handoff)

## Done
- Cleaned legacy rows from overnight history and rebuilt clean best summary.
- Clean overnight now keeps only gate-complete batches 24-30.
- Started clean rerun branch at `output/p3_edge_research/overnight_integrity_v1`.
- Batch 001 completed (0 accepted due drawdown gate failures), Batch 002 in progress.
- Prepared and tested deploy flow script for server move:
  - `ml/p3_edge_research/deploy_auto018_to_server.ps1`

## Current objective
- Move auto_018 live test workload to server for 24/7 run.

## Current blocker on server
- Remote install via `python3 -m pip install -r requirements.txt` failed due PEP668 (externally managed Python env).
- Need project venv on server before installing requirements.

## Next steps (tomorrow)
1. SSH and install venv package if missing:
   - `ssh root@200.200.201.4 "apt-get update; apt-get install -y python3-venv"`
2. Create and populate project venv:
   - `ssh root@200.200.201.4 "cd /root/macd-overlay; python3 -m venv .venv; ./.venv/bin/python -m pip install --upgrade pip; ./.venv/bin/python -m pip install -r requirements.txt"`
3. Start auto_018 launcher in background on server (using venv python).
4. Verify with process + logs:
   - `ssh root@200.200.201.4 "ps aux | grep -i auto018"`
   - `ssh root@200.200.201.4 "cd /root/macd-overlay; ls -la logs"`

## Notes
- Local machine was hot mainly because continuous rerun process is active.
- If needed, stop local rerun after server is stable and continue all heavy jobs on server.

## Update 2026-04-03 (late morning)
- Server path paused due HDD slowdown/issues on remote host.
- Switched auto_018 live-test back to local execution.
- Local launcher running via project venv and resumed in append mode at batch_002.
- Output path confirmed active: output/p3_edge_research/live_test_auto018/batch_002

## Update 2026-04-03 (backtest_sniper integration)
- Added auto_018 profile integration into ml/backtest_sniper.py via flags:
   - --use-auto018-profile (defaults to auto_018_live in auto_018_live_test.json)
   - --profile-path / --profile-name (generic profile loader)
- Added equity simulation/report modes:
   - --equity-mode event|mtm|both
- Added cost overrides and profile assumption import:
   - --fee-bps-per-side, --slippage-bps-per-side
- Smoke-tested command on BTCUSDT 2025 and confirmed outputs:
   - ml/backtest_equity_event_sniper.csv
   - ml/backtest_equity_mtm_sniper.csv

## Update 2026-04-03 (diagnostic + guard)
- Diagnosed severe drawdown case: all-symbol run (312 symbols) produced very high turnover and many losses from illiquid/meme tickers.
- Added `--max-files` to ml/backtest_sniper.py and auto-default `max_files=60` when using `--use-auto018-profile` without explicit max-files.
- Added runtime warning for broad universe without ML filter.
- Updated behavior: explicit `--max-files 0` now truly enables full-universe scan even with `--use-auto018-profile`.

## Update 2026-04-03 (full-market run01 review)
- Full-market standardized run completed at output/p3_edge_research/full_market_edge_20260403_run01.
- Experiments: 60; accepted_oos_gate: 5 (8.33%).
- Accepted configs: pullback_conservative, auto_008, auto_013, auto_023, auto_038.
- Best gate Sharpe: auto_038 (8.32), gate MDD: -5.87%, but very high turnover (wf_oos_turnover_raw ~900.8).
- auto_018 failed gate due oos_sharpe only (gate Sharpe 0.47, gate MDD -11.92%).

## Update 2026-04-03 (model A/B before next full batch)
- Added model profile switch in ml/p3_edge_research/run_research.py:
   - baseline
   - capacity_regularized (user proposal)
- Benchmark scope: baseline_grid (5 experiments), same full-universe settings, only model_profile changed.
- Results:
   - baseline accepted: 0/5
   - capacity_regularized accepted: 1/5 (pullback_conservative)
   - mean gate Sharpe: baseline -2.24 vs capacity_regularized -2.16
   - mean wf_oos net return: baseline -25.71% vs capacity_regularized -21.70%
   - trade-off: capacity_regularized has slightly worse median gate MDD and higher mean turnover.

## Update 2026-04-03 (extract-once reuse cache)
- Added persistent dataset cache in ml/p3_edge_research/run_research.py.
- New flags:
   - --dataset-cache-dir
   - --disable-dataset-cache
   - --refresh-dataset-cache
- Cache key includes extractor params, selected files signature, derivatives usage, and feature schema version.
- Verified behavior:
   - Run1 created cache ([CACHE SAVE]).
   - Run2 with same settings loaded cache ([CACHE HIT]) without re-extraction.

## Update 2026-04-03 (sniper selector train/run split)
- Added wrapper scripts:
   - ml/train_sniper_selector.py
   - ml/run_sniper_with_selector.py
- Intended workflow:
   1. Train-only selector and save artifact via --selector-train-only path.
   2. Run backtest loading pre-trained artifact (no retrain).
- Smoke test passed with auto_038_live profile on a small universe slice:
   - Artifact saved: output/selector_artifacts/auto_038_live_smoke.joblib
   - Run phase confirmed: loaded_artifact=True

## Update 2026-04-03 (leakage debug and split hardening)
- Found a real leakage risk source: selector split fallback (70/15/15 quantile) could silently trigger when train window was empty (e.g., start date >= selection_train_end).
- Hardening applied in ml/backtest_sniper.py:
   - Disabled fallback split for selector training path.
   - Fail-fast error when calendar split has empty train/val.
   - Artifact inference path now works without requiring local train/val split in the current run window.
- Added debug controls:
   - --selection-debug-checks
   - --selection-debug-shift-zscore
   - --selection-debug-permutation-runs
- Added diagnostics:
   - split integrity validation
   - real-vs-permuted validation AUC/logloss report
- Updated ml/train_sniper_selector.py to pass explicit --start/--end so train data exists before selection_train_end.

## Update 2026-04-03 (debug policy defaults)
- Selector leakage debug checks are now default-on in ml/backtest_sniper.py.
- Added opt-out flags:
   - --no-selection-debug-checks
   - --no-selection-debug-fail-on-suspect
- Added fail-fast leakage suspicion gate (default enabled):
   - Trigger condition: val_auc_real >= selection_debug_real_auc_suspect and val_auc_perm >= selection_debug_perm_auc_suspect
   - Defaults: real=0.70, perm=0.58
- Default permutation runs increased from 1 to 3.

## Update 2026-04-04 (sniper scan throughput)
- Implemented incremental candle history in `sniper_bot/sniper_scanner.py`:
   - First cycle does full fetch, then subsequent cycles fetch only recent window (`incremental_refresh_days`) and merge/dedup.
   - Cache key uses `symbol|timeframe` to avoid multi-timeframe collisions.
- Implemented batch prediction path for auto_038 scanner:
   - Collects latest setups across symbols and calls `predict_proba` once on the combined feature frame.
- Added strategy config knobs:
   - `selector_batch_predict`
   - `incremental_scan`
   - `incremental_refresh_days`
   - `scan_history_bars`
   - `progress_detail_log_path`
- Measured local smoke speedup on repeated scans:
   - 12 symbols: first pass ~77.5s, second pass ~5.4s (~14.3x faster).
   - 5 symbols: first pass ~37.2s, second pass ~2.0s (~19.1x faster).

## Update 2026-04-04 (auto_038 TP A/B)
- Ran `run_research` A/B under live-test-like assumptions (1h, 5/5 bps fee/slip, wf=4, embargo=24, risk_per_trade=0.005, max_concurrent=3).
- Config used: `ml/p3_edge_research/experiments/auto_038_tp_ab_20260404.json`.
- Summary output: `output/p3_edge_research/ab_tp_20260404_all/summary.csv`.
- Result snapshot:
   - baseline `tp_level=1.293,min_rr=0.668`: wf_oos_sharpe=8.228, wf_oos_mdd=-5.56%, wf_oos_net=407.91%.
   - `tp_level=1.2,min_rr=0.668`: wf_oos_sharpe=8.958, wf_oos_mdd=-5.64%, wf_oos_net=426.09%.
   - `tp_level=1.2,min_rr=1.0` (proxy RR floor): wf_oos_sharpe=8.103, wf_oos_mdd=-9.01%, wf_oos_net=364.61%.
- Important code note: `min_rr` filter is applied in short branch but missing in long branch in `ml/p3.py` (long around TP/risk/reward block; short has explicit min_rr check).

## Update 2026-04-05 (TP floor implementation + re-run)
- Implemented in `ml/p3.py`:
   - New extractor parameter `rr_floor_to_tp` (default 0.0).
   - TP adjustment rule: if `rr_floor_to_tp > 0`, enforce `reward >= rr_floor_to_tp * risk` by shifting TP outward.
   - Applied `min_rr` check symmetrically to both long and short branches.
- Threaded parameter through:
   - `ml/p3_edge_research/run_research.py` (extractor params)
   - `ml/backtest_sniper.py` (config/profile/CLI/extractor wiring)
   - `sniper_bot/sniper_scanner.py` (profile extractor params)
- New A/B config: `ml/p3_edge_research/experiments/auto_038_tp_rule_ab_20260405.json`.
- New output: `output/p3_edge_research/ab_tp_rule_20260405/summary.csv`.
- Result snapshot after symmetry fix:
   - baseline `tp=1.293,min_rr=0.668`: wf_oos_sharpe=5.287, wf_oos_mdd=-15.71% (fail gate)
   - fibo `tp=1.2,min_rr=0.668`: wf_oos_sharpe=6.156, wf_oos_mdd=-16.84% (fail gate)
   - exact rule `tp=1.2,min_rr=0.668,rr_floor_to_tp=1.0`: wf_oos_sharpe=6.769, wf_oos_mdd=-13.18% (pass gate)

## Update 2026-04-05 (high-risk scan + WFO integrity note)
- Ran high-risk grid with user-like costs (fee/slip = 4/6 bps per side):
   - `hr_c` (rpt=0.009,pos=6,lev=18): wf_sharpe=9.377, wf_mdd=-14.90%, wf_net=5202.36%.
   - `hr_d` (rpt=0.010,pos=7,lev=20): wf_sharpe=8.161, wf_mdd=-16.68%, wf_net=6883.55%.
   - `hr_f` (rpt=0.012,pos=8,lev=20): wf_sharpe=8.235, wf_mdd=-22.53%, wf_net=29168.88%.
- Cost sensitivity confirmed in stress tests (10/10 bps per side):
   - `risk_probe_r2_coststress`: wf_sharpe dropped to ~1.01, wf_mdd ~-19.92%.
   - `risk_probe_r3_coststress`: wf_sharpe < 0 and wf_mdd ~-45.71%.
- WFO split integrity detail:
   - Within each fold: no train/val/test timestamp overlap.
   - Across adjacent fold test windows: small overlap by row index (15, 15, 14 rows).
   - Interpretation: not classic future leakage in-fold, but aggregated WFO OOS is not strictly disjoint across folds.

## Update 2026-04-05 (sniper_testnet Ansible deploy + runtime fix)
- Deployed `sniper_testnet` with HRF high-risk config:
   - profile: `ml/p3_edge_research/experiments/auto_038_risk_probe_20260405.json` / `tp120_rr000_floor0`
   - selector artifact: `output/selector_artifacts/hrf_selector_20260405.joblib`
   - risk: `max_open_positions=8`, `max_risk_per_trade=0.012`, leverage 20 testnet.
- Found runtime compatibility issue on server clone: `RealDataQuantExtractor.__init__` missing `rr_floor_to_tp`.
- Fixed in `sniper_bot/sniper_scanner.py` by introspecting extractor signature and dropping unsupported kwargs before instantiation.
- Also hardened `sniper_bot/main.py` to enforce `max_open_positions` during per-cycle signal execution (not only before scan loop).

## Update 2026-04-05 (dashboard deploy clear option)
- Added optional one-time history reset in `ansible/deploy-dashboard.yml`:
   - `clear_old_balance_data: false` (safe default)
   - `dashboard_db_path: "{{ bot_dir }}/pnl_dashboard/pnl_history.db"`
   - When enabled, playbook stops service and removes DB + `-wal`/`-shm` before restart.
- Verified deployment with cleanup enabled via `--limit pnl_dashboard -e clear_old_balance_data=true` (service active after restart).

## Update 2026-04-05 (dashboard local-run robustness)
- In `pnl_dashboard/app.py`, fixed `UnboundLocalError` on `access_token` when running in bare mode (`python app.py`) by:
   - setting `access_token = None` before credential load,
   - adding explicit `return` immediately after each `st.stop()` guard.
- Also switched credentials path to script-relative (`pnl_dashboard/credentials.json`) to avoid CWD-dependent lookup failures.
- Reminder: proper launch is `streamlit run pnl_dashboard/app.py`; bare mode shows `ScriptRunContext` warnings and session-state limitations.
