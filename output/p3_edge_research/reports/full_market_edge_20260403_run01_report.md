# Full Market Edge Report - 2026-04-03 Run 01

## Objective
Evaluate full-market P3 setup search with strict OOS gate and realistic transaction costs, then promote the best accepted setup to live-test pipeline.

## Assumptions
- Timeframe: 1h
- Data scope: data/ohlcv/*.parquet (full market)
- Validation protocol: calendar split + walk-forward OOS (6 folds, embargo 24 bars)
- Cost model: 5 bps fee per side + 5 bps slippage per side
- Round-trip all-in: 20 bps
- Sizing: risk-per-trade 0.005
- Portfolio constraints: max concurrent positions 3, leverage 10
- OOS gate: min trades 60, Sharpe >= 1.5, max drawdown <= 15%
- Sharpe annualization base: 1h bars (8760), effective periods adjusted by trade-frequency sampling in run output
- Turnover definition: sum(abs(entry_notional)+abs(exit_notional))/average_equity

## Run Outcome
- Experiments evaluated: 60
- Accepted by OOS gate: 5
- Acceptance rate: 8.33%

Accepted experiments:
1. auto_038
2. auto_008
3. auto_023
4. auto_013
5. pullback_conservative

## Champion Selection
Champion: auto_038

Walk-forward OOS metrics:
- Trades: 2214
- Sharpe (annualized): 8.3227
- Max drawdown: -5.8669%
- Net return: 438.0610%
- Turnover raw: 900.7759

Held-out test metrics:
- Sharpe (annualized): 9.9379
- Max drawdown: -5.7823%
- Net return: 251.5831%

Extractor params and threshold:
- tp_level: 1.293
- max_hold_bars: 15
- min_mid_candles: 8
- min_price_pct: 2.526
- entry_pullback: 0.0151
- min_rr: 0.668
- threshold: 0.70

## Live Promotion Decision
Promote auto_038 to a dedicated live-test config for forward validation.

## Risk Notes Before Production
- Turnover is high; monitor implementation shortfall and slippage drift.
- Re-run stress checks at higher costs (30 to 40 bps round-trip) before production capital scaling.

## Source Artifacts
- output/p3_edge_research/full_market_edge_20260403_run01/summary.csv
- output/p3_edge_research/full_market_edge_20260403_run01/details.json
