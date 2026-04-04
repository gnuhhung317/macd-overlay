---
description: "Use when implementing, reviewing, or reporting trading/backtest metrics. Enforces repository-wide standards for Sharpe, max drawdown, turnover, and fee/slippage assumptions."
name: "Quant Metric Standards"
applyTo: "**/*.py, **/*.ipynb, **/*.md"
---
# Quant Metric Standards (Repo-Wide)

This instruction defines a mandatory metric standard for all strategy research, backtests, and portfolio evaluations in this repository.

## 1) Core Principle
- All performance metrics must be computed on net returns after transaction costs.
- Always report assumptions and formulas with the final metric values.
- Never compare two strategies if their fee/slippage assumptions differ.

## 2) Mandatory Assumption Block
Every result must include a reproducible assumptions block:
- Timeframe and bar frequency.
- Capital base and leverage model.
- Fee model (maker/taker or fixed bps).
- Slippage model (fixed bps or model-based).
- Position sizing and rebalancing logic.
- Turnover definition used.

If not explicitly specified by the task, use default baseline assumptions:
- Fee per side: 5 bps (0.05%).
- Slippage per side: 5 bps (0.05%).
- Total round-trip all-in cost: 20 bps.

If a script already has strategy-specific assumptions, keep them and report them explicitly.

## 3) Sharpe Standard
Use periodic net returns after all costs:
- net_ret_t = pnl_t / equity_{t-1} (after fees and slippage)

Annualized Sharpe:
- sharpe = sqrt(periods_per_year) * mean(net_ret) / std(net_ret)

Rules:
- Use risk-free rate = 0 by default unless user provides a different rate.
- Report sample count and periods_per_year used.
- Do not annualize if sample size is too small; report non-annualized and note limitation.

Recommended periods_per_year map:
- 1m: 525600
- 5m: 105120
- 15m: 35040
- 1h: 8760
- 4h: 2190
- 1d: 365

## 4) Max Drawdown Standard
Compute from cumulative equity curve built from net returns.
- equity_t = equity_{t-1} * (1 + net_ret_t)
- peak_t = cummax(equity_t)
- drawdown_t = equity_t / peak_t - 1
- MDD = min(drawdown_t)

Rules:
- Report MDD in percent.
- Report peak-to-trough timestamps when available.

## 5) Turnover Standard
Default turnover definition (portfolio-level):
- turnover = sum(abs(notional_traded_t)) / average_equity

Execution rules:
- Count both entry and exit notional.
- Report turnover for the same interval as return metrics.
- If turnover is reported annualized, include annualization formula.

Minimum required turnover outputs:
- Raw turnover ratio.
- Annualized turnover (if applicable).
- Average traded notional per rebalance or per trade.

## 6) Cost Application Standard
Costs must be applied as part of execution simulation, not as an afterthought.

Minimum implementation behavior:
- Entry cost on executed notional.
- Exit cost on executed notional.
- Slippage direction-aware (buy worse, sell worse).
- Liquidation or panic exits must include extra slippage assumptions if modeled.

## 7) Reporting Format (Required)
When presenting strategy performance, include this block:
- Sharpe (annualized), periods_per_year, sample size.
- Max Drawdown (%).
- Turnover (raw and annualized if used).
- Net Return (%), Win Rate (%), Profit Factor (if available).
- Fee/slippage assumptions (per side and round-trip all-in).

## 8) Validation Checklist
Before accepting results, verify:
- Metrics are net of costs.
- Sharpe uses correct period scaling.
- MDD comes from cumulative net equity curve.
- Turnover definition is explicitly stated.
- Fee/slippage assumptions are explicitly stated and consistent.
- OOS and walk-forward metrics are separated from in-sample metrics.

## 9) Non-Compliance Handling
If any required assumption or formula is missing:
- Mark result as non-comparable.
- Add missing assumption block.
- Recompute metrics before claiming edge.
