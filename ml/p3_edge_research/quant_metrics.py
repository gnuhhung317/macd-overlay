from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PERIODS_PER_YEAR_MAP: Dict[str, int] = {
    "1m": 525600,
    "5m": 105120,
    "15m": 35040,
    "1h": 8760,
    "4h": 2190,
    "1d": 365,
}

MIN_SHARPE_ANNUALIZATION_SAMPLES = 30


@dataclass
class PortfolioAssumptions:
    initial_capital: float = 10000.0
    leverage: float = 10.0
    risk_per_trade: float = 0.02
    max_concurrent_positions: int = 5
    max_margin_fraction: float = 0.10
    min_stop_distance: float = 0.005
    fee_bps_per_side: float = 5.0
    slippage_bps_per_side: float = 5.0
    panic_extra_slippage_bps: float = 10.0


def bps_to_rate(bps: float) -> float:
    return float(bps) / 10000.0


def _extract_future_path(row: pd.Series) -> Tuple[List[float], List[float], List[float]]:
    lows = list(row.get("future_lows", []))
    highs = list(row.get("future_highs", []))
    closes = list(row.get("future_closes", []))
    return lows, highs, closes


def _simulate_exit(
    row: pd.Series,
    side: int,
    entry_exec: float,
    assumptions: PortfolioAssumptions,
) -> Tuple[str, float]:
    lows, highs, closes = _extract_future_path(row)
    sl_p = float(row["sl_p"])
    tp_p = float(row["tp_p"])

    base_slip = bps_to_rate(assumptions.slippage_bps_per_side)
    panic_slip = base_slip + bps_to_rate(assumptions.panic_extra_slippage_bps)

    max_len = min(len(lows), len(highs))
    reason = "TIMEOUT"

    def _close_fallback() -> float:
        if closes:
            return float(closes[-1])
        if side == 1 and lows:
            return float(lows[-1])
        if side == -1 and highs:
            return float(highs[-1])
        return entry_exec

    for i in range(max_len):
        low_t = float(lows[i])
        high_t = float(highs[i])

        if side == 1:
            if (low_t / max(entry_exec, 1e-12) - 1.0) * assumptions.leverage <= -0.85:
                raw_exit = entry_exec * (1.0 - 0.85 / max(assumptions.leverage, 1e-8))
                exit_exec = raw_exit * (1.0 - panic_slip)
                reason = "LIQUIDATED"
                break
            if low_t <= sl_p:
                exit_exec = sl_p * (1.0 - panic_slip)
                reason = "STOP"
                break
            if high_t >= tp_p:
                exit_exec = tp_p * (1.0 - base_slip)
                reason = "TARGET"
                break
        else:
            if (high_t / max(entry_exec, 1e-12) - 1.0) * assumptions.leverage >= 0.85:
                raw_exit = entry_exec * (1.0 + 0.85 / max(assumptions.leverage, 1e-8))
                exit_exec = raw_exit * (1.0 + panic_slip)
                reason = "LIQUIDATED"
                break
            if high_t >= sl_p:
                exit_exec = sl_p * (1.0 + panic_slip)
                reason = "STOP"
                break
            if low_t <= tp_p:
                exit_exec = tp_p * (1.0 + base_slip)
                reason = "TARGET"
                break
    else:
        close_px = _close_fallback()
        if side == 1:
            exit_exec = close_px * (1.0 - base_slip)
        else:
            exit_exec = close_px * (1.0 + base_slip)

    directional_ret = (
        (exit_exec / max(entry_exec, 1e-12) - 1.0)
        if side == 1
        else (1.0 - exit_exec / max(entry_exec, 1e-12))
    )
    return reason, float(directional_ret)


def simulate_portfolio(
    trades_df: pd.DataFrame,
    assumptions: PortfolioAssumptions,
) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    sort_cols = ["timestamp"]
    sort_asc = [True]
    if "ai_prob" in trades_df.columns:
        sort_cols.append("ai_prob")
        sort_asc.append(False)
    if "coin" in trades_df.columns:
        sort_cols.append("coin")
        sort_asc.append(True)

    df = trades_df.sort_values(sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)

    equity = float(assumptions.initial_capital)
    fee_rate = bps_to_rate(assumptions.fee_bps_per_side)
    base_slip = bps_to_rate(assumptions.slippage_bps_per_side)

    active_positions: List[dict] = []
    records: List[dict] = []

    for _, row in df.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        end_time = pd.to_datetime(row["end_time"])

        active_positions = [p for p in active_positions if p["end_time"] > ts]
        if len(active_positions) >= assumptions.max_concurrent_positions:
            continue

        side = int(row.get("side", 1))
        entry_p = float(row["entry_p"])
        sl_p = float(row["sl_p"])

        if side == 1:
            dist_to_sl = (entry_p - sl_p) / max(entry_p, 1e-12)
            entry_exec = entry_p * (1.0 + base_slip)
        else:
            dist_to_sl = (sl_p - entry_p) / max(entry_p, 1e-12)
            entry_exec = entry_p * (1.0 - base_slip)

        if dist_to_sl <= assumptions.min_stop_distance:
            continue

        notional = (equity * assumptions.risk_per_trade) / max(dist_to_sl, 1e-12)
        margin_req = notional / max(assumptions.leverage, 1e-8)
        max_margin = equity * assumptions.max_margin_fraction

        if margin_req > max_margin:
            margin_req = max_margin
            notional = margin_req * assumptions.leverage

        reason, price_move_ret = _simulate_exit(row, side=side, entry_exec=entry_exec, assumptions=assumptions)

        equity_before = equity
        gross_pnl = notional * price_move_ret
        fees_paid = notional * fee_rate * 2.0
        net_pnl = gross_pnl - fees_paid
        equity = equity_before + net_pnl

        records.append(
            {
                "timestamp": ts,
                "end_time": end_time,
                "side": side,
                "exit_reason": reason,
                "equity_before": equity_before,
                "equity_after": equity,
                "notional": notional,
                "traded_notional": notional * 2.0,
                "fees_paid": fees_paid,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "net_ret": net_pnl / max(equity_before, 1e-12),
            }
        )

        active_positions.append({"end_time": end_time})

        if equity <= 0:
            break

    return pd.DataFrame(records)


def _profit_factor(pnl: pd.Series) -> float:
    positive = pnl[pnl > 0].sum()
    negative = -pnl[pnl < 0].sum()
    if negative <= 0:
        return float(np.inf) if positive > 0 else float("nan")
    return float(positive / negative)


def _infer_effective_periods_per_year(ledger: pd.DataFrame, fallback_periods_per_year: int) -> int:
    if ledger.empty:
        return int(max(fallback_periods_per_year, 1))

    ts = pd.to_datetime(ledger["timestamp"], errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return int(max(fallback_periods_per_year, 1))

    deltas = ts.diff().dt.total_seconds().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return int(max(fallback_periods_per_year, 1))

    median_delta_sec = float(deltas.median())
    if not np.isfinite(median_delta_sec) or median_delta_sec <= 0:
        return int(max(fallback_periods_per_year, 1))

    sec_per_year = 365.0 * 24.0 * 3600.0
    inferred = int(round(sec_per_year / median_delta_sec))
    return int(max(inferred, 1))


def summarize_portfolio(
    ledger: pd.DataFrame,
    assumptions: PortfolioAssumptions,
    periods_per_year: int,
) -> dict:
    if ledger.empty:
        return {
            "trades": 0,
            "net_return_pct": np.nan,
            "win_rate_pct": np.nan,
            "profit_factor": np.nan,
            "sharpe_annualized": np.nan,
            "sharpe_non_annualized": np.nan,
            "sample_size": 0,
            "periods_per_year": periods_per_year,
            "periods_per_year_input": periods_per_year,
            "periods_per_year_effective": periods_per_year,
            "sharpe_annualized_eligible": False,
            "sharpe_annualization_note": "insufficient_samples",
            "max_drawdown_pct": np.nan,
            "mdd_peak_time": None,
            "mdd_trough_time": None,
            "turnover_raw": np.nan,
            "turnover_annualized": np.nan,
            "avg_traded_notional": np.nan,
        }

    returns = ledger["net_ret"].astype(float)
    avg_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")

    if len(returns) > 1 and np.isfinite(std_ret) and std_ret > 0:
        sharpe_non_annualized = float(avg_ret / std_ret)
    else:
        sharpe_non_annualized = float("nan")

    effective_periods_per_year = _infer_effective_periods_per_year(
        ledger=ledger,
        fallback_periods_per_year=periods_per_year,
    )

    annualization_eligible = bool(
        len(returns) >= MIN_SHARPE_ANNUALIZATION_SAMPLES
        and np.isfinite(sharpe_non_annualized)
    )

    if annualization_eligible:
        sharpe = float(np.sqrt(effective_periods_per_year) * sharpe_non_annualized)
        sharpe_note = "annualized_from_trade_frequency"
    else:
        sharpe = float("nan")
        sharpe_note = "insufficient_samples"

    equity_curve = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "timestamp": pd.to_datetime(ledger["timestamp"].iloc[0]),
                        "equity": assumptions.initial_capital,
                    }
                ]
            ),
            ledger[["timestamp", "equity_after"]].rename(columns={"equity_after": "equity"}),
        ],
        ignore_index=True,
    )

    peak = equity_curve["equity"].cummax()
    drawdown = equity_curve["equity"] / peak - 1.0
    mdd_idx = int(drawdown.idxmin())
    max_dd = float(drawdown.iloc[mdd_idx] * 100.0)

    peak_idx = int(equity_curve["equity"].iloc[: mdd_idx + 1].idxmax())
    mdd_peak_time = pd.to_datetime(equity_curve["timestamp"].iloc[peak_idx])
    mdd_trough_time = pd.to_datetime(equity_curve["timestamp"].iloc[mdd_idx])

    avg_equity = float(ledger["equity_before"].mean())
    turnover_raw = float(ledger["traded_notional"].sum() / max(avg_equity, 1e-12))

    span_days = (
        pd.to_datetime(ledger["timestamp"].iloc[-1]) - pd.to_datetime(ledger["timestamp"].iloc[0])
    ).total_seconds() / 86400.0
    span_days = max(span_days, 1.0 / 24.0)
    turnover_annualized = float(turnover_raw * (365.0 / span_days))

    summary = {
        "trades": int(len(ledger)),
        "net_return_pct": float((ledger["equity_after"].iloc[-1] / assumptions.initial_capital - 1.0) * 100.0),
        "win_rate_pct": float((ledger["net_pnl"] > 0).mean() * 100.0),
        "profit_factor": _profit_factor(ledger["net_pnl"]),
        "sharpe_annualized": sharpe,
        "sharpe_non_annualized": sharpe_non_annualized,
        "sample_size": int(len(returns)),
        "periods_per_year": int(effective_periods_per_year),
        "periods_per_year_input": int(periods_per_year),
        "periods_per_year_effective": int(effective_periods_per_year),
        "sharpe_annualized_eligible": annualization_eligible,
        "sharpe_annualization_note": sharpe_note,
        "max_drawdown_pct": max_dd,
        "mdd_peak_time": mdd_peak_time,
        "mdd_trough_time": mdd_trough_time,
        "turnover_raw": turnover_raw,
        "turnover_annualized": turnover_annualized,
        "avg_traded_notional": float(ledger["traded_notional"].mean()),
    }
    return summary


def evaluate_trades(
    trades_df: pd.DataFrame,
    assumptions: PortfolioAssumptions,
    periods_per_year: int,
) -> Tuple[dict, pd.DataFrame]:
    ledger = simulate_portfolio(trades_df=trades_df, assumptions=assumptions)
    summary = summarize_portfolio(ledger=ledger, assumptions=assumptions, periods_per_year=periods_per_year)
    return summary, ledger
