import argparse
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


class PivotSetupExtractor:
    def __init__(self, max_hold_bars=24, min_mid_candles=6, min_price_pct=3.0):
        self.max_hold_bars = max_hold_bars
        self.min_mid_candles = min_mid_candles
        self.min_price_pct = min_price_pct
        self.candidates = []

    def check_ph(self, h1, h2, h3, l1, l2, l3):
        return (
            (h2 > h1 and h2 > h3 and l2 > l1 and l2 > l3)
            or (h2 >= h1 and h2 > h3 and l2 > l3 and l2 < l1)
            or (h2 > h1 and h2 >= h3 and l2 < l3 and l2 > l1)
            or (h2 >= h3 and h2 > h1 and l2 <= l3 and l2 > l1)
            or (h2 >= h3 and h2 >= h1 and l2 <= l3 and l2 > l1)
        )

    def check_pl(self, h1, h2, h3, l1, l2, l3):
        return (
            (l2 < l1 and l2 < l3 and h2 < h1 and h2 < h3)
            or (h2 >= h1 and h2 < h3 and l2 < l3 and l2 <= l1)
            or (l2 < l1 and l2 < l3 and h2 < h1 and h2 >= h3)
            or (h2 <= h3 and h2 < h1 and l2 >= l3 and l2 <= l1)
            or (h2 >= h3 and h2 < h1 and l2 <= l3 and l2 <= l1)
        )

    def is_valid_bc_long(self, close_arr, low_arr, b_idx, c_idx):
        if c_idx <= b_idx:
            return True
        for j in range(b_idx + 1, c_idx + 1):
            if close_arr[j] < low_arr[j - 1]:
                return False
        return True

    def is_valid_bc_short(self, close_arr, high_arr, b_idx, c_idx):
        if c_idx <= b_idx:
            return True
        for j in range(b_idx + 1, c_idx + 1):
            if close_arr[j] > high_arr[j - 1]:
                return False
        return True

    def has_all_cut(self, high_arr, low_arr, a_idx, p_a, b_idx, p_b):
        if b_idx <= a_idx:
            return False
        slope = (p_b - p_a) / max(1, b_idx - a_idx)
        intercept = p_a - slope * a_idx
        for j in range(a_idx + 1, b_idx):
            line_val = slope * j + intercept
            if not (line_val >= low_arr[j] and line_val <= high_arr[j]):
                return False
        return True

    def is_quality_wave(self, a_idx, p_a, b_idx, p_b):
        mid = b_idx - a_idx - 1
        pct = abs(p_b - p_a) / (abs(p_a) + 1e-8) * 100
        return (mid >= self.min_mid_candles) or (pct >= self.min_price_pct)

    def extract(self, df, coin_name):
        df = df.sort_values('timestamp' if 'timestamp' in df.columns else df.index).reset_index(drop=True)
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        v = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
        ts = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)

        range_sma_20 = pd.Series(h - l).rolling(20).mean().values
        vol_sma_20 = pd.Series(v).rolling(20).mean().values
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        atr_20 = pd.Series(tr).rolling(20).mean().values
        ema_20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
        ema_50 = pd.Series(c).ewm(span=50, adjust=False).mean().values

        pivots = []
        for i in range(250, len(df) - self.max_hold_bars):
            idx = i - 1
            h1, h2, h3 = h[idx - 1], h[idx], h[idx + 1]
            l1, l2, l3 = l[idx - 1], l[idx], l[idx + 1]
            is_ph = self.check_ph(h1, h2, h3, l1, l2, l3)
            is_pl = self.check_pl(h1, h2, h3, l1, l2, l3)

            if is_ph or is_pl:
                if pivots and pivots[-1][2] == is_ph:
                    if (is_ph and h[idx] > pivots[-1][1]) or ((not is_ph) and l[idx] < pivots[-1][1]):
                        pivots.pop()
                pivots.append([idx, h[idx] if is_ph else l[idx], is_ph])

            if len(pivots) < 3:
                continue

            c_idx, p_c, h_c = pivots[-1]
            b_idx, p_b, h_b = pivots[-2]
            a_idx, p_a, h_a = pivots[-3]

            f_lows = l[i + 1: i + 1 + self.max_hold_bars]
            f_highs = h[i + 1: i + 1 + self.max_hold_bars]
            f_closes = c[i + 1: i + 1 + self.max_hold_bars]
            if len(f_lows) == 0:
                continue

            if (not h_a) and h_b and (not h_c) and p_a < p_c and self.is_valid_bc_long(c, l, b_idx, c_idx):
                all_cut = self.has_all_cut(h, l, a_idx, p_a, b_idx, p_b)
                is_quality = self.is_quality_wave(a_idx, p_a, b_idx, p_b)
                if not (all_cut and is_quality):
                    continue

                range_b = max(h[b_idx] - l[b_idx], 1e-8)
                is_buy_climax = (
                    (range_b > range_sma_20[b_idx])
                    and ((h[b_idx] - c[b_idx]) <= range_b * 0.3)
                    and (v[b_idx] > vol_sma_20[b_idx] * 1.5)
                )
                if is_buy_climax:
                    continue

                trigger_p = h[c_idx]
                if c[i] > trigger_p and c[i - 1] <= trigger_p:
                    signal_volatility = atr_20[i] / (abs(c[i]) + 1e-8)
                    signal_volume_ratio = v[i] / (vol_sma_20[i] + 1e-8)
                    signal_trend_strength = (ema_20[i] - ema_50[i]) / (abs(ema_50[i]) + 1e-8)
                    self.candidates.append(
                        {
                            'coin': coin_name,
                            'timestamp': ts[i],
                            'side': 1,
                            'trigger_p': float(trigger_p),
                            'signal_close': float(c[i]),
                            'sl_pivot_c': float(l[c_idx]),
                            'p_a': float(p_a),
                            'p_b': float(p_b),
                            'signal_volatility': float(signal_volatility),
                            'signal_volume_ratio': float(signal_volume_ratio),
                            'signal_trend_strength': float(signal_trend_strength),
                            'future_lows': f_lows.tolist(),
                            'future_highs': f_highs.tolist(),
                            'future_closes': f_closes.tolist(),
                        }
                    )

            if h_a and (not h_b) and h_c and p_a > p_c and self.is_valid_bc_short(c, h, b_idx, c_idx):
                all_cut = self.has_all_cut(h, l, a_idx, p_a, b_idx, p_b)
                is_quality = self.is_quality_wave(a_idx, p_a, b_idx, p_b)
                if not (all_cut and is_quality):
                    continue

                range_b = max(h[b_idx] - l[b_idx], 1e-8)
                is_sell_climax = (
                    (range_b > range_sma_20[b_idx])
                    and ((c[b_idx] - l[b_idx]) <= range_b * 0.3)
                    and (v[b_idx] > vol_sma_20[b_idx] * 1.5)
                )
                if is_sell_climax:
                    continue

                trigger_p = l[c_idx]
                if c[i] < trigger_p and c[i - 1] >= trigger_p:
                    signal_volatility = atr_20[i] / (abs(c[i]) + 1e-8)
                    signal_volume_ratio = v[i] / (vol_sma_20[i] + 1e-8)
                    signal_trend_strength = (ema_20[i] - ema_50[i]) / (abs(ema_50[i]) + 1e-8)
                    self.candidates.append(
                        {
                            'coin': coin_name,
                            'timestamp': ts[i],
                            'side': -1,
                            'trigger_p': float(trigger_p),
                            'signal_close': float(c[i]),
                            'sl_pivot_c': float(h[c_idx]),
                            'p_a': float(p_a),
                            'p_b': float(p_b),
                            'signal_volatility': float(signal_volatility),
                            'signal_volume_ratio': float(signal_volume_ratio),
                            'signal_trend_strength': float(signal_trend_strength),
                            'future_lows': f_lows.tolist(),
                            'future_highs': f_highs.tolist(),
                            'future_closes': f_closes.tolist(),
                        }
                    )


def parse_float_list(raw_text):
    return [float(x.strip()) for x in raw_text.split(',') if x.strip()]


def assign_quantile_regime(series, labels):
    if series.isna().all():
        return pd.Series(['na'] * len(series), index=series.index)
    try:
        reg = pd.qcut(series, q=len(labels), labels=labels, duplicates='drop')
        return reg.astype(str).fillna('na')
    except Exception:
        return pd.Series(['na'] * len(series), index=series.index)


def build_distribution_frame(candidates_df, horizons):
    rows = []
    for row in candidates_df.itertuples(index=False):
        entry = float(row.trigger_p)
        if entry <= 0:
            continue

        closes = np.array(row.future_closes, dtype=float)
        highs = np.array(row.future_highs, dtype=float)
        lows = np.array(row.future_lows, dtype=float)
        side = int(row.side)

        if side == 1:
            directional_rets = closes / entry - 1.0
            mfe = float(np.max(highs / entry - 1.0)) if len(highs) > 0 else np.nan
            mae = float(np.min(lows / entry - 1.0)) if len(lows) > 0 else np.nan
            risk_pct = (entry - float(row.sl_pivot_c)) / entry if entry > 0 else np.nan
        else:
            directional_rets = 1.0 - closes / entry
            mfe = float(np.max(1.0 - lows / entry)) if len(lows) > 0 else np.nan
            mae = float(np.min(1.0 - highs / entry)) if len(highs) > 0 else np.nan
            risk_pct = (float(row.sl_pivot_c) - entry) / entry if entry > 0 else np.nan

        d = {
            'coin': row.coin,
            'timestamp': row.timestamp,
            'side': side,
            'signal_volatility': float(getattr(row, 'signal_volatility', np.nan)),
            'signal_volume_ratio': float(getattr(row, 'signal_volume_ratio', np.nan)),
            'signal_trend_strength': float(getattr(row, 'signal_trend_strength', np.nan)),
            'mfe_pct': mfe * 100 if pd.notna(mfe) else np.nan,
            'mae_pct': mae * 100 if pd.notna(mae) else np.nan,
            'risk_pct': risk_pct * 100 if pd.notna(risk_pct) else np.nan,
            'mfe_r': (mfe / (risk_pct + 1e-8)) if pd.notna(mfe) and pd.notna(risk_pct) and risk_pct > 0 else np.nan,
            'mae_r': (mae / (risk_pct + 1e-8)) if pd.notna(mae) and pd.notna(risk_pct) and risk_pct > 0 else np.nan,
            'immediate_fail': (directional_rets[0] < 0) if len(directional_rets) > 0 else np.nan,
        }

        pos_idx = np.where(directional_rets > 0)[0]
        d['time_to_positive_bars'] = int(pos_idx[0] + 1) if len(pos_idx) > 0 else np.nan

        for h in horizons:
            d[f'ret_{h}'] = directional_rets[h - 1] * 100 if len(directional_rets) >= h else np.nan

        rows.append(d)

    return pd.DataFrame(rows)


def summarize_event_study(dist_df, horizons):
    out = []
    for h in horizons:
        col = f'ret_{h}'
        s = dist_df[col].dropna()
        if s.empty:
            continue
        out.append(
            {
                'horizon_bars': h,
                'samples': int(s.shape[0]),
                'mean_ret_pct': float(s.mean()),
                'median_ret_pct': float(s.median()),
                'win_rate_pct': float((s > 0).mean() * 100),
                'q10_ret_pct': float(s.quantile(0.10)),
                'q25_ret_pct': float(s.quantile(0.25)),
                'q75_ret_pct': float(s.quantile(0.75)),
                'q90_ret_pct': float(s.quantile(0.90)),
            }
        )
    return pd.DataFrame(out)


def summarize_path(dist_df, horizons):
    out = []
    for h in horizons:
        col = f'ret_{h}'
        s = dist_df[col].dropna()
        if s.empty:
            continue
        out.append(
            {
                'horizon_bars': h,
                'mean_path_pct': float(s.mean()),
                'median_path_pct': float(s.median()),
                'q25_path_pct': float(s.quantile(0.25)),
                'q75_path_pct': float(s.quantile(0.75)),
            }
        )
    return pd.DataFrame(out)


def summarize_excursions(dist_df):
    mfe = dist_df['mfe_pct'].dropna()
    mae = dist_df['mae_pct'].dropna()
    mfe_r = dist_df['mfe_r'].dropna()
    mae_r = dist_df['mae_r'].dropna()

    return pd.DataFrame(
        [
            {
                'samples': int(dist_df.shape[0]),
                'mfe_mean_pct': float(mfe.mean()) if not mfe.empty else np.nan,
                'mfe_q10_pct': float(mfe.quantile(0.10)) if not mfe.empty else np.nan,
                'mfe_q25_pct': float(mfe.quantile(0.25)) if not mfe.empty else np.nan,
                'mfe_q50_pct': float(mfe.quantile(0.50)) if not mfe.empty else np.nan,
                'mfe_q75_pct': float(mfe.quantile(0.75)) if not mfe.empty else np.nan,
                'mfe_q90_pct': float(mfe.quantile(0.90)) if not mfe.empty else np.nan,
                'mae_mean_pct': float(mae.mean()) if not mae.empty else np.nan,
                'mae_q10_pct': float(mae.quantile(0.10)) if not mae.empty else np.nan,
                'mae_q25_pct': float(mae.quantile(0.25)) if not mae.empty else np.nan,
                'mae_q50_pct': float(mae.quantile(0.50)) if not mae.empty else np.nan,
                'mae_q75_pct': float(mae.quantile(0.75)) if not mae.empty else np.nan,
                'mae_q90_pct': float(mae.quantile(0.90)) if not mae.empty else np.nan,
                'mfe_ge_1r_pct': float((mfe_r >= 1.0).mean() * 100) if not mfe_r.empty else np.nan,
                'mfe_ge_1_5r_pct': float((mfe_r >= 1.5).mean() * 100) if not mfe_r.empty else np.nan,
                'mae_le_minus_0_8r_pct': float((mae_r <= -0.8).mean() * 100) if not mae_r.empty else np.nan,
                'immediate_fail_rate_pct': float(dist_df['immediate_fail'].dropna().mean() * 100) if dist_df['immediate_fail'].notna().any() else np.nan,
                'median_time_to_positive_bars': float(dist_df['time_to_positive_bars'].dropna().median()) if dist_df['time_to_positive_bars'].notna().any() else np.nan,
            }
        ]
    )


def summarize_conditional(dist_df, group_col, horizons):
    out = []
    for g, gdf in dist_df.groupby(group_col, dropna=False):
        row = {
            'group': str(g),
            'samples': int(gdf.shape[0]),
            'mfe_mean_pct': float(gdf['mfe_pct'].mean()) if gdf['mfe_pct'].notna().any() else np.nan,
            'mae_mean_pct': float(gdf['mae_pct'].mean()) if gdf['mae_pct'].notna().any() else np.nan,
        }
        for h in horizons:
            col = f'ret_{h}'
            s = gdf[col].dropna()
            row[f'mean_ret_{h}_pct'] = float(s.mean()) if not s.empty else np.nan
            row[f'win_rate_{h}_pct'] = float((s > 0).mean() * 100) if not s.empty else np.nan
        out.append(row)
    return pd.DataFrame(out)


def run_distribution_layers(candidates_df, horizons):
    dist_df = build_distribution_frame(candidates_df, horizons)
    if dist_df.empty:
        raise ValueError('Distribution analysis has no rows after feature build.')

    dist_df['volatility_regime'] = assign_quantile_regime(dist_df['signal_volatility'], ['low', 'mid', 'high'])
    dist_df['volume_regime'] = assign_quantile_regime(dist_df['signal_volume_ratio'], ['low', 'mid', 'high'])
    dist_df['trend_regime'] = assign_quantile_regime(dist_df['signal_trend_strength'], ['down', 'flat', 'up'])

    event_df = summarize_event_study(dist_df, horizons)
    path_df = summarize_path(dist_df, horizons)
    excursion_df = summarize_excursions(dist_df)
    cond_vol_df = summarize_conditional(dist_df, 'volatility_regime', horizons)
    cond_volume_df = summarize_conditional(dist_df, 'volume_regime', horizons)
    cond_trend_df = summarize_conditional(dist_df, 'trend_regime', horizons)

    return {
        'distribution': dist_df,
        'event': event_df,
        'path': path_df,
        'excursion': excursion_df,
        'cond_volatility': cond_vol_df,
        'cond_volume': cond_volume_df,
        'cond_trend': cond_trend_df,
    }


def simulate_one_trade(side, entry_p, sl_p, tp_p, f_lows, f_highs, f_closes):
    fill_idx = None
    if side == 1:
        for k, low_t in enumerate(f_lows):
            if low_t <= entry_p:
                fill_idx = k
                break
    else:
        for k, high_t in enumerate(f_highs):
            if high_t >= entry_p:
                fill_idx = k
                break

    if fill_idx is None:
        return False, 'no_fill', 0.0

    exit_price = None
    reason = None
    for k in range(fill_idx, len(f_lows)):
        low_t = f_lows[k]
        high_t = f_highs[k]
        if side == 1:
            if low_t <= sl_p:
                exit_price = sl_p
                reason = 'sl'
                break
            if high_t >= tp_p:
                exit_price = tp_p
                reason = 'tp'
                break
        else:
            if high_t >= sl_p:
                exit_price = sl_p
                reason = 'sl'
                break
            if low_t <= tp_p:
                exit_price = tp_p
                reason = 'tp'
                break

    if exit_price is None:
        if len(f_closes) > 0:
            exit_price = f_closes[-1]
        else:
            exit_price = (f_lows[-1] + f_highs[-1]) / 2
        reason = 'time'

    if side == 1:
        ret = exit_price / entry_p - 1.0
    else:
        ret = 1.0 - exit_price / entry_p

    return True, reason, float(ret)


def build_equity_curve(trades_df, initial_equity=1.0):
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=['timestamp', 'ret', 'reason', 'equity', 'drawdown_pct'])

    curve = trades_df.sort_values('timestamp').reset_index(drop=True).copy()
    curve['equity'] = initial_equity * np.cumprod(1.0 + curve['ret'])
    peak = curve['equity'].cummax()
    curve['drawdown_pct'] = (curve['equity'] / peak - 1.0) * 100
    return curve


def evaluate_combo(candidates_df, tp_level, entry_pullback, min_filled, return_trade_log=False):
    total = len(candidates_df)
    valid_rr = 0
    filled = 0
    tp_hits = 0
    sl_hits = 0
    time_exits = 0
    returns = []
    rr_vals = []
    trade_log = []

    ordered_df = candidates_df.sort_values('timestamp').reset_index(drop=True)
    for row in ordered_df.itertuples(index=False):
        side = row.side
        trigger = row.trigger_p
        sl_p = row.sl_pivot_c
        p_a = row.p_a
        p_b = row.p_b

        if side == 1:
            entry_p = trigger * (1.0 - entry_pullback)
            tp_p = p_a + tp_level * (p_b - p_a)
            risk = entry_p - sl_p
            reward = tp_p - entry_p
        else:
            entry_p = trigger * (1.0 + entry_pullback)
            tp_p = p_a - tp_level * (p_a - p_b)
            risk = sl_p - entry_p
            reward = entry_p - tp_p

        if risk <= 0 or reward <= 0:
            continue

        valid_rr += 1
        rr_vals.append(reward / risk)

        is_filled, reason, ret = simulate_one_trade(
            side,
            entry_p,
            sl_p,
            tp_p,
            row.future_lows,
            row.future_highs,
            row.future_closes,
        )

        if not is_filled:
            continue

        filled += 1
        returns.append(ret)
        trade_log.append({'timestamp': row.timestamp, 'ret': float(ret), 'reason': reason})
        if reason == 'tp':
            tp_hits += 1
        elif reason == 'sl':
            sl_hits += 1
        else:
            time_exits += 1

    if filled > 0:
        arr = np.array(returns)
        gross_profit = arr[arr > 0].sum()
        gross_loss = -arr[arr < 0].sum()
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate_frac = win_count / filled
        lose_rate_frac = loss_count / filled
        avg_win_pct = (wins.mean() * 100) if win_count > 0 else 0.0
        avg_loss_pct = (-losses.mean() * 100) if loss_count > 0 else 0.0
        expectancy_pct = win_rate_frac * avg_win_pct - lose_rate_frac * avg_loss_pct

        eq_curve = build_equity_curve(pd.DataFrame(trade_log), initial_equity=1.0)
        equity_final = float(eq_curve['equity'].iloc[-1]) if not eq_curve.empty else np.nan
        compounded_return_pct = (equity_final - 1.0) * 100 if pd.notna(equity_final) else np.nan
        max_drawdown_pct = float(eq_curve['drawdown_pct'].min()) if not eq_curve.empty else np.nan

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan
        avg_ret = arr.mean() * 100
        median_ret = np.median(arr) * 100
        total_ret = arr.sum() * 100
        win_rate = tp_hits / filled * 100
    else:
        gross_profit = 0.0
        gross_loss = 0.0
        profit_factor = np.nan
        avg_ret = np.nan
        median_ret = np.nan
        total_ret = np.nan
        avg_win_pct = np.nan
        avg_loss_pct = np.nan
        expectancy_pct = np.nan
        equity_final = np.nan
        compounded_return_pct = np.nan
        max_drawdown_pct = np.nan
        win_rate = np.nan

    metrics = {
        'tp_level': tp_level,
        'entry_pullback_pct': entry_pullback * 100,
        'total_candidates': total,
        'valid_rr_candidates': valid_rr,
        'filled_trades': filled,
        'fill_rate_valid_rr_pct': (filled / max(valid_rr, 1)) * 100,
        'fill_rate_total_pct': (filled / max(total, 1)) * 100,
        'tp_hits': tp_hits,
        'sl_hits': sl_hits,
        'time_exits': time_exits,
        'win_rate_pct': win_rate,
        'avg_win_pct': avg_win_pct,
        'avg_loss_pct': avg_loss_pct,
        'expectancy_pct': expectancy_pct,
        'avg_return_pct': avg_ret,
        'median_return_pct': median_ret,
        'total_return_pct': total_ret,
        'compounded_return_pct': compounded_return_pct,
        'equity_final': equity_final,
        'max_drawdown_pct': max_drawdown_pct,
        'gross_profit_pct': gross_profit * 100,
        'gross_loss_pct': gross_loss * 100,
        'profit_factor': profit_factor,
        'avg_rr': (float(np.mean(rr_vals)) if len(rr_vals) > 0 else np.nan),
        'passes_min_filled': filled >= min_filled,
    }

    if return_trade_log:
        return metrics, pd.DataFrame(trade_log)
    return metrics


def evaluate_grid(candidates_df, tp_levels, entry_pullbacks, min_filled):
    combos = [(tp, ep) for tp in tp_levels for ep in entry_pullbacks]
    results = []
    for tp_level, entry_pullback in tqdm(combos, desc='Evaluating grid'):
        results.append(evaluate_combo(candidates_df, tp_level, entry_pullback, min_filled))
    return pd.DataFrame(results)


def select_best_config(stats_df, objective):
    if stats_df.empty:
        return None

    candidate = stats_df[stats_df['passes_min_filled']].copy()
    if candidate.empty:
        candidate = stats_df[stats_df['filled_trades'] > 0].copy()
    if candidate.empty:
        return None

    score = candidate[objective].fillna(-np.inf)
    candidate = candidate.assign(_score=score)
    candidate = candidate.sort_values(['_score', 'filled_trades'], ascending=[False, False])
    return candidate.iloc[0]


def run_walk_forward(
    candidates_df,
    tp_levels,
    entry_pullbacks,
    min_filled,
    train_days,
    test_days,
    step_days,
    min_train_samples,
    min_test_samples,
    objective,
):
    if candidates_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    cdf = candidates_df.sort_values('timestamp').reset_index(drop=True)
    ts_min = cdf['timestamp'].min()
    ts_max = cdf['timestamp'].max()

    cur_start = ts_min
    fold = 1
    out = []
    oos_trade_logs = []

    while True:
        train_start = cur_start
        train_end = train_start + pd.Timedelta(days=train_days)
        test_end = train_end + pd.Timedelta(days=test_days)
        if test_end > ts_max:
            break

        train_df = cdf[(cdf['timestamp'] >= train_start) & (cdf['timestamp'] < train_end)]
        test_df = cdf[(cdf['timestamp'] >= train_end) & (cdf['timestamp'] < test_end)]

        if len(train_df) < min_train_samples or len(test_df) < min_test_samples:
            cur_start = cur_start + pd.Timedelta(days=step_days)
            fold += 1
            continue

        train_grid = evaluate_grid(train_df, tp_levels, entry_pullbacks, min_filled)
        best_cfg = select_best_config(train_grid, objective)
        if best_cfg is None:
            cur_start = cur_start + pd.Timedelta(days=step_days)
            fold += 1
            continue

        oos_metrics, oos_trade_log = evaluate_combo(
            test_df,
            tp_level=float(best_cfg['tp_level']),
            entry_pullback=float(best_cfg['entry_pullback_pct']) / 100.0,
            min_filled=min_filled,
            return_trade_log=True,
        )

        if not oos_trade_log.empty:
            oos_trade_log = oos_trade_log.copy()
            oos_trade_log['fold'] = fold
            oos_trade_logs.append(oos_trade_log)

        out.append(
            {
                'fold': fold,
                'train_start': train_start,
                'train_end': train_end,
                'test_end': test_end,
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'selected_tp_level': float(best_cfg['tp_level']),
                'selected_entry_pullback_pct': float(best_cfg['entry_pullback_pct']),
                'train_objective': float(best_cfg[objective]) if pd.notna(best_cfg[objective]) else np.nan,
                'train_filled_trades': int(best_cfg['filled_trades']),
                'oos_filled_trades': int(oos_metrics['filled_trades']),
                'oos_win_rate_pct': oos_metrics['win_rate_pct'],
                'oos_expectancy_pct': oos_metrics['expectancy_pct'],
                'oos_avg_return_pct': oos_metrics['avg_return_pct'],
                'oos_total_return_pct': oos_metrics['total_return_pct'],
                'oos_compounded_return_pct': oos_metrics['compounded_return_pct'],
                'oos_max_drawdown_pct': oos_metrics['max_drawdown_pct'],
                'oos_profit_factor': oos_metrics['profit_factor'],
                'oos_tp_hits': int(oos_metrics['tp_hits']),
                'oos_sl_hits': int(oos_metrics['sl_hits']),
                'oos_time_exits': int(oos_metrics['time_exits']),
            }
        )

        cur_start = cur_start + pd.Timedelta(days=step_days)
        fold += 1

    trade_log_df = pd.concat(oos_trade_logs, ignore_index=True) if oos_trade_logs else pd.DataFrame()
    return pd.DataFrame(out), trade_log_df


def main():
    parser = argparse.ArgumentParser(description='Entry/TP statistics sweep for p3 setup')
    parser.add_argument('--data-glob', default=r'data\ohlcv\*.parquet')
    parser.add_argument('--tp-levels', default='1.2,1.4,1.6,1.8,2.0')
    parser.add_argument('--entry-pullbacks', default='0,0.005,0.01,0.02')
    parser.add_argument('--max-hold-bars', type=int, default=24)
    parser.add_argument('--min-mid-candles', type=int, default=6)
    parser.add_argument('--min-price-pct', type=float, default=3.0)
    parser.add_argument('--min-filled', type=int, default=100)
    parser.add_argument('--walk-forward', action='store_true', help='Use walk-forward optimization instead of full-sample grid ranking')
    parser.add_argument('--wf-train-days', type=int, default=365)
    parser.add_argument('--wf-test-days', type=int, default=90)
    parser.add_argument('--wf-step-days', type=int, default=90)
    parser.add_argument('--wf-min-train-samples', type=int, default=1000)
    parser.add_argument('--wf-min-test-samples', type=int, default=300)
    parser.add_argument('--wf-objective', choices=['avg_return_pct', 'win_rate_pct', 'profit_factor'], default='avg_return_pct')
    parser.add_argument('--out-csv', default='output/p3_entry_tp_stats.csv')
    parser.add_argument('--equity-out-csv', default='output/p3_entry_tp_equity_curve.csv')
    parser.add_argument('--run-distribution-analysis', action='store_true', help='Run Layer1-3 distribution/path/excursion analysis')
    parser.add_argument('--analysis-only', action='store_true', help='Stop after distribution analysis, skip grid/WFO simulation')
    parser.add_argument('--analysis-horizons', default='1,3,6,12,24')
    parser.add_argument('--analysis-out-prefix', default='output/p3_event_study')
    args = parser.parse_args()

    tp_levels = parse_float_list(args.tp_levels)
    entry_pullbacks = parse_float_list(args.entry_pullbacks)
    analysis_horizons = [int(x) for x in parse_float_list(args.analysis_horizons)]

    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise FileNotFoundError(f'No files found with pattern: {args.data_glob}')

    extractor = PivotSetupExtractor(
        max_hold_bars=args.max_hold_bars,
        min_mid_candles=args.min_mid_candles,
        min_price_pct=args.min_price_pct,
    )

    print(f'Loading {len(files)} files...')
    for f in tqdm(files, desc='Extracting setups'):
        try:
            df_raw = pd.read_parquet(f)
            df_raw.columns = [x.lower() for x in df_raw.columns]
            extractor.extract(df_raw, os.path.basename(f))
        except Exception:
            continue

    candidates_df = pd.DataFrame(extractor.candidates)
    if candidates_df.empty:
        raise ValueError('No setup candidates extracted.')

    print(f'Total setup candidates: {len(candidates_df)}')
    print(f'TP levels: {tp_levels}')
    print(f'Entry pullbacks: {entry_pullbacks}')

    if args.run_distribution_analysis:
        layers = run_distribution_layers(candidates_df, analysis_horizons)
        layers['event'].to_csv(f"{args.analysis_out_prefix}_event.csv", index=False)
        layers['path'].to_csv(f"{args.analysis_out_prefix}_path.csv", index=False)
        layers['excursion'].to_csv(f"{args.analysis_out_prefix}_excursion.csv", index=False)
        layers['cond_volatility'].to_csv(f"{args.analysis_out_prefix}_cond_volatility.csv", index=False)
        layers['cond_volume'].to_csv(f"{args.analysis_out_prefix}_cond_volume.csv", index=False)
        layers['cond_trend'].to_csv(f"{args.analysis_out_prefix}_cond_trend.csv", index=False)

        print('\n[Layer 1-3] Distribution analysis saved:')
        print(f">> {args.analysis_out_prefix}_event.csv")
        print(f">> {args.analysis_out_prefix}_path.csv")
        print(f">> {args.analysis_out_prefix}_excursion.csv")
        print(f">> {args.analysis_out_prefix}_cond_volatility.csv")
        print(f">> {args.analysis_out_prefix}_cond_volume.csv")
        print(f">> {args.analysis_out_prefix}_cond_trend.csv")

        if not layers['event'].empty:
            print('\nEvent study summary:')
            print(layers['event'].to_string(index=False))

    if args.analysis_only:
        return

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.walk_forward:
        folds_df, oos_trade_log = run_walk_forward(
            candidates_df,
            tp_levels,
            entry_pullbacks,
            args.min_filled,
            train_days=args.wf_train_days,
            test_days=args.wf_test_days,
            step_days=args.wf_step_days,
            min_train_samples=args.wf_min_train_samples,
            min_test_samples=args.wf_min_test_samples,
            objective=args.wf_objective,
        )
        if folds_df.empty:
            raise ValueError('Walk-forward did not produce any valid fold. Adjust wf windows or sample thresholds.')

        folds_df.to_csv(args.out_csv, index=False)
        print(f'\nSaved walk-forward folds to: {args.out_csv}')

        total_oos_filled = folds_df['oos_filled_trades'].sum()
        weighted_avg_return = np.average(
            folds_df['oos_avg_return_pct'],
            weights=np.maximum(folds_df['oos_filled_trades'], 1),
        )
        total_tp = folds_df['oos_tp_hits'].sum()
        total_sl = folds_df['oos_sl_hits'].sum()
        total_time = folds_df['oos_time_exits'].sum()
        total_closed = total_tp + total_sl + total_time
        oos_win_rate = (total_tp / total_closed * 100) if total_closed > 0 else np.nan

        print('\nWalk-forward summary (OOS):')
        print(f'>> folds: {len(folds_df)}')
        print(f'>> total OOS filled trades: {int(total_oos_filled)}')
        print(f'>> weighted OOS avg return (%): {weighted_avg_return:.4f}')
        print(f'>> aggregate OOS win rate (%): {oos_win_rate:.2f}')
        print(f"\nTotal OOS return (%): {folds_df['oos_total_return_pct'].sum():.2f}")

        if not oos_trade_log.empty:
            oos_curve = build_equity_curve(oos_trade_log[['timestamp', 'ret', 'reason']], initial_equity=1.0)
            oos_curve.to_csv(args.equity_out_csv, index=False)
            print(f"Saved OOS equity curve to: {args.equity_out_csv}")
            print(f"OOS Max Drawdown (%): {oos_curve['drawdown_pct'].min():.2f}")
            print(f"OOS Final Equity: {oos_curve['equity'].iloc[-1]:.4f}")

        print('\nTop folds by OOS avg return:')
        print(
            folds_df.sort_values('oos_avg_return_pct', ascending=False)
            .head(10)[[
                'fold', 'selected_tp_level', 'selected_entry_pullback_pct',
                'oos_filled_trades', 'oos_win_rate_pct', 'oos_avg_return_pct', 'oos_profit_factor'
            ]]
            .to_string(index=False)
        )
    else:
        stats_df = evaluate_grid(candidates_df, tp_levels, entry_pullbacks, args.min_filled)
        stats_df = stats_df.sort_values(['passes_min_filled', 'avg_return_pct', 'win_rate_pct'], ascending=[False, False, False])
        stats_df.to_csv(args.out_csv, index=False)

        print(f'\nSaved full stats to: {args.out_csv}')
        print('\nTop configs (min_filled filtered):')
        top_df = stats_df[stats_df['passes_min_filled']].head(15)
        if top_df.empty:
            top_df = stats_df.head(15)
        print(top_df[['tp_level', 'entry_pullback_pct', 'filled_trades', 'fill_rate_valid_rr_pct', 'win_rate_pct', 'expectancy_pct', 'avg_return_pct', 'max_drawdown_pct', 'profit_factor', 'avg_rr']].to_string(index=False))

        best_cfg = select_best_config(top_df if not top_df.empty else stats_df, 'avg_return_pct')
        if best_cfg is not None:
            best_metrics, best_trade_log = evaluate_combo(
                candidates_df,
                tp_level=float(best_cfg['tp_level']),
                entry_pullback=float(best_cfg['entry_pullback_pct']) / 100.0,
                min_filled=args.min_filled,
                return_trade_log=True,
            )
            if not best_trade_log.empty:
                best_curve = build_equity_curve(best_trade_log[['timestamp', 'ret', 'reason']], initial_equity=1.0)
                best_curve.to_csv(args.equity_out_csv, index=False)
                print(f"Saved best-config equity curve to: {args.equity_out_csv}")
                print(f"Best config compounded return (%): {best_metrics['compounded_return_pct']:.2f}")
                print(f"Best config max drawdown (%): {best_metrics['max_drawdown_pct']:.2f}")

        focus = stats_df[
            (stats_df['tp_level'].round(4) == 1.6)
            & (stats_df['entry_pullback_pct'].round(4).isin([0.0, 2.0]))
        ]
        if not focus.empty:
            print('\nFocus on TP=1.6 and entry pullback 0%/2%:')
            print(focus[['tp_level', 'entry_pullback_pct', 'filled_trades', 'fill_rate_valid_rr_pct', 'win_rate_pct', 'expectancy_pct', 'avg_return_pct', 'max_drawdown_pct', 'profit_factor', 'avg_rr']].to_string(index=False))


if __name__ == '__main__':
    main()
