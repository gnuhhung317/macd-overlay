#!/usr/bin/env python3
"""
Market Breadth Risk System — Circuit Breaker Engine

Measures systemic risk by counting MACD Cross Down signals across all coins
on multiple timeframes (4H, 8H, 12H). When breadth exceeds thresholds,
triggers a Circuit Breaker: force-close all longs and sleep for N hours.

Anti Look-ahead Bias:
    A 4H candle closing at 04:00 is only available for decision-making
    from the NEXT period onward. We use shift(1) on the breadth timeline
    to ensure no future data leaks.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / 'bitget-data'
PROCESSED_DIR = DATA_DIR / 'processed'


@dataclass
class CircuitBreakerConfig:
    """Configuration for Circuit Breaker state machine."""
    # Trigger 1: Large-TF confluence (structural break)
    confluence_tf: str = '8h'           # '8h' or '12h'
    confluence_threshold: float = 0.35  # >= 35% of coins have MACD cross down

    # Trigger 2: Small-TF velocity (acceleration of selling)
    velocity_tf: str = '4h'
    velocity_lookback: int = 2          # Rolling window in TF bars
    velocity_threshold: float = 0.20   # Breadth increased >= 20% within lookback

    # Action
    sleep_duration_hours: int = 24      # Sleep duration after trigger


_GLOBAL_BREADTH_CACHE = {}

class BreadthEngine:
    """
    Computes Market Breadth timeline from multi-timeframe feature data.

    Market Breadth = fraction of coins showing MACD Cross Down at each timestamp.
    This measures systemic selling pressure across the entire market.
    """

    def __init__(self, config: CircuitBreakerConfig = None):
        self.config = config or CircuitBreakerConfig()
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def _load_tf_data(self, timeframe: str) -> pd.DataFrame:
        """Load and cache feature data for a timeframe."""
        if timeframe in self._data_cache:
            return self._data_cache[timeframe]

        path = PROCESSED_DIR / f'features_{timeframe}_full.parquet'
        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")

        df = pd.read_parquet(path, columns=['timestamp', 'symbol', 'macd_cross_down'])
        self._data_cache[timeframe] = df
        return df

    def _compute_breadth_for_tf(self, timeframe: str) -> pd.DataFrame:
        """
        Compute breadth (fraction of coins with MACD cross down) per timestamp
        for a given timeframe.

        Returns:
            DataFrame with columns ['timestamp', f'mb_{timeframe}', f'mb_{timeframe}_count']
        """
        if timeframe in _GLOBAL_BREADTH_CACHE:
            return _GLOBAL_BREADTH_CACHE[timeframe].copy()
            
        df = self._load_tf_data(timeframe)

        # Count total coins and cross_down coins per timestamp
        grouped = df.groupby('timestamp').agg(
            total_coins=('symbol', 'count'),
            cross_down_count=('macd_cross_down', 'sum')
        ).reset_index()

        col_name = f'mb_{timeframe}'
        grouped[col_name] = grouped['cross_down_count'] / grouped['total_coins']
        grouped[f'{col_name}_count'] = grouped['cross_down_count'].astype(int)

        result = grouped[['timestamp', col_name, f'{col_name}_count']].sort_values('timestamp')
        _GLOBAL_BREADTH_CACHE[timeframe] = result
        
        # Clear raw data cache to save memory since we only need aggregated data
        if timeframe in self._data_cache:
            del self._data_cache[timeframe]
            
        return result.copy()

    def _compute_velocity(self, breadth_series: pd.Series, lookback: int) -> pd.Series:
        """
        Compute velocity (acceleration) of breadth change.

        Velocity = current_breadth - breadth_N_bars_ago
        Positive velocity means breadth is increasing (more coins breaking down).
        """
        return breadth_series - breadth_series.shift(lookback)

    def build_breadth_timeline(self, base_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Build a breadth timeline aligned to the backtester's base timestamps.

        CRITICAL: Anti look-ahead bias implementation.
        A 4H candle at 04:00 only becomes "known" after it closes.
        We use merge_asof with backward direction to ensure we only use
        completed candles.

        Args:
            base_timestamps: Sorted timestamps from the backtester's main loop
                           (typically 1D at 00:00, or whatever TF the backtest runs on)

        Returns:
            DataFrame indexed by base_timestamps with columns:
            - mb_4h: Market breadth on 4H (fraction of coins with MACD cross down)
            - mb_8h: Market breadth on 8H
            - mb_12h: Market breadth on 12H
            - mb_4h_velocity: Rate of change of 4H breadth
        """
        # Determine which timeframes we need
        needed_tfs = set()
        needed_tfs.add(self.config.velocity_tf)     # e.g., '4h'
        needed_tfs.add(self.config.confluence_tf)    # e.g., '8h'
        # Always compute all 3 for visibility
        for tf in ['4h', '8h', '12h']:
            needed_tfs.add(tf)

        # Build base DataFrame
        result = pd.DataFrame({'timestamp': base_timestamps}).sort_values('timestamp')

        for tf in sorted(needed_tfs):
            try:
                breadth_df = self._compute_breadth_for_tf(tf)
            except FileNotFoundError:
                print(f"  ⚠️ Breadth data not available for {tf}, skipping")
                result[f'mb_{tf}'] = 0.0
                result[f'mb_{tf}_count'] = 0
                continue

            # ANTI LOOK-AHEAD BIAS:
            # Shift breadth timestamps FORWARD by 1 period of the TF.
            # This means a 4H candle closing at 04:00 only becomes available
            # at 08:00 (next 4H period). For the daily backtester (00:00 timestamps),
            # only candles that closed strictly BEFORE 00:00 are used.
            tf_hours = {'4h': 4, '8h': 8, '12h': 12, '1d': 24}
            shift_hours = tf_hours.get(tf, 4)
            breadth_df = breadth_df.copy()
            breadth_df['timestamp'] = breadth_df['timestamp'] + pd.Timedelta(hours=shift_hours)

            # merge_asof: for each base timestamp, find the most recent breadth
            # value that is <= that timestamp (backward lookup)
            result = pd.merge_asof(
                result,
                breadth_df,
                on='timestamp',
                direction='backward'
            )

        # Fill NaN (early periods with no data)
        for tf in needed_tfs:
            col = f'mb_{tf}'
            if col in result.columns:
                result[col] = result[col].fillna(0.0)
            count_col = f'mb_{tf}_count'
            if count_col in result.columns:
                result[count_col] = result[count_col].fillna(0).astype(int)

        # Compute velocity for the velocity timeframe
        vel_col = f'mb_{self.config.velocity_tf}'
        if vel_col in result.columns:
            result[f'{vel_col}_velocity'] = self._compute_velocity(
                result[vel_col],
                self.config.velocity_lookback
            ).fillna(0.0)
        else:
            result[f'{vel_col}_velocity'] = 0.0

        return result.set_index('timestamp')

    def check_trigger(self, breadth_row: pd.Series) -> tuple:
        """
        Check if Circuit Breaker should trigger at current timestamp.

        Args:
            breadth_row: A single row from the breadth timeline

        Returns:
            (should_trigger: bool, reason: str)
        """
        conf_col = f'mb_{self.config.confluence_tf}'
        vel_col = f'mb_{self.config.velocity_tf}_velocity'

        trigger_confluence = False
        trigger_velocity = False
        reasons = []

        if conf_col in breadth_row.index:
            conf_value = breadth_row[conf_col]
            if conf_value >= self.config.confluence_threshold:
                trigger_confluence = True
                reasons.append(
                    f"Confluence({self.config.confluence_tf}): "
                    f"{conf_value:.1%} >= {self.config.confluence_threshold:.1%}"
                )

        if vel_col in breadth_row.index:
            vel_value = breadth_row[vel_col]
            if vel_value >= self.config.velocity_threshold:
                trigger_velocity = True
                reasons.append(
                    f"Velocity({self.config.velocity_tf}): "
                    f"+{vel_value:.1%} >= {self.config.velocity_threshold:.1%}"
                )

        should_trigger = trigger_confluence or trigger_velocity
        reason = " | ".join(reasons) if reasons else ""

        return should_trigger, reason


def test_breadth_engine():
    """Quick sanity test for the breadth engine."""
    import sys

    config = CircuitBreakerConfig()
    engine = BreadthEngine(config)

    # Load base timestamps from 1D data
    df_1d = pd.read_parquet(PROCESSED_DIR / 'features_1d_full.parquet',
                            columns=['timestamp'])
    base_ts = pd.DatetimeIndex(sorted(df_1d['timestamp'].unique()))

    # Use last 6 months
    cutoff = base_ts.max() - pd.DateOffset(months=6)
    base_ts = base_ts[base_ts >= cutoff]

    print(f"Building breadth timeline for {len(base_ts)} timestamps...")
    timeline = engine.build_breadth_timeline(base_ts)

    print(f"\nTimeline shape: {timeline.shape}")
    print(f"Columns: {list(timeline.columns)}")
    print(f"\nSample (last 10 rows):")
    print(timeline.tail(10).to_string())

    # Check triggers
    triggers = []
    for ts, row in timeline.iterrows():
        triggered, reason = engine.check_trigger(row)
        if triggered:
            triggers.append((ts, reason))

    print(f"\n🚨 Circuit Breaker triggered {len(triggers)} times in test period:")
    for ts, reason in triggers[:20]:
        print(f"  {ts}: {reason}")

    if len(triggers) > 20:
        print(f"  ... and {len(triggers) - 20} more")

    # Stats
    for col in ['mb_4h', 'mb_8h', 'mb_12h']:
        if col in timeline.columns:
            vals = timeline[col]
            print(f"\n{col}: min={vals.min():.3f}, mean={vals.mean():.3f}, "
                  f"max={vals.max():.3f}, std={vals.std():.3f}")


if __name__ == '__main__':
    test_breadth_engine()
