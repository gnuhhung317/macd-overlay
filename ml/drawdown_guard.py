#!/usr/bin/env python3
"""
Drawdown Guard: Decision Tree Risk Management Filters (Exogenous Only)
Implements pure market-based rules to prevent entries during high-risk regimes.
Removed all bot-internal state (PnL, Position Counts) as requested.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

class DrawdownGuard:
    """
    Enforces risk management rules based ONLY on exogenous market data (BTC).
    No portfolio-internal state (PnL, equity volatility, or position counts).
    """
    
    def __init__(self, timeframe: str = '1d'):
        self.timeframe = timeframe
        self.btc_data: Optional[pd.DataFrame] = None
        
    def load_btc_data(self, df_full: pd.DataFrame):
        """Extract BTC features from the full dataset for real-time lookup."""
        # BTCUSDT rows
        btc = df_full[df_full['symbol'] == 'BTCUSDT'].copy()
        
        if btc.empty:
            # Fallback to loading from processed parquet if not in df_full
            data_dir = Path(__file__).parent.parent / 'bitget-data' / 'processed'
            path = data_dir / f'features_{self.timeframe}_full.parquet'
            if path.exists():
                df = pd.read_parquet(path)
                btc = df[df['symbol'] == 'BTCUSDT'].copy()
        
        if not btc.empty:
            # Set index to timestamp for fast lookup
            # Ensure index is unique and sorted
            self.btc_data = btc.groupby('timestamp').first().sort_index()

    def is_safe(self, current_time: pd.Timestamp) -> tuple:
        """
        Check if market conditions are safe based on exogenous BTC indicators.
        Returns (is_safe: bool, reason: str)
        """
        if self.btc_data is None:
            return True, "No BTC data loaded"
            
        # Find closest match if exact time not found (backtester might use different base timestamps)
        if current_time not in self.btc_data.index:
            # Try to find the nearest previous timestamp
            past_data = self.btc_data.index[self.btc_data.index <= current_time]
            if len(past_data) == 0:
                return True, "No historical BTC data"
            lookup_time = past_data[-1]
        else:
            lookup_time = current_time
            
        btc_row = self.btc_data.loc[lookup_time]
        
        # Exogenous features
        btc_rsi = btc_row.get('btc_rsi', 50)
        btc_adx = btc_row.get('btc_adx', 25)
        btc_chop = btc_row.get('btc_chop', 50)
        btc_atr_ratio = btc_row.get('btc_atr_ratio', 0.04)
        
        # ── 1D REGIME ─────────────────────────────────────────────
        if self.timeframe == '1d':
            if btc_chop > 44.71:
                if btc_atr_ratio <= 0.03 and btc_adx > 17.26:
                    return False, f"1D: Choppy ({btc_chop:.1f}) & Rising ADX ({btc_adx:.1f})"
                if btc_atr_ratio > 0.05:
                    return False, f"1D: High Volatility (ATR Ratio {btc_atr_ratio:.3f})"
            elif btc_rsi > 70:
                return False, "1D: BTC Overbought (RSI > 70)"
            
        # ── 12H REGIME ────────────────────────────────────────────
        elif self.timeframe == '12h':
            if btc_rsi > 58.04:
                if btc_atr_ratio > 0.08:
                    return False, f"12H: Extreme Volatility (ATR Ratio {btc_atr_ratio:.3f})"
            else:
                if btc_chop > 50:
                    return False, f"12H: Choppy Downtrend (RSI {btc_rsi:.1f}, CHOP {btc_chop:.1f})"

        # ── 8H REGIME ─────────────────────────────────────────────
        elif self.timeframe == '8h':
            if btc_chop > 48.2 and btc_atr_ratio > 0.045:
                return False, f"8H: High CHOP ({btc_chop:.1f}) & High ATR ({btc_atr_ratio:.3f})"
            if btc_adx > 71.22:
                return False, f"8H: Trend Blowoff (ADX {btc_adx:.1f})"

        return True, "Safe"
