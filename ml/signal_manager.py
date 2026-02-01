
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

SIGNAL_CACHE_FILE = Path(__file__).parent / "data" / "cached_signals.json"

class SignalManager:
    def __init__(self):
        self.cache_dir = SIGNAL_CACHE_FILE.parent
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.signals = self._load()

    def _load(self):
        if SIGNAL_CACHE_FILE.exists():
            try:
                with open(SIGNAL_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    print(f"[SignalManager] Loaded {len(data)} timeframes from {SIGNAL_CACHE_FILE}")
                    return data
            except Exception as e:
                print(f"[SignalManager] Error loading cache: {e}")
                return {}
        print(f"[SignalManager] Cache file not found at {SIGNAL_CACHE_FILE}")
        return {}

    def save(self, timeframe, new_signals):
        """
        Save signals for a specific timeframe.
        new_signals: list of dicts [{ symbol, type, timestamp, price, confidence, sl, tp }, ...]
        """
        # Convert timestamps to string for JSON
        for s in new_signals:
            if isinstance(s['timestamp'], (datetime, pd.Timestamp)):
                s['timestamp'] = s['timestamp'].isoformat()
        
        self.signals[timeframe] = new_signals
        
        try:
            with open(SIGNAL_CACHE_FILE, 'w') as f:
                json.dump(self.signals, f, indent=2)
            print(f"[SignalManager] Saved {len(new_signals)} signals for {timeframe} to disk.")
        except Exception as e:
            print(f"[SignalManager] Error saving to disk: {e}")

    def get_signals(self, timeframe):
        return self.signals.get(timeframe, [])

    def clear(self, timeframe=None):
        if timeframe:
            if timeframe in self.signals:
                del self.signals[timeframe]
        else:
            self.signals = {}
        
        with open(SIGNAL_CACHE_FILE, 'w') as f:
            json.dump(self.signals, f, indent=2)
