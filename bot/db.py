import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

DB_PATH = Path("data/bot_data.db")

class DatabaseManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize database schema"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                status TEXT NOT NULL,
                entry_price REAL,
                exit_price REAL,
                sl_price REAL,
                tp_price REAL,
                size REAL,
                pnl REAL,
                leverage INTEGER,
                exit_reason TEXT,
                entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                exit_time TIMESTAMP,
                raw_data TEXT
            )
            """)
            
            # Signals log table (for auditing/debugging ML performance)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                sl_pct REAL,
                tp_pct REAL,
                action TEXT,
                raw_data TEXT
            )
            """)
            
            conn.commit()

    def add_trade(self, trade_data: Dict[str, Any]) -> int:
        """Add a new trade to the database"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO trades (symbol, direction, status, entry_price, sl_price, tp_price, size, leverage, raw_data, entry_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['symbol'],
                trade_data['direction'],
                trade_data['status'],
                float(trade_data.get('entry_price')) if trade_data.get('entry_price') is not None else None,
                float(trade_data.get('sl_price')) if trade_data.get('sl_price') is not None else None,
                float(trade_data.get('tp_price')) if trade_data.get('tp_price') is not None else None,
                float(trade_data.get('size')) if trade_data.get('size') is not None else None,
                int(trade_data.get('leverage')) if trade_data.get('leverage') is not None else None,
                json.dumps(trade_data.get('raw_data', {}), cls=NumpyEncoder),
                trade_data.get('entry_time').isoformat() if isinstance(trade_data.get('entry_time'), (datetime, pd.Timestamp)) else trade_data.get('entry_time')
            ))
            return cursor.lastrowid

    def update_trade(self, trade_id: int, updates: Dict[str, Any]):
        """Update an existing trade"""
        if not updates:
            return
            
        columns = ", ".join([f"{k} = ?" for k in updates.keys()])
        
        # Serialize dicts/lists to JSON strings
        values = []
        for v in updates.values():
            if isinstance(v, (dict, list)):
                values.append(json.dumps(v, cls=NumpyEncoder))
            else:
                values.append(v)
                
        values.append(trade_id)
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE trades SET {columns} WHERE id = ?", values)

    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all open trades"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE status NOT IN ('CLOSED', 'CANCELED')")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def log_signal(self, signal_data: Dict[str, Any]):
        """Log a signal prediction"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if 'timestamp' in signal_data:
                cursor.execute("""
                INSERT INTO signals (symbol, timeframe, timestamp, confidence, sl_pct, tp_pct, action, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_data['symbol'],
                    signal_data['timeframe'],
                    signal_data['timestamp'],
                    signal_data.get('confidence'),
                    signal_data.get('sl_pct'),
                    signal_data.get('tp_pct'),
                    signal_data.get('action'),
                    json.dumps(signal_data.get('raw_data', {}), cls=NumpyEncoder)
                ))
            else:
                cursor.execute("""
                INSERT INTO signals (symbol, timeframe, confidence, sl_pct, tp_pct, action, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_data['symbol'],
                    signal_data['timeframe'],
                    signal_data.get('confidence'),
                    signal_data.get('sl_pct'),
                    signal_data.get('tp_pct'),
                    signal_data.get('action'),
                    json.dumps(signal_data.get('raw_data', {}), cls=NumpyEncoder)
                ))
            
    def get_last_signal(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Get the most recent signal for a symbol/timeframe"""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM signals 
            WHERE symbol = ? AND timeframe = ? 
            ORDER BY timestamp DESC LIMIT 1
            """, (symbol, timeframe))
            row = cursor.fetchone()
            return dict(row) if row else None

    def check_signal_exists(self, symbol: str, timeframe: str, timestamp: datetime) -> bool:
        """Check if a specific signal already exists to avoid duplicates"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # timestamp in DB is likely string, ensure format matches or use relaxed check
            # Best is to check symbol + timeframe and if we have a signal within small window?
            # Or exact match if we control the timestamp format. 
            # SmartScanner returns pandas Timestamp.
            
            ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(timestamp, (datetime, pd.Timestamp)) else str(timestamp)
            
            cursor.execute("""
            SELECT id FROM signals 
            WHERE symbol = ? AND timeframe = ? AND timestamp = ?
            """, (symbol, timeframe, ts_str))
            return cursor.fetchone() is not None

    def get_last_trade_exit(self, symbol: str) -> Optional[datetime]:
        """Get the exit time of the last closed trade for a symbol"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT exit_time FROM trades 
            WHERE symbol = ? AND status = 'CLOSED' 
            ORDER BY exit_time DESC LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            
            if row and row[0]:
                try:
                    # SQLite stores timestamps as strings usually, need to parse
                    # Format is usually "YYYY-MM-DD HH:MM:SS.ssssss" or similar
                    return datetime.fromisoformat(row[0]) if isinstance(row[0], str) else row[0]
                except ValueError:
                    # Fallback for simple date strings if isoformat fails
                    try:
                        return datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                    except:
                        return None
            return None
