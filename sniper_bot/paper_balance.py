import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List

from sniper_bot.config import SniperBotConfig
from bot.data_provider import DataProvider


def _load_trades(db_path: Path) -> List[Dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, symbol, direction, status, entry_price, size, pnl
            FROM trades
            """
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_paper_balance(config_path: Path, initial_balance: float) -> Dict:
    cfg = SniperBotConfig.load(config_path)

    db_path = Path("data") / "bot_data.db"
    if not db_path.exists():
        return {
            "initial_balance": initial_balance,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "equity": initial_balance,
            "realized_balance": initial_balance,
            "open_positions": 0,
            "closed_positions": 0,
            "missing_price_symbols": [],
            "db_path": str(db_path),
        }

    trades = _load_trades(db_path)

    closed = [t for t in trades if str(t.get("status", "")).upper() == "CLOSED"]
    open_or_pending = [
        t for t in trades if str(t.get("status", "")).upper() not in {"CLOSED", "CANCELED"}
    ]

    realized_pnl = sum(_safe_float(t.get("pnl"), 0.0) for t in closed)

    # Reuse live data provider to mark open local-paper trades to market.
    provider = DataProvider(cfg)

    unrealized_pnl = 0.0
    missing_price_symbols: List[str] = []

    for t in open_or_pending:
        symbol = str(t.get("symbol", ""))
        direction = str(t.get("direction", "")).upper()
        entry = _safe_float(t.get("entry_price"), 0.0)
        size = _safe_float(t.get("size"), 0.0)

        if not symbol or entry <= 0 or size <= 0:
            continue

        current = _safe_float(provider.get_current_price(symbol), 0.0)
        if current <= 0:
            missing_price_symbols.append(symbol)
            continue

        if direction == "LONG":
            pnl_pct = (current - entry) / entry
        else:
            pnl_pct = (entry - current) / entry

        unrealized_pnl += size * pnl_pct

    realized_balance = initial_balance + realized_pnl
    equity = realized_balance + unrealized_pnl

    return {
        "initial_balance": initial_balance,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "realized_balance": realized_balance,
        "equity": equity,
        "open_positions": len(open_or_pending),
        "closed_positions": len(closed),
        "missing_price_symbols": sorted(set(missing_price_symbols)),
        "db_path": str(db_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Show local-paper balance/equity")
    parser.add_argument(
        "--config-path",
        type=str,
        default="sniper_bot/sniper_bot_config.json",
        help="Path to sniper config JSON",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Initial virtual balance used by DryRunExecutor",
    )
    args = parser.parse_args()

    summary = calculate_paper_balance(
        config_path=Path(args.config_path),
        initial_balance=float(args.initial_balance),
    )

    print("[local-paper] ===== BALANCE SUMMARY =====")
    print(f"db_path={summary['db_path']}")
    print(f"initial_balance={summary['initial_balance']:.2f}")
    print(f"realized_pnl={summary['realized_pnl']:.2f}")
    print(f"unrealized_pnl={summary['unrealized_pnl']:.2f}")
    print(f"realized_balance={summary['realized_balance']:.2f}")
    print(f"equity={summary['equity']:.2f}")
    print(f"open_positions={summary['open_positions']}")
    print(f"closed_positions={summary['closed_positions']}")

    missing = summary["missing_price_symbols"]
    if missing:
        print(f"missing_price_symbols={','.join(missing)}")


if __name__ == "__main__":
    main()
