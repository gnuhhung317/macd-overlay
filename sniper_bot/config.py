import os
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

# Default config path
CONFIG_PATH = Path("sniper_bot_config.json")

class ExchangeConfig(BaseModel):
    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    password: str = "" # Used by some exchanges via CCXT
    leverage: int = 1
    dry_run: bool = True
    slippage: float = 0.0005 # 0.05% default
    margin_mode: str = "ISOLATED"
    use_testnet: bool = False

class RiskConfig(BaseModel):
    max_open_positions: int = 5
    max_risk_per_trade: float = 0.02
    max_position_size_usd: float = 10000.0

class StrategyConfig(BaseModel):
    timeframes: List[str] = ["1h"]
    profile_path: str = "ml/p3_edge_research/experiments/auto_038_live_test.json"
    profile_name: str = "auto_038_live"
    selector_artifact_path: str = "output/selector_artifacts/auto_038_selector_fullasset.joblib"
    selector_threshold_override: float = -1.0
    selector_lookback_days: int = 450
    selector_batch_predict: bool = True
    incremental_scan: bool = True
    incremental_refresh_days: int = 7
    scan_history_bars: int = 1200
    progress_detail_log_path: str = "logs/sniper_scan_progress.log"
    min_volume_usdt: float = 10000.0
    timeout_candles: int = 48

    # Still used by PositionManager to cancel stale limit orders.
    limit_wait_bars: int = 5

class TelegramConfig(BaseModel):
    enabled: bool = False
    token: str = ""
    chat_id: str = ""

class SniperBotConfig(BaseModel):
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    coins: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    use_all_symbols: bool = True
    max_symbols: int = 0
    
    def save(self, path: Path = CONFIG_PATH):
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "SniperBotConfig":
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        
        with open(path, "r") as f:
            data = json.load(f)
            return cls(**data)
