import os
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

# Default config path
CONFIG_PATH = Path("bot_config.json")

class ExchangeConfig(BaseModel):
    name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    leverage: int = 20
    dry_run: bool = True  # Safety first! Default to simulation mode
    slippage: float = 0.0005  # 0.05% slippage
    margin_mode: str = "ISOLATED"  # ISOLATED or CROSS

class RiskConfig(BaseModel):
    max_open_positions: int = 5
    max_risk_per_trade: float = 0.02  # 2% of account per trade
    max_concentration: float = 0.20   # Max 20% of account in one coin
    use_kelly: bool = True
    kelly_fraction: float = 0.5

class StrategyConfig(BaseModel):
    timeframes: List[str] = ["1d"]
    entry_threshold: float = 0.65
    min_volume_usdt: float = 5.0  # Reduced to 5$ per user request
    cooldown_candles: int = 3
    require_fresh_crossover: bool = True
    min_rr_ratio: float = 1.0

class TelegramConfig(BaseModel):
    enabled: bool = False
    token: str = ""
    chat_id: str = ""

class BotConfig(BaseModel):
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    coins: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    use_all_symbols: bool = True # If True, fetches all liquid pairs
    max_symbols: int = 0 # 0 for all acceptable pairs
    
    def save(self, path: Path = CONFIG_PATH):
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "BotConfig":
        if not path.exists():
            # Create default if not exists
            cfg = cls()
            cfg.save(path)
            return cfg
        
        with open(path, "r") as f:
            data = json.load(f)
            return cls(**data)
