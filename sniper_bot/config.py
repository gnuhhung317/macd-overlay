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
    leverage: int = 1
    dry_run: bool = True
    margin_mode: str = "ISOLATED"
    use_testnet: bool = False

class RiskConfig(BaseModel):
    max_open_positions: int = 5
    max_risk_per_trade: float = 0.02
    max_position_size_usd: float = 10000.0

class StrategyConfig(BaseModel):
    timeframes: List[str] = ["1h"]
    entry_threshold: float = 0.6
    min_volume_usdt: float = 10000.0
    timeout_candles: int = 48
    
    # Sniper ATR Multipliers
    sl_atr_multiplier_long: float = 1.0
    tp_atr_multiplier_long: float = 2.0
    sl_atr_multiplier_short: float = 1.5
    tp_atr_multiplier_short: float = 2.5
    
    # Entry Lùi (ATR Offset)
    long_atr_offset: float = -0.1
    short_atr_offset: float = 0.5
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
