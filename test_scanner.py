import json
from ml.scanner import SmartScanner
from ccxt_data_processor import CCXTDataProcessor
from bot.config import BotConfig

with open("bot_config.json", "r") as f:
    config_dict = json.load(f)

config = BotConfig(**config_dict)

processor = CCXTDataProcessor(
    exchange_id="bitget",
    api_key=config.exchange.api_key,
    api_secret=config.exchange.api_secret,
    password=config.exchange.password,
    use_futures=True
)

scanner = SmartScanner(config=config, data_processor=processor)

# Let's test a couple symbols
test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
print("Scanning symbols:", test_symbols)

import builtins
# Override scanner print to actually print
scanner._print = builtins.print

import sys
def debug_print(*args, **kwargs):
    print("[Scanner Debug]", *args, **kwargs)
    
scanner.scan_debug = True

# Monkey patch scanner to enable its internal commented debugs
import re
with open("ml/scanner.py", "r", encoding="utf-8") as f:
    code = f.read()

# Make the test use the modified scanner code dynamically
import ast
import types

code = code.replace("# print(f\"[Scanner Debug]", "print(f\"[Scanner Debug]")
code = code.replace("__file__", "'ml/scanner.py'")
code = code.replace("if df.empty or len(df) < 50: continue", "if df.empty or len(df) < 50: continue\n                if symbol == 'BTCUSDT': print('DF TAIL:', df.tail())")
module = types.ModuleType("hacked_scanner")
exec(code, module.__dict__)

scanner = module.SmartScanner(config=config, data_processor=processor)

signals = scanner.scan(test_symbols, timeframe="12h", lookback_days=10)

print(f"Found {len(signals)} signals!")
for sig in signals:
    print(sig)
