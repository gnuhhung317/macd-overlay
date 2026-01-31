"""
Script to migrate monitor_config.json to new multi-timeframe structure
"""

import json
import shutil
from datetime import datetime

# Backup original config
backup_file = f'monitor_config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
shutil.copy('monitor_config.json', backup_file)
print(f"✅ Backup created: {backup_file}")

# Load current config
with open('monitor_config.json', 'r', encoding='utf-8') as f:
    old_config = json.load(f)

# Extract coin list (remove interval field)
coins = [
    {
        'symbol': coin['symbol'],
        'enabled': coin['enabled']
    }
    for coin in old_config['coins']
]

# Create new config structure
new_config = {
    "telegram_token": old_config.get('telegram_token', ''),
    "telegram_enabled": old_config.get('telegram_enabled', True),
    
    "timeframes": {
        "4h": {
            "enabled": True,
            "telegram_chat_id": "-5113373606",  # Original channel for 4h
            "scan_interval": 7200,  # 2 hours in seconds
            "models_dir": "ml/models/4h",
            "entry_threshold": 0.55,
            "description": "4-hour timeframe - Primary timeframe with best Sharpe ratio"
        },
        "8h": {
            "enabled": True,
            "telegram_chat_id": "-5113373607",  # TODO: Create new channel
            "scan_interval": 14400,  # 4 hours in seconds
            "models_dir": "ml/models/8h",
            "entry_threshold": 0.55,
            "description": "8-hour timeframe"
        },
        "12h": {
            "enabled": True,
            "telegram_chat_id": "-5113373608",  # TODO: Create new channel
            "scan_interval": 21600,  # 6 hours in seconds
            "models_dir": "ml/models/12h",
            "entry_threshold": 0.55,
            "description": "12-hour timeframe"
        },
        "1d": {
            "enabled": True,
            "telegram_chat_id": "-5113373609",  # TODO: Create new channel
            "scan_interval": 43200,  # 12 hours in seconds
            "models_dir": "ml/models/1d",
            "entry_threshold": 0.6,
            "description": "Daily timeframe - Lower frequency, higher confidence"
        },
        "1h": {
            "enabled": False,
            "telegram_chat_id": "-5113373610",  # TODO: Create new channel
            "scan_interval": 3600,  # 1 hour in seconds
            "models_dir": "ml/models/4h",  # Reuse 4h models temporarily
            "entry_threshold": 0.5,
            "description": "1-hour timeframe - Disabled by default (needs trained models)"
        }
    },
    
    "coins": coins,
    
    "global_settings": {
        "base_scan_interval": 300,  # 5 minutes - base loop cycle
        "max_memory_mb": 1000,  # Auto cleanup if exceeds
        "model_cache_ttl": 3600,  # Cache models for 1 hour
        "enable_model_caching": True
    }
}

# Save new config
with open('monitor_config.json', 'w', encoding='utf-8') as f:
    json.dump(new_config, f, indent=2, ensure_ascii=False)

print(f"✅ Migrated {len(coins)} coins")
print(f"✅ Configured {len(new_config['timeframes'])} timeframes")
print("\n📋 Timeframe Configuration:")
for tf, cfg in new_config['timeframes'].items():
    status = "✅ ENABLED" if cfg['enabled'] else "❌ DISABLED"
    print(f"  {tf:4s}: {status} | Scan every {cfg['scan_interval']//60}min | Threshold: {cfg['entry_threshold']} | Chat: {cfg['telegram_chat_id']}")

print("\n⚠️  TODO: Create 4 new Telegram channels and update chat IDs:")
print("  - 8h:  Replace -5113373607")
print("  - 12h: Replace -5113373608")
print("  - 1d:  Replace -5113373609")
print("  - 1h:  Replace -5113373610 (if enabling)")

print(f"\n💾 Original config backed up to: {backup_file}")
print("✅ Migration complete!")
