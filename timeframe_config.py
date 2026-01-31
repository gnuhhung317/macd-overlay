"""
Multi-Timeframe Configuration Manager
Handles loading and accessing timeframe-specific configurations
"""

import json
import os
from typing import Dict, List, Optional


class MultiTimeframeConfig:
    """
    Configuration manager for multi-timeframe monitoring
    
    Features:
    - Load timeframe-specific settings
    - Get enabled timeframes
    - Access telegram chat IDs per timeframe
    - Manage global settings
    """
    
    def __init__(self, config_path='monitor_config.json'):
        """
        Initialize config manager
        
        Args:
            config_path: Path to config JSON file
        """
        self.config_path = config_path
        self.config = self.load_config()
        self._validate_config()
    
    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def _validate_config(self):
        """Validate configuration structure"""
        required_keys = ['telegram_token', 'timeframes', 'coins']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate timeframes
        for interval, tf_config in self.config['timeframes'].items():
            required_tf_keys = ['enabled', 'telegram_chat_id', 'scan_interval', 
                              'models_dir', 'entry_threshold']
            for key in required_tf_keys:
                if key not in tf_config:
                    raise ValueError(f"Missing key '{key}' in timeframe '{interval}'")
    
    def get_timeframe_config(self, interval: str) -> Optional[Dict]:
        """
        Get configuration for specific timeframe
        
        Args:
            interval: Timeframe interval (e.g., '4h', '1d')
            
        Returns:
            Dict with timeframe config or None if not found
        """
        return self.config['timeframes'].get(interval)
    
    def get_enabled_timeframes(self) -> List[str]:
        """
        Get list of enabled timeframes
        
        Returns:
            List of interval strings (e.g., ['4h', '1d'])
        """
        return [
            interval 
            for interval, cfg in self.config['timeframes'].items() 
            if cfg.get('enabled', False)
        ]
    
    def get_telegram_token(self) -> str:
        """Get telegram bot token"""
        return self.config.get('telegram_token', '')
    
    def get_telegram_chat_id(self, interval: str) -> Optional[str]:
        """
        Get telegram chat ID for specific timeframe
        
        Args:
            interval: Timeframe interval
            
        Returns:
            Chat ID string or None
        """
        tf_config = self.get_timeframe_config(interval)
        return tf_config.get('telegram_chat_id') if tf_config else None
    
    def is_telegram_enabled(self) -> bool:
        """Check if telegram notifications are enabled globally"""
        return self.config.get('telegram_enabled', False)
    
    def get_coins(self) -> List[Dict]:
        """
        Get list of all coins
        
        Returns:
            List of coin dicts with 'symbol' and 'enabled' keys
        """
        return self.config.get('coins', [])
    
    def get_enabled_coins(self) -> List[str]:
        """
        Get list of enabled coin symbols
        
        Returns:
            List of symbol strings
        """
        return [
            coin['symbol'] 
            for coin in self.config.get('coins', []) 
            if coin.get('enabled', False)
        ]
    
    def get_global_settings(self) -> Dict:
        """Get global settings dict"""
        return self.config.get('global_settings', {})
    
    def get_base_scan_interval(self) -> int:
        """Get base scan loop interval in seconds"""
        return self.get_global_settings().get('base_scan_interval', 300)
    
    def get_model_cache_ttl(self) -> int:
        """Get model cache TTL in seconds"""
        return self.get_global_settings().get('model_cache_ttl', 3600)
    
    def is_model_caching_enabled(self) -> bool:
        """Check if model caching is enabled"""
        return self.get_global_settings().get('enable_model_caching', True)
    
    def get_max_memory_mb(self) -> int:
        """Get max memory threshold in MB"""
        return self.get_global_settings().get('max_memory_mb', 1000)
    
    def update_timeframe_enabled(self, interval: str, enabled: bool):
        """
        Enable or disable a timeframe
        
        Args:
            interval: Timeframe interval
            enabled: True to enable, False to disable
        """
        if interval in self.config['timeframes']:
            self.config['timeframes'][interval]['enabled'] = enabled
            self.save_config()
    
    def update_telegram_chat_id(self, interval: str, chat_id: str):
        """
        Update telegram chat ID for a timeframe
        
        Args:
            interval: Timeframe interval
            chat_id: New chat ID
        """
        if interval in self.config['timeframes']:
            self.config['timeframes'][interval]['telegram_chat_id'] = chat_id
            self.save_config()
    
    def get_priority_order(self) -> List[str]:
        """
        Get timeframes in priority order for scanning
        
        Returns:
            List of intervals in priority order
        """
        # Priority: 4h > 1d > 8h > 12h > 1h
        priority = ['4h', '1d', '8h', '12h', '1h']
        enabled = self.get_enabled_timeframes()
        return [tf for tf in priority if tf in enabled]
    
    def __repr__(self):
        enabled_tfs = self.get_enabled_timeframes()
        return f"<MultiTimeframeConfig: {len(enabled_tfs)} enabled timeframes, {len(self.get_enabled_coins())} coins>"


if __name__ == '__main__':
    # Test config manager
    config = MultiTimeframeConfig()
    
    print("=" * 70)
    print("MULTI-TIMEFRAME CONFIGURATION")
    print("=" * 70)
    
    print(f"\n📊 Enabled Timeframes: {config.get_enabled_timeframes()}")
    print(f"📈 Priority Order: {config.get_priority_order()}")
    print(f"💰 Total Coins: {len(config.get_coins())}")
    print(f"✅ Enabled Coins: {len(config.get_enabled_coins())}")
    
    print("\n📱 Telegram Configuration:")
    print(f"  Token: {config.get_telegram_token()[:20]}...")
    print(f"  Enabled: {config.is_telegram_enabled()}")
    
    print("\n⚙️  Timeframe Details:")
    for interval in config.get_priority_order():
        tf_cfg = config.get_timeframe_config(interval)
        print(f"\n  {interval}:")
        print(f"    Scan Interval: {tf_cfg['scan_interval']}s ({tf_cfg['scan_interval']//60}min)")
        print(f"    Entry Threshold: {tf_cfg['entry_threshold']}")
        print(f"    Models Dir: {tf_cfg['models_dir']}")
        print(f"    Chat ID: {tf_cfg['telegram_chat_id']}")
        print(f"    Description: {tf_cfg.get('description', 'N/A')}")
    
    print("\n🌐 Global Settings:")
    global_settings = config.get_global_settings()
    for key, value in global_settings.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print(config)
    print("=" * 70)
