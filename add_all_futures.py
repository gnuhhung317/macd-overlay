#!/usr/bin/env python3
"""
Script to add all Binance Futures symbols to monitor_config.json with 1d interval
Only includes symbols that have been listed for at least 6 months
"""
import json
import requests
from typing import List, Dict
from datetime import datetime, timedelta

def get_binance_futures_symbols(min_age_months: int = 6) -> List[str]:
    """Fetch all USDT futures symbols from Binance that are at least min_age_months old"""
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        symbols = []
        current_time = datetime.now()
        min_age_ms = int((current_time - timedelta(days=min_age_months * 30)).timestamp() * 1000)
        
        print(f"Filtering coins listed before: {datetime.fromtimestamp(min_age_ms / 1000).strftime('%Y-%m-%d')}")
        
        for symbol_info in data['symbols']:
            symbol = symbol_info['symbol']
            # Only add USDT perpetual futures that are TRADING
            if not (symbol.endswith('USDT') and symbol_info['status'] == 'TRADING'):
                continue
            
            # Check onboardDate (listing date)
            onboard_date = symbol_info.get('onboardDate')
            if onboard_date and onboard_date <= min_age_ms:
                symbols.append(symbol)
                age_days = (current_time.timestamp() * 1000 - onboard_date) / (1000 * 86400)
                print(f"  ✓ {symbol}: Listed {int(age_days)} days ago")
            elif onboard_date:
                age_days = (current_time.timestamp() * 1000 - onboard_date) / (1000 * 86400)
                print(f"  ✗ {symbol}: Only {int(age_days)} days old (too new)")
        
        print(f"\nFound {len(symbols)} symbols older than {min_age_months} months")
        return sorted(symbols)
    
    except Exception as e:
        print(f"Error fetching Binance futures: {e}")
        return []

def load_config(config_path: str = "monitor_config.json") -> Dict:
    """Load existing monitor config"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def add_futures_to_config(config: Dict, symbols: List[str], interval: str = "1d") -> Dict:
    """Add all futures symbols to config with specified interval"""
    if not config or 'coins' not in config:
        print("Invalid config structure")
        return config
    
    # Get existing symbol-interval combinations
    existing = set()
    for coin in config['coins']:
        existing.add((coin['symbol'], coin['interval']))
    
    # Add new symbols
    added = 0
    for symbol in symbols:
        if (symbol, interval) not in existing:
            config['coins'].append({
                "symbol": symbol,
                "interval": interval,
                "enabled": True
            })
            added += 1
    
    print(f"Added {added} new symbols with interval {interval}")
    print(f"Total coins in config: {len(config['coins'])}")
    return config

def save_config(config: Dict, config_path: str = "monitor_config.json") -> bool:
    """Save config to file"""
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Config saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def main():
    """Main function"""
    config_path = "monitor_config.json"
    interval = "1d"
    
    print("=" * 60)
    print("Adding all Binance Futures to monitor config")
    print("=" * 60)
    
    # Step 1: Get futures symbols
    print("\n[1/4] Fetching Binance Futures symbols...")
    symbols = get_binance_futures_symbols()
    if not symbols:
        print("No symbols found. Exiting.")
        return
    
    # Step 2: Load config
    print(f"\n[2/4] Loading config from {config_path}...")
    config = load_config(config_path)
    if not config:
        print("Failed to load config. Exiting.")
        return
    
    # Step 3: Add symbols
    print(f"\n[3/4] Adding symbols with interval '{interval}'...")
    config = add_futures_to_config(config, symbols, interval)
    
    # Step 4: Save config
    print(f"\n[4/4] Saving updated config...")
    if save_config(config, config_path):
        print("\n✓ Successfully updated monitor_config.json")
        print(f"✓ Total monitoring pairs: {len(config['coins'])}")
    else:
        print("\n✗ Failed to save config")

if __name__ == "__main__":
    main()
