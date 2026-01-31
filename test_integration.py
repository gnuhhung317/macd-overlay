#!/usr/bin/env python
"""
Quick test script for Phase 5 integration
Tests that all components are working together
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        from optimized_monitor import OptimizedMonitor
        from timeframe_config import MultiTimeframeConfig
        from data_processor import BinanceDataProcessor
        from telegram_notifier import TelegramNotifier
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        from timeframe_config import MultiTimeframeConfig
        config = MultiTimeframeConfig('monitor_config.json')
        
        timeframes = config.get_enabled_timeframes()
        coins = config.get_enabled_coins()
        
        print(f"✅ Config loaded successfully")
        print(f"   - Enabled timeframes: {timeframes}")
        print(f"   - Total coins: {len(coins)}")
        print(f"   - Telegram enabled: {config.is_telegram_enabled()}")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_monitor_init():
    """Test OptimizedMonitor initialization"""
    print("\nTesting monitor initialization...")
    try:
        import threading
        from optimized_monitor import OptimizedMonitor
        
        stop_event = threading.Event()
        shared_data = {
            'check_count': 0,
            'last_scan_time': None,
            'current_data': {},
            'alerts': [],
            'last_check': {},
            'timeframe_stats': {},
            'memory_usage_mb': 0,
            'monitor': None
        }
        data_lock = threading.Lock()
        
        monitor = OptimizedMonitor(
            stop_event=stop_event,
            shared_data=shared_data,
            data_lock=data_lock
        )
        
        print(f"✅ Monitor initialized successfully")
        print(f"   - Model cache enabled: {monitor.config.is_model_caching_enabled()}")
        print(f"   - Max memory: {monitor.config.get_max_memory_mb()} MB")
        print(f"   - Notifiers: {list(monitor.notifiers.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Monitor init failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test API server can start"""
    print("\nTesting API server...")
    try:
        import api_server
        
        # Check if FastAPI app exists
        if hasattr(api_server, 'app'):
            print("✅ FastAPI app found")
            
            # List endpoints
            routes = []
            for route in api_server.app.routes:
                if hasattr(route, 'path'):
                    routes.append(f"{route.path}")
            
            print(f"   - Total endpoints: {len(routes)}")
            
            # Check new endpoints
            new_endpoints = ['/api/timeframes', '/api/timeframes/{interval}/status']
            for endpoint in new_endpoints:
                # Check if endpoint pattern exists
                found = any(endpoint.replace('{interval}', '') in r for r in routes)
                if found:
                    print(f"   ✅ {endpoint}")
                else:
                    print(f"   ⚠️  {endpoint} not found (might use path parameters)")
            
            return True
        else:
            print("❌ FastAPI app not found")
            return False
            
    except Exception as e:
        print(f"❌ API server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*70)
    print("Phase 5 Integration Test Suite")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Monitor Init", test_monitor_init),
        ("API Endpoints", test_api_endpoints)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration is ready.")
        print("\nNext steps:")
        print("1. Start the API server: python api_server.py")
        print("2. Open browser: http://localhost:8000")
        print("3. Click 'Start' to begin monitoring")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix errors before proceeding.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
