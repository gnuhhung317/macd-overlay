from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import threading
import time
import json
import os
import math
from datetime import datetime
from zoneinfo import ZoneInfo
from data_processor import BinanceDataProcessor
from telegram_notifier import TelegramNotifier
from optimized_monitor import OptimizedMonitor
import pandas as pd

# ML Predictor (optional)
try:
    from ml.realtime_predictor import get_predictor
    ml_predictor = get_predictor(entry_threshold=0.4)
    print(f"[ML] Predictor loaded: {ml_predictor.is_loaded}")
except ImportError as e:
    ml_predictor = None
    print(f"[ML] Predictor not available: {e}")

APP_DIR = os.path.dirname(__file__)
CONFIG_FILE = os.path.join(APP_DIR, 'monitor_config.json')
SCAN_INTERVAL = 300  # seconds

app = FastAPI(title="MACD Monitor API")

def sanitize_json_value(obj):
    """Replace NaN and Infinity with None for JSON compatibility, convert numpy types and datetime"""
    import numpy as np
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    if isinstance(obj, dict):
        return {k: sanitize_json_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_value(v) for v in obj]
    elif isinstance(obj, datetime):
        # Convert datetime to ISO string
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=ZoneInfo('UTC'))
        return obj.astimezone(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

# Mount static files
static_dir = os.path.join(APP_DIR, 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get('/')
def root():
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail='index.html not found')

# Shared state
stop_event = threading.Event()
data_lock = threading.Lock()
processor = None
telegram = None

shared_data = {
    'check_count': 0,
    'last_scan_time': None,
    'current_data': {},
    'alerts': [],
    'last_check': {},
    'timeframe_stats': {},  # Per-timeframe statistics
    'memory_usage_mb': 0,
    'monitor': None  # OptimizedMonitor instance
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        'coins': [
            {'symbol': 'BTCUSDT', 'interval': '30m', 'enabled': True},
            {'symbol': 'ETHUSDT', 'interval': '30m', 'enabled': True}
        ],
        'telegram_enabled': False,
        'telegram_token': '',
        'telegram_chat_id': ''
    }


def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def setup_telegram_if_needed(config):
    global telegram
    if config.get('telegram_enabled') and config.get('telegram_token') and config.get('telegram_chat_id'):
        try:
            tg = TelegramNotifier(config['telegram_token'], config['telegram_chat_id'])
            if tg.test_connection():
                telegram = tg
                return True
        except Exception:
            telegram = None
            return False
    telegram = None
    return False


def setup_processor(config):
    """Instantiate or re-instantiate the BinanceDataProcessor with API keys from config."""
    global processor
    api_key = config.get('binance_api_key', '')
    api_secret = config.get('binance_api_secret', '')
    try:
        processor = BinanceDataProcessor(api_key=api_key, api_secret=api_secret, use_futures=True)
        return True
    except Exception as e:
        processor = BinanceDataProcessor(use_futures=True)
        print('[PROCESSOR] failed to init with keys, using public client:', e)
        return False


def get_futures_symbols():
    try:
        if processor is None:
            setup_processor(load_config())
        exchange_info = processor.client.futures_exchange_info()
        usdt_symbols = []
        for symbol_info in exchange_info['symbols']:
            if (symbol_info['symbol'].endswith('USDT') and symbol_info['status'] == 'TRADING' and symbol_info.get('contractType') == 'PERPETUAL'):
                usdt_symbols.append(symbol_info['symbol'])
        try:
            tickers = processor.client.futures_ticker()
            volume_map = {t['symbol']: float(t.get('quoteVolume', 0)) for t in tickers}
            usdt_symbols.sort(key=lambda s: volume_map.get(s, 0), reverse=True)
            usdt_symbols = usdt_symbols[:50]
        except Exception:
            usdt_symbols.sort()
        return usdt_symbols
    except Exception:
        return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']


def iso(dt):
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo('UTC'))
        return dt.astimezone(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat()
    return str(dt)


def check_coin(symbol, interval, config):
    try:
        if processor is None:
            # ensure we have a processor
            setup_processor(config)
        # Determine lookback period based on interval
        interval_to_lookback = {
            '1m': '6 hours ago UTC',
            '3m': '12 hours ago UTC',
            '5m': '1 day ago UTC',
            '15m': '2 days ago UTC',
            '30m': '3 days ago UTC',
            '1h': '5 days ago UTC',
            '2h': '10 days ago UTC',
            '4h': '20 days ago UTC',
            '6h': '30 days ago UTC',
            '8h': '40 days ago UTC',
            '12h': '60 days ago UTC',
            '1d': '90 days ago UTC',  # 90 days for daily chart
            '3d': '270 days ago UTC',
            '1w': '1 year ago UTC',
        }
        lookback_period = interval_to_lookback.get(interval, '5 days ago UTC')
        df = processor.get_historical_data(symbol, interval, lookback_period, 'now UTC')
        time.sleep(1)  # avoid rate limits
        
        # ENFORCE CLOSED CANDLE LOGIC:
        # Drop the last (forming) candle to prevent repaint/unstable signals
        if not df.empty:
            df = df.iloc[:-1].copy()
            
        if df.empty:
            return False
            
        df = processor.calculate_macd(df)
        current = {
            'price': float(df['close'].iloc[-1]) if not df.empty else None,
            'macd': float(df['macd'].iloc[-1]) if not df.empty else None,
            'signal': float(df['signal'].iloc[-1]) if not df.empty else None,
            'histogram': float(df['histogram'].iloc[-1]) if not df.empty else None,
            'timestamp': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')),
            'trend': 'BULLISH' if (not df.empty and df['macd'].iloc[-1] > df['signal'].iloc[-1]) else 'BEARISH',
            'has_new_alert': False
        }

        recent_crossovers = processor.detect_crossovers(df.tail(20))
        if recent_crossovers:
            latest = recent_crossovers[-1]
            ts = latest['timestamp']
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=ZoneInfo('UTC'))
            ts = ts.astimezone(ZoneInfo('Asia/Ho_Chi_Minh'))
            key = f"{symbol}_{interval}"
            with data_lock:
                last_alert = shared_data['last_check'].get(key)
            if last_alert is None or ts > last_alert:
                alert = {
                    'symbol': symbol,
                    'interval': interval,
                    'type': latest['type'],
                    'timestamp': ts,
                    'price': latest['price'],
                    'macd': latest['macd'],
                    'signal': latest['signal'],
                    'histogram': latest['histogram']
                }
                with data_lock:
                    shared_data['alerts'].insert(0, alert)
                    shared_data['last_check'][key] = ts
                
                # Get ML prediction if available
                ml_prediction = None
                features_df = None
                if ml_predictor is not None and ml_predictor.is_loaded:
                    try:
                        # Calculate features for ML prediction with dynamic parity
                        # Fetch funding rate
                        funding_rate = processor.get_current_funding_rate(symbol)
                        
                        features_df = ml_predictor.calculate_features(
                            df, 
                            timeframe=interval,
                            funding_rate=funding_rate
                        )
                        ml_prediction = ml_predictor.predict(features_df)
                        if ml_prediction:
                            print(f"[ML] {symbol}: confidence={ml_prediction['entry_confidence']:.1%}, "
                                  f"SL={ml_prediction['sl_pct']:.1%}, TP={ml_prediction['tp_pct']:.1%}")
                            alert['ml_prediction'] = ml_prediction
                    except Exception as e:
                        print(f"[ML] Prediction error for {symbol}: {e}")
                
                if telegram:
                    try:
                        # Pass features_df for ML prediction in telegram message
                        telegram.send_crossover_alert(alert, symbol, interval, features_df)
                    except Exception as e:
                        print(f"[TELEGRAM] Error: {e}")
                current['has_new_alert'] = True

        with data_lock:
            shared_data['current_data'][f"{symbol}_{interval}"] = current
        return True
    except Exception as e:
        with data_lock:
            shared_data['current_data'][f"{symbol}_{interval}"] = {'error': str(e), 'timestamp': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))}
        return False


def scan_coins_worker():
    config = load_config()
    enabled = [c for c in config['coins'] if c.get('enabled')]
    if not enabled:
        return
    with data_lock:
        shared_data['check_count'] += 1
    for coin in enabled:
        check_coin(coin['symbol'], coin['interval'], config)
    with data_lock:
        shared_data['last_scan_time'] = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh'))


def crawler_worker():
    print('[CRAWLER] Multi-timeframe monitor started')
    
    with data_lock:
        monitor = shared_data['monitor']
    
    if monitor is None:
        print('[CRAWLER] Monitor not initialized')
        return
    
    # Run the optimized monitor's main loop (runs continuously until stop_event)
    # Using run() instead of run_scan_cycle() to get continuous monitoring
    try:
        monitor.run()  # This runs the continuous loop with smart scheduling
    except Exception as e:
        print(f'[CRAWLER] Monitor error: {e}')
        import traceback
        traceback.print_exc()
    
    print('[CRAWLER] stopped')


crawler_thread = None


@app.post('/api/start')
def api_start():
    global crawler_thread
    config = load_config()
    setup_telegram_if_needed(config)
    setup_processor(config)
    
    # Initialize OptimizedMonitor
    with data_lock:
        if shared_data['monitor'] is None:
            try:
                shared_data['monitor'] = OptimizedMonitor(
                    stop_event=stop_event,
                    shared_data=shared_data,
                    data_lock=data_lock
                )
                print('[MONITOR] OptimizedMonitor initialized')
            except Exception as e:
                print(f'[MONITOR] Failed to initialize: {e}')
                return {'status': 'error', 'message': str(e)}
    
    if crawler_thread is None or not crawler_thread.is_alive():
        stop_event.clear()
        crawler_thread = threading.Thread(target=crawler_worker, daemon=True)
        crawler_thread.start()
        return {'status': 'started'}
    return {'status': 'already_running'}


@app.post('/api/stop')
def api_stop():
    stop_event.set()
    return {'status': 'stopping'}


@app.get('/api/status')
def api_status():
    with data_lock:
        data = {
            'check_count': shared_data['check_count'],
            'last_scan_time': iso(shared_data['last_scan_time']),
            'current_data': {},
            'alerts': [],
            'timeframe_stats': {},
            'memory_usage_mb': shared_data.get('memory_usage_mb', 0),
            'monitor_active': shared_data.get('monitor') is not None
        }
        
        # Convert timeframe_stats datetimes
        for interval, stats in shared_data.get('timeframe_stats', {}).items():
            stats_copy = stats.copy() if isinstance(stats, dict) else {}
            if 'last_scan' in stats_copy and isinstance(stats_copy['last_scan'], datetime):
                stats_copy['last_scan'] = iso(stats_copy['last_scan'])
            data['timeframe_stats'][interval] = stats_copy
        
        # convert current_data datetimes
        for k, v in shared_data['current_data'].items():
            entry = v.copy()
            if isinstance(entry.get('timestamp'), datetime):
                entry['timestamp'] = iso(entry['timestamp'])
            data['current_data'][k] = entry
            
        # ensure alerts are sorted newest-first
        alerts_sorted = sorted(shared_data['alerts'], key=lambda x: x.get('timestamp') or datetime.min, reverse=True)
        for a in alerts_sorted[:100]:
            a2 = a.copy()
            if isinstance(a2.get('timestamp'), datetime):
                a2['timestamp'] = iso(a2['timestamp'])
            data['alerts'].append(a2)
            
    # Sanitize NaN/Infinity values before JSON serialization
    data = sanitize_json_value(data)
    return JSONResponse(data)


@app.get('/api/history')
def api_history(symbol: str, interval: str = '30m', limit: int = 500):
    """Return historical OHLCV + MACD indicators for a symbol and interval."""
    try:
        if processor is None:
            setup_processor(load_config())
        df = processor.get_historical_data(symbol, interval, f"{limit} hours ago UTC", 'now UTC')
        if df.empty:
            return JSONResponse({'candles': []})
            
        # ENFORCE CLOSED CANDLE LOGIC:
        # Drop the last (forming) candle to prevent repaint/unstable signals
        df = df.iloc[:-1].copy()
        
        if df.empty:
            return JSONResponse({'candles': []})
            
        df = processor.calculate_macd(df)

        # Limit to last `limit` rows
        df = df.tail(limit)

        candles = []
        for _, row in df.iterrows():
            candles.append({
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'macd': float(row['macd']) if 'macd' in row.index and not pd.isna(row['macd']) else None,
                'signal': float(row['signal']) if 'signal' in row.index and not pd.isna(row['signal']) else None,
                'histogram': float(row['histogram']) if 'histogram' in row.index and not pd.isna(row['histogram']) else None,
            })

        return JSONResponse({'candles': candles})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/fetch_now')
def api_fetch_now(symbol: str, interval: str = '30m'):
    """Trigger an immediate fetch for a symbol/interval and return latest processed data."""
    cfg = load_config()
    setup_processor(cfg)
    ok = check_coin(symbol, interval, cfg)
    with data_lock:
        data = shared_data['current_data'].get(f"{symbol}_{interval}")
        alerts = [a for a in shared_data['alerts'] if a['symbol'] == symbol and a['interval'] == interval]
        alerts = sorted(alerts, key=lambda x: x.get('timestamp') or datetime.min, reverse=True)
    return JSONResponse({'ok': ok, 'data': data, 'alerts': alerts})


@app.get('/api/config')
def api_get_config():
    return load_config()


@app.post('/api/config')
def api_post_config(cfg: dict):
    save_config(cfg)
    # apply new config immediately
    setup_telegram_if_needed(cfg)
    setup_processor(cfg)
    return {'status': 'saved'}


@app.post('/api/add_coin')
def api_add_coin(symbol: str, interval: str = '30m'):
    cfg = load_config()
    exists = any(c['symbol'] == symbol and c['interval'] == interval for c in cfg['coins'])
    if not exists:
        cfg['coins'].append({'symbol': symbol, 'interval': interval, 'enabled': True})
        save_config(cfg)
        return {'status': 'added'}
    return {'status': 'exists'}


@app.post('/api/remove_coin')
def api_remove_coin(symbol: str, interval: str = '30m'):
    cfg = load_config()
    before = len(cfg['coins'])
    cfg['coins'] = [c for c in cfg['coins'] if not (c['symbol'] == symbol and c['interval'] == interval)]
    if len(cfg['coins']) < before:
        save_config(cfg)
        return {'status': 'removed'}
    return {'status': 'not_found'}


@app.get('/api/symbols')
def api_symbols():
    return {'symbols': get_futures_symbols()}


@app.get('/api/timeframes')
def api_timeframes():
    """Get list of all enabled timeframes with configuration"""
    with data_lock:
        monitor = shared_data.get('monitor')
    
    if monitor is None:
        return JSONResponse({'timeframes': [], 'message': 'Monitor not initialized'})
    
    timeframes_list = []
    for interval in monitor.config.get_enabled_timeframes():
        tf_config = monitor.config.get_timeframe_config(interval)
        timeframes_list.append({
            'interval': interval,
            'scan_interval': tf_config.get('scan_interval', 300),
            'telegram_chat_id': tf_config.get('telegram_chat_id', ''),
            'enabled': tf_config.get('enabled', True)
        })
    
    return JSONResponse({'timeframes': timeframes_list})


@app.get('/api/timeframes/{interval}/status')
def api_timeframe_status(interval: str):
    """Get status for a specific timeframe"""
    with data_lock:
        stats = shared_data.get('timeframe_stats', {}).get(interval, {})
        # Get alerts for this timeframe
        timeframe_alerts = [
            a for a in shared_data['alerts'] 
            if a.get('interval') == interval
        ][:20]  # Last 20 alerts for this timeframe
        
        # Get current data for coins in this timeframe
        timeframe_data = {
            k: v for k, v in shared_data['current_data'].items() 
            if k.endswith(f'_{interval}')
        }
    
    # Convert datetimes
    for a in timeframe_alerts:
        a['timestamp'] = iso(a.get('timestamp'))
    
    for k, v in timeframe_data.items():
        if isinstance(v.get('timestamp'), datetime):
            v['timestamp'] = iso(v['timestamp'])
    
    result = {
        'interval': interval,
        'stats': stats,
        'alerts': timeframe_alerts,
        'current_data': timeframe_data
    }
    
    return JSONResponse(sanitize_json_value(result))


@app.get('/ping')
def ping():
    return {'ping': 'pong'}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api_server:app', host='0.0.0.0', port=8000, reload=False)
