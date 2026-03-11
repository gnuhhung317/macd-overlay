"""
Streamlit Dashboard cho MACD Real-time Monitor
Theo dõi nhiều coin với cấu hình linh hoạt
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
from zoneinfo import ZoneInfo
from data_processor import BinanceDataProcessor
from telegram_notifier import TelegramNotifier
import json
import os

# Cấu hình trang
st.set_page_config(
    page_title="MACD Monitor Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File lưu cấu hình
CONFIG_FILE = "monitor_config.json"
# Scan interval cố định: 5 phút
SCAN_INTERVAL = 300  # seconds

# Thread control - global variables accessible from thread
stop_event = threading.Event()
data_lock = threading.Lock()

# Global references for crawler (set when starting)
crawler_config = None
crawler_processor = None
crawler_telegram = None

# Shared data structures (accessed by both crawler thread and UI)
shared_data = {
    'check_count': 0,
    'last_scan_time': None,
    'current_data': {},  # {"symbol_interval": {...}}
    'alerts': [],  # List of alert dicts
    'last_check': {}  # {"symbol_interval": timestamp}
}

def load_config():
    """Load cấu hình từ file"""
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
    """Lưu cấu hình vào file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def initialize_session_state():
    """Khởi tạo session state"""
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False
    
    if 'processor' not in st.session_state:
        st.session_state.processor = BinanceDataProcessor(use_futures=True)
    
    if 'telegram' not in st.session_state:
        st.session_state.telegram = None
    
    if 'futures_symbols' not in st.session_state:
        st.session_state.futures_symbols = None
    
    if 'crawler_thread' not in st.session_state:
        st.session_state.crawler_thread = None

def setup_telegram():
    """Khởi tạo Telegram"""
    config = st.session_state.config
    if config['telegram_enabled'] and config['telegram_token'] and config['telegram_chat_id']:
        try:
            telegram = TelegramNotifier(config['telegram_token'], config['telegram_chat_id'])
            if telegram.test_connection():
                st.session_state.telegram = telegram
                return True
            else:
                st.session_state.telegram = None
                return False
        except Exception as e:
            st.error(f"Lỗi Telegram: {e}")
            st.session_state.telegram = None
            return False
    return False

def check_coin(symbol, interval):
    """Kiểm tra một coin"""
    try:
        # Lấy dữ liệu
        print(f"[CHECK] {symbol} {interval}...")
        df = crawler_processor.get_historical_data(symbol, interval, '2 days ago UTC', 'now UTC')
        df = crawler_processor.calculate_macd(df)
        
        # Lưu dữ liệu hiện tại
        current_data = {
            'price': float(df['close'].iloc[-1]),
            'macd': float(df['macd'].iloc[-1]),
            'signal': float(df['signal'].iloc[-1]),
            'histogram': float(df['histogram'].iloc[-1]),
            'timestamp': datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")),
            'trend': 'BULLISH' if df['macd'].iloc[-1] > df['signal'].iloc[-1] else 'BEARISH',
            'has_new_alert': False
        }
        
        # Kiểm tra crossover
        recent_crossovers = crawler_processor.detect_crossovers(df.tail(20))
        
        if recent_crossovers:
            latest_cross = recent_crossovers[-1]
            
            # Kiểm tra xem có phải crossover mới không
            coin_key = f"{symbol}_{interval}"
            
            with data_lock:
                last_alert_time = shared_data['last_check'].get(coin_key)
            
            # Handle timestamp
            timestamp = latest_cross['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=ZoneInfo("UTC"))
            timestamp = timestamp.astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))
            
            if last_alert_time is None or timestamp > last_alert_time:
                # Crossover mới!
                alert = {
                    'symbol': symbol,
                    'interval': interval,
                    'type': latest_cross['type'],
                    'timestamp': timestamp,
                    'price': latest_cross['price'],
                    'macd': latest_cross['macd'],
                    'signal': latest_cross['signal'],
                    'histogram': latest_cross['histogram']
                }
                
                # Thêm vào alerts (thread-safe)
                with data_lock:
                    shared_data['alerts'].insert(0, alert)
                    shared_data['last_check'][coin_key] = timestamp
                
                # Gửi Telegram
                if crawler_telegram:
                    try:
                        crawler_telegram.send_crossover_alert(alert, symbol)
                    except Exception as tg_error:
                        print(f"[TELEGRAM] Error: {tg_error}")
                
                current_data['has_new_alert'] = True
        
        # Thread-safe update
        with data_lock:
            shared_data['current_data'][f"{symbol}_{interval}"] = current_data
        print(f"[CHECK] {symbol} {interval} OK")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"[CHECK] {symbol} {interval} ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Thread-safe update
        with data_lock:
            shared_data['current_data'][f"{symbol}_{interval}"] = {
                'error': error_msg,
                'timestamp': datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
            }
        return False

def get_futures_symbols():
    """Lấy danh sách symbols từ Binance Futures API"""
    if st.session_state.futures_symbols is not None:
        return st.session_state.futures_symbols
    
    try:
        print("[FETCH] Getting futures symbols from API...")
        processor = st.session_state.processor
        
        # Lấy thông tin tất cả symbols từ Futures
        exchange_info = processor.client.futures_exchange_info()
        
        # Filter ra các cặp USDT và đang hoạt động
        usdt_symbols = []
        for symbol_info in exchange_info['symbols']:
            if (symbol_info['symbol'].endswith('USDT') and 
                symbol_info['status'] == 'TRADING' and
                symbol_info['contractType'] == 'PERPETUAL'):
                usdt_symbols.append(symbol_info['symbol'])
        
        # Lấy 24h ticker để sort theo volume
        try:
            tickers = processor.client.futures_ticker()
            volume_map = {t['symbol']: float(t['quoteVolume']) for t in tickers}
            
            # Sort theo volume (cao nhất lên đầu)
            usdt_symbols.sort(key=lambda s: volume_map.get(s, 0), reverse=True)
            
            # Lấy top 50 symbols có volume cao nhất
            usdt_symbols = usdt_symbols[:50]
        except:
            # Fallback: sort alphabet nếu không lấy được volume
            usdt_symbols.sort()
        
        st.session_state.futures_symbols = usdt_symbols
        print(f"[FETCH] Got {len(usdt_symbols)} symbols")
        return usdt_symbols
    
    except Exception as e:
        print(f"[FETCH] Error getting symbols: {e}")
        # Fallback về list cũ
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'DOTUSDT',
            'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'NEARUSDT',
            'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'INJUSDT'
        ]

def scan_coins():
    """Quét tất cả coins được bật"""
    try:
        with data_lock:
            shared_data['check_count'] += 1
            check_num = shared_data['check_count']
        
        print(f"\n[SCAN] ===== Check #{check_num} =====")
        
        # Use global config
        enabled_coins = [c for c in crawler_config['coins'] if c['enabled']]
        
        success_count = 0
        error_count = 0
        
        for coin in enabled_coins:
            if check_coin(coin['symbol'], coin['interval']):
                success_count += 1
            else:
                error_count += 1
        
        # Cập nhật thời gian scan
        with data_lock:
            shared_data['last_scan_time'] = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
        
        print(f"[SCAN] Done: {success_count} OK, {error_count} errors")
        return True
    except Exception as e:
        print(f"[SCAN] Critical error: {e}")
        import traceback
        traceback.print_exc()
        # Vẫn update last_scan_time để không bị stuck
        with data_lock:
            shared_data['last_scan_time'] = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
        return False

def crawler_worker():
    """Worker thread cho crawler - chạy độc lập mỗi 5 phút"""
    print("[CRAWLER] Thread started")
    
    # First scan immediately
    try:
        print(f"[CRAWLER] Starting initial scan...")
        scan_coins()
        print(f"[CRAWLER] Initial scan complete.")
    except Exception as e:
        print(f"[CRAWLER] Error in initial scan: {e}")
        import traceback
        traceback.print_exc()
    
    while not stop_event.is_set():
        try:
            # Wait for interval or until stop signal
            if stop_event.wait(timeout=SCAN_INTERVAL):
                break  # Stop signal received
            
            # Time to scan
            print(f"[CRAWLER] Starting scan cycle...")
            scan_coins()
            print(f"[CRAWLER] Scan complete.")
        
        except Exception as e:
            print(f"[CRAWLER] Error in worker: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(10)  # Wait a bit before retrying
    
    print("[CRAWLER] Thread stopped")

def start_crawler():
    """Bắt đầu crawler thread"""
    global crawler_config, crawler_processor, crawler_telegram
    
    if st.session_state.crawler_thread is None or not st.session_state.crawler_thread.is_alive():
        # Set global references for crawler to use
        crawler_config = st.session_state.config
        crawler_processor = st.session_state.processor
        crawler_telegram = st.session_state.telegram
        
        stop_event.clear()  # Reset stop signal
        st.session_state.crawler_thread = threading.Thread(target=crawler_worker, daemon=True)
        st.session_state.crawler_thread.start()
        print("[CRAWLER] Started new thread")

def stop_crawler():
    """Dừng crawler thread"""
    stop_event.set()  # Signal thread to stop
    print("[CRAWLER] Stop signal sent")

def update_crawler_config():
    """Cập nhật config cho crawler khi thay đổi trong UI"""
    global crawler_config
    if crawler_config is not None:
        crawler_config = st.session_state.config
        print("[CRAWLER] Config updated")

def render_sidebar():
    """Render sidebar cấu hình"""
    st.sidebar.title("⚙️ Cấu hình Monitor")
    
    config = st.session_state.config
    
    # Hiển thị scan interval cố định
    st.sidebar.subheader("🔄 Tần suất quét")
    st.sidebar.info(f"⏱️ Quét mỗi {SCAN_INTERVAL // 60} phút")
    st.sidebar.caption("Tần suất quét cố định để tối ưu hiệu năng")
    
    st.sidebar.divider()
    
    # Telegram settings
    st.sidebar.subheader("📱 Telegram Alerts")
    
    telegram_enabled = st.sidebar.checkbox(
        "Bật thông báo Telegram",
        value=config['telegram_enabled']
    )
    
    telegram_token = st.sidebar.text_input(
        "Bot Token",
        value=config['telegram_token'],
        type="password",
        disabled=not telegram_enabled
    )
    
    telegram_chat_id = st.sidebar.text_input(
        "Chat ID",
        value=config['telegram_chat_id'],
        disabled=not telegram_enabled
    )
    
    if st.sidebar.button("💾 Lưu Telegram", disabled=not telegram_enabled):
        config['telegram_enabled'] = telegram_enabled
        config['telegram_token'] = telegram_token
        config['telegram_chat_id'] = telegram_chat_id
        save_config(config)
        
        if setup_telegram():
            st.sidebar.success("✅ Telegram đã kết nối!")
        else:
            st.sidebar.error("❌ Không thể kết nối Telegram")
    
    st.sidebar.divider()
    
    # Danh sách coins
    st.sidebar.subheader("💰 Danh sách Coins")
    
    # Hiển thị các coin hiện tại
    coins_to_remove = []
    for i, coin in enumerate(config['coins']):
        col1, col2, col3 = st.sidebar.columns([3, 2, 1])
        
        with col1:
            coin['enabled'] = st.checkbox(
                coin['symbol'],
                value=coin['enabled'],
                key=f"enable_{i}"
            )
        
        with col2:
            coin['interval'] = st.selectbox(
                "TF",
                options=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'],
                index=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'].index(coin['interval']),
                key=f"interval_{i}",
                label_visibility="collapsed"
            )
        
        with col3:
            if st.button("🗑️", key=f"remove_{i}"):
                coins_to_remove.append(i)
    
    # Xóa coins
    for i in sorted(coins_to_remove, reverse=True):
        config['coins'].pop(i)
        save_config(config)
        update_crawler_config()  # Update crawler's config
        st.rerun()
    
    # Thêm coin mới
    st.sidebar.subheader("➕ Thêm Coin")
    
    # Lấy danh sách symbols từ Futures API
    with st.spinner("Đang tải danh sách coins..."):
        available_symbols = get_futures_symbols()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        new_symbol = st.selectbox(
            "Symbol",
            options=available_symbols,
            key="new_symbol",
            help=f"Top {len(available_symbols)} coins theo volume"
        )
    
    with col2:
        new_interval = st.selectbox(
            "Interval",
            options=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'],
            index=4,
            key="new_interval"
        )
    
    if st.sidebar.button("➕ Thêm"):
        # Kiểm tra trùng lặp
        exists = any(c['symbol'] == new_symbol and c['interval'] == new_interval 
                    for c in config['coins'])
        
        if not exists:
            config['coins'].append({
                'symbol': new_symbol,
                'interval': new_interval,
                'enabled': True
            })
            save_config(config)
            update_crawler_config()  # Update crawler's config
            st.rerun()
        else:
            st.sidebar.error("Coin này đã tồn tại!")
    
    st.sidebar.divider()
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    
    # Kiểm tra có coins enabled không
    enabled_coins = [c for c in config['coins'] if c['enabled']]
    has_coins = len(enabled_coins) > 0
    
    with col1:
        if not st.session_state.monitoring:
            button_disabled = not has_coins
            button_label = "▶️ Bắt đầu" if has_coins else "⚠️ Thêm coin"
            if st.button(button_label, width="stretch", type="primary", disabled=button_disabled):
                st.session_state.monitoring = True
                
                # Reset shared data
                with data_lock:
                    shared_data['check_count'] = 0
                    shared_data['alerts'] = []
                    shared_data['last_check'] = {}
                    shared_data['last_scan_time'] = None
                    shared_data['current_data'] = {}
                
                # Setup Telegram nếu enabled
                if config['telegram_enabled']:
                    setup_telegram()
                
                # Save current config to ensure it's fresh
                save_config(config)
                
                # Start crawler thread (nó sẽ quét ngay lập tức)
                start_crawler()
                
                enabled_coins = [c for c in config['coins'] if c['enabled']]
                st.success(f"✅ Đã bắt đầu quét {len(enabled_coins)} coins!")
                time.sleep(1)
                st.rerun()
    
    with col2:
        if st.session_state.monitoring:
            if st.button("⏸️ Dừng", width="stretch", type="secondary"):
                st.session_state.monitoring = False
                stop_crawler()
                st.rerun()
    
    # Stats
    if st.session_state.monitoring:
        st.sidebar.success("🟢 Đang chạy")
        
        with data_lock:
            check_count = shared_data['check_count']
            alerts_count = len(shared_data['alerts'])
            last_scan_time = shared_data['last_scan_time']
        
        st.sidebar.metric("Số lần quét", check_count)
        st.sidebar.metric("Alerts", alerts_count)
        
        # Hiển thị coins đang theo dõi
        enabled_coins = [c for c in config['coins'] if c['enabled']]
        st.sidebar.caption(f"Đang theo dõi {len(enabled_coins)} coins")
        
        # Hiển thị lần quét cuối
        if last_scan_time:
            last_scan = last_scan_time.strftime('%H:%M:%S')
            st.sidebar.caption(f"Quét lần cuối: {last_scan}")
    else:
        st.sidebar.info("🔴 Đang dừng")
        enabled_coins = [c for c in config['coins'] if c['enabled']]
        if enabled_coins:
            st.sidebar.caption(f"Sẵn sàng quét {len(enabled_coins)} coins")
        else:
            st.sidebar.warning("⚠️ Chưa có coins nào được bật")

def render_current_status():
    """Hiển thị trạng thái hiện tại của các coin"""
    st.header("📊 Trạng thái hiện tại")
    
    config = st.session_state.config
    enabled_coins = [c for c in config['coins'] if c['enabled']]
    
    if not enabled_coins:
        st.warning("Không có coin nào được bật. Vui lòng bật coin từ sidebar.")
        return
    
    # Lấy dữ liệu từ shared_data
    with data_lock:
        current_data = shared_data['current_data'].copy()
    
    print(f"[UI] Rendering status for {len(enabled_coins)} coins, got {len(current_data)} data entries")
    
    # Tạo bảng
    rows = []
    for coin in enabled_coins:
        coin_key = f"{coin['symbol']}_{coin['interval']}"
        data = current_data.get(coin_key, {})
        
        if 'error' in data:
            rows.append({
                'Symbol': coin['symbol'],
                'Interval': coin['interval'],
                'Status': '❌ Lỗi',
                'Price': '-',
                'MACD': '-',
                'Signal': '-',
                'Trend': '-',
                'Distance': '-',
                'Last Update': data.get('timestamp', '-')
            })
        elif data:
            distance = abs(data['macd'] - data['signal'])
            trend_icon = "🟢" if data['trend'] == 'BULLISH' else "🔴"
            
            rows.append({
                'Symbol': coin['symbol'],
                'Interval': coin['interval'],
                'Status': '✅' + (' 🆕' if data.get('has_new_alert') else ''),
                'Price': f"${data['price']:.2f}",
                'MACD': f"{data['macd']:.4f}",
                'Signal': f"{data['signal']:.4f}",
                'Trend': f"{trend_icon} {data['trend']}",
                'Distance': f"{distance:.4f}",
                'Last Update': data['timestamp'].strftime('%H:%M:%S')
            })
        else:
            rows.append({
                'Symbol': coin['symbol'],
                'Interval': coin['interval'],
                'Status': '⏳ Chờ',
                'Price': '-',
                'MACD': '-',
                'Signal': '-',
                'Trend': '-',
                'Distance': '-',
                'Last Update': '-'
            })
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)

def render_alerts():
    """Hiển thị lịch sử alerts"""
    st.header("🔔 Lịch sử Crossover Alerts")
    
    # Lấy alerts từ shared_data
    with data_lock:
        alerts = shared_data['alerts'][:50]  # Giới hạn 50 alerts gần nhất
    
    if not alerts:
        st.info("Chưa có alerts nào. Alerts sẽ hiển thị khi phát hiện crossover.")
        return
    
    for alert in alerts:
        type_color = "green" if alert['type'] == 'BULLISH' else "red"
        emoji = "🟢" if alert['type'] == 'BULLISH' else "🔴"
        
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 3])
            
            with col1:
                st.markdown(f"### {emoji} {alert['symbol']}")
            
            with col2:
                st.caption(alert['interval'])
            
            with col3:
                st.markdown(f"**:{type_color}[{alert['type']}]**")
            
            with col4:
                st.metric("Price", f"${alert['price']:.2f}")
            
            with col5:
                st.caption(alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
            
            # Chi tiết
            with st.expander("Chi tiết"):
                col1, col2, col3 = st.columns(3)
                col1.metric("MACD", f"{alert['macd']:.4f}")
                col2.metric("Signal", f"{alert['signal']:.4f}")
                col3.metric("Histogram", f"{alert['histogram']:.4f}")
            
            st.divider()

def render_header():
    """Render header"""
    st.title("📊 MACD Real-time Monitor Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        enabled_count = len([c for c in st.session_state.config['coins'] if c['enabled']])
        st.metric("Coins theo dõi", enabled_count)
    
    with col2:
        st.metric("Tần suất quét", f"{SCAN_INTERVAL // 60} phút")
    
    with col3:
        telegram_status = "🟢 Bật" if st.session_state.telegram else "🔴 Tắt"
        st.metric("Telegram", telegram_status)
    
    with col4:
        status = "🟢 Running" if st.session_state.monitoring else "🔴 Stopped"
        st.metric("Trạng thái", status)
    
    st.divider()

def main():
    """Main function"""
    initialize_session_state()
    
    render_sidebar()
    render_header()
    
    # Debug: hiển thị số lượng data
    with data_lock:
        data_count = len(shared_data['current_data'])
        alerts_count = len(shared_data['alerts'])
    
    if data_count > 0 or alerts_count > 0:
        st.sidebar.caption(f"📊 Data: {data_count} coins | 🔔 {alerts_count} alerts")
    
    # Hiển thị thông tin monitoring
    if st.session_state.monitoring:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("🟢 Crawler đang chạy độc lập")
        with col2:
            with data_lock:
                last_scan_time = shared_data['last_scan_time']
            
            if last_scan_time:
                now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
                elapsed = (now - last_scan_time).total_seconds()
                next_scan = max(0, SCAN_INTERVAL - elapsed)
                minutes = int(next_scan // 60)
                seconds = int(next_scan % 60)
                st.info(f"⏱️ Quét tiếp sau: {minutes}m {seconds}s")
            else:
                st.info("⏱️ Đang chuẩn bị quét đầu tiên...")
        with col3:
            last_update = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime('%H:%M:%S')
            st.caption(f"UI cập nhật: {last_update}")
    
    # Tab layout
    tab1, tab2 = st.tabs(["📊 Trạng thái", "🔔 Alerts"])
    
    with tab1:
        render_current_status()
    
    with tab2:
        render_alerts()
    
    # Auto-refresh UI để cập nhật trạng thái
    if st.session_state.monitoring:
        # Refresh mỗi 2 giây khi đang monitor
        time.sleep(2)
        st.rerun()
    elif data_count > 0:
        # Nếu có data nhưng đã dừng, vẫn cho phép xem
        pass
    else:
        # Không có gì, refresh chậm hơn
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
