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
        'scan_interval': 60,
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
    
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    if 'last_check' not in st.session_state:
        st.session_state.last_check = {}
    
    if 'current_data' not in st.session_state:
        st.session_state.current_data = {}
    
    if 'processor' not in st.session_state:
        st.session_state.processor = BinanceDataProcessor()
    
    if 'telegram' not in st.session_state:
        st.session_state.telegram = None
    
    if 'check_count' not in st.session_state:
        st.session_state.check_count = 0

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
        processor = st.session_state.processor
        
        # Lấy dữ liệu
        df = processor.get_historical_data(symbol, interval, '2 days ago UTC', 'now UTC')
        df = processor.calculate_macd(df)
        
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
        recent_crossovers = processor.detect_crossovers(df.tail(20))
        
        if recent_crossovers:
            latest_cross = recent_crossovers[-1]
            
            # Kiểm tra xem có phải crossover mới không
            coin_key = f"{symbol}_{interval}"
            last_alert_time = st.session_state.last_check.get(coin_key)
            
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
                
                # Thêm vào alerts
                st.session_state.alerts.insert(0, alert)
                st.session_state.last_check[coin_key] = timestamp
                
                # Gửi Telegram
                if st.session_state.telegram:
                    st.session_state.telegram.send_crossover_alert(alert, symbol)
                
                current_data['has_new_alert'] = True
        
        st.session_state.current_data[f"{symbol}_{interval}"] = current_data
        return True
        
    except Exception as e:
        st.session_state.current_data[f"{symbol}_{interval}"] = {
            'error': str(e),
            'timestamp': datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
        }
        return False

def monitoring_loop():
    """Vòng lặp monitoring"""
    while st.session_state.get('monitoring', False):
        try:
            st.session_state.check_count += 1
            
            config = st.session_state.config
            enabled_coins = [c for c in config['coins'] if c['enabled']]
            
            for coin in enabled_coins:
                if not st.session_state.get('monitoring', False):
                    break
                check_coin(coin['symbol'], coin['interval'])
            
            # Chờ scan_interval giây
            if st.session_state.get('monitoring', False):
                time.sleep(config['scan_interval'])
        except Exception as e:
            print(f"Monitoring loop error: {e}")
            time.sleep(5)

def render_sidebar():
    """Render sidebar cấu hình"""
    st.sidebar.title("⚙️ Cấu hình Monitor")
    
    config = st.session_state.config
    
    # Khoảng cách giữa các lần quét
    st.sidebar.subheader("🔄 Tần suất quét")
    scan_interval = st.sidebar.slider(
        "Giây giữa mỗi lần quét",
        min_value=10,
        max_value=300,
        value=config['scan_interval'],
        step=10,
        help="Thời gian chờ giữa mỗi lần quét tất cả các coin"
    )
    
    if scan_interval != config['scan_interval']:
        config['scan_interval'] = scan_interval
        save_config(config)
    
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
                options=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'],
                index=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'].index(coin['interval']),
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
        st.rerun()
    
    # Thêm coin mới
    st.sidebar.subheader("➕ Thêm Coin")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        new_symbol = st.text_input("Symbol", value="BTCUSDT", key="new_symbol")
    
    with col2:
        new_interval = st.selectbox(
            "Interval",
            options=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'],
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
                st.session_state.check_count = 0
                st.session_state.alerts = []  # Reset alerts
                st.session_state.last_check = {}  # Reset last check
                
                # Setup Telegram nếu enabled
                if config['telegram_enabled']:
                    setup_telegram()
                
                # Bắt đầu monitoring thread
                thread = threading.Thread(target=monitoring_loop, daemon=True)
                thread.start()
                
                # Thực hiện quét đầu tiên ngay lập tức
                enabled_coins = [c for c in config['coins'] if c['enabled']]
                for coin in enabled_coins:
                    check_coin(coin['symbol'], coin['interval'])
                
                st.success(f"✅ Đã bắt đầu quét {len(enabled_coins)} coins!")
                time.sleep(1)
                st.rerun()
    
    with col2:
        if st.session_state.monitoring:
            if st.button("⏸️ Dừng", width="stretch", type="secondary"):
                st.session_state.monitoring = False
                st.rerun()
    
    # Stats
    if st.session_state.monitoring:
        st.sidebar.success("🟢 Đang chạy")
        st.sidebar.metric("Số lần quét", st.session_state.check_count)
        st.sidebar.metric("Alerts", len(st.session_state.alerts))
        
        # Hiển thị coins đang theo dõi
        enabled_coins = [c for c in config['coins'] if c['enabled']]
        st.sidebar.caption(f"Đang theo dõi {len(enabled_coins)} coins")
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
    
    # Tạo bảng
    rows = []
    for coin in enabled_coins:
        coin_key = f"{coin['symbol']}_{coin['interval']}"
        data = st.session_state.current_data.get(coin_key, {})
        
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
    
    if not st.session_state.alerts:
        st.info("Chưa có alerts nào. Alerts sẽ hiển thị khi phát hiện crossover.")
        return
    
    # Giới hạn hiển thị 50 alerts gần nhất
    alerts = st.session_state.alerts[:50]
    
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
        st.metric("Tần suất quét", f"{st.session_state.config['scan_interval']}s")
    
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
    
    # Hiển thị thông tin monitoring
    if st.session_state.monitoring:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("🟢 Đang quét...")
        with col2:
            next_scan = st.session_state.config['scan_interval']
            st.info(f"⏱️ Quét tiếp sau: ~{next_scan}s")
        with col3:
            last_update = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime('%H:%M:%S')
            st.caption(f"Cập nhật lần cuối: {last_update}")
    
    # Tab layout
    tab1, tab2 = st.tabs(["📊 Trạng thái", "🔔 Alerts"])
    
    with tab1:
        render_current_status()
    
    with tab2:
        render_alerts()
    
    # Auto-refresh nếu đang monitoring
    if st.session_state.monitoring:
        time.sleep(2)  # Refresh mỗi 2 giây
        st.rerun()

if __name__ == "__main__":
    main()
