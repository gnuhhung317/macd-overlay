import streamlit as st
import pandas as pd
import plotly.express as px
import ccxt
import json
import logging
import sqlite3
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 1. DATABASE MANAGER
# ==========================================
class DatabaseManager:
    def __init__(self, db_path: str = 'pnl_history.db'):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                account_name TEXT NOT NULL,
                exchange TEXT NOT NULL,
                total_equity REAL NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_equity REAL NOT NULL,
                total_unrealized_pnl REAL NOT NULL,
                total_open_positions INTEGER NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def save_balance_snapshot(self, df_balances: pd.DataFrame, df_positions: pd.DataFrame):
        if df_balances.empty:
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            for _, row in df_balances.iterrows():
                cursor.execute('''
                    INSERT INTO balance_history (timestamp, account_name, exchange, total_equity)
                    VALUES (?, ?, ?, ?)
                ''', (now, row['Account'], row['Exchange'], float(row['Total Equity (USDT)'])))

            global_equity = float(df_balances['Total Equity (USDT)'].sum())
            global_pnl = float(df_positions['Unrealized PnL'].astype(float).sum()) if not df_positions.empty else 0.0
            total_positions = len(df_positions) if not df_positions.empty else 0
            
            cursor.execute('''
                INSERT INTO global_equity_history (timestamp, total_equity, total_unrealized_pnl, total_open_positions)
                VALUES (?, ?, ?, ?)
            ''', (now, global_equity, global_pnl, total_positions))
            
            conn.commit()
        except Exception as e:
            logging.error(f"DB Save Error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_global_history(self, limit: int = None) -> pd.DataFrame:
        conn = self._get_connection()
        query = "SELECT timestamp, total_equity, total_unrealized_pnl FROM global_equity_history ORDER BY timestamp DESC"
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        return df

# ==========================================
# 2. DATA FETCHER (CCXT)
# ==========================================
class DataFetcher:
    def __init__(self, credentials_path: str = 'credentials.json'):
        self.credentials_path = credentials_path
        self.accounts_config = self._load_credentials()
        self.exchanges = self._initialize_exchanges()

    def _load_credentials(self) -> Dict[str, Any]:
        try:
            with open(self.credentials_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _initialize_exchanges(self) -> Dict[str, ccxt.Exchange]:
        exchanges = {}
        for account_name, config in self.accounts_config.items():
            if not isinstance(config, dict):
                continue
            exchange_id = config.get('exchange')
            if not exchange_id or not hasattr(ccxt, exchange_id):
                continue
            try:
                exchange_class = getattr(ccxt, exchange_id)
                auth_params = {k: v for k, v in config.items() if k != 'exchange'}
                exchanges[account_name] = exchange_class(auth_params)
            except Exception as e:
                logging.error(f"Init Error for {account_name}: {e}")
        return exchanges

    def get_balances(self) -> pd.DataFrame:
        records = []
        for account_name, exchange in self.exchanges.items():
            try:
                balance = exchange.fetch_balance()
                total_equity = None
                
                if 'info' in balance:
                    if exchange.id == 'binance':
                        info = balance['info']
                        total_equity = next((asset['marginBalance'] for asset in info.get('assets', []) if asset['asset'] == 'USDT'), None)
                        if total_equity is None:
                             total_equity = float(info.get('totalMarginBalance', 0))
                    elif exchange.id == 'bitget':
                        total_equity = float(balance.get('USDT', {}).get('total', 0))
                
                if total_equity is None:
                    total_equity = float(balance.get('total', {}).get('USDT', 0))
                
                records.append({
                    'Account': account_name,
                    'Exchange': exchange.id.capitalize(),
                    'Total Equity (USDT)': float(total_equity) if total_equity else 0.0
                })
            except Exception as e:
                records.append({'Account': account_name, 'Exchange': exchange.id.capitalize(), 'Total Equity (USDT)': 0.0, 'Error': str(e)})
        return pd.DataFrame(records)

    def get_positions(self) -> pd.DataFrame:
        records = []
        for account_name, exchange in self.exchanges.items():
            if not exchange.has['fetchPositions']: continue
            try:
                positions = exchange.fetch_positions()
                for pos in positions:
                    if 'contracts' in pos and float(pos['contracts'] or 0) > 0:
                        records.append({
                            'Account': account_name,
                            'Exchange': exchange.id.capitalize(),
                            'Symbol': pos.get('symbol', 'N/A'),
                            'Side': pos.get('side', 'N/A'),
                            'Size': pos.get('contracts', 0),
                            'Entry Price': pos.get('entryPrice', 0),
                            'Mark Price': pos.get('markPrice', 0),
                            'Unrealized PnL': pos.get('unrealizedPnl', 0),
                            'Leverage': pos.get('leverage', 1),
                        })
            except Exception as e:
                logging.error(f"Position Fetch Error for {account_name}: {e}")
        return pd.DataFrame(records)

# ==========================================
# 3. BACKGROUND COLLECTOR THREAD
# ==========================================
@st.cache_resource
def start_background_collector(credentials_file):
    """Khởi động luồng chạy ngầm để thu thập dữ liệu tự động mà không block UI."""
    def collection_loop():
        logging.info("Starting background collector thread.")
        fetcher = DataFetcher(credentials_path=credentials_file)
        db = DatabaseManager('pnl_history.db')
        
        while True:
            try:
                # Cập nhật số liệu để ghi vào DB mỗi 5 phút
                df_balances = fetcher.get_balances()
                df_positions = fetcher.get_positions()
                if not df_balances.empty:
                    db.save_balance_snapshot(df_balances, df_positions)
                    logging.info("Background update OK.")
            except Exception as e:
                logging.error(f"Collector Error: {e}")
            
            # Đợi 5 phút (300 giây)
            time.sleep(300)

    # Đánh dấu luồng là daemon để nó tự tắt khi app đóng
    thread = threading.Thread(target=collection_loop, daemon=True)
    thread.start()
    return thread

# ==========================================
# 4. STREAMLIT APP UI
# ==========================================
st.set_page_config(page_title="Multi-Account PnL Dashboard", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1f2937; font-family: 'Inter', sans-serif; }
    h2, h3 { color: #374151; font-family: 'Inter', sans-serif; }
    .metric-card {
        background-color: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
    }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .positive-value { color: #10b981; font-weight: bold; }
    .negative-value { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📈 Multi-Account PnL & Asset Dashboard")
    st.markdown("___")

    credentials_file = "credentials.json"
    if not os.path.exists(credentials_file):
        st.warning(f"Chưa tìm thấy file `{credentials_file}`. Vui lòng tạo cấu hình API.", icon="⚠️")
        st.stop()
    
    # LOAD ACCESS TOKEN
    try:
        with open(credentials_file, 'r') as f:
            creds = json.load(f)
            access_token = creds.get('access_token')
    except Exception as e:
        st.error(f"Lỗi khi load access_token từ {credentials_file}: {e}")
        st.stop()

    if not access_token:
        st.error(f"File `{credentials_file}` thiếu trường `access_token`. Vui lòng cấu hình.", icon="🔒")
        st.stop()

    # AUTHENTICATION UI
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("### 🔒 Authentication Required")
        input_token = st.text_input("Nhập Access Token để tiếp tục:", type="password")
        if st.button("Truy cập Dashboard"):
            if input_token == access_token:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Token không chính xác!")
        st.stop()
    
    # ==========================================
    # Nếu đã authenticated thì mới chạy tiếp bên dưới
    # ==========================================
    
    # Kích hoạt luồng lấy dữ liệu tự động ở background (chạy 1 lần duy nhất)
    start_background_collector(credentials_file)

    st.success("Hệ thống background đã được kích hoạt. Đang lấy dữ liệu Snapshot hiện tại...", icon="✅")

    # Hiển thị snapshot realtime
    with st.spinner('Đang kết nối API...'):
        fetcher = DataFetcher(credentials_path=credentials_file)
        if not fetcher.exchanges:
            st.error("Không thể khởi tạo sàn giao dịch từ cấu hình.")
            st.stop()
        df_balances = fetcher.get_balances()
        df_positions = fetcher.get_positions()
    
    # ======= LỊCH SỬ THAY ĐỔI & BIỂU ĐỒ =======
    st.header("1. Equity History Chart")
    
    # Timeframe selection
    timeframe_options = {
        "Raw (5m)": None,
        "1H": "H",
        "4H": "4H",
        "1D": "D",
        "1W": "W",
        "1M": "ME"
    }
    
    selected_tf_label = st.selectbox("Select Timeframe:", list(timeframe_options.keys()), index=0)
    selected_tf = timeframe_options[selected_tf_label]

    db = DatabaseManager('pnl_history.db')
    # Fetch more data if resampled, otherwise keep 500 for low latency "Raw" view
    fetch_limit = None if selected_tf else 500
    df_history = db.get_global_history(limit=fetch_limit)
    
    if not df_history.empty and selected_tf:
        # Sort by timestamp (project rule reminder)
        df_history = df_history.sort_values('timestamp')
        df_history.set_index('timestamp', inplace=True)
        # Resample and take the last value of each bucket
        df_history = df_history.resample(selected_tf).last().dropna().reset_index()

    col_chart, col_live = st.columns([3, 1])
    
    with col_chart:
        if df_history.empty:
            st.info("Chưa có đủ dữ liệu lịch sử để vẽ biểu đồ. Hãy chờ 5 phút hệ thống sẽ cập nhật điểm dữ liệu đầu tiên.")
        else:
            fig = px.line(
                df_history, x='timestamp', y='total_equity', 
                title='Total Equity Over Time (Trailling)',
                labels={'timestamp': 'Thời gian', 'total_equity': 'Total Equity (USDT)'},
                markers=True, line_shape='spline'
            )
            fig.update_layout(hovermode="x unified", plot_bgcolor='white', paper_bgcolor='white')
            fig.update_traces(line_color='#2563eb', marker=dict(size=4))
            st.plotly_chart(fig, use_container_width=True)

    with col_live:
         # TỔNG QUAN HIỆN TẠI
        total_balance = df_balances['Total Equity (USDT)'].sum() if not df_balances.empty else 0
        total_unrealized_pnl = df_positions['Unrealized PnL'].astype(float).sum() if not df_positions.empty else 0
        pnl_color = "🟢" if total_unrealized_pnl >= 0 else "🔴"
        pnl_css_class = "positive-value" if total_unrealized_pnl >= 0 else "negative-value"
        
        st.markdown(f'<div class="metric-card" style="margin-bottom:20px;"><h3>Total Equity</h3><h2 style="color:#2563eb;">${total_balance:,.2f}</h2></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><h3>Unrealized PnL</h3><h2 class="{pnl_css_class}">{pnl_color} ${total_unrealized_pnl:,.2f}</h2></div>', unsafe_allow_html=True)

    st.markdown("___")

    # ======= CHI TIẾT SỐ DƯ (ACCOUNT BALANCES) =======
    st.header("2. Account Balances")
    if df_balances.empty:
        st.info("Không có dữ liệu số dư nào.")
    else:
        st.dataframe(
            df_balances, use_container_width=True, hide_index=True,
            column_config={"Total Equity (USDT)": st.column_config.NumberColumn("Total Equity (USDT)", format="$ %.2f")}
        )

    st.markdown("___")

    # ======= CHI TIẾT VỊ THẾ (OPEN POSITIONS) =======
    st.header("3. Open Positions & PnL")
    if df_positions.empty:
        st.info("Hiện không có vị thế nào đang mở (hoặc không lấy được dữ liệu).")
    else:
        def format_pnl(val):
            try:
                val = float(val)
                return f'color: {"green" if val > 0 else "red" if val < 0 else "grey"}'
            except:
                return ''
        st.dataframe(
            df_positions.style.applymap(format_pnl, subset=['Unrealized PnL']),
            use_container_width=True, hide_index=True
        )

if __name__ == "__main__":
    main()
