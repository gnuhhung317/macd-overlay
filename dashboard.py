import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# -----------------------------------------------------------------------------
# CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🤖 TradeBot Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Premium" Look
st.markdown("""
<style>
    /* Gradient Background for Header */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #4CAF50; /* Success Green */
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #111;
        border-right: 1px solid #333;
    }
    
    /* Tables */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    
    /* Status Badges */
    .badge-open {
        background-color: #2196F3;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    .badge-closed {
        background-color: #757575;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
    }
    .badge-long {
        color: #4CAF50 !important;
        font-weight: bold;
    }
    .badge-short {
        color: #FF5252 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = os.path.join("data", "bot_data.db")

# -----------------------------------------------------------------------------
# DATA LOADING FUNCTIONS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_trades_data():
    """Fetch all trades from the database."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM trades ORDER BY entry_time DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Datetime Conversion
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])
            
        return df
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def get_signals_data():
    """Fetch recent signals from the database."""
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
        SELECT * FROM signals 
        WHERE action != 'WAIT' 
        ORDER BY timestamp DESC LIMIT 500
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df
    except Exception as e:
        st.error(f"Error loading signals: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def color_pnl(val):
    """Colorize PnL values."""
    if val > 0:
        return 'color: #4CAF50'
    elif val < 0:
        return 'color: #FF5252'
    return ''

def format_currency(val):
    return f"${val:,.2f}"

def format_pct(val):
    return f"{val*100:,.2f}%"

import requests

def fetch_current_prices():
    """Fetch real-time prices from Binance Futures public API."""
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/price"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        return {t['symbol']: float(t['price']) for t in data}
    except Exception as e:
        # st.error(f"Error fetching prices: {e}")
        return {}

# --- Constants (From ML Analysis Report) ---
AVG_MAE_STATS = {
    '4h': 0.035,   # ~3.5% (Bin 0.6-0.7)
    '8h': 0.045,   # ~4.5%
    '12h': 0.055,  # ~5.5%
    '1d': 0.065    # ~6.5%
}

AVG_MFE_STATS = {
    '4h': 0.11,
    '8h': 0.14,
    '12h': 0.16,
    '1d': 0.22
}

def get_entry_status(signal_type, cross_price, current_price, mae_pct, mfe_pct):
    """
    Determine the status of the signal based on current price relative to Entry Zone and MFE.
    """
    status = "UNKNOWN"
    zone_min = 0.0
    zone_max = 0.0
    color = "gray"
    
    if signal_type == 'LONG': # Normalized to internal "LONG"
        # Long Logic
        limit_price = cross_price * (1 - mae_pct) # Deepest dip expected
        fomo_limit = cross_price * (1 + 0.01) # Allow 1% slippage
        profit_limit = cross_price * (1 + mfe_pct * 0.5) # 50% of potential
        
        zone_min = limit_price
        zone_max = fomo_limit
        
        if current_price < limit_price:
            status = "⚠️ DEEP MERGE (Wait)" # Dropped below expected MAE
            color = "red"
        elif limit_price <= current_price <= cross_price:
            status = "💎 DISCOUNT ENTRY" # Better than AI
            color = "green"
        elif cross_price < current_price <= fomo_limit:
            status = "✅ GOOD ENTRY"
            color = "lightgreen"
        elif current_price > profit_limit:
            status = "❌ TOO LATE (High Risk)"
            color = "orange"
        else:
            status = "⚠️ CHASING" # In between good and too late
            color = "yellow"
            
    else:
        # Short Logic
        limit_price = cross_price * (1 + mae_pct) # Highest pump expected
        fomo_limit = cross_price * (1 - 0.01)
        profit_limit = cross_price * (1 - mfe_pct * 0.5)
        
        zone_max = limit_price
        zone_min = fomo_limit # Technically inverted for display
        
        if current_price > limit_price:
            status = "⚠️ DEEP MERGE (Wait)"
            color = "red"
        elif cross_price <= current_price <= limit_price:
            status = "💎 DISCOUNT ENTRY"
            color = "green"
        elif fomo_limit <= current_price < cross_price:
            status = "✅ GOOD ENTRY"
            color = "lightgreen"
        elif current_price < profit_limit:
            status = "❌ TOO LATE"
            color = "orange"
        else:
            status = "⚠️ CHASING"
            color = "yellow"
            
    return status, zone_min, zone_max, color

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------

# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620601.png", width=50)
    st.title("Bot Dashboard")
    st.write("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "⚡ Active Positions", "📜 Trade History", "📡 AI Signals"],
        index=0
    )
    
    st.write("---")
    st.subheader("System Status")
    
    # Simple check if DB exists and has data
    if os.path.exists(DB_PATH):
        st.success("🟢 Database Connected")
        st.caption(f"Path: `{DB_PATH}`")
    else:
        st.error("🔴 Database Not Found")

    if st.button("🔄 Refresh Data"):
        st.rerun()

    st.write("---")
    st.subheader("⚙️ Settings")
    
    # Load Config
    CONFIG_PATH = "bot_config.json"
    import json
    
    def load_config():
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        return {}

    def save_config(cfg):
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f, indent=4)
            
    config = load_config()
    
    with st.expander("Telegram Alerts"):
        tg_config = config.get("telegram", {"enabled": False, "token": "", "chat_id": ""})
        
        tg_enabled = st.checkbox("Enable Alerts", value=tg_config.get("enabled", False))
        tg_token = st.text_input("Bot Token", value=tg_config.get("token", ""), type="password")
        tg_chat_id = st.text_input("Chat ID", value=tg_config.get("chat_id", ""))
        
        if st.button("Save Settings"):
            # Update config structure
            if "telegram" not in config:
                config["telegram"] = {}
                
            config["telegram"]["enabled"] = tg_enabled
            config["telegram"]["token"] = tg_token
            config["telegram"]["chat_id"] = tg_chat_id
            
            save_config(config)
            st.success("Settings Saved!")
            
        if st.button("Test Connection"):
             try:
                 from telegram_notifier import TelegramNotifier
                 if tg_token and tg_chat_id:
                     notifier = TelegramNotifier(tg_token, tg_chat_id)
                     if notifier.test_connection():
                         st.success("✅ Connected!")
                     else:
                         st.error("❌ Failed to connect.")
                 else:
                     st.warning("Missing Token or Chat ID")
             except Exception as e:
                 st.error(f"Error: {e}")

# Data Fetching
df_trades = get_trades_data()
df_signals = get_signals_data()

# -----------------------------------------------------------------------------
# PAGE: OVERVIEW
# -----------------------------------------------------------------------------
if page == "🏠 Overview":
    st.title("🏠 Trading Overview")
    
    if df_trades.empty:
        st.warning("No trade data found yet. Start the bot to generate trades.")
    else:
        # separate active and closed
        df_closed = df_trades[df_trades['status'] == 'CLOSED'].copy()
        df_active = df_trades[df_trades['status'] == 'OPEN'].copy()
        
        # --- Top Metrics Row ---
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_pnl = df_closed['pnl'].sum() if not df_closed.empty else 0.0
        win_count = len(df_closed[df_closed['pnl'] > 0])
        total_closed = len(df_closed)
        win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0
        
        active_exposure = df_active['size'].sum() if not df_active.empty else 0.0
        active_count = len(df_active)
        
        with col1:
            st.metric("💰 Total PnL", format_currency(total_pnl), delta_color="normal")
        with col2:
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%", f"{win_count}/{total_closed} Trades")
        with col3:
            st.metric("⚡ Active Trades", f"{active_count}", f"${active_exposure:,.0f} Exposure")
        with col4:
            avg_pnl = df_closed['pnl'].mean() if not df_closed.empty else 0
            st.metric("📊 Avg PnL / Trade", format_currency(avg_pnl))
        with col5:
             # Basic Sharpe Ratio proxy (Avg PnL / StdDev PnL)
             std_pnl = df_closed['pnl'].std()
             sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0
             st.metric("Risk Factor (Sharpe)", f"{sharpe:.2f}")

        st.markdown("---")
        
        # --- Charts Row ---
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📈 Equity Curve (PnL Accumulation)")
            if not df_closed.empty:
                df_closed = df_closed.sort_values('exit_time')
                df_closed['cumulative_pnl'] = df_closed['pnl'].cumsum()
                
                fig = px.area(df_closed, x='exit_time', y='cumulative_pnl', 
                              labels={'exit_time': 'Date', 'cumulative_pnl': 'Cumulative PnL ($)'},
                              template="plotly_dark")
                fig.update_traces(line_color='#2196F3', fillcolor='rgba(33, 150, 243, 0.2)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for equity curve.")
                
        with c2:
            st.subheader("🍩 Win/Loss Ratio")
            if not df_closed.empty:
                loss_count = total_closed - win_count
                fig_pie = px.pie(names=['Win', 'Loss'], values=[win_count, loss_count], 
                                 color_discrete_sequence=['#4CAF50', '#FF5252'],
                                 hole=0.4, template="plotly_dark")
                st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE: ACTIVE POSITIONS
# -----------------------------------------------------------------------------
elif page == "⚡ Active Positions":
    st.title("⚡ Active Positions")

    # Load Config & Init Executor
    from bot.config import BotConfig
    from bot.executor import get_executor
    
    try:
        config = BotConfig.load()
        # Force non-dry run to see real positions if configured
        # config.exchange.dry_run = False 
        executor = get_executor(config)
        external_positions = executor.get_open_positions()
    except Exception as e:
        st.error(f"Could not fetch external positions: {e}")
        external_positions = []

    # Get DB Positions
    df_active_db = pd.DataFrame()
    if not df_trades.empty:
        df_active_db = df_trades[df_trades['status'] == 'OPEN'].copy()

    # Merge Logic
    # We want to show all external positions.
    # If a position exists in DB, we use DB metadata (entry time, SL/TP)
    # If not in DB, it's a Manual/External trade.
    
    # Convert DB positions to dict for easy lookup
    db_positions = {}
    if not df_active_db.empty:
        for _, row in df_active_db.iterrows():
            db_positions[row['symbol']] = row

    all_positions = []
    
    # 1. Process External Positions (Real Exchange Data)
    for ext_pos in external_positions:
        symbol = ext_pos['symbol']
        is_bot = symbol in db_positions
        
        pos_data = {
            'symbol': symbol,
            'source': '🤖 Bot' if is_bot else '👤 Manual',
            'size': ext_pos['size'],
            'entry_price': ext_pos['entry_price'],
            'mark_price': ext_pos['mark_price'],
            'pnl': ext_pos['pnl'],
            'leverage': ext_pos['leverage'],
            'side': ext_pos['side'],
            'entry_time': db_positions[symbol]['entry_time'] if is_bot else None,
            'sl': db_positions[symbol]['sl_price'] if is_bot else 0,
            'tp': db_positions[symbol]['tp_price'] if is_bot else 0,
        }
        all_positions.append(pos_data)
        
        # Remove from db_positions tracker to see if any DB positions are missing from exchange
        if is_bot:
            del db_positions[symbol]

    # 2. Process Remaining DB Positions (Ghost Positions?)
    # These are positions the Bot thinks are open, but Exchange says are closed.
    for symbol, row in db_positions.items():
        all_positions.append({
            'symbol': symbol,
            'source': '👻 Ghost (Sync Error)',
            'size': row['size'], # This might be USDT size in DB, need conversion or careful display
            'entry_price': row['entry_price'],
            'mark_price': row['entry_price'], # Unknown current price
            'pnl': 0.0,
            'leverage': row['leverage'],
            'side': row['direction'],
            'entry_time': row['entry_time'],
            'sl': row['sl_price'],
            'tp': row['tp_price']
        })

    if not all_positions:
        st.info("No active positions found on Exchange or Database.")
    else:
        st.write(f"Showing **{len(all_positions)}** open positions:")
        
        for pos in all_positions:
            with st.container():
                # Card Styling
                card_border = "1px solid #4CAF50" if pos['source'] == '🤖 Bot' else "1px solid #2196F3"
                if "Ghost" in pos['source']: card_border = "1px solid #FF5252"
                
                st.markdown(f"""
                <div style="border: {card_border}; border-radius: 10px; padding: 10px; margin-bottom: 10px;">
                """, unsafe_allow_html=True)
                
                cols = st.columns([1, 2, 2, 2, 2, 2])
                
                with cols[0]:
                    st.subheader(f"{pos['symbol']}")
                    st.caption(pos['source'])
                    
                with cols[1]:
                    side_color = "green" if pos['side'] == "LONG" else "red"
                    st.markdown(f"**{pos['side']}** x{pos['leverage']}")
                    st.markdown(f":{side_color}[PnL: ${pos['pnl']:.2f}]")

                with cols[2]:
                    st.markdown("**Entry / Mark**")
                    st.write(f"{pos['entry_price']:.4f}")
                    st.caption(f"{pos['mark_price']:.4f}")

                with cols[3]:
                    st.markdown("**Size (Qty)**")
                    st.write(f"{pos['size']}")
                    
                with cols[4]:
                    if pos['sl'] > 0:
                        st.markdown("**SL / TP**")
                        st.write(f"🛑 {pos['sl']:.4f}")
                        st.write(f"🎯 {pos['tp']:.4f}")
                    else:
                        st.info("No TP/SL (Manual)")
                
                with cols[5]:
                    if pos['entry_time']:
                        duration = datetime.now() - pos['entry_time']
                        st.write(f"⌛ {str(duration).split('.')[0]}")
                    else:
                        st.write("Unknown duration")
                        
                st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE: TRADE HISTORY
# -----------------------------------------------------------------------------
elif page == "📜 Trade History":
    st.title("📜 Trade History")
    
    if df_trades.empty:
        st.info("No trade history available.")
    else:
        df_closed = df_trades[df_trades['status'] != 'OPEN'].copy()
        df_closed = df_closed.sort_values('exit_time', ascending=False)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            symbol_filter = st.multiselect("Filter by Symbol", options=df_closed['symbol'].unique())
        with col2:
            status_filter = st.multiselect("Filter by Result", options=['Win', 'Loss']) # Logic needed
            
        # Apply Filters
        if symbol_filter:
            df_closed = df_closed[df_closed['symbol'].isin(symbol_filter)]
            
        # Display Table
        # Select columns to show
        display_cols = ['id', 'symbol', 'direction', 'entry_price', 'exit_price', 'size', 'pnl', 'exit_reason', 'entry_time', 'exit_time']
        
        # Formatting for the table
        styled_df = df_closed[display_cols].style.applymap(color_pnl, subset=['pnl'])\
            .format({
                'entry_price': "{:.4f}",
                'exit_price': "{:.4f}",
                'size': "${:,.0f}",
                'pnl': "${:,.2f}",
                'entry_time': "{:%Y-%m-%d %H:%M}",
                'exit_time': "{:%Y-%m-%d %H:%M}"
            })
            
        st.dataframe(styled_df, use_container_width=True, height=600)

# -----------------------------------------------------------------------------
# PAGE: ML SIGNALS
# -----------------------------------------------------------------------------
elif page == "📡 AI Signals":
    st.title("📡 Smart AI Signals Feed")
    st.caption("Live feed from Bot Scanner. (Updates every candle close)")
    
    # Imports
    import json
    
    # 1. Fetch Signals from DB
    if not os.path.exists(DB_PATH):
        st.error("Database not found")
    else:
        conn = sqlite3.connect(DB_PATH)
        # Fetch last 1000 signals
        df_sig = pd.read_sql_query("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 1000", conn)
        conn.close()
        
        if df_sig.empty:
            st.info("No signals generated by the Bot yet. Please wait for the next scan cycle.")
        else:
            # Parse `raw_data` JSON
            def parse_raw(x):
                try: 
                    return json.loads(x)
                except: 
                    return {}
            
            df_sig['meta'] = df_sig['raw_data'].apply(parse_raw)
            df_sig['type'] = df_sig['action'] # Assuming we stored LONG/SHORT in action
            df_sig['timestamp'] = pd.to_datetime(df_sig['timestamp'])

            st.write(f"Displaying **{len(df_sig)}** recent signals")
            
            # 2. Fetch Real-time Prices for Context
            with st.spinner("Fetching live prices..."):
                current_prices = fetch_current_prices()
                
            # 3. Build Display Data
            display_data = []
            
            for _, row in df_sig.iterrows():
                symbol = row['symbol']
                timeframe = row['timeframe']
                
                # Get Stats for Timeframe
                mae_stat = AVG_MAE_STATS.get(timeframe, 0.04)
                mfe_stat = AVG_MFE_STATS.get(timeframe, 0.12)
                
                # Parsed Meta
                meta = row['meta']
                signal_price = meta.get('signal_price', 0)
                if signal_price == 0: continue # Skip invalid
                
                curr_price = current_prices.get(symbol, meta.get('current_price', signal_price))
                
                # Get Intelligent Status
                status_msg, z_min, z_max, color = get_entry_status(
                    row['type'], signal_price, curr_price, mae_stat, mfe_stat
                )
                
                # Calc PnL based on Signal Price
                if row['type'] == 'LONG':
                    pnl = (curr_price - signal_price) / signal_price
                else: # SHORT
                    pnl = (signal_price - curr_price) / signal_price
                
                # Time since cross
                time_diff = datetime.now() - row['timestamp'].replace(tzinfo=None) # naive for diff
                hours = time_diff.total_seconds() / 3600
                
                # Format Entry Zone
                if row['type'] == 'LONG':
                    zone_str = f"{z_min:.4f} - {z_max:.4f}"
                else:
                    zone_str = f"{z_max:.4f} - {z_min:.4f}"
                
                display_data.append({
                    'Tín hiệu': "🟢 LONG" if row['type'] == 'LONG' else "🔴 SHORT",
                    'Symbol': symbol,
                    'Timeframe': timeframe,
                    'Timestamp': row['timestamp'],
                    'Confidence': row['confidence'] * 100,
                    'Giá Signal': f"{signal_price:.4f}",
                    'Giá Hiện Tại': f"{curr_price:.4f}",
                    'Entry Zone': zone_str,
                    'Trạng Thái': status_msg,
                    'PnL (Signal)': pnl * 100,
                    'Độ trễ (h)': hours,
                    'TP': row['tp_pct'],
                    'SL': row['sl_pct'],
                    'Meta': meta
                })
            
            if not display_data:
                st.warning("No valid signal data to display.")
            else:
                df_display = pd.DataFrame(display_data)
                
                # --- Filters ---
                c1, c2 = st.columns(2)
                with c1:
                    # Filter by Status
                    all_statuses = sorted(df_display['Trạng Thái'].unique().tolist())
                    selected_status = st.multiselect("Lọc Trạng Thái", options=all_statuses, default=[s for s in all_statuses if "TOO LATE" not in s])
                
                with c2:
                    # Filter by Confidence
                    min_conf = st.slider("Min Confidence", 0, 100, 60)
                
                # Apply Filters
                if selected_status:
                    df_display = df_display[df_display['Trạng Thái'].isin(selected_status)]
                
                df_display = df_display[df_display['Confidence'] >= min_conf]
                
                if df_display.empty:
                    st.info("No signals match filters.")
                else:
                    # Sort by Timestamp DESC
                    df_display = df_display.sort_values('Timestamp', ascending=False)
                    
                    # Main Table
                    st.dataframe(
                        df_display[[
                            'Tín hiệu', 'Symbol', 'Timeframe', 'Confidence', 
                            'Giá Signal', 'Giá Hiện Tại', 'Entry Zone', 
                            'Trạng Thái', 'PnL (Signal)', 'Độ trễ (h)'
                        ]],
                        column_config={
                            "Confidence": st.column_config.ProgressColumn("Conf (%)", min_value=0, max_value=100, format="%.0f%%"),
                            "PnL (Signal)": st.column_config.NumberColumn("PnL (%)", format="%.2f%%"),
                            "Độ trễ (h)": st.column_config.NumberColumn("Wait (h)", format="%.1f h"),
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    

