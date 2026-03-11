
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import os

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processor import BinanceDataProcessor
from ml.inference import InferenceEngine
from ml.signal_manager import SignalManager
from ml.realtime_predictor import get_predictor

# Load predictor for refined filtering
predictor = get_predictor()

# --- Page Configuration ---
st.set_page_config(
    page_title="High Profit Signal Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium feel
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    .status-in-zone {
        color: #00ff00;
        font-weight: bold;
    }
    .status-out-zone {
        color: #ff4b4b;
        font-weight: bold;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #3e4150;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
@st.cache_resource
def get_inference_engine(timeframe):
    try:
        return InferenceEngine(timeframe)
    except:
        return None

@st.cache_resource
def get_processor():
    return BinanceDataProcessor(use_futures=True)

def initialize_state():
    if 'signal_manager' not in st.session_state:
        st.session_state.signal_manager = SignalManager()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = {}
    if 'ticker_cache' not in st.session_state:
        st.session_state.ticker_cache = {}

# --- Data Logic ---
def get_top_symbols(limit=None):
    processor = get_processor()
    try:
        tickers = processor.client.futures_ticker()
        # Sort by quote volume
        sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
        symbols = [t['symbol'] for t in sorted_tickers if t['symbol'].endswith('USDT')]
        if limit:
            return symbols[:limit]
        return symbols
    except Exception as e:
        st.error(f"Error fetching symbols: {e}")
        return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

def fetch_symbol_signal(symbol, timeframe, lookback_days):
    """Fetch signals for a single symbol (sequential)"""
    processor = get_processor()
    engine = get_inference_engine(timeframe)
    if not engine: return None
    
    # Always fetch enough data for indicators (e.g. 200 EMA)
    # 4h: 365 days ~2190 bars
    # 1d: 365 days ~365 bars
    # This is safe for all supported timeframes
    fetch_start = "400 days ago UTC" 
    
    try:
        df = processor.get_historical_data(symbol, timeframe, fetch_start, 'now UTC')
        
        # ENFORCE CLOSED CANDLE LOGIC:
        # Drop the last (forming) candle to prevent repaint/unstable signals
        if not df.empty:
            df = df.iloc[:-1].copy()
            
        if len(df) < 50: # Standard InferenceEngine requirement
            return None
            
        # Instead of importing from data_pipeline (which might have different logic),
        # we'll use the processor's calculation for crossovers to match monitoring logic
        # OR better: use InferenceEngine.calculate_features if it exists
        df = processor.calculate_macd(df)
        
        # Detect crossovers manually to be consistent with monitor
        df['macd_cross_up'] = ((df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1))).astype(int)
        
        # Filter for recent crossovers based on user selection
        cutoff_date = pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)
        if df['timestamp'].dt.tz is None:
             cutoff_date = cutoff_date.tz_localize(None)
             
        recent = df[df['timestamp'] >= cutoff_date]
        
        if recent.empty:
            return None

        cross_up = recent[recent['macd_cross_up'] == 1]
        cross_down = recent[recent['macd_cross_down'] == 1]
        
        if cross_up.empty and cross_down.empty:
            return None
            
        is_up = not cross_up.empty
        if not cross_up.empty and not cross_down.empty:
            is_up = cross_up.index[-1] > cross_down.index[-1]
            
        if is_up:
            row = cross_up.iloc[-1]
            since_cross = df.loc[row.name:]
            highest = since_cross['high'].max()
            lowest = since_cross['low'].min() 
        else:
            row = cross_down.iloc[-1]
            since_cross = df.loc[row.name:]
            highest = since_cross['high'].max()
            lowest = since_cross['low'].min()
        
        # Predict uses the dataframe up to the signal point
        # engine.predict internally calculates all required features correctly for the timeframe
        prediction = engine.predict(symbol, df.loc[:row.name])
        
        if prediction and not prediction.get('error'):
            # The key in InferenceEngine is 'entry_confidence', not 'confidence'
            confidence = prediction.get('entry_confidence', prediction.get('confidence', 0.0))
            
            # Calculate Refined Score (2/3 logic)
            refined_data = {'score': 1.0, 'factors': []}
            try:
                feat_df = predictor.calculate_features(df.loc[:row.name], timeframe=timeframe)
                refined_data = predictor.get_refined_score(feat_df, timeframe=timeframe)
            except Exception as e:
                print(f"Refined score calculation error: {e}")

            return {
                'symbol': symbol,
                'type': 'BULLISH' if is_up else 'BEARISH',
                'timestamp': row['timestamp'],
                'cross_price': row['close'],
                'highest_since': highest,
                'lowest_since': lowest,
                'confidence': float(confidence),
                'refined_score': float(refined_data['score']),
                'refined_factors': refined_data['factors'],
                'sl_pct': prediction.get('sl_pct', 0.02),
                'tp_pct': prediction.get('tp_pct', 0.04),
                'sl_price': prediction.get('sl_price', 0.0),
                'tp_price': prediction.get('tp_price', 0.0)
            }
    except Exception as e:
        print(f"Error processing {symbol} on {timeframe}: {e}")
        return None
    return None

def scan_timeframe(timeframe, symbols, lookback_days):
    signals = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Sequential scan for maximum stability
    for i, symbol in enumerate(symbols):
        status_text.text(f"Scanning {symbol} ({i+1}/{len(symbols)})...")
        progress_bar.progress((i + 1) / len(symbols))
        
        result = fetch_symbol_signal(symbol, timeframe, lookback_days)
        if result:
            signals.append(result)
            # Show live updates if signals are found (optional)
                
    st.session_state.signal_manager.save(timeframe, signals)
    st.session_state.last_refresh[timeframe] = datetime.now()
    return signals

def fetch_current_prices(symbols):
    processor = get_processor()
    try:
        tickers = processor.client.futures_ticker()
        price_map = {t['symbol']: float(t['lastPrice']) for t in tickers}
        return price_map
    except:
        return {}

# --- Constants (From ML Analysis Report) ---
AVG_MAE_STATS = {
    '4h': 0.035,   # ~3.5% (Bin 0.6-0.7)
    '8h': 0.045,   # ~4.5%
    '12h': 0.055,  # ~5.5%
    '1d': 0.065,   # ~6.5%
    '1w': 0.10     # ~10% (Weekly estimate)
}

AVG_MFE_STATS = {
    '4h': 0.11,
    '8h': 0.14,
    '12h': 0.16,
    '1d': 0.22,
    '1w': 0.35     # ~35% (Weekly estimate)
}

def get_entry_status(signal_type, cross_price, current_price, mae_pct, mfe_pct):
    """
    Determine the status of the signal based on current price relative to Entry Zone and MFE.
    """
    status = "UNKNOWN"
    zone_min = 0.0
    zone_max = 0.0
    color = "gray"
    
    if signal_type == 'BULLISH':
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

# --- UI Helper ---
def display_signal_table(timeframe, signals, current_prices, use_refined=False):
    if not signals:
        st.info(f"No active signals cached for {timeframe}. Press 'Load Signals' to scan.")
        return

    data = []
    
    # Get Stats for Timeframe
    mae_stat = AVG_MAE_STATS.get(timeframe, 0.04)
    mfe_stat = AVG_MFE_STATS.get(timeframe, 0.12)
    
    for s in signals:
        symbol = s['symbol']
        curr_price = current_prices.get(symbol, s['cross_price'])
        
        # Calc PnL based on Signal Price (Not User Entry)
        if s['type'] == 'BULLISH': # LONG
            pnl = (curr_price - s['cross_price']) / s['cross_price']
            max_profit = (s.get('highest_since', curr_price) - s['cross_price']) / s['cross_price']
            max_pullback = (s.get('lowest_since', curr_price) - s['cross_price']) / s['cross_price']
        else: # SHORT
            pnl = (s['cross_price'] - curr_price) / s['cross_price']
            max_profit = (s['cross_price'] - s.get('lowest_since', curr_price)) / s['cross_price']
            max_pullback = (s['cross_price'] - s.get('highest_since', curr_price)) / s['cross_price']
            
        # Get Intelligent Status
        status_msg, z_min, z_max, color = get_entry_status(
            s['type'], s['cross_price'], curr_price, mae_stat, mfe_stat
        )
        
        # Time since cross
        cross_time = pd.to_datetime(s['timestamp'])
        time_diff = datetime.now() - cross_time.replace(tzinfo=None) # naive for diff
        hours = time_diff.total_seconds() / 3600
        
        # Format Entry Zone
        if s['type'] == 'BULLISH':
            zone_str = f"{z_min:.4f} - {z_max:.4f}"
        else:
            zone_str = f"{z_max:.4f} - {z_min:.4f}"
        
        data.append({
            'Tín hiệu': "🟢 LONG" if s['type'] == 'BULLISH' else "🔴 SHORT",
            'Symbol': symbol,
            'Confidence': s['confidence'] * 100,
            'Giá Signal': f"{s['cross_price']:.4f}",
            'Giá Hiện Tại': f"{curr_price:.4f}",
            'Entry Zone': zone_str,
            'Trạng Thái': status_msg,
            'Score': s.get('refined_score', 1.0),
            'PnL (Signal)': pnl * 100,
            'Max Profit': max_profit * 100,
            'Max Pullback': max_pullback * 100,
            'Độ trễ (h)': hours,
        })

    df = pd.DataFrame(data)
    
    if df.empty:
        return
        
    # --- Confidence Filter ---
    # User requirement: min confidence 0.5 (50%) for 1w, 0.6 (60%) for others
    min_conf = 50.0 if timeframe == '1w' else 60.0
    df = df[df['Confidence'] >= min_conf]
    
    # --- Refined Filter (If enabled) ---
    if use_refined:
        # 2/3 logic means refined_score >= 0.6
        df = df[df['Score'] >= 0.6]
    
    if df.empty:
        st.info(f"Không có tín hiệu nào đạt mức tin cậy tối thiểu ({min_conf:.0f}%) hoặc không khớp bộ lọc Refined.")
        return
        
    # --- Status Filter (Above Table) ---
    all_statuses = sorted(df['Trạng Thái'].unique().tolist())
    options = ["All"] + all_statuses
    
    selected_status = st.selectbox(
        f"Lọc Trạng Thái ({timeframe.upper()})",
        options=options,
        index=0,
        key=f"filter_status_{timeframe}"
    )
    
    if selected_status != "All":
        df = df[df['Trạng Thái'] == selected_status]
        
    if df.empty:
        st.warning("Không có tín hiệu nào khớp với bộ lọc.")
        return
        
    # Sort by confidence
    df = df.sort_values('Confidence', ascending=False)
    
    # Styled Dataframe
    st.dataframe(
        df,
        column_config={
            "Confidence": st.column_config.ProgressColumn("Confidence (%)", min_value=0, max_value=100, format="%.0f%%"),
            "Score": st.column_config.NumberColumn("Refined Score", format="%.2f"),
            "PnL (Signal)": st.column_config.NumberColumn("PnL (Signal) (%)", format="%.2f%%"),
            "Max Profit": st.column_config.NumberColumn("Max Profit (%)", format="%.2f%%"),
            "Max Pullback": st.column_config.NumberColumn("Max Pullback (%)", format="%.2f%%"),
            "Độ trễ (h)": st.column_config.NumberColumn("Độ trễ", format="%.1f h"),
            "Entry Zone": st.column_config.TextColumn("Entry Zone (Discount -> FOMO)"),
        },
        width='stretch',
        hide_index=True
    )
    
    # --- Detailed Journal Helper (Expanders below table) ---
    st.markdown("### 📝 Trade Journal Helper")
    
    # Filter signals that match table filters first
    journal_signals = []
    for s in signals:
        conf = s['confidence'] * 100
        score = s.get('refined_score', 1.0)
        min_conf = 50.0 if timeframe == '1w' else 60.0
        if conf >= min_conf and (not use_refined or score >= 0.6):
            journal_signals.append(s)
            
    if journal_signals:
        search_query = st.text_input(f"🔍 Tìm Symbol để ghi chú ({timeframe.upper()})", key=f"search_{timeframe}").upper()
        
        for s in journal_signals:
            symbol = s['symbol']
            if search_query and search_query not in symbol:
                continue
                
            score = s.get('refined_score', 1.0)
            conf = s['confidence'] * 100
            
            with st.expander(f"📔 Nhật ký: {symbol} ({s['type']}) - Score: {score:.2f}"):
                factors_str = ""
                for f in s.get('refined_factors', []):
                    icon = "✅" if f['passed'] else "❌"
                    factors_str += f"- {icon} **{f['feature']}**: {f['value']:.4f} (Mục tiêu: {f['target']:.4f})\n"
                
                note = f"""### TRADING NOTE: {symbol} ({s['type']})
- **Timeframe**: {timeframe.upper()}
- **Timestamp**: {s['timestamp']}
- **ML Confidence**: {conf:.1f}%
- **Refined Score**: {score:.2f} ({(score*3):.0f}/3)
- **Success Factors**:
{factors_str}
- **Entry Price**: {s['cross_price']:.4f}
- **Stop Loss**: {s['sl_price']:.4f} ({s['sl_pct']*100:.1f}%)
- **Take Profit**: {s['tp_price']:.4f} ({s['tp_pct']*100:.1f}%)
- **Notes**: [Điền bối cảnh thị trường tại đây]
"""
                st.code(note, language="markdown")
                if st.button("📋 Copy Note to Clipboard", key=f"copy_{timeframe}_{symbol}"):
                    try:
                        # Attempt to use built-in copy (Streamlit >= 1.34)
                        st.copy_to_clipboard(note)
                        st.toast("Đã copy vào bộ nhớ tạm!", icon="✅")
                    except:
                        st.warning("Vui lòng nhấn nút Copy ở góc trên bên phải của khung mã bên trên.")


# --- Main App ---
def main():
    initialize_state()
    
    st.sidebar.title("🚀 Signal Admin")
    use_refined = st.sidebar.checkbox("✨ Refined Filter (2/3 logic)", value=False)
    scan_all = st.sidebar.checkbox("Scan All Symbols", value=False)
    
    if scan_all:
        limit = None
    else:
        limit = st.sidebar.slider("Scan limit (Top by Volume)", 10, 300, 100)
        
    # Removed Global Lookback
    
    st.title("📊 High Profit MACD Dashboard")
    st.caption("Dữ liệu được lấy từ Binance Futures và lọc qua mô hình ML tối ưu 20% TP.")

    # --- Strategy Performance Guide ---
    with st.expander("📊 Chi tiết Hiệu suất & Hướng dẫn Chiến thuật (Cheat Sheet)", expanded=False):
        st.markdown("""
### 📈 Thống kê Hiệu suất (Dựa trên Backtest 8H)
Việc sử dụng Điểm số (Score) giúp bạn cân bằng giữa **Số lượng kèo** và **Chất lượng kèo**.

| Phân hạng | Score | Win Rate (8H) | Số lượng kèo | Đặc điểm |
| :--- | :--- | :--- | :--- | :--- |
| **🏆 ELITE** | 3/3 | **82.2%** | Ít | Cực kỳ an toàn, Volume và Trend đồng thuận tuyệt đối. |
| **🥈 BALANCED** | **>= 2/3** | **79.1%** | Trung bình | **Sweet Spot**: Tăng gấp đôi số kèo mà vẫn giữ Winrate cao. |
| **⚠️ RELAXED** | >= 1/3 | 76.0% | Nhiều | Rủi ro hơn, chỉ dùng khi thị trường có xu hướng mạnh. |
| **❌ RAW** | 0/3 | 73.5% | Rất nhiều | Tín hiệu gốc, không khuyến khích đánh trực tiếp. |

### 💡 Hướng dẫn vào lệnh chuẩn (SOP):
1. **Lọc**: Luôn bật `Refined Filter (2/3 logic)`.
2. **Chọn**: Ưu tiên **ELITE (3/3)**. Nếu ít kèo, dùng **BALANCED (2/3)**.
3. **Vào**: Sử dụng **Limit Price** gợi ý để tối ưu Risk/Reward.
4. **Cắt**: Tuyệt đối cài **SL** và **TP** theo AI gợi ý.
        """)
    
    # Global Refresh Price Button
    if st.button("📈 Cập nhật giá Hiện tại", width='stretch'):
        st.session_state.ticker_cache = fetch_current_prices([])
        st.toast("Updated Prices!")

    current_prices = st.session_state.ticker_cache
    if not current_prices:
        current_prices = fetch_current_prices([])
        st.session_state.ticker_cache = current_prices

    # Tabs for Timeframes
    timeframes = ['4h', '8h', '12h', '1d', '1w']
    tabs = st.tabs([tf.upper() for tf in timeframes])
    
    for i, tf in enumerate(timeframes):
        with tabs[i]:
            # Control Row
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.subheader(f"Tín hiệu Khung {tf.upper()}")
                
            with col2:
                # Per-Timeframe Lookback Configuration
                # Set default lookback based on timeframe
                default_idx = 1 # 7 days
                if tf == '4h': default_idx = 0 # 3 days
                if tf == '1d': default_idx = 3 # 30 days
                if tf == '1w': default_idx = 4 # 60 days
                
                lookback_days = st.selectbox(
                    "Lookback (Days)", 
                    [3, 7, 14, 30, 60, 90, 180, 365], 
                    index=default_idx,
                    key=f"lookback_{tf}"
                )
                
            with col3:
                # Align button with input
                st.write("") # Spacer
                st.write("") # Spacer
                if st.button(f"🔄 Scan {tf.upper()}", key=f"btn_{tf}", width='stretch'):
                    with st.status(f"Scanning {tf}...", expanded=True) as status:
                        symbols = get_top_symbols(limit)
                        signals = scan_timeframe(tf, symbols, lookback_days)
                        status.update(label=f"Scan {tf} Complete! Found {len(signals)} signals.", state="complete", expanded=False)
                    st.rerun()

            signals = st.session_state.signal_manager.get_signals(tf)
            display_signal_table(tf, signals, current_prices, use_refined)

    st.divider()
    st.caption(f"Last Price Sync: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
