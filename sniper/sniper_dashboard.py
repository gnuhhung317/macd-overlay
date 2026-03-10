"""
🎯 AI Sniper Dashboard - Prop-shop Grade
Kiến trúc 3 vùng: Signal Feed | AI X-Ray | Market Pulse
"""
import os, sys, time, json, threading
import pandas as pd
import numpy as np
from pathlib import Path
import concurrent.futures
import ccxt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import dash
from dash import dcc, html, Input, Output, State, ctx, no_update
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

# ============================================================
# CONFIG & PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
SYMBOLS_DIR = BASE_DIR / "data" / "processed" / "symbols_v2"
MODEL_PATH  = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_lgbm_tabular.joblib"
META_PATH   = BASE_DIR / "ml" / "training" / "models" / "1h" / "ensemble_meta.joblib"

# ============================================================
# XAI FEATURE MAP - mô tả ý nghĩa cho con người
# ============================================================
FEATURE_EXPLAIN = {
    'upper_wick_ratio':   {'label': 'Lực xả ngầm (Wick)',    'unit': '%', 'scale': 100, 'bad_high': True},
    'dist_to_ema50_atr':  {'label': 'Kiệt sức EMA50',        'unit': 'xATR', 'scale': 1,   'bad_high': True},
    'volume_ratio':       {'label': 'Khối lượng bất thường',  'unit': 'x TB', 'scale': 1,  'bad_high': False},
    'rsi_14':             {'label': 'RSI 14',                 'unit': '',   'scale': 1,   'bad_high': False},
    'vol_compression':    {'label': 'Nén biến động (BB)',     'unit': '',   'scale': 1,   'bad_high': False},
    'volume_zscore':      {'label': 'Volume Z-Score',         'unit': 'σ',  'scale': 1,   'bad_high': False},
    'btc_is_bull_regime': {'label': 'BTC Bull Regime',        'unit': '',   'scale': 1,   'bad_high': False},
    'adx':                {'label': 'ADX (Xu hướng)',         'unit': '',   'scale': 1,   'bad_high': False},
    'macd_slope':         {'label': 'MACD Slope',             'unit': '',   'scale': 1,   'bad_high': False},
    'stoch_k':            {'label': 'Stoch %K',               'unit': '%',  'scale': 1,   'bad_high': False},
}

# ============================================================
# SHARED STATE
# ============================================================
state_lock = threading.Lock()
shared_state = {
    'signals':     [],    # List[dict] - all active signals
    'btc_regime':  'N/A',
    'market_atr':  0.0,
    'today_wins':  0,
    'today_loss':  0,
    'last_scan':   None,
    'is_scanning': False,
    'live_data':   {},    # {sig_id: {'price': float, 'mfe': float, 'mae': float}}
}

# Result files from sync_worker.py
SIGNALS_JSON = Path(__file__).parent / "signals_live.json"
STATS_JSON   = Path(__file__).parent / "market_stats.json"

def load_json_data():
    """Load latest signals and stats from files."""
    signals = []
    stats = {
        'last_scan': 'N/A',
        'btc_regime': 'N/A',
        'market_atr': 0.0
    }
    
    try:
        if SIGNALS_JSON.exists():
            with open(SIGNALS_JSON, 'r') as f:
                signals = json.load(f)
        if STATS_JSON.exists():
            with open(STATS_JSON, 'r') as f:
                stats = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        
    return signals, stats

# Initialize Binance client for USD-M Futures
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}
})

def sync_symbol_data(symbol):
    """Fetch latest data from Binance and update local parquet file."""
    try:
        # Check if it's already a Binance symbol format
        binance_symbol = symbol.replace('USDT', '/USDT') if symbol.endswith('USDT') else f"{symbol}/USDT"
        
        # Fetch OHLCV (1h timeframe, last 500 candles for stability)
        limit = 500
        ohlcv = exchange.fetch_ohlcv(binance_symbol, timeframe='1h', limit=limit)
        if not ohlcv: return False
        
        # Convert to DataFrame
        df_new = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_new['timestamp'] = pd.to_datetime(df_new['timestamp'], unit='ms')
        df_new['symbol'] = symbol
        
        # Save to parquet
        file_path = SYMBOLS_DIR / f"{symbol}.parquet"
        df_new.to_parquet(file_path, index=False)
        return True
    except Exception as e:
        # Don't print full trace, just the symbol and error
        return False

def calc_atr(df):
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def scan_file(file_path, features, clf, horizon=48):
    try:
        # Optimize: only read necessary columns
        cols = features + ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        avail_cols = pd.read_parquet(file_path).columns
        read_cols = [c for c in cols if c in avail_cols]
        
        df = pd.read_parquet(file_path, columns=read_cols)
        if df.empty: return None

        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.sort_values('timestamp')
        df = df.tail(200).reset_index(drop=True)
        if len(df) < 20: return None
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
        df = df[df['timestamp'] > cutoff]

        # ATR
        df['atr_14'] = calc_atr(df)

        # MFE/MAE for validation
        df['f_high'] = df['high'].shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon-1))
        df['f_low']  = df['low'].shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon-1))
        df['mfe']   = ((df['f_high'] - df['close']) / df['close']) * 100
        df['mae']   = ((df['f_low']  - df['close']) / df['close']) * 100

        # Indicators
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        vol_sma = df['volume'].rolling(20).mean().shift(1)

        # Filter Tầng 1 (Ignition bar)
        c1 = (df['close'] > df['open']) & (df['close'] > df['ema_20'])
        c2 = ((df['close'] - df['open']) / df['open']) > 0.015
        c3 = (df['volume'] > vol_sma * 1.5) & (df['volume'] < vol_sma * 4.0)
        c4 = df.get('rsi_14', pd.Series(60, index=df.index)).between(55, 72)
        if 'rsi_14' not in df.columns:
            c4 = pd.Series(True, index=df.index)

        hits = df[c1 & c2 & c3 & c4].copy()
        if hits.empty: return None

        # Tầng 2: AI
        missing = [f for f in features if f not in hits.columns]
        for m in missing:
            hits[m] = 0
        X = hits[features].apply(pd.to_numeric, errors='coerce').fillna(0)
        probas = clf.predict_proba(X)

        hits['prob_long']  = probas[:, 1]
        hits['prob_short'] = probas[:, 2]
        hits['final_signal'] = 'WAIT'
        hits.loc[hits['prob_long']  > 0.45, 'final_signal'] = 'LONG'
        hits.loc[hits['prob_short'] > 0.45, 'final_signal'] = 'SHORT'

        active = hits[hits['final_signal'] != 'WAIT'].copy()
        if active.empty: return None

        active['atr_pct'] = (active['atr_14'] / active['close']) * 100
        results = []
        for _, row in active.iterrows():
            signal_type = row['final_signal']
            prob = row['prob_long'] if signal_type == 'LONG' else row['prob_short']
            atr_pct = row.get('atr_pct', 2.0)
            tp_pct  = round(atr_pct * 2.0, 2)
            sl_pct  = round(atr_pct * 1.0, 2)

            # Tổng hợp XAI features
            xai = {}
            for feat_key in FEATURE_EXPLAIN:
                if feat_key in row.index:
                    xai[feat_key] = float(row[feat_key])

            results.append({
                'symbol':      row.get('symbol', file_path.stem),
                'timestamp':   row['timestamp'].isoformat(),
                'side':        signal_type,
                'prob':        round(float(prob), 3),
                'prob_long':   round(float(row['prob_long']), 3),
                'prob_short':  round(float(row['prob_short']), 3),
                'price':       float(row['close']),
                'tp_pct':      tp_pct,
                'sl_pct':      sl_pct,
                'mfe':         round(float(row.get('mfe', 0)), 2),
                'mae':         round(float(row.get('mae', 0)), 2),
                'xai':         xai,
                'id':          f"{row.get('symbol', file_path.stem)}_{row['timestamp'].isoformat()}",
            })
            
        if results:
            print(f"   [Scanner] Found {len(results)} signals in {file_path.name}")
        return results
    except Exception:
        return None

def run_scan_thread():
    """Background thread to reload JSON data."""
    global shared_state
    
    # Reload from disk
    all_signals, stats = load_json_data()

    # Tally performance based on is_win field from sync_worker or scanned results
    # We count all-time stats for the loaded feed
    wins   = sum(1 for s in all_signals if s.get('is_win') is True)
    losses = sum(1 for s in all_signals if s.get('is_win') is False)

    with state_lock:
        shared_state['signals']     = all_signals
        shared_state['btc_regime']  = stats.get('btc_regime', 'N/A')
        shared_state['market_atr']  = stats.get('market_atr', 0.0)
        shared_state['today_wins']  = wins
        shared_state['today_loss']  = losses
        shared_state['last_scan']   = stats.get('last_scan', datetime.now().strftime('%H:%M:%S'))
        shared_state['is_scanning'] = False
    
    print(f"   [Dashboard] Reloaded {len(all_signals)} signals from disk.")
    
def track_live_signals_thread():
    """Background thread to update active signals with real-time prices."""
    global shared_state
    print("   [Live] Tracking thread started.", flush=True)
    
    try:
        print("   [Live] Fetching market list from Binance...", flush=True)
        markets = exchange.load_markets()
        # Internal symbols look like BTCUSDT
        # CCXT symbol e.g. BTC/USDT:USDT -> split(':')[0] -> BTC/USDT -> replace('/') -> BTCUSDT
        valid_symbols = {m['symbol'].split(':')[0].replace('/', '') for m in markets.values() 
                         if m.get('type') == 'swap' and m.get('linear')}
        print(f"   [Live] Loaded {len(valid_symbols)} valid swap symbols.", flush=True)
    except Exception as e:
        print(f"   [Live] Warning: Could not pre-fetch market list: {e}", flush=True)
        valid_symbols = None

    while True:
        try:
            with state_lock:
                active_signals = list(shared_state['signals'])
            
            if not active_signals:
                time.sleep(10)
                continue
            
            # Filter unique symbols and ensure they are valid for swap market
            symbols_to_fetch = []
            for s in set(sig['symbol'] for sig in active_signals):
                if valid_symbols is not None and s not in valid_symbols:
                    # Silently skip if we know it's not a swap symbol
                    continue
                symbols_to_fetch.append(s)

            if not symbols_to_fetch:
                time.sleep(10)
                continue

            # CCXT swap symbols usually look like 'BTC/USDT:USDT' or 'BTC/USDT'
            tickers_input = []
            symbol_to_api = {} # Map BTCUSDT -> BTC/USDT:USDT
            
            for s in symbols_to_fetch:
                # Find the original CCXT symbol
                found = False
                for m_key, m_val in markets.items():
                    api_sym = m_val['symbol']
                    # Clean the API symbol to match our internal symbol (e.g. BTC/USDT:USDT -> BTCUSDT)
                    clean_api = api_sym.split(':')[0].replace('/', '')
                    if clean_api == s and m_val.get('type') == 'swap':
                        tickers_input.append(api_sym)
                        symbol_to_api[s] = api_sym
                        found = True
                        break
                if not found:
                    pass # Silent skip
            
            # Fetch prices in chunks
            all_tickers = {}
            for i in range(0, len(tickers_input), 50):
                chunk = tickers_input[i:i+50]
                try:
                    tickers = exchange.fetch_tickers(chunk)
                    all_tickers.update(tickers)
                except Exception as e:
                    # Fallback to fetch single tickers
                    for sym in chunk:
                        try:
                            t = exchange.fetch_ticker(sym)
                            all_tickers[sym] = t
                        except:
                            pass
            
            # Update state
            updated_count = 0
            with state_lock:
                for sig in active_signals:
                    sig_id = sig['id']
                    # Find if we have a ticker for this signal
                    symbol_stem = sig['symbol']
                    api_sym = symbol_to_api.get(symbol_stem)
                    current_ticker = all_tickers.get(api_sym) if api_sym else None

                    if current_ticker:
                        current_price = current_ticker['last']
                        entry_price = sig['price']
                        side = sig['side']
                        
                        # Initialize or get cached live data
                        # We use sig.get('mfe', 0.0) to start from historical excursion if available
                        data = shared_state['live_data'].get(sig_id, {
                            'price': current_price,
                            'mfe': sig.get('mfe', 0.0),
                            'mae': sig.get('mae', 0.0)
                        })
                        
                        data['price'] = current_price
                        
                        # Calculate ROI
                        if side == 'LONG':
                            roi = (current_price - entry_price) / (entry_price + 1e-9) * 100
                        else:
                            roi = (entry_price - current_price) / (entry_price + 1e-9) * 100
                        
                        # Update MFE/MAE
                        data['mfe'] = max(data['mfe'], roi)
                        data['mae'] = min(data['mae'], roi)
                            
                        shared_state['live_data'][sig_id] = data
                        updated_count += 1
            
            if updated_count > 0:
                print(f"   [Live] Updated {updated_count} signals with new prices.", flush=True)
            else:
                if not active_signals:
                    pass
                else:
                    # Debug: Why was nothing updated?
                    pass
                    # print(f"   [Live] No updates. Active: {len(active_signals)}, Fetch: {len(symbols_to_fetch)}", flush=True)

            time.sleep(60)
        except Exception as e:
            print(f"   [Live] Background update error: {e}")
            time.sleep(10)

def trigger_scan():
    with state_lock:
        if shared_state['is_scanning']:
            return
        shared_state['is_scanning'] = True
    t = threading.Thread(target=run_scan_thread, daemon=True)
    t.start()

# ============================================================
# CANDLE COUNTDOWN HELPER
# ============================================================
def get_candle_countdown():
    now = datetime.now()
    mins = now.minute
    secs = now.second
    remaining = (59 - mins) * 60 + (60 - secs)
    m = remaining // 60
    s = remaining % 60
    return f"{m:02d}:{s:02d}"

# ============================================================
# DASH APP
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        'https://fonts.googleapis.com/css2?family=Fira+Code:wght@300;400;500;600;700&display=swap',
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap',
    ],
    title='🎯 AI Sniper Dashboard',
    suppress_callback_exceptions=True,
)

# ============================================================
# STYLES
# ============================================================
COLORS = {
    'bg':        '#0d1117',
    'surface':   '#161b22',
    'surface2':  '#1c2128',
    'border':    '#30363d',
    'long':      '#39d353',
    'long_rgb':  '57, 211, 83',
    'long_bg':   'rgba(57,211,83,0.12)',
    'short':     '#f85149',
    'short_rgb': '248, 81, 73',
    'short_bg':  'rgba(248,81,73,0.12)',
    'text':      '#e6edf3',
    'muted':     '#7d8590',
    'accent':    '#58a6ff',
    'accent_rgb': '88, 166, 255',
    'gold':      '#d29922',
    'gold_rgb':  '210, 153, 34',
    'gauge_bg':  '#21262d',
}

BASE_STYLE = {
    'fontFamily': '"Fira Code", "Courier New", monospace',
    'backgroundColor': COLORS['bg'],
    'color': COLORS['text'],
    'minHeight': '100vh',
}

CARD_BASE = {
    'background':    COLORS['surface'],
    'border':        f"1px solid {COLORS['border']}",
    'borderRadius':  '8px',
    'padding':       '12px',
    'marginBottom':  '8px',
    'cursor':        'pointer',
    'transition':    'all 0.2s ease',
}

# ============================================================
# AUDIO — injected via clientside callback (Dash 4 compatible)
# ============================================================
AUDIO_CLIENTSIDE = """
function(signalType) {
    if (!signalType) return '';
    var ACtx = window.AudioContext || window.webkitAudioContext;
    if (!ACtx) return '';
    var actx = new ACtx();
    var o = actx.createOscillator();
    var g = actx.createGain();
    o.connect(g); g.connect(actx.destination);
    if (signalType === 'LONG') {
        o.type = 'sine';
        o.frequency.setValueAtTime(880, actx.currentTime);
        o.frequency.exponentialRampToValueAtTime(1320, actx.currentTime + 0.15);
        g.gain.setValueAtTime(0.3, actx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + 0.5);
        o.start(); o.stop(actx.currentTime + 0.5);
    } else if (signalType === 'SHORT') {
        o.type = 'sawtooth';
        o.frequency.setValueAtTime(220, actx.currentTime);
        o.frequency.exponentialRampToValueAtTime(80, actx.currentTime + 0.3);
        g.gain.setValueAtTime(0.4, actx.currentTime);
        g.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + 0.6);
        o.start(); o.stop(actx.currentTime + 0.6);
    }
    return '';
}
"""

# ============================================================
# LAYOUT
# ============================================================
app.layout = html.Div([
    # Stores
    dcc.Store(id='prev-signal-count', data=0),
    dcc.Store(id='selected-signal-id', data=None),
    dcc.Store(id='known-signal-ids', data=[]),
    dcc.Store(id='audio-signal-type', data=None),   # drives audio

    # Hidden audio output div
    html.Div(id='audio-output', style={'display': 'none'}),

    # Intervals
    dcc.Interval(id='interval-countdown', interval=5000,  n_intervals=0),  # 5 sec (giảm tải polling)
    dcc.Interval(id='interval-refresh',   interval=60000, n_intervals=0),  # 1 min
    dcc.Interval(id='interval-fast',      interval=30000, n_intervals=0),  # 30 sec live update (trước là 5s - quá nhanh)

    # ── TOP HEADER BAR ─────────────────────────────────────
    html.Div([
        html.Div([
            html.Span('🎯', style={'fontSize': '24px'}),
            html.Span(' AI SNIPER', style={
                'fontSize': '18px', 'fontWeight': '700',
                'color': COLORS['accent'], 'letterSpacing': '3px',
            }),
            html.Span(' DASHBOARD', style={
                'fontSize': '18px', 'fontWeight': '300',
                'color': COLORS['muted'], 'letterSpacing': '2px',
            }),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),

        # Center: Confidence Slider
        html.Div([
            html.Label('MIN CONFIDENCE', style={
                'color': COLORS['muted'], 'fontSize': '10px',
                'letterSpacing': '2px', 'marginBottom': '4px',
            }),
            dcc.Slider(
                id='confidence-slider',
                min=0.45, max=0.95, step=0.05, value=0.60,
                marks={v/100: {'label': f'{v}%', 'style': {'color': COLORS['muted'], 'fontSize': '10px'}}
                       for v in range(45, 100, 10)},
                tooltip={"placement": "top", "always_visible": True},
            ),
        ], style={'flex': '1', 'maxWidth': '500px', 'margin': '0 40px'}),

        # Right: Clock & Candle Countdown
        html.Div([
            html.Div([
                html.Span('CANDLE CLOSE IN', style={'color': COLORS['muted'], 'fontSize': '10px', 'letterSpacing': '2px'}),
                html.Div(id='candle-countdown', style={
                    'fontSize': '28px', 'fontWeight': '700',
                    'color': COLORS['gold'], 'letterSpacing': '4px',
                }),
            ], style={'textAlign': 'right'}),
            html.Div([
                html.Button('⟳ SCAN', id='scan-btn', n_clicks=0, style={
                    'background': COLORS['accent'], 'border': 'none',
                    'color': '#000', 'fontFamily': '"Fira Code", monospace',
                    'fontWeight': '700', 'fontSize': '12px', 'padding': '8px 16px',
                    'borderRadius': '6px', 'cursor': 'pointer', 'letterSpacing': '1px',
                    'marginLeft': '16px',
                }),
            ]),
        ], style={'display': 'flex', 'alignItems': 'center'}),

    ], style={
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between',
        'background': COLORS['surface'], 'borderBottom': f"1px solid {COLORS['border']}",
        'padding': '12px 24px', 'margin': '0',
    }),

    # Scan status
    html.Div(id='scan-status-bar', style={
        'background': '#0d1117', 'padding': '4px 24px',
        'fontSize': '11px', 'color': COLORS['muted'],
        'borderBottom': f"1px solid {COLORS['border']}",
    }),

    # ── MAIN 3-ZONE LAYOUT ──────────────────────────────────
    html.Div([

        # ── ZONE 1: SIGNAL FEED (30%) ───────────────────────
        html.Div([
            html.Div([
                html.Span('SIGNAL FEED', style={
                    'color': COLORS['muted'], 'fontSize': '10px',
                    'letterSpacing': '3px', 'fontWeight': '600',
                }),
                html.Span(id='signal-count-badge', children='0 signals', style={
                    'background': COLORS['surface2'], 'color': COLORS['muted'],
                    'fontSize': '10px', 'padding': '2px 8px',
                    'borderRadius': '12px', 'marginLeft': '8px',
                }),
            ], style={
                'display': 'flex', 'alignItems': 'center',
                'marginBottom': '12px', 'borderBottom': f"1px solid {COLORS['border']}",
                'paddingBottom': '8px',
            }),
            html.Div(id='signal-feed', style={
                'overflowY': 'auto', 'maxHeight': 'calc(100vh - 140px)',
                'paddingRight': '4px',
            }),
        ], style={
            'width': '28%', 'minWidth': '280px',
            'padding': '16px',
            'borderRight': f"1px solid {COLORS['border']}",
            'background': COLORS['bg'],
        }),

        # ── ZONE 2: AI X-RAY (50%) ──────────────────────────
        html.Div([
            html.Div([
                html.Span('AI X-RAY', style={
                    'color': COLORS['muted'], 'fontSize': '10px',
                    'letterSpacing': '3px', 'fontWeight': '600',
                }),
                html.Span('• Explainable AI', style={
                    'color': COLORS['accent'], 'fontSize': '10px',
                    'marginLeft': '8px',
                }),
            ], style={
                'marginBottom': '12px', 'borderBottom': f"1px solid {COLORS['border']}",
                'paddingBottom': '8px',
            }),
            html.Div(id='xray-panel', children=[
                html.Div('← Chọn một tín hiệu để phân tích', style={
                    'color': COLORS['muted'], 'textAlign': 'center',
                    'marginTop': '40px', 'fontSize': '14px',
                }),
            ]),
        ], style={
            'flex': '1', 'padding': '16px',
            'borderRight': f"1px solid {COLORS['border']}",
            'background': COLORS['bg'],
        }),

        # ── ZONE 3: MARKET PULSE (20%) ──────────────────────
        html.Div([
            html.Div([
                html.Span('MARKET PULSE', style={
                    'color': COLORS['muted'], 'fontSize': '10px',
                    'letterSpacing': '3px', 'fontWeight': '600',
                }),
            ], style={
                'marginBottom': '12px', 'borderBottom': f"1px solid {COLORS['border']}",
                'paddingBottom': '8px',
            }),

            # BTC Compass
            html.Div([
                html.Div('BTC REGIME', style={'color': COLORS['muted'], 'fontSize': '10px', 'letterSpacing': '2px'}),
                html.Div(id='btc-regime-text', style={
                    'fontSize': '13px', 'fontWeight': '600',
                    'color': COLORS['text'], 'marginTop': '4px',
                }),
                dcc.Graph(id='btc-gauge', config={'displayModeBar': False},
                          style={'height': '120px'}),
            ], style={
                'background': COLORS['surface'],
                'border': f"1px solid {COLORS['border']}",
                'borderRadius': '8px', 'padding': '12px',
                'marginBottom': '12px',
            }),

            # Market ATR
            html.Div([
                html.Div('MARKET VOLATILITY (ATR%)', style={'color': COLORS['muted'], 'fontSize': '10px', 'letterSpacing': '2px'}),
                html.Div(id='market-atr-text', style={
                    'fontSize': '22px', 'fontWeight': '700',
                    'color': COLORS['accent'], 'marginTop': '4px',
                    'letterSpacing': '2px',
                }),
                html.Div('avg ATR/Close across market', style={
                    'color': COLORS['muted'], 'fontSize': '10px', 'marginTop': '2px',
                }),
            ], style={
                'background': COLORS['surface'],
                'border': f"1px solid {COLORS['border']}",
                'borderRadius': '8px', 'padding': '12px',
                'marginBottom': '12px',
            }),

            # Today's Stats
            html.Div([
                html.Div('TODAY\'S AI PERFORMANCE', style={'color': COLORS['muted'], 'fontSize': '10px', 'letterSpacing': '2px', 'marginBottom': '8px'}),
                html.Div(id='today-stats'),
            ], style={
                'background': COLORS['surface'],
                'border': f"1px solid {COLORS['border']}",
                'borderRadius': '8px', 'padding': '12px',
                'marginBottom': '12px',
            }),

            # Last scan info
            html.Div([
                html.Div('LAST SCAN', style={'color': COLORS['muted'], 'fontSize': '10px', 'letterSpacing': '2px'}),
                html.Div(id='last-scan-time', style={'color': COLORS['text'], 'fontSize': '12px', 'marginTop': '4px'}),
            ], style={
                'background': COLORS['surface'],
                'border': f"1px solid {COLORS['border']}",
                'borderRadius': '8px', 'padding': '12px',
            }),
        ], style={
            'width': '22%', 'minWidth': '200px',
            'padding': '16px',
            'background': COLORS['bg'],
        }),

    ], style={
        'display': 'flex',
        'height': 'calc(100vh - 96px)',
        'overflow': 'hidden',
    }),

], style=BASE_STYLE)


# ============================================================
# CALLBACKS
# ============================================================

# 1. Candle Countdown
@app.callback(Output('candle-countdown', 'children'),
              Input('interval-countdown', 'n_intervals'))
def update_countdown(_):
    return get_candle_countdown()


# 2. Trigger scan on button or interval
@app.callback(
    Output('scan-status-bar', 'children'),
    Input('scan-btn', 'n_clicks'),
    Input('interval-refresh', 'n_intervals'),
    prevent_initial_call=False,
)
def on_scan(n_clicks, n_intervals):
    trigger_scan()
    with state_lock:
        scanning = shared_state['is_scanning']
    if scanning:
        return '⏳ Đang quét... Parquet files đang được xử lý.'
    with state_lock:
        last = shared_state['last_scan']
    return f'✅ Scan hoàn tất lúc {last}  |  Tự động refresh mỗi 60s' if last else '💤 Chưa scan. Nhấn ⟳ SCAN để bắt đầu.'


# 3. Render signal feed
@app.callback(
    Output('signal-feed', 'children'),
    Output('signal-count-badge', 'children'),
    Output('prev-signal-count', 'data'),
    Output('known-signal-ids', 'data'),
    # Đã bỏ audio-signal-type khỏi đây để khớp signature cũ của trình duyệt
    Input('interval-refresh', 'n_intervals'),
    Input('scan-btn', 'n_clicks'),
    State('confidence-slider', 'value'),
    State('prev-signal-count', 'data'),
    State('known-signal-ids', 'data'),
)
def render_signals(_, n_clicks, min_conf, prev_count, known_ids):
    time.sleep(0.5)  # small delay to let scanner thread complete
    with state_lock:
        raw_signals = list(shared_state['signals'])

    # Filter by confidence
    signals = [s for s in raw_signals if s['prob'] >= min_conf]
    # Keep only last 7 days
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
    signals = [s for s in signals if pd.to_datetime(s['timestamp']) > cutoff]
    # Sort newest first
    signals.sort(key=lambda x: x['timestamp'], reverse=True)

    known_set = set(known_ids)
    new_ids = [s['id'] for s in signals if s['id'] not in known_set]
    all_ids = [s['id'] for s in signals]

    cards = []
    for sig in signals[:50]:  # Max 50 cards
        is_long = sig['side'] == 'LONG'
        border_color = COLORS['long'] if is_long else COLORS['short']
        bg_color     = COLORS['long_bg'] if is_long else COLORS['short_bg']
        sig_icon     = '🚀' if is_long else '💀'
        sig_label    = 'LONG' if is_long else 'SHORT'
        pct_conf     = round(sig['prob'] * 100)

        ts = pd.to_datetime(sig['timestamp'])
        ts_str = ts.strftime('%d/%m %H:%M')

        card = html.Div([
            # Header row
            html.Div([
                html.Div([
                    html.Span(sig['symbol'], style={
                        'fontWeight': '700', 'fontSize': '14px',
                        'color': COLORS['text'], 'letterSpacing': '1px',
                    }),
                    html.Span(ts_str, style={
                        'color': COLORS['muted'], 'fontSize': '10px',
                        'marginLeft': '8px',
                    }),
                ]),
                html.Div([
                    html.Span(f'{sig_icon} {sig_label}', style={
                        'background': border_color,
                        'color': '#000' if is_long else '#fff',
                        'fontSize': '10px', 'fontWeight': '700',
                        'padding': '2px 8px', 'borderRadius': '12px',
                        'letterSpacing': '1px',
                    }),
                    html.Span(f'{pct_conf}%', style={
                        'color': border_color, 'fontSize': '12px',
                        'fontWeight': '700', 'marginLeft': '6px',
                    }),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={
                'display': 'flex', 'justifyContent': 'space-between',
                'alignItems': 'center', 'marginBottom': '8px',
            }),

            # Confidence bar
            html.Div([
                html.Div(style={
                    'height': '3px', 'background': border_color,
                    'width': f'{pct_conf}%', 'borderRadius': '2px',
                    'transition': 'width 0.5s ease',
                }),
            ], style={
                'background': COLORS['gauge_bg'], 'borderRadius': '2px',
                'height': '3px', 'marginBottom': '8px',
            }),

            # TP / SL / ROI Row
            html.Div([
                html.Div([
                    html.Span('TP ', style={'color': COLORS['muted'], 'fontSize': '10px'}),
                    html.Span(f"+{sig['tp_pct']}%", style={'color': COLORS['long'], 'fontWeight': '600', 'fontSize': '12px'}),
                ]),
                html.Div([
                    html.Span('SL ', style={'color': COLORS['muted'], 'fontSize': '10px'}),
                    html.Span(f"-{sig['sl_pct']}%", style={'color': COLORS['short'], 'fontWeight': '600', 'fontSize': '12px'}),
                ]),
                # Live ROI
                html.Div((lambda data: [
                    html.Span('ROI ', style={'color': COLORS['muted'], 'fontSize': '10px'}),
                    html.Span(f"{((data['price'] - sig['price']) / sig['price'] * 100):+.2f}%" if sig['side'] == 'LONG' else f"{((sig['price'] - data['price']) / sig['price'] * 100):+.2f}%", 
                              style={'color': COLORS['long'] if (data['price'] >= sig['price'] if sig['side'] == 'LONG' else data['price'] <= sig['price']) else COLORS['short'], 'fontWeight': '700', 'fontSize': '12px'})
                ] if sig['id'] in shared_state['live_data'] else [
                    html.Span('ROI ', style={'color': COLORS['muted'], 'fontSize': '10px'}),
                    html.Span('0.00%', style={'color': COLORS['muted'], 'fontSize': '12px'})
                ])(shared_state['live_data'].get(sig['id'], {}))),
                
                html.Div([
                    html.Span('@ ', style={'color': COLORS['muted'], 'fontSize': '10px'}),
                    html.Span(f"${(shared_state['live_data'].get(sig['id'], {}).get('price', sig['price'])):,.4f}", style={'color': COLORS['accent'], 'fontSize': '11px'}),
                ]),
            ], style={'display': 'flex', 'gap': '12px', 'alignItems': 'center', 'flexWrap': 'wrap'}),

        ], id={'type': 'signal-card', 'index': sig['id']},
        n_clicks=0,
        style={
            **CARD_BASE,
            'borderLeft': f"3px solid {border_color}",
            'background': bg_color if sig['id'] in new_ids else COLORS['surface'],
        })
        cards.append(card)

    badge = f'{len(signals)} signals'
    
    return cards, badge, len(signals), all_ids


# 4. Confidence slider also re-renders signals (via scan status -> but we delegate to interval)
@app.callback(
    Output('signal-feed', 'children', allow_duplicate=True),
    Output('signal-count-badge', 'children', allow_duplicate=True),
    Input('confidence-slider', 'value'),
    prevent_initial_call=True,
)
def filter_by_confidence(min_conf):
    with state_lock:
        raw_signals = list(shared_state['signals'])

    signals = [s for s in raw_signals if s['prob'] >= min_conf]
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
    signals = [s for s in signals if pd.to_datetime(s['timestamp']) > cutoff]
    signals.sort(key=lambda x: x['timestamp'], reverse=True)

    cards = []
    for sig in signals[:50]:
        is_long = sig['side'] == 'LONG'
        border_color = COLORS['long'] if is_long else COLORS['short']
        bg_color     = COLORS['long_bg'] if is_long else COLORS['short_bg']
        sig_icon     = '🚀' if is_long else '💀'
        sig_label    = 'LONG' if is_long else 'SHORT'
        pct_conf     = round(sig['prob'] * 100)
        ts = pd.to_datetime(sig['timestamp'])
        ts_str = ts.strftime('%d/%m %H:%M')

        card = html.Div([
            html.Div([
                html.Div([
                    html.Span(sig['symbol'], style={'fontWeight': '700', 'fontSize': '14px', 'color': COLORS['text'], 'letterSpacing': '1px'}),
                    html.Span(ts_str, style={'color': COLORS['muted'], 'fontSize': '10px', 'marginLeft': '8px'}),
                ]),
                html.Div([
                    html.Span(f'{sig_icon} {sig_label}', style={
                        'background': border_color, 'color': '#000' if is_long else '#fff',
                        'fontSize': '10px', 'fontWeight': '700', 'padding': '2px 8px',
                        'borderRadius': '12px', 'letterSpacing': '1px',
                    }),
                    html.Span(f'{pct_conf}%', style={'color': border_color, 'fontSize': '12px', 'fontWeight': '700', 'marginLeft': '6px'}),
                ], style={'display': 'flex', 'alignItems': 'center'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '8px'}),
            html.Div([html.Div(style={'height': '3px', 'background': border_color, 'width': f'{pct_conf}%', 'borderRadius': '2px'})],
                     style={'background': COLORS['gauge_bg'], 'borderRadius': '2px', 'height': '3px', 'marginBottom': '8px'}),
            html.Div([
                html.Div([html.Span('TP ', style={'color': COLORS['muted'], 'fontSize': '10px'}),
                          html.Span(f"+{sig['tp_pct']}%", style={'color': COLORS['long'], 'fontWeight': '600', 'fontSize': '12px'})]),
                html.Div([html.Span('SL ', style={'color': COLORS['muted'], 'fontSize': '10px'}),
                          html.Span(f"-{sig['sl_pct']}%", style={'color': COLORS['short'], 'fontWeight': '600', 'fontSize': '12px'})]),
                html.Div([html.Span('@ ', style={'color': COLORS['muted'], 'fontSize': '10px'}),
                          html.Span(f"${sig['price']:,.4f}", style={'color': COLORS['accent'], 'fontSize': '11px'})]),
            ], style={'display': 'flex', 'gap': '16px', 'alignItems': 'center'}),
        ], id={'type': 'signal-card', 'index': sig['id']},
        n_clicks=0,
        style={**CARD_BASE, 'borderLeft': f"3px solid {border_color}"})
        cards.append(card)

    return cards, f'{len(signals)} signals'


# 5. Select signal -> update X-Ray panel
@app.callback(
    Output('xray-panel', 'children'),
    Output('selected-signal-id', 'data'),
    Output('audio-signal-type', 'data'), # Khôi phục lại signature 3 output cho trình duyệt khỏi lỗi
    Input({'type': 'signal-card', 'index': dash.ALL}, 'n_clicks'),
    Input('interval-fast', 'n_intervals'),
    State('selected-signal-id', 'data'),
    prevent_initial_call=False,
)
def select_signal(clicks, n_intervals, selected_id):
    # Determine the trigger
    trigger = ctx.triggered_id
    
    # If it's a card click
    if isinstance(trigger, dict) and trigger.get('type') == 'signal-card':
        selected_id = trigger.get('index')
    
    if not selected_id:
        return no_update, no_update, no_update

    # Find this signal
    with state_lock:
        signals = list(shared_state['signals'])

    sig = next((s for s in signals if s['id'] == selected_id), None)
    if sig is None:
        return html.Div([
            html.H3("SIGNAL EXPIRED", style={'color': COLORS['short'], 'textAlign': 'center', 'marginTop': '50px'}),
            html.P("This setup is no longer in the active feed. Please select a fresh signal.", 
                   style={'color': COLORS['muted'], 'textAlign': 'center'})
        ], style={'padding': '40px'}), no_update, no_update

    return build_xray_panel(sig), selected_id, no_update # Trả về no_update để không kêu bậy


def build_gauge(label, value, max_val, color, unit='', desc=''):
    pct = min(max(value / max_val, 0), 1) * 100
    bar_color = color

    return html.Div([
        html.Div([
            html.Span(label, style={'color': COLORS['muted'], 'fontSize': '11px', 'letterSpacing': '1px'}),
            html.Span(f'{value:.2f}{unit}', style={
                'color': color, 'fontSize': '13px', 'fontWeight': '700',
                'fontFamily': '"Fira Code", monospace',
            }),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'}),
        html.Div([
            html.Div(style={
                'height': '8px', 'background': bar_color,
                'width': f'{pct:.1f}%', 'borderRadius': '4px',
                'transition': 'width 0.8s ease',
                'boxShadow': f'0 0 8px {bar_color}55',
            }),
        ], style={
            'background': COLORS['gauge_bg'], 'borderRadius': '4px',
            'height': '8px', 'marginBottom': '4px',
        }),
        html.Div(desc, style={'color': COLORS['muted'], 'fontSize': '10px', 'marginBottom': '12px'}),
    ])


def build_xray_panel(sig):
    is_long = sig['side'] == 'LONG'
    accent  = COLORS['long'] if is_long else COLORS['short']
    accent_rgb = COLORS['long_rgb'] if is_long else COLORS['short_rgb']
    sig_icon = '🚀' if is_long else '💀'

    xai = sig.get('xai', {})

    wick    = xai.get('upper_wick_ratio', 0)
    ema_dist = xai.get('dist_to_ema50_atr', 0)
    vol_r   = xai.get('volume_ratio', 1)
    rsi     = xai.get('rsi_14', 50)
    adx     = xai.get('adx', 20)
    stoch   = xai.get('stoch_k', 50)
    btc_bull = xai.get('btc_is_bull_regime', 0)

    # Interpretive texts
    wick_desc = f"{'⚠️ Râu trên dài — xả ngầm' if wick > 0.4 else '✅ Nến khoẻ, ít râu'}"
    ema_desc  = f"{'⚠️ Kiệt sức, xa EMA50 ' + str(round(abs(ema_dist), 1)) + 'x ATR' if abs(ema_dist) > 2 else '✅ Gần EMA50 — còn sức đẩy'}"
    vol_desc  = f"{'🔥 Vol gấp ' + str(round(vol_r, 1)) + 'x trung bình — xác nhận lực' if vol_r > 2 else '⚠️ Vol bình thường'}"
    rsi_desc  = f"RSI {rsi:.0f} — {'Mua quá' if rsi > 70 else ('Kiệt sức' if rsi < 40 else 'Hợp lý')}"

    # BTC context
    btc_context_color = COLORS['long'] if btc_bull else COLORS['short']
    btc_context_text  = '🟢 BTC đang Bull' if btc_bull else '🔴 BTC không Bull — Thận trọng'

    # AI verdict
    verdict_text = ''
    reasons = []
    if wick > 0.4:
        reasons.append('râu xả cao')
    if abs(ema_dist) > 2:
        reasons.append(f'xa EMA50 {abs(ema_dist):.1f}x ATR')
    if vol_r > 3:
        reasons.append(f'vol gấp {vol_r:.1f}x')
    if reasons:
        verdict_text = f"AI phát hiện: {', '.join(reasons)}."
    else:
        verdict_text = "Setups cơ bản ổn, không có dấu hiệu bất thường nổi trội."

    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span(f'{sig_icon} {sig["symbol"]}', style={
                    'fontSize': '22px', 'fontWeight': '700', 'color': COLORS['text'],
                    'letterSpacing': '2px',
                }),
                html.Span(f'{sig["side"]}', style={
                    'background': accent, 'color': '#000' if is_long else '#fff',
                    'fontSize': '11px', 'fontWeight': '700', 'padding': '3px 10px',
                    'borderRadius': '12px', 'marginLeft': '10px', 'letterSpacing': '1px',
                }),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Div(f"AI Confidence: {round(sig['prob']*100)}%", style={
                'color': accent, 'fontSize': '20px', 'fontWeight': '700',
                'fontFamily': '"Fira Code", monospace',
            }),
        ], style={
            'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
            'background': COLORS['surface2'], 'borderRadius': '8px',
            'padding': '12px 16px', 'marginBottom': '16px',
            'border': f"1px solid rgba({accent_rgb}, 0.25)",
        }),

        # Position Details
        html.Div([
            html.Div('POSITION DETAILS', style={
                'color': COLORS['muted'], 'fontSize': '10px',
                'letterSpacing': '2px', 'marginBottom': '8px',
            }),
            html.Div([
                _detail_col('Entry', f"${sig['price']:,.4f}", COLORS['accent']),
                _detail_col('TP', f"+{sig['tp_pct']}%", COLORS['long']),
                _detail_col('SL', f"-{sig['sl_pct']}%", COLORS['short']),
                _detail_col('Live PnL', (lambda data: f"{((data['price'] - sig['price']) / sig['price'] * 100):+.2f}%" if sig['side'] == 'LONG' else f"{((sig['price'] - data['price']) / sig['price'] * 100):+.2f}%")(shared_state['live_data'].get(sig['id'], {'price': sig['price']})), COLORS['accent']),
            ], style={'display': 'flex', 'gap': '0', 'justifyContent': 'space-around'}),
            html.Div([
                _detail_col('MFE (Live)', f"{shared_state['live_data'].get(sig['id'], {}).get('mfe', 0.0):+.2f}%", COLORS['long']),
                _detail_col('MAE (Live)', f"{shared_state['live_data'].get(sig['id'], {}).get('mae', 0.0):+.2f}%", COLORS['short']),
                _detail_col('R:R', f"1:{round(sig['tp_pct']/sig['sl_pct'],1)}", COLORS['gold']),
            ], style={'display': 'flex', 'gap': '0', 'justifyContent': 'space-around', 'marginTop': '8px'}),
        ], style={
            'background': COLORS['surface'], 'border': f"1px solid {COLORS['border']}",
            'borderRadius': '8px', 'padding': '12px 16px', 'marginBottom': '16px',
        }),

        # Gauges Section
        html.Div([
            html.Div('AI FEATURE ANALYSIS', style={
                'color': COLORS['muted'], 'fontSize': '10px',
                'letterSpacing': '2px', 'marginBottom': '12px',
            }),
            build_gauge('Lực xả ngầm (Wick Rejection)', wick * 100, 100, COLORS['short'], '%', wick_desc),
            build_gauge('Kiệt sức EMA50 (Dist)', abs(ema_dist), 5, COLORS['gold'], 'xATR', ema_desc),
            build_gauge('Khối lượng bất thường (Vol)', vol_r, 6, COLORS['accent'], 'x', vol_desc),
            build_gauge('Momentum (RSI)', rsi, 100, COLORS['long'] if rsi < 70 else COLORS['short'], '', rsi_desc),
            build_gauge('Xu hướng (ADX)', adx, 60, COLORS['accent'], '', f"ADX {adx:.0f} — {'Xu hướng mạnh' if adx > 25 else 'Sideway'}"),
        ], style={
            'background': COLORS['surface'], 'border': f"1px solid {COLORS['border']}",
            'borderRadius': '8px', 'padding': '16px', 'marginBottom': '16px',
        }),

        # BTC Context
        html.Div([
            html.Div('BTC CONTEXT', style={'color': COLORS['muted'], 'fontSize': '10px', 'letterSpacing': '2px', 'marginBottom': '6px'}),
            html.Div(btc_context_text, style={'color': btc_context_color, 'fontSize': '13px', 'fontWeight': '600'}),
        ], style={
            'background': COLORS['surface'], 'border': f"1px solid {COLORS['border']}",
            'borderRadius': '8px', 'padding': '12px 16px', 'marginBottom': '16px',
        }),

        # AI Verdict
        html.Div([
            html.Div('AI VERDICT', style={'color': COLORS['muted'], 'fontSize': '10px', 'letterSpacing': '2px', 'marginBottom': '6px'}),
            html.Div(verdict_text, style={
                'color': COLORS['text'], 'fontSize': '13px', 'lineHeight': '1.6',
                'fontFamily': '"Inter", sans-serif',
            }),
            html.Div(
                f"{'Vào lệnh với sự TỰ TIN — Các chỉ số hội tụ rõ ràng ✅' if len(reasons) >= 2 else 'Cân nhắc kỹ — Tín hiệu chưa đủ mạnh ⚠️'}",
                style={
                    'color': COLORS['long'] if len(reasons) >= 2 else COLORS['gold'],
                    'fontSize': '12px', 'fontWeight': '700', 'marginTop': '8px',
                    'fontFamily': '"Fira Code", monospace',
                }
            ),
        ], style={
            'background': f"rgba({accent_rgb}, 0.05)",
            'border': f"1px solid rgba({accent_rgb}, 0.25)",
            'borderRadius': '8px', 'padding': '12px 16px',
        }),

    ])


def _detail_col(label, value, color):
    return html.Div([
        html.Div(label, style={'color': COLORS['muted'], 'fontSize': '10px', 'letterSpacing': '1px'}),
        html.Div(value, style={'color': color, 'fontSize': '16px', 'fontWeight': '700',
                               'fontFamily': '"Fira Code", monospace', 'marginTop': '2px'}),
    ], style={'textAlign': 'center', 'padding': '8px 12px'})


# 6. Market Pulse callbacks
@app.callback(
    Output('btc-regime-text', 'children'),
    Output('btc-gauge', 'figure'),
    Output('market-atr-text', 'children'),
    Output('today-stats', 'children'),
    Output('last-scan-time', 'children'),
    Input('interval-refresh', 'n_intervals'),
    Input('scan-btn', 'n_clicks'),
)
def update_market_pulse(_, __):
    with state_lock:
        regime = shared_state['btc_regime']
        m_atr  = shared_state['market_atr']
        wins   = shared_state['today_wins']
        losses = shared_state['today_loss']
        last   = shared_state['last_scan']

    # BTC Gauge (speedometer style)
    color = COLORS['long'] if 'Uptrend' in regime else (
            COLORS['gold'] if 'Sideway' in regime else COLORS['short'])
    val = 80 if 'Uptrend' in regime else (50 if 'Sideway' in regime else 20)
    fig = go.Figure(go.Indicator(
        mode='gauge',
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': COLORS['muted'], 'tickwidth': 1, 'tickfont': {'size': 8}},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': COLORS['surface'],
            'bordercolor': COLORS['border'],
            'steps': [
                {'range': [0, 33],  'color': f"rgba({COLORS['short_rgb']}, 0.15)"},
                {'range': [33, 66], 'color': f"rgba({COLORS['gold_rgb']}, 0.15)"},
                {'range': [66, 100],'color': f"rgba({COLORS['long_rgb']}, 0.15)"},
            ],
            'threshold': {'line': {'color': color, 'width': 2}, 'thickness': 0.8, 'value': val},
        },
    ))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=110,
        font={'color': COLORS['muted'], 'size': 9},
    )

    # ATR
    atr_text = f'{m_atr:.3f}%' if m_atr > 0 else 'N/A'

    # Today stats
    total = wins + losses
    acc   = wins / total * 100 if total > 0 else 0
    stats = html.Div([
        html.Div([
            html.Div([
                html.Div(f'{wins}', style={'color': COLORS['long'],  'fontSize': '24px', 'fontWeight': '700'}),
                html.Div('WIN', style={'color': COLORS['muted'], 'fontSize': '9px', 'letterSpacing': '2px'}),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Div(f'{losses}', style={'color': COLORS['short'], 'fontSize': '24px', 'fontWeight': '700'}),
                html.Div('LOSS', style={'color': COLORS['muted'], 'fontSize': '9px', 'letterSpacing': '2px'}),
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Div(f'{acc:.0f}%', style={'color': COLORS['accent'] if acc >= 55 else COLORS['gold'], 'fontSize': '24px', 'fontWeight': '700'}),
                html.Div('ACC', style={'color': COLORS['muted'], 'fontSize': '9px', 'letterSpacing': '2px'}),
            ], style={'textAlign': 'center'}),
        ], style={'display': 'flex', 'justifyContent': 'space-around'}),
        html.Div([
            html.Div(style={
                'height': '4px',
                'background': f'linear-gradient(to right, {COLORS["long"]} {acc:.0f}%, {COLORS["short"]} {acc:.0f}%)',
                'borderRadius': '2px', 'marginTop': '10px',
            }),
        ]),
    ])

    last_text = last if last else '— Chưa scan'
    return regime, fig, atr_text, stats, last_text


# 7. Dedicated Audio Alert when new signals appear
@app.callback(
    Output('audio-signal-type', 'data', allow_duplicate=True),
    Input('known-signal-ids', 'data'),
    prevent_initial_call=True,
)
def trigger_audio_on_new_signals(known_ids):
    if not known_ids:
        return no_update
    
    with state_lock:
        signals = list(shared_state['signals'])
        
    if not signals:
        return no_update

    # Lấy kèo mới nhất (timestamp lớn nhất)
    latest_sig = max(signals, key=lambda x: x['timestamp'])
    
    # Nếu kèo mới nhất này nằm trong danh sách IDs vừa được cập nhật, và nó "mới"
    # Thực tế known_ids thay đổi là tín hiệu có scan mới.
    # Để đơn giản, ta chỉ kêu nếu kèo mới nhất vừa xuất hiện.
    return latest_sig['side']


# ============================================================
# CLIENTSIDE CALLBACK: Audio Alert
# ============================================================
app.clientside_callback(
    AUDIO_CLIENTSIDE,
    Output('audio-output', 'children'),
    Input('audio-signal-type', 'data'),
    prevent_initial_call=True,
)

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("🎯 AI Sniper Dashboard đang khởi động...")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Data:  {SYMBOLS_DIR}")
    print("   Mở trình duyệt tại: http://127.0.0.1:8050")

    # Kick off initial scan
    trigger_scan()
    
    # Start live tracking thread
    print("   [Main] Starting Live Tracking Thread...", flush=True)
    lt = threading.Thread(target=track_live_signals_thread, daemon=True)
    lt.start()

    # Suppress flask logs to see our our own logs better
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(debug=False, host='127.0.0.1', port=8050)
