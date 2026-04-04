import pandas as pd
import numpy as np
import glob
import argparse
import lightgbm as lgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PINE-SCRIPT SYNCHRONIZED EXTRACTOR
# ==========================================
class RealDataQuantExtractor:
    def __init__(
        self,
        tp_level=1.6,
        max_hold_bars=24,
        min_mid_candles=6,
        min_price_pct=3.0,
        entry_pullback=0.0,
        min_rr=1.0,
    ):
        self.tp_level = tp_level
        self.max_hold_bars = max_hold_bars
        self.min_mid_candles = min_mid_candles
        self.min_price_pct = min_price_pct
        self.entry_pullback = entry_pullback
        self.min_rr = min_rr
        self.dataset = []

    def get_fill_index(self, side, entry_p, f_lows, f_highs):
        # Market-style entry (no pullback) is considered filled immediately at signal bar.
        if self.entry_pullback <= 0:
            return 0

        if side == 1:
            for k, low_t in enumerate(f_lows):
                if low_t <= entry_p:
                    return k
        else:
            for k, high_t in enumerate(f_highs):
                if high_t >= entry_p:
                    return k
        return None

    def check_ph(self, h1, h2, h3, l1, l2, l3):
        return (
            (h2 > h1 and h2 > h3 and l2 > l1 and l2 > l3) or
            (h2 >= h1 and h2 > h3 and l2 > l3 and l2 < l1) or
            (h2 > h1 and h2 >= h3 and l2 < l3 and l2 > l1) or
            (h2 >= h3 and h2 > h1 and l2 <= l3 and l2 > l1) or
            (h2 >= h3 and h2 >= h1 and l2 <= l3 and l2 > l1)
        )

    def check_pl(self, h1, h2, h3, l1, l2, l3):
        return (
            (l2 < l1 and l2 < l3 and h2 < h1 and h2 < h3) or
            (h2 >= h1 and h2 < h3 and l2 < l3 and l2 <= l1) or
            (l2 < l1 and l2 < l3 and h2 < h1 and h2 >= h3) or
            (h2 <= h3 and h2 < h1 and l2 >= l3 and l2 <= l1) or
            (h2 >= h3 and h2 < h1 and l2 <= l3 and l2 <= l1)
        )

    def is_valid_bc_long(self, close_arr, low_arr, b_idx, c_idx):
        if c_idx <= b_idx:
            return True
        for j in range(b_idx + 1, c_idx + 1):
            if close_arr[j] < low_arr[j - 1]:
                return False
        return True

    def is_valid_bc_short(self, close_arr, high_arr, b_idx, c_idx):
        if c_idx <= b_idx:
            return True
        for j in range(b_idx + 1, c_idx + 1):
            if close_arr[j] > high_arr[j - 1]:
                return False
        return True

    def has_all_cut(self, high_arr, low_arr, a_idx, p_a, b_idx, p_b):
        if b_idx <= a_idx:
            return False
        slope = (p_b - p_a) / max(1, b_idx - a_idx)
        intercept = p_a - slope * a_idx
        for j in range(a_idx + 1, b_idx):
            line_val = slope * j + intercept
            if not (line_val >= low_arr[j] and line_val <= high_arr[j]):
                return False
        return True

    def is_quality_wave(self, a_idx, p_a, b_idx, p_b):
        mid = b_idx - a_idx - 1
        pct = abs(p_b - p_a) / (abs(p_a) + 1e-8) * 100
        return (mid >= self.min_mid_candles) or (pct >= self.min_price_pct)

    def get_dynamic_zscore(self, series, span=200):
        ewm_mean = series.ewm(span=span, adjust=False).mean()
        ewm_std = series.ewm(span=span, adjust=False).std()
        return (series - ewm_mean) / (ewm_std + 1e-8)

    def extract(self, df, coin_name):
        df = df.sort_values('timestamp' if 'timestamp' in df.columns else df.index).reset_index(drop=True)
        h, l, c, o = df['high'].values, df['low'].values, df['close'].values, df['open'].values
        v = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
        ts = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
        
        # Core Indicators
        tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
        atr_20 = pd.Series(tr).rolling(20).mean().values # Dùng SMA20 cho ATR giống Pine
        range_sma_20 = pd.Series(h - l).rolling(20).mean().values
        vol_sma_20 = pd.Series(v).rolling(20).mean().values # SMA20 Volume giống Pine
        
        ema_20 = pd.Series(c).ewm(span=20, adjust=False).mean()
        ema_50 = pd.Series(c).ewm(span=50, adjust=False).mean()
        ema_200 = pd.Series(c).ewm(span=200, adjust=False).mean()
        
        z_trend_20_50 = self.get_dynamic_zscore((ema_20 - ema_50) / (ema_50 + 1e-8)).values
        z_price_ema200 = self.get_dynamic_zscore((pd.Series(c) - ema_200) / (ema_200 + 1e-8)).values
        z_atr_ratio = self.get_dynamic_zscore(pd.Series(atr_20), span=200).values
        
        pivots = []
        for i in range(250, len(df) - self.max_hold_bars):
            idx = i - 1
            h1, h2, h3 = h[idx - 1], h[idx], h[idx + 1]
            l1, l2, l3 = l[idx - 1], l[idx], l[idx + 1]
            is_ph = self.check_ph(h1, h2, h3, l1, l2, l3)
            is_pl = self.check_pl(h1, h2, h3, l1, l2, l3)
            
            if is_ph or is_pl:
                if pivots and pivots[-1][2] == is_ph:
                    if (is_ph and h[idx] > pivots[-1][1]) or (not is_ph and l[idx] < pivots[-1][1]): 
                        pivots.pop()
                pivots.append([idx, h[idx] if is_ph else l[idx], is_ph])
            
            if len(pivots) < 3: continue
            c_idx, p_c, h_c = pivots[-1]
            b_idx, p_b, h_b = pivots[-2]
            a_idx, p_a, h_a = pivots[-3]
            
            # LONG: Đồng bộ setup Pine
            if (not h_a) and h_b and (not h_c) and p_a < p_c and self.is_valid_bc_long(c, l, b_idx, c_idx):
                all_cut = self.has_all_cut(h, l, a_idx, p_a, b_idx, p_b)
                is_quality = self.is_quality_wave(a_idx, p_a, b_idx, p_b)
                if not (all_cut and is_quality):
                    continue
                
                # B FILTER: Lọc Buying Climax tại đỉnh B (Dịch sát 100% Pine)
                range_b = max(h[b_idx] - l[b_idx], 1e-8)
                is_buy_climax = (range_b > range_sma_20[b_idx]) and ((h[b_idx] - c[b_idx]) <= range_b * 0.3) and (v[b_idx] > vol_sma_20[b_idx] * 1.5)
                if is_buy_climax: continue
                
                # TRIGGER: Pine kích hoạt khi nến đóng cửa vọt qua High của nến tạo Pivot C
                trigger_p = h[c_idx] 
                
                # Nếu cây nến hiện tại [i] break qua trigger (và cây trước đó chưa break)
                if c[i] > trigger_p and c[i-1] <= trigger_p:
                    
                    # Khớp lệnh sau khi nến xác nhận đóng, tránh fill ảo tại trigger.
                    base_entry = max(trigger_p, c[i])
                    entry_p = base_entry * (1.0 - self.entry_pullback)
                    sl_p = l[c_idx]
                    
                    dist_to_sl = (entry_p - sl_p) / entry_p
                    if dist_to_sl <= 0:
                        continue
                    
                    tp_p = p_a + self.tp_level * (p_b - p_a)
                    if tp_p <= entry_p:
                        continue

                    risk_abs = entry_p - sl_p
                    reward_abs = tp_p - entry_p

                    
                    # Labels & Win Logic
                    f_lows = l[i+1 : i+1+self.max_hold_bars]
                    f_highs = h[i+1 : i+1+self.max_hold_bars]
                    f_closes = c[i+1 : i+1+self.max_hold_bars]

                    fill_idx = self.get_fill_index(1, entry_p, f_lows, f_highs)
                    if fill_idx is None:
                        continue

                    f_lows = f_lows[fill_idx:]
                    f_highs = f_highs[fill_idx:]
                    f_closes = f_closes[fill_idx:]
                    if len(f_lows) == 0:
                        continue

                    entry_ts = ts[i] if self.entry_pullback <= 0 else ts[i + 1 + fill_idx]
                    
                    target_win = 0
                    for low_t, high_t in zip(f_lows, f_highs):
                        # Ghi chú: Zip đi chung nên nếu trong cùng 1 nến giá quét cả 2 đầu, 
                        # lệnh IF check Cháy/SL đứng trước sẽ kích hoạt -> View bi quan (rất tốt để test)
                        if (low_t / entry_p - 1) * 10 <= -0.85: target_win = 0; break 
                        if low_t <= sl_p: target_win = 0; break 
                        if high_t >= tp_p: target_win = 1; break
                    self.dataset.append({
                        'coin': coin_name, 'timestamp': entry_ts,
                        'side': 1,
                        'z_trend_20_50': z_trend_20_50[i],
                        'z_price_to_ema200': z_price_ema200[i],
                        'z_volatility_atr': z_atr_ratio[i],
                        'structure_size': abs(p_b - p_a) / (p_a + 1e-8),
                        'pullback_depth': (p_b - p_c) / (p_b - p_a) if p_b != p_a else 0,
                        'dist_to_sl_pct': dist_to_sl, # Tính năng mới để AI học độ nguy hiểm của Margin
                        'entry_p': entry_p, 'sl_p': sl_p, 'tp_p': tp_p,
                        'future_lows': f_lows.tolist(), 'future_highs': f_highs.tolist(), 'future_closes': f_closes.tolist(),
                        'target_win': target_win,
                        'end_time': ts[i + self.max_hold_bars]
                    })

            # SHORT: Đồng bộ setup Pine đối xứng
            if h_a and (not h_b) and h_c and p_a > p_c and self.is_valid_bc_short(c, h, b_idx, c_idx):
                all_cut = self.has_all_cut(h, l, a_idx, p_a, b_idx, p_b)
                is_quality = self.is_quality_wave(a_idx, p_a, b_idx, p_b)
                if not (all_cut and is_quality):
                    continue

                range_b = max(h[b_idx] - l[b_idx], 1e-8)
                is_sell_climax = (range_b > range_sma_20[b_idx]) and ((c[b_idx] - l[b_idx]) <= range_b * 0.3) and (v[b_idx] > vol_sma_20[b_idx] * 1.5)
                if is_sell_climax:
                    continue

                trigger_p = l[c_idx]
                if c[i] < trigger_p and c[i-1] >= trigger_p:
                    # Short đối xứng: chỉ được fill ở giá đóng nến xác nhận hoặc tệ hơn.
                    base_entry = min(trigger_p, c[i])
                    entry_p = base_entry * (1.0 + self.entry_pullback)
                    sl_p = h[c_idx]
                    dist_to_sl = (sl_p - entry_p) / entry_p
                    if dist_to_sl <= 0:
                        continue
                    tp_p = p_a - self.tp_level * (p_a - p_b)
                    if tp_p >= entry_p:
                        continue

                    risk_abs = sl_p - entry_p
                    reward_abs = entry_p - tp_p
                    if (reward_abs / (risk_abs + 1e-8)) < self.min_rr:
                        continue

                    f_lows = l[i+1 : i+1+self.max_hold_bars]
                    f_highs = h[i+1 : i+1+self.max_hold_bars]
                    f_closes = c[i+1 : i+1+self.max_hold_bars]

                    fill_idx = self.get_fill_index(-1, entry_p, f_lows, f_highs)
                    if fill_idx is None:
                        continue

                    f_lows = f_lows[fill_idx:]
                    f_highs = f_highs[fill_idx:]
                    f_closes = f_closes[fill_idx:]
                    if len(f_lows) == 0:
                        continue

                    entry_ts = ts[i] if self.entry_pullback <= 0 else ts[i + 1 + fill_idx]

                    target_win = 0
                    for low_t, high_t in zip(f_lows, f_highs):
                        if (high_t / entry_p - 1) * 10 >= 0.85:
                            target_win = 0
                            break
                        if high_t >= sl_p:
                            target_win = 0
                            break
                        if low_t <= tp_p:
                            target_win = 1
                            break

                    self.dataset.append({
                        'coin': coin_name, 'timestamp': entry_ts,
                        'side': -1,
                        'z_trend_20_50': z_trend_20_50[i],
                        'z_price_to_ema200': z_price_ema200[i],
                        'z_volatility_atr': z_atr_ratio[i],
                        'structure_size': abs(p_a - p_b) / (p_a + 1e-8),
                        'pullback_depth': (p_c - p_b) / (p_a - p_b) if p_a != p_b else 0,
                        'dist_to_sl_pct': dist_to_sl,
                        'entry_p': entry_p, 'sl_p': sl_p, 'tp_p': tp_p,
                        'future_lows': f_lows.tolist(), 'future_highs': f_highs.tolist(), 'future_closes': f_closes.tolist(),
                        'target_win': target_win,
                        'end_time': ts[i + self.max_hold_bars]
                    })

# ==========================================
# 2. PORTFOLIO ENGINE
# ==========================================
class StressTestPortfolio:
    def __init__(self, initial_cap=10000, leverage=10, risk_per_trade=0.02):
        self.balance = initial_cap
        self.leverage = leverage
        self.risk = risk_per_trade
        self.commission = 0.0006
        self.history = []

    def simulate(self, trades_df):
        if trades_df.empty: return pd.DataFrame()
        sort_cols = ['timestamp']
        sort_asc = [True]
        if 'ai_prob' in trades_df.columns:
            sort_cols.append('ai_prob')
            sort_asc.append(False)
        if 'coin' in trades_df.columns:
            sort_cols.append('coin')
            sort_asc.append(True)
        trades_df = trades_df.sort_values(sort_cols, ascending=sort_asc, kind='mergesort').reset_index(drop=True)
        active_positions = []
        
        for _, row in trades_df.iterrows():
            active_positions = [p for p in active_positions if p['end_time'] > row['timestamp']]
            if len(active_positions) >= 5: continue 
            
            side = row.get('side', 1)
            if side == 1:
                dist_to_sl = (row['entry_p'] - row['sl_p']) / row['entry_p']
            else:
                dist_to_sl = (row['sl_p'] - row['entry_p']) / row['entry_p']
            if dist_to_sl <= 0.005: continue # Bảo vệ kép 
            
            notional = (self.balance * self.risk) / dist_to_sl
            margin_req = notional / self.leverage
            if margin_req > self.balance * 0.1: 
                margin_req = self.balance * 0.1; notional = margin_req * self.leverage

            sl_panic_slippage = 0.002 
            exit_pnl = 0
            
            for low_t, high_t in zip(row['future_lows'], row['future_highs']):
                if side == 1:
                    if (low_t / row['entry_p'] - 1) * self.leverage <= -0.85:
                        exit_pnl = -1.0 * margin_req
                        break
                    if low_t <= row['sl_p']:
                        exit_price = row['sl_p'] * (1 - sl_panic_slippage)
                        exit_pnl = margin_req * (exit_price / row['entry_p'] - 1) * self.leverage
                        break
                    if high_t >= row['tp_p']:
                        exit_pnl = margin_req * (row['tp_p'] / row['entry_p'] - 1) * self.leverage
                        break
                else:
                    if (high_t / row['entry_p'] - 1) * self.leverage >= 0.85:
                        exit_pnl = -1.0 * margin_req
                        break
                    if high_t >= row['sl_p']:
                        exit_price = row['sl_p'] * (1 + sl_panic_slippage)
                        exit_pnl = margin_req * (1 - exit_price / row['entry_p']) * self.leverage
                        break
                    if low_t <= row['tp_p']:
                        exit_pnl = margin_req * (1 - row['tp_p'] / row['entry_p']) * self.leverage
                        break
            
            if exit_pnl == 0 and len(row['future_lows']) > 0:
                has_close_path = 'future_closes' in row and len(row['future_closes']) > 0
                if side == 1:
                    if has_close_path:
                        exit_pnl = margin_req * (row['future_closes'][-1] / row['entry_p'] - 1) * self.leverage
                    else:
                        exit_pnl = margin_req * (row['future_lows'][-1] / row['entry_p'] - 1) * self.leverage
                else:
                    if has_close_path:
                        exit_pnl = margin_req * (1 - row['future_closes'][-1] / row['entry_p']) * self.leverage
                    else:
                        exit_pnl = margin_req * (1 - row['future_highs'][-1] / row['entry_p']) * self.leverage
            
            self.balance += (exit_pnl - notional * self.commission * 2)
            self.history.append({'time': row['timestamp'], 'equity': self.balance})
            active_positions.append({'end_time': row['end_time']})
            
        return pd.DataFrame(self.history)

# ==========================================
# 3. PIPELINE: TỐI ƯU & BACKTEST
# ==========================================
def optimize_threshold(y_true, y_probs, min_samples=50):
    thresholds = np.arange(0.9, 0.99, 0.01)
    best_thresh, best_ev, best_winrate, best_count = 0.5, -999, 0, 0
    
    for th in thresholds:
        preds_idx = (y_probs >= th)
        count = sum(preds_idx)
        if count < min_samples: continue
            
        win_rate = np.mean(y_true[preds_idx])
        ev = (win_rate * 1.5) - ((1 - win_rate) * 1) 
        if ev > best_ev:
            best_ev = ev; best_thresh = th; best_winrate = win_rate; best_count = count
            
    return 0.82, best_winrate, best_count

def build_parser():
    parser = argparse.ArgumentParser(description='Run p3 pipeline with injectable strategy params')
    parser.add_argument('--data-glob', default=r'data\ohlcv\*.parquet')
    parser.add_argument('--tp-level', type=float, default=1.6)
    parser.add_argument('--entry-pullback', type=float, default=0.0, help='Fractional pullback for entry. Ex: 0.02 = 2%%')
    parser.add_argument('--min-rr', type=float, default=0.5, help='Minimum reward:risk ratio')
    parser.add_argument('--max-hold-bars', type=int, default=24)
    parser.add_argument('--min-mid-candles', type=int, default=6)
    parser.add_argument('--min-price-pct', type=float, default=3.0)
    parser.add_argument('--train-end', default='2025-01-01')
    parser.add_argument('--val-end', default='2025-05-01')
    parser.add_argument('--initial-capital', type=float, default=100)
    parser.add_argument('--leverage', type=float, default=10)
    parser.add_argument('--risk-per-trade', type=float, default=0.02)
    return parser


def main():
    args = build_parser().parse_args()

    extractor = RealDataQuantExtractor(
        tp_level=args.tp_level,
        max_hold_bars=args.max_hold_bars,
        min_mid_candles=args.min_mid_candles,
        min_price_pct=args.min_price_pct,
        entry_pullback=args.entry_pullback,
        min_rr=args.min_rr,
    )
    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise FileNotFoundError(f'No files found with pattern: {args.data_glob}')

    print('--- [1/4] Trích xuất dữ liệu (Pine Script Sync) ---')
    print(f'>> Params: tp_level={args.tp_level}, entry_pullback={args.entry_pullback}, min_rr={args.min_rr}')
    for f in tqdm(files):
        try:
            df_raw = pd.read_parquet(f)
            df_raw.columns = [x.lower() for x in df_raw.columns]
            extractor.extract(df_raw, os.path.basename(f))
        except Exception:
            continue

    full_df = pd.DataFrame(extractor.dataset)
    if full_df.empty:
        raise ValueError('Không có dữ liệu setup để train/backtest.')

    full_df = full_df.sort_values('timestamp').reset_index(drop=True)

    # Split mặc định theo lịch thời gian để tránh leakage
    train = full_df[full_df['timestamp'] < args.train_end].copy()
    val = full_df[(full_df['timestamp'] >= args.train_end) & (full_df['timestamp'] < args.val_end)].copy()
    test = full_df[full_df['timestamp'] >= args.val_end].copy()
    split_mode = 'calendar'

    # Fallback nếu mốc lịch không tạo đủ cả 3 tập
    if min(len(train), len(val), len(test)) == 0:
        n = len(full_df)
        if n < 3:
            raise ValueError('Không đủ mẫu để tách Train/Validation/Test.')
        cut1 = max(1, int(n * 0.70))
        cut2 = max(cut1 + 1, int(n * 0.85))
        cut2 = min(cut2, n - 1)
        train = full_df.iloc[:cut1].copy()
        val = full_df.iloc[cut1:cut2].copy()
        test = full_df.iloc[cut2:].copy()
        split_mode = 'time-quantile-fallback'

    print(f'\n--- [2/4] Split dữ liệu ({split_mode}) ---')
    print(f'>> Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}')

    feats = [
        'side',
        'z_trend_20_50', 'z_price_to_ema200', 'z_volatility_atr',
        'structure_size', 'pullback_depth', 'dist_to_sl_pct'
    ]

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.01, max_depth=5,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=50,
        class_weight='balanced', verbose=-1
    )

    print('\n--- [3/4] Huấn luyện AI (Early Stopping trên Validation) ---')
    model.fit(
        train[feats],
        train['target_win'],
        eval_set=[(val[feats], val['target_win'])],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    val_probs = model.predict_proba(val[feats])[:, 1]
    val_min_samples = max(20, int(len(val) * 0.01))
    optimal_th, val_wr, val_count = optimize_threshold(val['target_win'], val_probs, min_samples=val_min_samples)

    print(f'>> Threshold tối ưu từ Validation: {optimal_th:.2f}')
    print(f'>> Validation Winrate tại Threshold này: {val_wr*100:.1f}% (Dựa trên {val_count} lệnh)')

    print('\n--- [4/4] Backtest thực chiến (Test giữ kín) ---')
    test['ai_prob'] = model.predict_proba(test[feats])[:, 1]
    vip_trades = test[test['ai_prob'] >= optimal_th].copy()

    if vip_trades.empty:
        print('Không có lệnh thỏa mãn.')
    else:
        print(f">> Winrate thực tế đạt {vip_trades['target_win'].mean()*100:.1f}%")
        portfolio = StressTestPortfolio(
            initial_cap=args.initial_capital,
            leverage=args.leverage,
            risk_per_trade=args.risk_per_trade,
        )
        result_curve = portfolio.simulate(vip_trades)

        if not result_curve.empty:
            net_profit = result_curve['equity'].iloc[-1] - args.initial_capital
            peak = result_curve['equity'].cummax()
            drawdown = ((result_curve['equity'] - peak) / peak).min() * 100

            print('\n[KẾT QUẢ ĐẦU TƯ]')
            print(f'Tổng số lệnh VIP trade: {len(vip_trades)}')
            print(f'Net Profit: ${net_profit:.2f}')
            print(f'Max Drawdown: {drawdown:.2f}%')

            result_curve.set_index('time')['equity'].plot(figsize=(12, 6), title='Pine-Synced System | Equity', color='teal')
            plt.grid(True, alpha=0.3)
            plt.show()


if __name__ == '__main__':
    main()