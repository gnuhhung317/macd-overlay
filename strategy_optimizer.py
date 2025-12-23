"""
Test các tổ hợp bộ lọc để tìm chiến lược MACD tối ưu
"""

import pandas as pd
from data_processor import BinanceDataProcessor
from datetime import datetime
import sys


def assign_regime(df):
    """
    Gán regime labels dựa trên ATR percentiles
    
    Args:
        df (pd.DataFrame): DataFrame với ATR
        
    Returns:
        pd.DataFrame: DataFrame với regime column
    """
    # Tính percentiles
    p30 = df['atr'].quantile(0.30)
    p70 = df['atr'].quantile(0.70)
    
    # Gán regime
    df['regime'] = 'Normal'
    df.loc[df['atr'] < p30, 'regime'] = 'Regime A (Low Volatility)'
    df.loc[df['atr'] > p70, 'regime'] = 'Regime B (High Volatility)'
    
    print(f"\n📊 REGIME DISTRIBUTION:")
    print(f"   ATR P30: {p30:.4f}")
    print(f"   ATR P70: {p70:.4f}")
    print(f"   Regime A (ATR < P30): {len(df[df['regime'] == 'Regime A (Low Volatility)'])} candles")
    print(f"   Normal   (P30 ≤ ATR ≤ P70): {len(df[df['regime'] == 'Normal'])} candles")
    print(f"   Regime B (ATR > P70): {len(df[df['regime'] == 'Regime B (High Volatility)'])} candles\n")
    
    return df


def filter_crossovers_by_regime(crossovers, df, regime=None):
    """
    Lọc crossovers theo regime
    
    Args:
        crossovers (list): Danh sách crossovers
        df (pd.DataFrame): DataFrame với regime
        regime (str): None (all), 'Regime A (Low Volatility)', 'Regime B (High Volatility)'
        
    Returns:
        list: Crossovers filtered by regime
    """
    if regime is None:
        return crossovers
    
    filtered = []
    for cross in crossovers:
        idx = cross['index']
        if df.iloc[idx]['regime'] == regime:
            filtered.append(cross)
    
    return filtered


def apply_filters(crossovers, df, use_ema=False, use_bb=False, use_rsi=False, 
                  bb_threshold=0.02, rsi_overbought=70, rsi_oversold=30):
    """
    Áp dụng các bộ lọc cho crossovers
    
    Args:
        crossovers (list): Danh sách crossovers
        df (pd.DataFrame): DataFrame với indicators
        use_ema (bool): Sử dụng EMA 200 filter
        use_bb (bool): Sử dụng Bollinger Bands width filter
        use_rsi (bool): Sử dụng RSI filter
        bb_threshold (float): Ngưỡng BB width tối thiểu
        rsi_overbought (int): Ngưỡng RSI quá mua
        rsi_oversold (int): Ngưỡng RSI quá bán
        
    Returns:
        list: Crossovers sau khi lọc
    """
    filtered = []
    
    for cross in crossovers:
        idx = cross['index']
        
        # Kiểm tra có đủ dữ liệu không
        if pd.isna(df['ema_200'].iloc[idx]):
            continue
            
        price = df['close'].iloc[idx]
        ema_200 = df['ema_200'].iloc[idx]
        rsi = df['rsi'].iloc[idx]
        bb_width = df['bb_width'].iloc[idx]
        
        # Áp dụng filters
        passed = True
        
        # A. EMA 200 Trend Filter
        if use_ema:
            if cross['type'] == 'BULLISH' and price < ema_200:
                passed = False  # Không long khi giá dưới EMA 200
            elif cross['type'] == 'BEARISH' and price > ema_200:
                passed = False  # Không short khi giá trên EMA 200
        
        # B. Bollinger Bands Width Filter (proxy cho ADX)
        if use_bb and passed:
            if pd.isna(bb_width) or bb_width < bb_threshold:
                passed = False  # Thị trường đi ngang, bỏ qua
        
        # C. RSI Filter
        if use_rsi and passed:
            if cross['type'] == 'BULLISH' and rsi > rsi_overbought:
                passed = False  # Không long khi RSI quá mua
            elif cross['type'] == 'BEARISH' and rsi < rsi_oversold:
                passed = False  # Không short khi RSI quá bán
        
        if passed:
            filtered.append(cross)
    
    return filtered


def calculate_trades(crossovers, df=None, tp_method='crossover', tp_fixed_pct=2.0, tp_atr_mult=2.0, tp_time_hours=24):
    """
    Tính toán trades từ crossovers với các phương pháp TP khác nhau
    
    Args:
        crossovers (list): Danh sách crossovers
        df (pd.DataFrame): DataFrame với price data (cần cho TP methods)
        tp_method (str): 'crossover', 'fixed', 'atr', 'time'
        tp_fixed_pct (float): % TP cho fixed method
        tp_atr_mult (float): ATR multiplier cho atr method
        tp_time_hours (int): Số giờ cho time-based method
        
    Returns:
        list: Danh sách trades
    """
    trades = []
    in_position = False
    entry_price = 0
    entry_time = None
    entry_type = None
    entry_idx = 0
    
    for i, cross in enumerate(crossovers):
        # Check TP trước khi xử lý crossover mới (nếu đang có position)
        if in_position and tp_method != 'crossover' and df is not None:
            should_exit = False
            exit_price = None
            exit_time = None
            exit_reason = None
            
            # Tìm giá từ entry đến crossover hiện tại
            start_idx = entry_idx
            end_idx = cross['index']
            
            if start_idx < end_idx and end_idx < len(df):
                price_range = df.iloc[start_idx:end_idx+1]
                
                # Fixed % TP
                if tp_method == 'fixed':
                    if entry_type == 'LONG':
                        target_price = entry_price * (1 + tp_fixed_pct / 100)
                        # Check nếu có giá đạt target
                        hit_rows = price_range[price_range['high'] >= target_price]
                        if not hit_rows.empty:
                            should_exit = True
                            exit_price = target_price
                            exit_time = hit_rows.iloc[0]['timestamp']
                            exit_reason = f'TP Fixed {tp_fixed_pct}%'
                    else:  # SHORT
                        target_price = entry_price * (1 - tp_fixed_pct / 100)
                        hit_rows = price_range[price_range['low'] <= target_price]
                        if not hit_rows.empty:
                            should_exit = True
                            exit_price = target_price
                            exit_time = hit_rows.iloc[0]['timestamp']
                            exit_reason = f'TP Fixed {tp_fixed_pct}%'
                
                # ATR-based TP
                elif tp_method == 'atr':
                    entry_atr = df.iloc[start_idx]['atr']
                    if not pd.isna(entry_atr):
                        if entry_type == 'LONG':
                            target_price = entry_price + (entry_atr * tp_atr_mult)
                            hit_rows = price_range[price_range['high'] >= target_price]
                            if not hit_rows.empty:
                                should_exit = True
                                exit_price = target_price
                                exit_time = hit_rows.iloc[0]['timestamp']
                                exit_reason = f'TP ATR {tp_atr_mult}x'
                        else:  # SHORT
                            target_price = entry_price - (entry_atr * tp_atr_mult)
                            hit_rows = price_range[price_range['low'] <= target_price]
                            if not hit_rows.empty:
                                should_exit = True
                                exit_price = target_price
                                exit_time = hit_rows.iloc[0]['timestamp']
                                exit_reason = f'TP ATR {tp_atr_mult}x'
                
                # Time-based TP
                elif tp_method == 'time':
                    time_diff = (cross['timestamp'] - entry_time).total_seconds() / 3600
                    if time_diff >= tp_time_hours:
                        should_exit = True
                        exit_price = cross['price']
                        exit_time = cross['timestamp']
                        exit_reason = f'TP Time {tp_time_hours}h'
            
            # Đóng lệnh nếu hit TP
            if should_exit:
                if entry_type == 'LONG':
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                
                duration = (exit_time - entry_time).total_seconds() / 3600
                
                trades.append({
                    'type': entry_type,
                    'pnl_percent': pnl_percent,
                    'duration_hours': duration,
                    'result': 'WIN' if pnl_percent > 0 else 'LOSS',
                    'exit_reason': exit_reason
                })
                
                in_position = False
        
        # Xử lý crossover (đóng position cũ và mở mới)
        if cross['type'] == 'BULLISH':
            if in_position:
                exit_price = cross['price']
                exit_time = cross['timestamp']
                
                if entry_type == 'LONG':
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                
                duration = (exit_time - entry_time).total_seconds() / 3600
                
                trades.append({
                    'type': entry_type,
                    'pnl_percent': pnl_percent,
                    'duration_hours': duration,
                    'result': 'WIN' if pnl_percent > 0 else 'LOSS',
                    'exit_reason': 'Crossover'
                })
            
            in_position = True
            entry_price = cross['price']
            entry_time = cross['timestamp']
            entry_type = 'LONG'
            entry_idx = cross['index']
            
        elif cross['type'] == 'BEARISH':
            if in_position:
                exit_price = cross['price']
                exit_time = cross['timestamp']
                
                if entry_type == 'LONG':
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                
                duration = (exit_time - entry_time).total_seconds() / 3600
                
                trades.append({
                    'type': entry_type,
                    'pnl_percent': pnl_percent,
                    'duration_hours': duration,
                    'result': 'WIN' if pnl_percent > 0 else 'LOSS',
                    'exit_reason': 'Crossover'
                })
            
            in_position = True
            entry_price = cross['price']
            entry_time = cross['timestamp']
            entry_type = 'SHORT'
            entry_idx = cross['index']
    
    return trades


def analyze_performance(trades, scenario_name):
    """
    Phân tích performance của chiến lược
    """
    if not trades:
        return {
            'scenario': scenario_name,
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'sharpe_ratio': 0
        }
    
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['result'] == 'WIN']
    
    total_pnl = sum(t['pnl_percent'] for t in trades)
    avg_pnl = total_pnl / total_trades
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    
    # Tính Sharpe Ratio (simplified)
    pnl_list = [t['pnl_percent'] for t in trades]
    std_dev = pd.Series(pnl_list).std() if len(pnl_list) > 1 else 1
    sharpe_ratio = (avg_pnl / std_dev) if std_dev > 0 else 0
    
    return {
        'scenario': scenario_name,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'sharpe_ratio': sharpe_ratio,
        'avg_win': sum(t['pnl_percent'] for t in winning_trades) / len(winning_trades) if winning_trades else 0,
        'avg_loss': sum(t['pnl_percent'] for t in [t for t in trades if t['result'] == 'LOSS']) / len([t for t in trades if t['result'] == 'LOSS']) if len([t for t in trades if t['result'] == 'LOSS']) > 0 else 0
    }


def test_all_scenarios(symbol='BTCUSDT', interval='1h', start_date='1 year ago UTC', end_date='now UTC'):
    """
    Test tất cả các tổ hợp bộ lọc
    """
    print("="*80)
    print(f"STRATEGY OPTIMIZER - {symbol} {interval}")
    print("="*80)
    print(f"Từ: {start_date} → Đến: {end_date}\n")
    
    # Lấy dữ liệu
    processor = BinanceDataProcessor()
    df = processor.get_historical_data(symbol, interval, start_date, end_date)
    
    # Tính indicators
    print("Đang tính toán indicators...")
    df = processor.calculate_macd(df)
    df = processor.add_indicators(df)
    
    # Gán regime labels
    df = assign_regime(df)
    
    # Phát hiện crossovers
    print("Đang phát hiện crossovers...")
    all_crossovers = processor.detect_crossovers(df)
    print(f"✓ Tìm thấy {len(all_crossovers)} crossovers\n")
    
    # Test cho từng regime
    regimes = [
        ('All Data', None),
        ('Regime A (Low Volatility)', 'Regime A (Low Volatility)'),
        ('Regime B (High Volatility)', 'Regime B (High Volatility)')
    ]
    
    all_results = {}
    
    for regime_name, regime_filter in regimes:
        print("\n" + "="*80)
        print(f"📈 TESTING: {regime_name}")
        print("="*80)
        
        # Filter crossovers by regime
        regime_crossovers = filter_crossovers_by_regime(all_crossovers, df, regime_filter)
        print(f"Crossovers trong {regime_name}: {len(regime_crossovers)}\n")
        
        if len(regime_crossovers) == 0:
            print(f"⚠️  Không có crossovers trong {regime_name}, bỏ qua...\n")
            continue
        
        # Định nghĩa các scenarios - Grid test cho Filters x TP Methods
        filter_configs = [
            {'name': 'No Filter', 'ema': False, 'bb': False, 'rsi': False},
            {'name': 'EMA Only', 'ema': True, 'bb': False, 'rsi': False},
            {'name': 'EMA+RSI', 'ema': True, 'bb': False, 'rsi': True},
            {'name': 'All Filters', 'ema': True, 'bb': True, 'rsi': True},
        ]
        
        tp_configs = [
            {'name': 'Crossover', 'method': 'crossover'},
            {'name': 'Fixed 2%', 'method': 'fixed', 'fixed_pct': 2.0},
            {'name': 'Fixed 3%', 'method': 'fixed', 'fixed_pct': 3.0},
            {'name': 'ATR 1.5x', 'method': 'atr', 'atr_mult': 1.5},
            {'name': 'ATR 2x', 'method': 'atr', 'atr_mult': 2.0},
            {'name': 'Time 24h', 'method': 'time', 'time_hours': 24},
            {'name': 'Time 48h', 'method': 'time', 'time_hours': 48},
        ]
        
        scenarios = []
        for filter_cfg in filter_configs:
            for tp_cfg in tp_configs:
                scenarios.append({
                    'name': f"{filter_cfg['name']} + {tp_cfg['name']}",
                    'filter': filter_cfg,
                    'tp': tp_cfg
                })
        
        results = []
        
        print(f"Đang test {len(scenarios)} scenarios (Grid: {len(filter_configs)} filters x {len(tp_configs)} TP methods)...\n")
        for i, scenario in enumerate(scenarios, 1):
            print(f"  [{i}/{len(scenarios)}] Testing: {scenario['name'][:50]}...", end='', flush=True)
            
            # Áp dụng filters
            filter_cfg = scenario['filter']
            filtered_crossovers = apply_filters(
                regime_crossovers, df,
                use_ema=filter_cfg['ema'],
                use_bb=filter_cfg['bb'],
                use_rsi=filter_cfg['rsi']
            )
            
            # Tính trades với TP method
            tp_cfg = scenario['tp']
            trades = calculate_trades(
                filtered_crossovers, 
                df=df,
                tp_method=tp_cfg['method'],
                tp_fixed_pct=tp_cfg.get('fixed_pct', 2.0),
                tp_atr_mult=tp_cfg.get('atr_mult', 2.0),
                tp_time_hours=tp_cfg.get('time_hours', 24)
            )
            
            # Phân tích performance
            perf = analyze_performance(trades, scenario['name'])
            perf['regime'] = regime_name  # Thêm regime info
            results.append(perf)
            
            print(f" ✓ ({perf['total_trades']} trades, {'+' if perf['total_pnl'] >= 0 else ''}{perf['total_pnl']:.1f}%)")
        
        all_results[regime_name] = results
    
    # Hiển thị kết quả comparison
    print("\n" + "="*80)
    print("📊 KẾT QUẢ SO SÁNH THEO REGIME (Top 10 mỗi regime)")
    print("="*80)
    
    for regime_name in all_results:
        print(f"\n{'='*80}")
        print(f"🎯 {regime_name.upper()}")
        print('='*80)
        
        results_sorted = sorted(all_results[regime_name], key=lambda x: x['total_pnl'], reverse=True)
        
        for i, result in enumerate(results_sorted[:10], 1):
            rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            
            print(f"\n{rank_emoji} {result['scenario']}")
            print(f"   Tổng Trades:        {result['total_trades']}")
            print(f"   Win Rate:           {result['win_rate']:.1f}%")
            print(f"   Tổng P&L:           {'+' if result['total_pnl'] >= 0 else ''}{result['total_pnl']:.2f}%")
            print(f"   P&L TB/Trade:       {'+' if result['avg_pnl'] >= 0 else ''}{result['avg_pnl']:.2f}%")
            print(f"   Sharpe Ratio:       {result['sharpe_ratio']:.2f}")
    
    # Lưu kết quả combined
    all_results_combined = []
    for regime_name, results in all_results.items():
        all_results_combined.extend(results)
    
    results_df = pd.DataFrame(all_results_combined)
    output_file = f"strategy_comparison_by_regime_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Đã lưu kết quả vào {output_file}")
    
    # Tìm best strategy cho từng regime
    print("\n" + "="*80)
    print("🏆 CHIẾN LƯỢC TỐI ƯU THEO REGIME")
    print("="*80)
    
    for regime_name in all_results:
        if all_results[regime_name]:
            best = sorted(all_results[regime_name], key=lambda x: x['total_pnl'], reverse=True)[0]
            print(f"\n📌 {regime_name}:")
            print(f"   Strategy: {best['scenario']}")
            print(f"   Total P&L: {'+' if best['total_pnl'] >= 0 else ''}{best['total_pnl']:.2f}%")
            print(f"   Win Rate: {best['win_rate']:.1f}%")
            print(f"   Sharpe: {best['sharpe_ratio']:.2f}")
    
    print()


def main():
    """Main function"""
    print("\n🔬 STRATEGY OPTIMIZER - MACD FILTER COMBINATIONS\n")
    
    # Cấu hình
    SYMBOL = 'BTCUSDT'
    INTERVAL = '1h'
    START_DATE = '1 year ago UTC'
    END_DATE = 'now UTC'
    
    # Override từ command line
    if len(sys.argv) > 1:
        SYMBOL = sys.argv[1]
    if len(sys.argv) > 2:
        INTERVAL = sys.argv[2]
    if len(sys.argv) > 3:
        START_DATE = sys.argv[3]
    if len(sys.argv) > 4:
        END_DATE = sys.argv[4]
    
    # Chạy test
    test_all_scenarios(SYMBOL, INTERVAL, START_DATE, END_DATE)
    
    print(f"\nSử dụng: python strategy_optimizer.py [SYMBOL] [INTERVAL] [START_DATE] [END_DATE]")
    print(f"Ví dụ:   python strategy_optimizer.py ETHUSDT 4h '2024-01-01' '2024-12-31'")


if __name__ == "__main__":
    main()
