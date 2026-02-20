
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Fix Unicode output for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Import Backtester
from backtest_3stage import ThreeStageBacktester, BacktestConfig, PROCESSED_DIR

def load_data():
    # Correct path to bitget-data/processed
    data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / 'features_1d_full.parquet'
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return None
        
    print(f"Loading data from {data_path.name}...", flush=True)
    df = pd.read_parquet(data_path)
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    return df

def run_signal_analysis():
    print("Running SIGNAL-BASED Pullback Analysis (Threshold > 0.65)...", flush=True)
    
    df = load_data()
    if df is None: return

    # Use last 6 months for relevant data
    latest_date = df['timestamp'].max()
    # start_date = latest_date - pd.DateOffset(months=2)
    # df_test = df[df['timestamp'] >= start_date].copy()
    df_test = df.copy()
    
    print(f"Analyzing period: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}", flush=True)

    # Initialize Backtester just to get the Model & Predictor
    config = BacktestConfig(
        entry_threshold=0.65,
        timeframe='1d'
    )
    backtester = ThreeStageBacktester(config)
    
    # 1. IDENTIFY SIGNALS
    print("Scanning for High Confidence Signals (>0.65)...", flush=True)
    
    # Filter for crossovers strictly
    crossovers = df_test[df_test['macd_crossover'] != 0].copy()
    
    signals = []
    
    # Optimize prediction: checking one by one is slow, but robust
    count = 0
    total = len(crossovers)
    
    for idx, row in crossovers.iterrows():
        count += 1
        if count % 100 == 0:
            print(f"Scanning... {count}/{total}", end='\r', flush=True)
            
        # Standardize input for model
        # The backtester handles feature selection internally
        should_enter, confidence = backtester.predict_entry(row)
        
        if confidence >= 0.65:
            signals.append({
                'symbol': row['symbol'],
                'timestamp': row['timestamp'],
                'entry_time': row['timestamp'], # Signal time
                'direction': 'LONG' if row['macd_crossover'] > 0 else 'SHORT',
                'signal_price': row['close'],
                'confidence': confidence,
                'atr': row.get('atr_14', row['close']*0.02) # Fallback
            })
            
    print(f"\nFound {len(signals)} confirmed signals.", flush=True)
    
    if not signals:
        return

    # 3. STRATEGY SIMULATION CONFIGURATION
    STRATEGIES = [
        {"name": "Baseline (100% Market)", "orders": [(0.0, 1.0)]},
        {"name": "DCA_3_6 (40% Market, 30% @ -3%, 30% @ -6%)", "orders": [(0.0, 0.0), (0.03, 0.0), (0.05, 1)]},
        {"name": "DCA_5_10 (20% Market, 40% @ -5%, 40% @ -10%)", "orders": [(0.0, 0.2), (0.05, 0.4), (0.10, 0.4)]}
    ]
    
    strat_results = {s['name']: {'pnl': [], 'wins': 0, 'losses': 0} for s in STRATEGIES}
    
    # Global TP/SL for all strategies
    GLOBAL_TP = 0.20
    GLOBAL_SL = 0.15 

    winning_signals = []
    pullbacks = []

    # Pre-group data for O(1) lookup
    # Need full df_test to look ahead
    data_by_symbol = {sym: group.sort_values('timestamp') for sym, group in df_test.groupby('symbol')}

    for sig in signals:
        if sig['symbol'] not in data_by_symbol: continue
        
        sym_df = data_by_symbol[sig['symbol']]
        future_df = sym_df[sym_df['timestamp'] > sig['entry_time']].head(30) # 30 days max hold
        
        if future_df.empty: continue
        
        entry_price = sig['signal_price']
        
        # Track max pullback for general stats
        max_pb = 0
        min_low = entry_price
        
        # --- Simulate Strategies ---
        # For each strategy, we need to track:
        # 1. Total Position Size & Cost Basis (as orders get filled)
        # 2. Status (Active, Won, Lost)
        
        active_strats = []
        for strat in STRATEGIES:
            active_strats.append({
                'name': strat['name'],
                'orders': strat['orders'], # List of (offset, size)
                'filled_size': 0.0,
                'total_cost': 0.0,
                'avg_price': 0.0,
                'status': 'OPEN', # OPEN, WIN, LOSS
                'pnl': 0.0
            })
            
            # Initial Fill (Market Orders at 0.0 offset)
            for offset, size in strat['orders']:
                if offset == 0.0:
                    cost = size * entry_price
                    active_strats[-1]['filled_size'] += size
                    active_strats[-1]['total_cost'] += cost
                    if active_strats[-1]['filled_size'] > 0:
                        active_strats[-1]['avg_price'] = active_strats[-1]['total_cost'] / active_strats[-1]['filled_size']
                    else:
                        active_strats[-1]['avg_price'] = 0.0

        # Scan Future Candles
        for _, row in future_df.iterrows():
            low = row['low']
            high = row['high']
            
            # Update general stats
            curr_pb = (entry_price - low) / entry_price
            max_pb = max(max_pb, curr_pb)
            
            # Process Strategies
            for s in active_strats:
                if s['status'] != 'OPEN': continue
                
                # 1. Check Limits (Fill orders)
                # If Low price dips below Limit Price, fill the order
                for offset, size in s['orders']:
                    limit_price = entry_price * (1 - offset)
                    
                    # If this order part isn't filled yet... 
                    # Simpler way: Check if current Low covers this offset limit
                    # We need to know WHICH orders are already filled. 
                    # Let's simplify: 
                    # Calculate current filled size based on lowest low seen so far in this candle
                    
                    if offset > 0 and low <= limit_price:
                        # Check if we already accounted for this size? 
                        # Using floating math is risky. Let's use a flag or just assume
                        # total filled size increases.
                        
                        # Better: calculate `max_filled_size` allowed by `max_pb` 
                        pass

        # --- SIMPLIFIED LOOP FOR ROBUSTNESS ---
        # Since logic inside loop is complex, let's process the candle sequence cleaner.
        
        strat_states = [] 
        for strat in STRATEGIES:
            strat_states.append({
                'filled_tiers': [False] * len(strat['orders']),
                'avg_price': 0,
                'filled_qty': 0, # Normalized to 1.0 total
                'status': 'OPEN'
            })

        for _, row in future_df.iterrows():
            low = row['low']
            high = row['high']
            
            # Update general stats
            curr_pb = (entry_price - low) / entry_price
            max_pb = max(max_pb, curr_pb)
            
            all_closed = True
            
            for i, strat in enumerate(STRATEGIES):
                state = strat_states[i]
                if state['status'] != 'OPEN': continue
                
                all_closed = False
                
                # 1. FILL ORDERS
                for idx, (offset, size) in enumerate(strat['orders']):
                    if not state['filled_tiers'][idx]:
                        limit_price = entry_price * (1 - offset)
                        if low <= limit_price:
                            state['filled_tiers'][idx] = True
                            cost = size * limit_price
                            # Update Average Price
                            prev_cost = state['avg_price'] * state['filled_qty']
                            new_cost = prev_cost + cost
                            state['filled_qty'] += size
                            if state['filled_qty'] > 0:
                                state['avg_price'] = new_cost / state['filled_qty']
                            else:
                                state['avg_price'] = 0.0
                
                # 2. CHECK EXIT (TP/SL)
                # TP based on AVG PRICE
                if state['avg_price'] > 0:
                    tp_price = state['avg_price'] * (1 + GLOBAL_TP)
                    sl_price = state['avg_price'] * (1 - GLOBAL_SL)
                    
                    # Check SL (Conservative: Low first)
                    if low <= sl_price:
                        state['status'] = 'LOSS'
                        strat_results[strat['name']]['losses'] += 1
                        strat_results[strat['name']]['pnl'].append(-GLOBAL_SL) # -15%
                        continue
                        
                    # Check TP
                    if high >= tp_price:
                        state['status'] = 'WIN'
                        strat_results[strat['name']]['wins'] += 1
                        strat_results[strat['name']]['pnl'].append(GLOBAL_TP) # +20%
                        continue
            
            if all_closed: break

        # End of Symbol Loop -> Check Timeouts
        for i, strat in enumerate(STRATEGIES):
            state = strat_states[i]
            if state['status'] == 'OPEN':
                # Timed out after 30 days
                # Mark to market exit at last Close
                last_close = future_df.iloc[-1]['close']
                if state['avg_price'] > 0:
                    pnl = (last_close - state['avg_price']) / state['avg_price']
                    strat_results[strat['name']]['pnl'].append(pnl)
                    if pnl > 0: strat_results[strat['name']]['wins'] += 1
                    else: strat_results[strat['name']]['losses'] += 1
                else:
                    # No fill? (e.g. limit order never hit)
                    # For Baseline this shouldn't happen. For deep limits it might.
                    strat_results[strat['name']]['pnl'].append(0.0)

        # Store basic pullback for distribution stats (using Baseline logic)
        # If it hit +20% before -50% (stats only)
        # Re-calc "basic winner" for distribution graph
        basic_win = False
        basic_max_pb = 0
        basic_outcome = "UNKNOWN"
        
        for _, row in future_df.iterrows():
            l, h = row['low'], row['high']
            curr_p = (l - entry_price)/entry_price
            if (h - entry_price)/entry_price >= 0.20:
                basic_win = True
                break
            if (entry_price - l)/entry_price >= 0.50:
                break
            basic_max_pb = max(basic_max_pb, (entry_price - l)/entry_price)
            
        if basic_win:
            winning_signals.append(sig)
            pullbacks.append(basic_max_pb * 100)

    # OUTPUT RESULTS
    print("\n" + "="*80, flush=True)
    print(f"📊 STRATEGY COMPARISON (TP: +{GLOBAL_TP:.0%}, SL: -{GLOBAL_SL:.0%})")
    print("="*80, flush=True)
    print(f"{'Strategy Name':<40} | {'Win Rate':<10} | {'Avg PnL':<10} | {'Total Return (Compound)'}")
    print("-" * 80, flush=True)
    
    for name, res in strat_results.items():
        total_trades = len(res['pnl'])
        if total_trades == 0: continue
        
        win_rate = res['wins'] / total_trades
        avg_pnl = np.mean(res['pnl'])
        
        # Simple compound simulation
        capital = 100.0
        for p in res['pnl']:
            capital *= (1 + p)
            
        print(f"{name:<40} | {win_rate:.1%}     | {avg_pnl:.2%}     | {capital:.2f}x")

    print("\n" + "="*80, flush=True)
    
    # Original Distribution output for context
    pullbacks = np.array(pullbacks)
    print("\nDISTRIBUTION OF PULLBACKS (For Winners):", flush=True)
    hist, bins = np.histogram(pullbacks, bins=[0, 1, 3, 5, 7, 10, 20, 50])
    labels = ["0-1%", "1-3%", "3-5%", "5-7%", "7-10%", "10-20%", "20-50%", "50%+"]
    
    for i, count in enumerate(hist):
        bar = "#" * int(count / len(pullbacks) * 50)
        if i < len(labels):
             print(f"{labels[i]:<10} | {bar} {count} ({count/len(pullbacks):.1%})", flush=True)

if __name__ == "__main__":
    run_signal_analysis()
