
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from backtest_3stage import ThreeStageBacktester, BacktestConfig, PROCESSED_DIR

# Suppress warnings
warnings.filterwarnings('ignore')

def run_analysis():
    print("Starting Trade Insight Analysis...")
    
    # Load Data (hardcoded 1d as default)
    data_path = PROCESSED_DIR / 'features_1d_full.parquet'
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return

    print(f"Loading data from {data_path.name}...")
    df = pd.read_parquet(data_path)
    df = df.sort_values('timestamp')
    
    # Use last 2 months for faster analysis
    latest_date = df['timestamp'].max()
    test_start_date = latest_date - pd.DateOffset(months=2) # 2 months
    # df_test = df[df['timestamp'] >= test_start_date].copy()
    df_test = df.copy() 
    print(f"Analyzing period: {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")

    # Config 
    config = BacktestConfig(
        initial_capital=10_000,
        leverage=5,              
        risk_per_trade=0.01,
        entry_threshold=0.55,    # Lower threshold
        use_trailing_stop=True,
        trailing_start_pct=0.02,
        trailing_step_pct=0.01,
        timeframe='1d'
    )
    
    print("\nRunning Reference Backtest (Leverage 5x, Trailing Stop)...")
    backtester = ThreeStageBacktester(config)
    result = backtester.run_backtest(df_test, verbose=False)
    
    trades = result.trades
    print(f"Generated {len(trades)} trades for analysis.")
    
    if not trades:
        print("No trades generated. Cannot perform analysis.")
        return

    # --- ENRICHMENT STAGE ---
    print("Enriching trades with market context data...")
    
    analysis_data = []
    
    # Create lookup for O(1) access
    df_lookup = df_test.set_index(['symbol', 'timestamp'])
    
    for t in trades:
        try:
            # We need the row AT entry time
            if (t.symbol, t.entry_time) not in df_lookup.index:
                continue
                
            row = df_lookup.loc[(t.symbol, t.entry_time)]
            
            # --- Extract Key Metrics ---
            
            # 1. RSI (Momentum)
            rsi = row.get('rsi_14', 50)
            
            # 2. Trend (Price vs EMA200)
            ema_200 = row.get('ema_200', 0)
            close = row.get('close', 0)
            
            is_above_ema200 = close > ema_200 if ema_200 > 0 else True
            
            # 3. Pullback / Extension (Distance from MA20)
            ma_20 = row.get('bb_middle', close)
            # Distance as a percentage
            if ma_20 > 0:
                dist_ma_20_pct = (close - ma_20) / ma_20 
            else:
                dist_ma_20_pct = 0
            
            # 4. ADX (Trend Strength)
            adx = row.get('adx', 0)
            
            analysis_data.append({
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'result': 'WIN' if t.pnl > 0 else 'LOSS',
                'direction': t.direction,
                'rsi': rsi,
                'above_ema200': is_above_ema200,
                'dist_ma_20_pct': dist_ma_20_pct,
                'adx': adx
            })
            
        except Exception as e:
            continue
            
    df_stats = pd.DataFrame(analysis_data)
    
    if df_stats.empty:
        print("Failed to link trades to data.")
        return

    # --- INSIGHTS REPORT ---
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n" + "="*80)
    print("TRADING INSIGHTS & PATTERN RECOGNITION")
    print("="*80)
    print(f"Based on {len(df_stats)} trades. PnL includes fees & funding.")
    
    # --- 1. RSI ANALYSIS ---
    print("\n1. RSI ANALYSIS (Entry Momentum)")
    print("    Are we buying the top (Overbought) or catching a knife (Oversold)?")
    
    df_stats['rsi_bin'] = pd.cut(df_stats['rsi'], 
                                 bins=[0, 30, 45, 55, 70, 100], 
                                 labels=['Oversold (<30)', 'Weak (30-45)', 'Neutral (45-55)', 'Strong (55-70)', 'Overbought (>70)'])
    
    rsi_summary = df_stats.groupby('rsi_bin', observed=False).agg({
        'pnl': ['count', 'mean'],
        'result': lambda x: (x == 'WIN').mean()
    })
    rsi_summary.columns = ['Count', 'Avg PnL ($)', 'Win Rate']
    rsi_summary['Win Rate'] = rsi_summary['Win Rate'].map('{:.1%}'.format)
    rsi_summary['Avg PnL ($)'] = rsi_summary['Avg PnL ($)'].map('{:,.2f}'.format)
    print(rsi_summary)
    
    # --- 2. TREND FILTER ANALYSIS ---
    print("\n2. TREND ALIGNMENT (EMA 200)")
    print("    Does trading WITH the trend improve results?")
    
    def get_trend_type(row):
        if row['direction'] == 'LONG':
            return "With Trend" if row['above_ema200'] else "Counter Trend"
        else: # SHORT
            return "With Trend" if not row['above_ema200'] else "Counter Trend"
            
    df_stats['trend_type'] = df_stats.apply(get_trend_type, axis=1)
    
    trend_summary = df_stats.groupby('trend_type').agg({
        'pnl': ['count', 'mean'],
        'result': lambda x: (x == 'WIN').mean()
    })
    trend_summary.columns = ['Count', 'Avg PnL ($)', 'Win Rate']
    trend_summary['Win Rate'] = trend_summary['Win Rate'].map('{:.1%}'.format)
    trend_summary['Avg PnL ($)'] = trend_summary['Avg PnL ($)'].map('{:,.2f}'.format)
    print(trend_summary)
    
    # --- 3. PULLBACK DEPTH ANALYSIS ---
    print("\n3. PULLBACK DEPTH (vs MA20)")
    print("    Are we buying at a good discount or chasing price?")
    
    # For Longs: Negative distance is a pullback (below MA)
    # For Shorts: Positive distance is a pullback (above MA)
    def get_pullback_type(row):
        dist = row['dist_ma_20_pct']
        # Normalize relative to direction: "Higher" is extended, "Lower" is pullback
        if row['direction'] == 'SHORT':
            dist = -dist # Flip for shorts
        
        # Now dist > 0 means "Extended in favor of trade direction"
        # dist < 0 means "Pullback discount"
        
        if dist < -0.05: return "Deep Pullback (<-5%)"
        if dist < -0.01: return "Minor Pullback (-1% to -5%)"
        if dist < 0.01: return "At MA20 (+/-1%)"
        if dist < 0.05: return "Minor Extension (1% to 5%)"
        return "Overextended (>5%)"

    df_stats['pullback_type'] = df_stats.apply(get_pullback_type, axis=1)
    # Define custom sort order
    pullback_order = ["Deep Pullback (<-5%)", "Minor Pullback (-1% to -5%)", "At MA20 (+/-1%)", "Minor Extension (1% to 5%)", "Overextended (>5%)"]
    df_stats['pullback_type'] = pd.Categorical(df_stats['pullback_type'], categories=pullback_order, ordered=True)
    
    pullback_summary = df_stats.groupby('pullback_type', observed=False).agg({
        'pnl': ['count', 'mean'],
        'result': lambda x: (x == 'WIN').mean()
    })
    pullback_summary.columns = ['Count', 'Avg PnL ($)', 'Win Rate']
    pullback_summary['Win Rate'] = pullback_summary['Win Rate'].map('{:.1%}'.format)
    pullback_summary['Avg PnL ($)'] = pullback_summary['Avg PnL ($)'].map('{:,.2f}'.format)
    print(pullback_summary)
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS FOR OPTIMIZATION")
    
    # Check RSI
    high_rsi_wr = float(rsi_summary.loc['Overbought (>70)', 'Win Rate'].strip('%')) if 'Overbought (>70)' in rsi_summary.index else 0
    neutral_rsi_wr = float(rsi_summary.loc['Neutral (45-55)', 'Win Rate'].strip('%')) if 'Neutral (45-55)' in rsi_summary.index else 0
    
    if high_rsi_wr < neutral_rsi_wr - 5:
        print(f"RSI Warning: Win Rate drops significantly when Overbought ({high_rsi_wr}% vs {neutral_rsi_wr}%).")
        print("   -> SUGGESTION: Avoid Long entries when RSI > 70.")
    
    # Check Trend
    try:
        trend_wr = float(trend_summary.loc['With Trend', 'Win Rate'].strip('%'))
        counter_wr = float(trend_summary.loc['Counter Trend', 'Win Rate'].strip('%'))
        diff = trend_wr - counter_wr
        
        if diff > 5:
            print(f"Trend Filter: Trading WITH Trend increases WR by +{diff:.1f}% ({trend_wr}% vs {counter_wr}%).")
            print("   -> SUGGESTION: Implement strict EMA200 filter.")
        elif diff < -2:
             print(f"Counter-Trend: Surprisingly, Counter-Trend trades performed better ({counter_wr}% vs {trend_wr}%).")
             print("   -> SUGGESTION: Do NOT filter by EMA200 (Mean reversion strategy?).")
        else:
            print(f"Trend Neutral: Little difference (+{diff:.1f}%). Trend filter may simply reduce opportunity.")
    except:
        pass

    print("\nDone.")

if __name__ == "__main__":
    run_analysis()
