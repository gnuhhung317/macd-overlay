#!/usr/bin/env python3
"""
Monte Carlo Trade Resampling Simulator
Performs Post-Backtest Bootstrap Resampling with Replacement
to generate realistic probability distributions of equity curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pathlib import Path
import logging
import argparse
from tqdm import tqdm

# Import backtest module
from backtest_3stage import ThreeStageBacktester, BacktestConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_baseline_backtest(args):
    """Run the standard backtest to get the original trades."""
    backtest_start = args.start
    backtest_end = args.end
    warm_up_months = args.warmup
    timeframe = args.timeframe
    leverage = args.leverage
    initial_capital = args.capital
    margin_mode = args.margin_mode
    use_kelly = args.kelly
    
    logger.info(f"🚀 Running Baseline Backtest (Timeframe: {timeframe}, Leverage: {leverage}x)")
    
    # Load data
    data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / f'features_{timeframe}_full.parquet'
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / 'bitget-data' / 'processed' / f'features_{timeframe}.parquet'
        
    if getattr(args, 'data', None):
        data_path = Path(args.data)
        
    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        return None, None
    
    df = pd.read_parquet(data_path)
    
    # Calculate warm-up start date
    backtest_start_dt = pd.to_datetime(backtest_start)
    warm_up_start_dt = backtest_start_dt - pd.DateOffset(months=warm_up_months)
    backtest_end_dt = pd.to_datetime(backtest_end)
    
    # Filter with warm-up period
    df_with_warmup = df[
        (df['timestamp'] >= warm_up_start_dt) & 
        (df['timestamp'] <= backtest_end_dt)
    ].copy()
    
    if df_with_warmup.empty:
        logger.error("❌ No data found for the specified period!")
        return None, None
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=initial_capital,
        risk_per_trade=args.risk,
        entry_threshold=args.threshold,
        fee_rate=args.fee,
        slippage=args.slippage,
        leverage=leverage,
        timeframe=timeframe,
        margin_mode=margin_mode,
        use_kelly=use_kelly,
        fixed_position_size=args.fixed_size,
        position_size_usd=args.size_usd,
        max_open_trades=args.max_positions,
        require_fresh_crossover_after_exit=True,
        # Trailing Stop arguments
        use_trailing_stop=args.trailing,
        trailing_start_pct=args.trailing_start,
        trailing_step_pct=args.trailing_step,
        # Pullback options
        entry_pullback_pct=args.entry_pullback,
        entry_pullback_timeout=args.entry_timeout,
        max_bars=args.max_bars,
        # Scanner options
        use_scanner_filter=args.use_scanner,
        scanner_mae=args.scanner_mae,
        scanner_mfe=args.scanner_mfe,
        scanner_lookback_days=args.scanner_lookback,
        # global_cb_pct=args.global_cb,
        # global_cb_cooldown=args.global_cb_cooldown
    )
    
    # Create backtester and run
    backtester = ThreeStageBacktester(config)
    result = backtester.run_backtest(df_with_warmup, verbose=False)
    
    # Filter trades to only include those starting after start_date
    analysis_start_ts = pd.to_datetime(backtest_start).tz_localize(None)
    
    valid_trades = [t for t in result.trades if t.entry_time.replace(tzinfo=None) >= analysis_start_ts]
    
    logger.info(f"✅ Baseline complete. Extracted {len(valid_trades)} completed trades.")
    
    # Determine actual initial capital (in case of warm-up or active trades)
    actual_initial_capital = config.initial_capital
    if result.equity_curve and len(result.equity_curve) > 0:
        # Get equity value at the start of analysis period
        start_idx = 0
        for i, ts in enumerate(result.timestamps):
            if ts.replace(tzinfo=None) >= analysis_start_ts:
                start_idx = i
                break
        actual_initial_capital = result.equity_curve[start_idx] if start_idx < len(result.equity_curve) else config.initial_capital
        
    return valid_trades, actual_initial_capital

def extract_trade_returns(trades):
    """
    Extract Trade Return % (Account Impact).
    We mimic the actual PnL generated relative to the account capital available when it executed.
    """
    trade_impacts = []
    
    # Simulate a running Realized Equity to extract true percentage impact per trade
    working_equity = 100.0  # Normalized Base capital
    
    # Sort by exit time to determine realized equity timeline
    sorted_trades = sorted(trades, key=lambda x: x.exit_time if x.exit_time else x.entry_time)
    
    for t in sorted_trades:
        if t.position_size <= 0: continue
            
        # PnL scaling
        # Assuming Fixed Risk, impact_pct relies on SL risk ratio.
        # But for Monte Carlo, we just take the relative impact it had on the exact balance
        impact_pct = t.pnl / working_equity
        working_equity += t.pnl
        
        trade_impacts.append({
            'symbol': t.symbol,
            'impact_pct': impact_pct, 
            'pnl': t.pnl,
            'original_equity': working_equity - t.pnl
        })
        
    return trade_impacts

def simulate_equity_curve_fast(trade_impacts, initial_capital, num_trades):
    """
    Simulate a single equity curve utilizing bootstrap sampling of returns using numpy.
    Compounding mathematically: E_{t} = E_{t-1} * (1 + impact_pct)
    """
    # Sample impacts with replacement
    sampled_indices = np.random.randint(0, len(trade_impacts), size=num_trades)
    sampled_impacts = np.array([trade_impacts[i]['impact_pct'] for i in sampled_indices])
    
    # Compounding Equity Curve calculation
    # E_t = Initial * cumprod(1 + r_t)
    returns = 1.0 + sampled_impacts
    
    # Prevent negative equity blowups (bankruptcy)
    returns = np.clip(returns, 0.0, None)
    
    cumulative_growth = np.cumprod(returns)
    equity_curve = initial_capital * cumulative_growth
    
    # Prepend initial capital
    equity_curve = np.insert(equity_curve, 0, initial_capital)
    
    return equity_curve

def run_monte_carlo(args, trade_impacts, actual_initial_capital):
    """Run Monte Carlo simulation N times and aggregate results."""
    logger.info(f"\n🎲 Starting Monte Carlo Simulation: {args.simulations:,} Iterations")
    
    num_trades = len(trade_impacts)
    if num_trades == 0:
        logger.error("No trades extracted. Simulation aborted.")
        return
        
    all_equity_curves = np.zeros((args.simulations, num_trades + 1))
    
    for i in tqdm(range(args.simulations), desc="Simulating"):
        curve = simulate_equity_curve_fast(trade_impacts, actual_initial_capital, num_trades)
        all_equity_curves[i] = curve
        
    # Statistical Analysis
    final_equities = all_equity_curves[:, -1]
    
    # Calculate Percentiles for the Fan Chart
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    curve_percentiles = np.percentile(all_equity_curves, percentiles, axis=0)
    
    # Calculate Max Drawdown for each simulation
    peaks = np.maximum.accumulate(all_equity_curves, axis=1)
    drawdowns = (peaks - all_equity_curves) / peaks
    max_drawdowns = np.max(drawdowns, axis=1) * 100  # Convert to %
    
    # Risk of Ruin calculation (Equity drops below 50% of peak or initial)
    ruin_threshold = actual_initial_capital * 0.5
    ruined_sims = np.any(all_equity_curves <= ruin_threshold, axis=1)
    risk_of_ruin_pct = (np.sum(ruined_sims) / args.simulations) * 100
    
    # 3. Print Report
    print("\n" + "="*80)
    print("🎲 MONTE CARLO SIMULATION RESULTS (Bootstrap Resampling)")
    print("="*80)
    print(f"   Original Trades Count:    {num_trades}")
    print(f"   Simulations Run:          {args.simulations:,}")
    print(f"   Initial Capital:          ${actual_initial_capital:,.2f}")
    print("\n💰 Final Equity Percentiles:")
    print(f"   99th Percentile (Luckiest): ${np.percentile(final_equities, 99):,.2f}")
    print(f"   95th Percentile (Great):    ${np.percentile(final_equities, 95):,.2f}")
    print(f"   75th Percentile (Good):     ${np.percentile(final_equities, 75):,.2f}")
    print(f"   50th Percentile (Median):   ${np.percentile(final_equities, 50):,.2f}")
    print(f"   25th Percentile (Poor):     ${np.percentile(final_equities, 25):,.2f}")
    print(f"   5th  Percentile (Unlucky):  ${np.percentile(final_equities, 5):,.2f}")
    print(f"   1st  Percentile (Worst):    ${np.percentile(final_equities, 1):,.2f}")
    
    print("\n📉 Expected Maximum Drawdown:")
    print(f"   Median Max Drawdown:      {np.percentile(max_drawdowns, 50):.2f}%")
    print(f"   95th Percentile Max DD:   {np.percentile(max_drawdowns, 95):.2f}% (95% chance to not exceed this)")
    print(f"   Maximum Simulated DD:     {np.max(max_drawdowns):.2f}%")
    
    print(f"\n⚠️ Risk Metrics:")
    print(f"   Risk of Ruin (< 50% Init):{risk_of_ruin_pct:.2f}% chance of catastrophic loss")
    print("="*80)
    
    # 4. Plot Fan Chart
    logger.info("\n📈 Generating Monte Carlo Fan Chart...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(num_trades + 1)
    
    # Plot confidence bands
    ax.fill_between(x, curve_percentiles[1], curve_percentiles[5], color='blue', alpha=0.1, label='5th-95th Percentile (90% Conf)')
    ax.fill_between(x, curve_percentiles[2], curve_percentiles[4], color='blue', alpha=0.2, label='25th-75th Percentile (50% Conf)')
    
    # Plot median
    ax.plot(x, curve_percentiles[3], color='blue', linewidth=2, label='Median Expected Equity')
    
    # Format axes (log scale optional for massive returns)
    median_final = curve_percentiles[3][-1]
    if median_final > actual_initial_capital * 10:
        ax.set_yscale('log')
        ax.set_ylabel('Equity ($) - Log Scale')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    else:
        ax.set_ylabel('Equity ($)')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    
    ax.set_title(f'Monte Carlo Bootstrap Trade Resampling ({args.simulations:,} paths)', fontsize=15, fontweight='bold')
    ax.set_xlabel('Trade Number in Sequence')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left')
    
    # Add stats box
    stats_text = (
        f"Simulations: {args.simulations:,}\n"
        f"Median Final: ${np.median(final_equities):,.0f}\n"
        f"95% Worst Case: ${np.percentile(final_equities, 5):,.0f}\n"
        f"Median Max DD: {np.median(max_drawdowns):.1f}%\n"
        f"Risk of Ruin: {risk_of_ruin_pct:.1f}%"
    )
    ax.text(0.02, 0.65, stats_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent / 'monte_carlo_equity.png'
    fig.savefig(save_path, dpi=300)
    logger.info(f"💾 Plot saved to: {save_path.name}")
    
def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Trade Resampling Simulator")
    
    # Baseline Backtest arguments
    parser.add_argument('--data', type=str, default=None, help='Path to data file')
    parser.add_argument('--capital', type=float, default=100.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.01, help='Risk per trade (0.01 = 1%)')
    parser.add_argument('--threshold', type=float, default=0.65, help='Entry confidence threshold')
    parser.add_argument('--fee', type=float, default=0.001, help='Fee rate (0.001 = 0.1%)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage (0.0005 = 0.05%)')
    parser.add_argument('--kelly', action='store_true', help='Use Kelly Criterion')
    parser.add_argument('--fixed-size', action='store_true', help='Use fixed position size')
    parser.add_argument('--size-usd', type=float, default=1000, help='Fixed position size in USD')
    parser.add_argument('--leverage', type=float, default=20.0, help='Leverage multiplier (e.g. 1, 3, 5, 7, 10, 20)')
    parser.add_argument('--max-positions', type=int, default=10, help='Max open positions (default: 10)')
    
    # Trailing Stop arguments
    parser.add_argument('--trailing', action='store_true', help='Enable Trailing Stop')
    parser.add_argument('--trailing-start', type=float, default=0.1, help='Trailing start pct (e.g. 0.02 for 2%)')
    parser.add_argument('--trailing-step', type=float, default=0.05, help='Trailing step pct (e.g. 0.01 for 1%)')
    
    # Pullback options
    parser.add_argument('--entry-pullback', type=float, default=0.0, help='Pullback pct for limit entry (e.g. 0.005 for 0.5%)')
    parser.add_argument('--entry-timeout', type=int, default=3, help='Timeout bars for limit entry')
    parser.add_argument('--max-bars', type=int, default=10, help='Max bars to hold trade (timeout)')
    
    # Scanner Filter arguments
    parser.add_argument('--use-scanner', action='store_true', help='Enable SmartScanner Entry Zone filtering')
    parser.add_argument('--scanner-mae', type=float, default=0.04, help='Max Adverse Excursion for zone (default: 0.04)')
    parser.add_argument('--scanner-mfe', type=float, default=0.12, help='Max Favorable Excursion for zone (default: 0.12)')
    parser.add_argument('--scanner-lookback', type=int, default=6, help='Lookback days for scanner entry (default: 6)')
    
    parser.add_argument("--start", type=str, default='2025-01-01', help="Analysis start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default='2026-02-22', help="Analysis end date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default='1d', help="Timeframe (1d, 4h, etc.)")
    parser.add_argument("--margin-mode", type=str, default='ISOLATED', choices=['ISOLATED', 'CROSS'], help="Margin mode")
    parser.add_argument("--warmup", type=int, default=0, help="Warm-up months for indicators")
    
    # Global Circuit Breaker
    parser.add_argument('--global-cb', type=float, default=0.0, help='Global Circuit Breaker %% (e.g. 0.15 for 15%% Max Drawdown from peak)')
    parser.add_argument('--global-cb-cooldown', type=int, default=0, help='Number of bars to cooldown after Global Circuit Breaker triggers')
    
    # Monte Carlo Specific argument
    parser.add_argument('--simulations', type=int, default=10000, help='Number of Monte Carlo simulations to run')
    
    args = parser.parse_args()
    
    # 1. Run baseline
    trades, actual_capital = run_baseline_backtest(args)
    if not trades:
        return
        
    # 2. Extract trade impacts
    impacts = extract_trade_returns(trades)
    
    # 3. Process simulations
    run_monte_carlo(args, impacts, actual_capital)

if __name__ == "__main__":
    main()
