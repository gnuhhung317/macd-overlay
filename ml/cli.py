#!/usr/bin/env python3
"""
Main CLI for ML Pipeline

Commands:
    prepare  - Prepare data for all timeframes
    train    - Train models for all timeframes
    backtest - Run backtests
    full     - Run complete pipeline (prepare -> train -> backtest)
"""
import sys
import argparse
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from config import SUPPORTED_TIMEFRAMES


def cmd_prepare(args):
    """Prepare data for timeframes"""
    from multi_timeframe_pipeline import main as run_pipeline
    
    # Override sys.argv for the pipeline
    if args.timeframe == 'all':
        sys.argv = ['multi_timeframe_pipeline.py', 'all']
    else:
        sys.argv = ['multi_timeframe_pipeline.py', args.timeframe]
    
    run_pipeline()


def cmd_train(args):
    """Train models for timeframes"""
    from training.train_all import main as run_training
    
    # Build sys.argv
    argv = ['train_all.py', args.timeframe]
    if args.model != 'all':
        argv.extend(['--model', args.model])
    if args.tune:
        argv.append('--tune')
    
    sys.argv = argv
    run_training()


def cmd_predict(args):
    """Run prediction for a symbol"""
    from inference import InferenceEngine, load_data_for_symbol
    
    print(f"Loading {args.timeframe} models...")
    engine = InferenceEngine(args.timeframe)
    
    print(f"Loading data for {args.symbol}...")
    df = load_data_for_symbol(args.symbol, "1h") # Always load 1h and resample? Or load timeframe specifically?
    # Context: The training data was built from 1h. inference.py assumes input is adequate.
    # We should probably use the same logic as training: load 1h, resample if needed.
    # For simplicity, let's assume we load the raw 1h data and let the engine/pipeline handle features.
    # Wait, inference.py logic above calls calculate_features on the passed df. 
    # If timeframe is 4h, we need to pass 4h data.
    
    # Correction: The logic in multi_timeframe_pipeline resamples 1h to target timeframe.
    # We should replicate that or assume user has target timeframe data.
    # Let's try to load 1h and resample using the pipeline util if possible.
    from multi_timeframe_pipeline import resample_to_timeframe
    from data_pipeline import load_ohlcv_1h
    import pandas as pd
    
    # Load 1h data
    # Try data/ohlcv (Parquet) first as it seems to be the main source
    print("Fetching data...")
    df = None
    
    # Check data/ohlcv
    parquet_path = Path(f"data/ohlcv/{args.symbol}USDT_USDT.parquet")
    if not parquet_path.exists():
        # Try without extra USDT
        parquet_path = Path(f"data/ohlcv/{args.symbol}_USDT.parquet")
        
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            print(f"❌ Error loading parquet: {e}")
            return
    else:
        # Fallback to data/raw/1h (CSV)
        csv_path = Path(f"data/raw/1h/{args.symbol}USDT.csv") 
        if not csv_path.exists():
             csv_path = Path(f"data/raw/1h/{args.symbol}.csv")
         
        if csv_path.exists():
             try:
                 df = pd.read_csv(csv_path)
                 df['timestamp'] = pd.to_datetime(df['timestamp'])
             except Exception as e:
                 print(f"❌ Error loading CSV: {e}")
                 return
    
    if df is None or df.empty:
         print(f"❌ Data for {args.symbol} not found in data/ohlcv or data/raw/1h")
         return
         
    # Resample
    if args.timeframe != '1h':
        print(f"Resampling to {args.timeframe}...")
        df = resample_to_timeframe(df, args.timeframe)
        
    print(f"Analyzing {len(df)} candles...")
    result = engine.predict(args.symbol, df)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
        
    print("\n" + "="*50)
    print(f"PREDICTION: {args.symbol} ({args.timeframe})")
    print("="*50)
    print(f"Time:       {result['timestamp']}")
    print(f"Close:      {result['close_price']:.4f}")
    print(f"Action:     {result['action']}")
    
    if result['can_enter']:
        print("-" * 50)
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Stop Loss:  {result['sl_price']:.4f} (-{result['sl_pct']:.2%})")
        print(f"Take Profit:{result['tp_price']:.4f} (+{result['tp_pct']:.2%})")
        print(f"Risk/Reward: 1:{result['tp_pct']/result['sl_pct']:.1f}")
    print("="*50)


def cmd_backtest(args):
    """Run backtests"""
    from backtesting.backtest_timeframes import main as run_backtest
    
    argv = ['backtest_timeframes.py', args.timeframe]
    if args.capital:
        argv.extend(['--capital', str(args.capital)])
    if args.fixed_size:
        argv.append('--fixed-size')
        argv.extend(['--size-usd', str(args.size_usd)])
    if args.leverage:
        argv.extend(['--leverage', str(args.leverage)])
    if args.start_date:
        argv.extend(['--start-date', args.start_date])
    if args.end_date:
        argv.extend(['--end-date', args.end_date])
    
    sys.argv = argv
    run_backtest()


def cmd_full(args):
    """Run full pipeline"""
    print("="*70)
    print("FULL ML PIPELINE")
    print("="*70)
    
    # 1. Prepare data
    print("\n" + "="*70)
    print("Step 1: Preparing data...")
    print("="*70)
    cmd_prepare(args)
    
    # 2. Train models
    print("\n" + "="*70)
    print("Step 2: Training models...")
    print("="*70)
    cmd_train(args)
    
    # 3. Backtest
    print("\n" + "="*70)
    print("Step 3: Running backtests...")
    print("="*70)
    cmd_backtest(args)
    
    print("\n" + "="*70)
    print("✅ FULL PIPELINE COMPLETE!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='MACD ML Pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare data for all timeframes
    python cli.py prepare all
    
    # Train models for 1d timeframe
    python cli.py train 1d
    
    # Backtest 4h with 5x leverage
    python cli.py backtest 4h --leverage 5
    
    # Predict signal for BTC
    python cli.py predict BTC 1h
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Prepare command
    prep_parser = subparsers.add_parser('prepare', help='Prepare data')
    prep_parser.add_argument('timeframe', default='all', nargs='?',
                            help=f'Timeframe ({", ".join(SUPPORTED_TIMEFRAMES)}, all)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('timeframe', default='all', nargs='?',
                             help=f'Timeframe ({", ".join(SUPPORTED_TIMEFRAMES)}, all)')
    train_parser.add_argument('--model', type=str, default='all',
                             choices=['all', 'entry', 'sl', 'tp'],
                             help='Model type to train')
    train_parser.add_argument('--tune', action='store_true',
                             help='Enable hyperparameter tuning')
    
    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Run backtests')
    bt_parser.add_argument('timeframe', default='all', nargs='?',
                          help=f'Timeframe ({", ".join(SUPPORTED_TIMEFRAMES)}, all)')
    bt_parser.add_argument('--capital', type=float, default=10000)
    bt_parser.add_argument('--fixed-size', action='store_true')
    bt_parser.add_argument('--size-usd', type=float, default=1000)
    bt_parser.add_argument('--leverage', type=float, default=None)
    bt_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    bt_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict signal')
    pred_parser.add_argument('symbol', help='Symbol to predict (e.g. BTC)')
    pred_parser.add_argument('timeframe', default='4h', nargs='?',
                            help='Timeframe to analyze')
    
    # Full command
    full_parser = subparsers.add_parser('full', help='Run full pipeline')
    full_parser.add_argument('timeframe', default='all', nargs='?',
                            help=f'Timeframe ({", ".join(SUPPORTED_TIMEFRAMES)}, all)')
    full_parser.add_argument('--model', type=str, default='all',
                            choices=['all', 'entry', 'sl', 'tp'])
    full_parser.add_argument('--tune', action='store_true')
    full_parser.add_argument('--capital', type=float, default=10000)
    full_parser.add_argument('--fixed-size', action='store_true')
    full_parser.add_argument('--size-usd', type=float, default=1000)
    full_parser.add_argument('--leverage', type=float, default=None)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'full':
        cmd_full(args)


if __name__ == '__main__':
    main()
