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
    
    # Run full pipeline for 1d
    python cli.py full 1d
    
    # Run full pipeline for all timeframes
    python cli.py full all
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
    elif args.command == 'full':
        cmd_full(args)


if __name__ == '__main__':
    main()
