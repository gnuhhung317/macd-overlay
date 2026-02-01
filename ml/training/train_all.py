#!/usr/bin/env python3
"""
Multi-Timeframe Model Trainer
Orchestrator for training Entry, SL, and TP models.
"""
import sys
import argparse
from pathlib import Path
from typing import List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import SUPPORTED_TIMEFRAMES

# Local imports
# Add parent directory to path to find training_utils and model trainers
sys.path.insert(0, str(Path(__file__).parent))
from training_utils import TrainingResult, PROCESSED_DIR
from train_entry import train_entry_filter
from train_sl import train_sl_predictor
from train_tp import train_tp_predictor


def train_all_models_for_timeframe(timeframe: str, tune: bool = False) -> List[TrainingResult]:
    """Train all 3 models for a specific timeframe"""
    results = []
    
    try:
        results.append(train_entry_filter(timeframe, tune))
    except Exception as e:
        print(f"❌ Entry Filter failed for {timeframe}: {e}")
    
    try:
        results.append(train_sl_predictor(timeframe, tune))
    except Exception as e:
        print(f"❌ SL Predictor failed for {timeframe}: {e}")
    
    try:
        results.append(train_tp_predictor(timeframe, tune))
    except Exception as e:
        print(f"❌ TP Predictor failed for {timeframe}: {e}")
    
    return results


def print_summary(all_results: List[TrainingResult]):
    """Print training summary table"""
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    
    # Group by timeframe
    by_timeframe = {}
    for r in all_results:
        if r.timeframe not in by_timeframe:
            by_timeframe[r.timeframe] = {}
        by_timeframe[r.timeframe][r.model_type] = r
    
    print(f"\n{'Timeframe':<10} {'Entry Filter':<25} {'SL Predictor':<25} {'TP Predictor':<25}")
    print("-"*100)
    
    for tf in SUPPORTED_TIMEFRAMES:
        if tf not in by_timeframe:
            print(f"{tf:<10} {'N/A':<25} {'N/A':<25} {'N/A':<25}")
            continue
        
        models = by_timeframe[tf]
        
        entry_str = "N/A"
        if 'entry_filter' in models:
            r = models['entry_filter']
            entry_str = f"{r.best_model_name} AUC={r.test_score:.3f}"
        
        sl_str = "N/A"
        if 'sl_predictor' in models:
            r = models['sl_predictor']
            sl_str = f"{r.best_model_name} MAE={r.test_score:.4f}"
        
        tp_str = "N/A"
        if 'tp_predictor' in models:
            r = models['tp_predictor']
            tp_str = f"{r.best_model_name} MAE={r.test_score:.4f}"
        
        print(f"{tf:<10} {entry_str:<25} {sl_str:<25} {tp_str:<25}")
    
    # Total time
    total_time = sum(r.training_time for r in all_results)
    print(f"\nTotal training time: {total_time/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Train ML models for all timeframes')
    parser.add_argument('timeframe', nargs='?', default='all', 
                       help='Timeframe to train (1h, 4h, 8h, 12h, 1d, all)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'entry', 'sl', 'tp'],
                       help='Model type to train')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Multi-Timeframe ML Model Trainer")
    print("="*60)
    
    # Determine timeframes
    if args.timeframe == 'all':
        timeframes = SUPPORTED_TIMEFRAMES
    else:
        timeframes = [args.timeframe]
    
    # Check data exists
    for tf in timeframes:
        data_path = PROCESSED_DIR / f'features_{tf}_full.parquet'
        if not data_path.exists():
            print(f"⚠️ Data not found for {tf}: {data_path}")
            print(f"   Run: python multi_timeframe_pipeline.py {tf}")
            timeframes.remove(tf)
    
    if not timeframes:
        print("No valid timeframes to train!")
        return
    
    print(f"\nTimeframes: {timeframes}")
    print(f"Model type: {args.model}")
    print(f"Hyperparameter tuning: {args.tune}")
    
    all_results = []
    
    for tf in timeframes:
        if args.model == 'all':
            results = train_all_models_for_timeframe(tf, args.tune)
            all_results.extend(results)
        elif args.model == 'entry':
            all_results.append(train_entry_filter(tf, args.tune))
        elif args.model == 'sl':
            all_results.append(train_sl_predictor(tf, args.tune))
        elif args.model == 'tp':
            all_results.append(train_tp_predictor(tf, args.tune))
    
    print_summary(all_results)
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
