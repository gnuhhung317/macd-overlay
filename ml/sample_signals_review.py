#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import random
import traceback
import pyarrow.parquet as pq

# Fix paths to allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.three_stage_ml import ThreeStageMLSystem

# Constants
DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
MODEL_DIR = Path(__file__).parent / 'models'
RESULTS_DIR = Path(__file__).parent / 'results' / 'review'
SAMPLES_PER_TIER = 20
CANDIDATES_TO_PROCESS = 500 

TIMEFRAMES = ['1d', '12h', '8h', '4h']

CONFIDENCE_TIERS = {
    'Elite': (0.7, 1.1),
    'High': (0.6, 0.7),
    'Medium': (0.5, 0.6),
    'Low': (0.3, 0.5),
    'Poor': (0.0, 0.3)
}

def plot_candlestick(df: pd.DataFrame, signal_row: pd.Series, pred: Dict, timeframe: str, tier: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    df = df.copy()
    if 'timestamp' in df.columns: df = df.set_index('timestamp')
    
    is_long = signal_row['macd_cross_up'] == 1
    entry_price = signal_row['close']
    sl_price = signal_row['close'] * (1 - pred['sl_pct']) if is_long else signal_row['close'] * (1 + pred['sl_pct'])
    tp_price = signal_row['close'] * (1 + pred['tp_pct']) if is_long else signal_row['close'] * (1 - pred['tp_pct'])
    
    up_color, down_color, bg_color = '#26a69a', '#ef5350', '#131722'
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    width = 0.6
    for i, (idx, row) in enumerate(df.iterrows()):
        color = up_color if row['close'] >= row['open'] else down_color
        ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        ax.add_patch(plt.Rectangle((i - width/2, min(row['open'], row['close'])), 
                                  width, abs(row['open'] - row['close']), 
                                  color=color, zorder=3))
        
    signal_time = signal_row.get('timestamp')
    try: current_signal_idx = list(df.index).index(signal_time)
    except: current_signal_idx = len(df) // 2
        
    marker = '^' if is_long else 'v'
    marker_color = up_color if is_long else down_color
    
    ax.scatter(current_signal_idx, entry_price, marker=marker, color=marker_color, s=200, label=f'Entry ({"LONG" if is_long else "SHORT"})', zorder=5, edgecolors='white')
    ax.axhline(y=sl_price, color='#ff9800', linestyle='--', alpha=0.8, label=f'SL ({pred["sl_pct"]*100:.1f}%)')
    ax.axhline(y=tp_price, color='#2196f3', linestyle='--', alpha=0.8, label=f'TP ({pred["tp_pct"]*100:.1f}%)')
    
    if is_long:
        ax.fill_between(range(len(df)), entry_price, tp_price, color=up_color, alpha=0.1)
        ax.fill_between(range(len(df)), entry_price, sl_price, color=down_color, alpha=0.1)
    else:
        ax.fill_between(range(len(df)), entry_price, tp_price, color=down_color, alpha=0.1)
        ax.fill_between(range(len(df)), entry_price, sl_price, color=up_color, alpha=0.1)

    ax.set_title(f"{signal_row.get('symbol', 'UNKNOWN')} | {timeframe} | Conf: {pred['entry_confidence']:.1%} ({tier})", color='white', fontsize=14, pad=20)
    ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
    ax.grid(True, alpha=0.1, color='white')
    for spine in ax.spines.values(): spine.set_color('#2a2e39')
    ax.legend(facecolor=bg_color, edgecolor='#2a2e39', labelcolor='white', loc='upper left')
    
    step = max(1, len(df) // 4)
    indices = [i for i in range(0, len(df), step)]
    labels = [df.index[idx].strftime('%H:%M:%S') if hasattr(df.index[idx], 'strftime') else str(df.index[idx])[-8:] for idx in indices]
    ax.set_xticks(indices); ax.set_xticklabels(labels, rotation=15, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=bg_color)
    plt.close()

def generate_markdown_report(data: List[Dict]):
    report_path = Path(__file__).parent / 'SIGNAL_REVIEW_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Signal Quality Review Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        summary = pd.DataFrame(data)
        if not summary.empty:
            f.write("## 📊 Summary\n\n")
            counts = summary.groupby(['timeframe', 'tier']).size().unstack(fill_value=0)
            f.write(counts.to_markdown() + "\n\n")
        for tf in ['1d', '12h', '8h', '4h']:
            tf_data = [d for d in data if d['timeframe'] == tf]
            if not tf_data: continue
            f.write(f"## 🕓 Timeframe: {tf}\n\n")
            for tier in ['Elite', 'High', 'Medium', 'Low', 'Poor']:
                tier_data = [d for d in tf_data if d['tier'] == tier]
                if not tier_data: continue
                f.write(f"### 🛡️ {tier} Confidence\n\n")
                for item in tier_data:
                    f.write(f"#### {item['symbol']} - {item['confidence']:.1%} - {item['timestamp']} | SL: {item['pred']['sl_pct']:.1%} TP: {item['pred']['tp_pct']:.1%}\n")
                    f.write(f"![{item['symbol']}]({item['rel_path']})\n\n")
                f.write("---\n\n")

def main():
    print(f"🚀 Starting Signal Review Sampling...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_data = []
    
    for tf in TIMEFRAMES:
        print(f"\n--- Processing {tf} ---")
        data_path = DATA_DIR / f"features_{tf}_full.parquet"
        model_dir = MODEL_DIR / tf
        if not data_path.exists(): continue
            
        try:
            parquet_file = pq.ParquetFile(data_path)
            all_available = parquet_file.schema.names
            system = ThreeStageMLSystem(
                entry_model_path=str(model_dir / 'entry_filter.joblib'),
                sl_model_path=str(model_dir / 'sl_predictor.joblib'),
                tp_model_path=str(model_dir / 'tp_predictor.joblib')
            )
            if system.entry_model is None: continue
            
            essential_cols = ['timestamp', 'symbol', 'macd_cross_up', 'macd_cross_down', 'open', 'high', 'low', 'close']
            feat_cols = list(set(system.entry_features + system.sl_features + system.tp_features))
            needed_cols = [c for c in list(set(essential_cols + feat_cols)) if c in all_available]
            
            print(f"  Finding signals...")
            df_small = pd.read_parquet(data_path, columns=essential_cols)
            df_small['timestamp'] = pd.to_datetime(df_small['timestamp'])
            crossovers = (df_small['macd_cross_up'] == 1) | (df_small['macd_cross_down'] == 1)
            valid_idx = df_small[crossovers & (df_small.index > 35) & (df_small.index < len(df_small) - 20)].index.tolist()
            if not valid_idx: del df_small; continue
                
            sample_idx = random.sample(valid_idx, min(len(valid_idx), CANDIDATES_TO_PROCESS))
            sample_idx.sort()
            print(f"  Processing {len(sample_idx)} candidates...")
            
            tier_buckets = {tier: [] for tier in CONFIDENCE_TIERS}
            
            # Memory-safe extraction using batches
            current_candidate_ptr = 0
            row_idx_offset = 0
            for batch in parquet_file.iter_batches(columns=needed_cols, batch_size=50000):
                batch_df = batch.to_pandas()
                batch_end = row_idx_offset + len(batch_df)
                
                while current_candidate_ptr < len(sample_idx) and sample_idx[current_candidate_ptr] < batch_end:
                    idx = sample_idx[current_candidate_ptr]
                    rel_idx = idx - row_idx_offset
                    row_feats = batch_df.iloc[[rel_idx]]
                    
                    pred = system.predict(row_feats, max_sl=0.30, max_tp=0.60)
                    conf = pred['entry_confidence']
                    res = {
                        'idx': idx, 'symbol': df_small.loc[idx, 'symbol'], 'timestamp': df_small.loc[idx, 'timestamp'],
                        'confidence': conf, 'pred': pred, 'row': df_small.loc[idx]
                    }
                    for tier, (low, high) in CONFIDENCE_TIERS.items():
                        if low <= conf < high:
                            tier_buckets[tier].append(res)
                            break
                    current_candidate_ptr += 1
                
                row_idx_offset = batch_end
                del batch_df
                if current_candidate_ptr >= len(sample_idx): break

            print(f"  Plotting results...")
            for tier, samples in tier_buckets.items():
                if not samples: continue
                selected = random.sample(samples, min(len(samples), SAMPLES_PER_TIER))
                print(f"    - Tier {tier}: {len(selected)}")
                tf_tier_dir = RESULTS_DIR / tf / tier
                tf_tier_dir.mkdir(parents=True, exist_ok=True)
                for i, item in enumerate(selected):
                    plot_df = df_small.iloc[item['idx']-35 : item['idx']+20]
                    save_path = tf_tier_dir / f"{i:02d}_{str(item['symbol']).replace('/', '_')}_{item['idx']}.png"
                    plot_candlestick(plot_df, item['row'], item['pred'], tf, tier, save_path)
                    report_data.append({
                        'timeframe': tf, 'tier': tier, 'symbol': item['symbol'],
                        'confidence': item['confidence'], 'timestamp': item['timestamp'], 'pred': item['pred'],
                        'rel_path': f"results/review/{tf}/{tier}/{save_path.name}"
                    })
            del df_small
        except Exception as e:
            print(f"  ❌ Error processing {tf}: {e}")
            traceback.print_exc()

    if report_data: generate_markdown_report(report_data); print(f"\n✅ All done!")
    else: print(f"\n❌ Fail: No data collected.")

if __name__ == "__main__": main()
