#!/usr/bin/env python3
"""
GUI Backtester for Multi-Portfolio System.
Runs the backtests defined in a config and provides a Tkinter UI to step through the combined portfolio bar-by-bar.
"""
import json
import argparse
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from pathlib import Path
import copy

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import get_timeframe_config
from backtest_3stage import ThreeStageBacktester, BacktestConfig, DATA_DIR, PROCESSED_DIR, BacktestResult
from run_multi_portfolio import run_portfolio

class MultiPortfolioGUI(tk.Tk):
    def __init__(self, results_dict, all_timestamps, df_equity, total_initial_capital, all_trades, symbols_data):
        super().__init__()
        self.title("Multi-Portfolio ML Backtest - Bar by Bar")
        self.geometry("1400x1000")
        
        self.results_dict = results_dict
        self.timestamps = all_timestamps
        self.df_equity = df_equity
        self.total_initial_capital = total_initial_capital
        self.all_trades = all_trades
        self.symbols_data = symbols_data
        
        self.current_idx = 0
        self.max_idx = len(self.timestamps) - 1
        
        self.is_playing = False
        self.play_speed_ms = 100
        
        # We pre-calculated the MTM equity curve in df_equity['Total']
        self.equity_curve = self.df_equity['Total'].values
        
        self._build_ui()
        self._update_ui_for_current_bar()
        
    def _build_ui(self):
        # --- Top Info Frame ---
        info_frame = ttk.LabelFrame(self, text="Combined Portfolio State", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_time = ttk.Label(info_frame, text="Time: N/A", font=("Helvetica", 14, "bold"))
        self.lbl_time.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.lbl_equity = ttk.Label(info_frame, text="Equity: $0.00", font=("Helvetica", 14))
        self.lbl_equity.grid(row=0, column=1, padx=20, pady=5, sticky="w")
        
        self.lbl_realized = ttk.Label(info_frame, text="Realized PnL: $0.00", font=("Helvetica", 12))
        self.lbl_realized.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.lbl_floating = ttk.Label(info_frame, text="Floating PnL: $0.00", font=("Helvetica", 12))
        self.lbl_floating.grid(row=1, column=1, padx=20, pady=5, sticky="w")
        
        self.lbl_open_pos = ttk.Label(info_frame, text="Open Positions: 0", font=("Helvetica", 12))
        self.lbl_open_pos.grid(row=1, column=2, padx=20, pady=5, sticky="w")
        
        # --- Chart Frame ---
        chart_frame = ttk.LabelFrame(self, text="Aggregate Equity Curve", padding=10)
        chart_frame.pack(fill="x", expand=False, padx=10, pady=5)
        
        self.fig = Figure(figsize=(12, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Plot individual strategies with lower alpha
        col_count = len(self.df_equity.columns) - 1 # excluding Total
        colors = matplotlib.cm.get_cmap('tab10', col_count)
        
        for i, col in enumerate(self.df_equity.columns):
            if col == 'Total':
                continue
            # Normalize to starting value (Relative Return)
            start_val = self.df_equity[col].iloc[0]
            if start_val == 0: start_val = 1 # avoid division by zero
            normalized = self.df_equity[col] / start_val
            self.ax.plot(self.timestamps, normalized, label=col, alpha=0.6, linewidth=1)
            
        # Plot the Total equity curve prominently (normalized)
        total_start_val = self.df_equity['Total'].iloc[0]
        if total_start_val == 0: total_start_val = 1
        normalized_total = self.df_equity['Total'] / total_start_val
        self.ax.plot(self.timestamps, normalized_total, label='Portfolio Total', color="black", linewidth=2.5)
        
        self.ax.set_ylabel("Relative Return (x-times)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper left', fontsize='small', ncol=2)
        self.fig.tight_layout()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Add the vertical cursor line
        self.vline = self.ax.axvline(x=self.timestamps[0], color='red', linestyle='--', linewidth=2)
        
        # Bind click event
        self.canvas.mpl_connect('button_press_event', self._on_chart_click)
        
        # --- Middle Table Frame ---
        table_frame = ttk.LabelFrame(self, text="Open Positions across All Strategies", padding=10)
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        columns = ("strategy", "symbol", "direction", "entry_time", "entry_price", "current_price", "size", "pnl")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        
        self.tree.heading("strategy", text="Strategy")
        self.tree.heading("symbol", text="Symbol")
        self.tree.heading("direction", text="Direction")
        self.tree.heading("entry_time", text="Entry Time")
        self.tree.heading("entry_price", text="Entry Price")
        self.tree.heading("current_price", text="Current Price")
        self.tree.heading("size", text="Size ($)")
        self.tree.heading("pnl", text="Floating PnL ($)")
        
        self.tree.column("strategy", width=120, anchor="w")
        self.tree.column("symbol", width=100, anchor="center")
        self.tree.column("direction", width=80, anchor="center")
        self.tree.column("entry_time", width=150, anchor="center")
        self.tree.column("entry_price", width=100, anchor="e")
        self.tree.column("current_price", width=100, anchor="e")
        self.tree.column("size", width=100, anchor="e")
        self.tree.column("pnl", width=120, anchor="e")
        
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add tags for colors
        self.tree.tag_configure("profit", foreground="green")
        self.tree.tag_configure("loss", foreground="red")
        
        # --- Bottom Controls Frame ---
        controls_frame = ttk.Frame(self, padding=10)
        controls_frame.pack(fill="x", padx=10, pady=10)
        
        btn_start = ttk.Button(controls_frame, text="|< Start", command=self._go_start)
        btn_start.pack(side="left", padx=5)
        
        btn_prev = ttk.Button(controls_frame, text="< Prev", command=self._step_prev)
        btn_prev.pack(side="left", padx=5)
        
        self.btn_play = ttk.Button(controls_frame, text="Play", command=self._toggle_play)
        self.btn_play.pack(side="left", padx=5)
        
        btn_next = ttk.Button(controls_frame, text="Next >", command=self._step_next)
        btn_next.pack(side="left", padx=5)
        
        btn_end = ttk.Button(controls_frame, text="End >|", command=self._go_end)
        btn_end.pack(side="left", padx=5)
        
        self.slider = ttk.Scale(controls_frame, from_=0, to=self.max_idx, orient="horizontal", command=self._on_slider_move)
        self.slider.pack(side="left", fill="x", expand=True, padx=20)
        
    def _go_start(self):
        self.current_idx = 0
        self._update_ui_for_current_bar()
        
    def _go_end(self):
        self.current_idx = self.max_idx
        self._update_ui_for_current_bar()
        
    def _step_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._update_ui_for_current_bar()
            
    def _step_next(self):
        if self.current_idx < self.max_idx:
            self.current_idx += 1
            self._update_ui_for_current_bar()
            
    def _on_slider_move(self, val):
        if getattr(self, '_updating_slider', False):
            return
        
        new_idx = int(float(val))
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self._update_ui_for_current_bar()
            
    def _on_chart_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
            
        try:
            from matplotlib.dates import num2date
        except ImportError:
            pass
            
        clicked_time = matplotlib.dates.num2date(event.xdata).replace(tzinfo=None)
        
        closest_idx = min(range(len(self.timestamps)), key=lambda i: abs(self.timestamps[i] - clicked_time))
        
        if closest_idx != self.current_idx:
            self.current_idx = closest_idx
            self._update_ui_for_current_bar()
        
    def _toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="Pause")
            self._play_loop()
        else:
            self.btn_play.config(text="Play")
            
    def _play_loop(self):
        if self.is_playing and self.current_idx < self.max_idx:
            self.current_idx += 1
            self._update_ui_for_current_bar()
            self.after(self.play_speed_ms, self._play_loop)
        else:
            self.is_playing = False
            self.btn_play.config(text="Play")
            
    def _get_current_price(self, symbol, t):
        try:
            return self.symbols_data[symbol][t]
        except KeyError:
            return None
            
    def _update_ui_for_current_bar(self):
        t = self.timestamps[self.current_idx]
        
        # Find trades
        closed_trades = [tr for tr in self.all_trades if tr.exit_time and tr.exit_time <= t]
        open_trades = [tr for tr in self.all_trades if tr.entry_time <= t and (not tr.exit_time or tr.exit_time > t)]
        
        # Calculate Realized
        realized_pnl = sum(tr.pnl for tr in closed_trades)
        capital = self.total_initial_capital + realized_pnl
        
        floating_pnl_total = 0.0
        
        # Clear existing rows
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for tr in open_trades:
            curr_price = self._get_current_price(tr.symbol, t)
            if curr_price is None:
                curr_price = tr.entry_price # fallback
                
            if tr.direction == 'LONG':
                pnl_pct = (curr_price - tr.entry_price) / tr.entry_price
            else:
                pnl_pct = (tr.entry_price - curr_price) / tr.entry_price
            
            # Use fee rate of 0.1% per side (approx) if context fee rate is unavailable directly on trade
            # actually fee logic: we can extract it or compute approx:
            fee_est = tr.position_size * 0.001 * 2  # default 0.1%
            
            float_pnl = (tr.position_size * pnl_pct) - fee_est
            floating_pnl_total += float_pnl
            
            tag = "profit" if float_pnl >= 0 else "loss"
            self.tree.insert("", "end", values=(
                getattr(tr, 'strategy', 'Unknown'),
                tr.symbol,
                tr.direction,
                tr.entry_time.strftime("%Y-%m-%d %H:%M"),
                f"{tr.entry_price:.5f}",
                f"{curr_price:.5f}",
                f"${tr.position_size:.2f}",
                f"${float_pnl:.2f}"
            ), tags=(tag,))
            
        mtm_equity = capital + floating_pnl_total
            
        # Update Labels
        self.lbl_time.config(text=f"Time: {t.strftime('%Y-%m-%d %H:%M:%S')}")
        self.lbl_equity.config(text=f"Equity (MTM): ${mtm_equity:.2f}")
        
        if realized_pnl >= 0:
            self.lbl_realized.config(text=f"Realized PnL: +${realized_pnl:.2f}", foreground="green")
        else:
            self.lbl_realized.config(text=f"Realized PnL: -${abs(realized_pnl):.2f}", foreground="red")
            
        if floating_pnl_total >= 0:
            self.lbl_floating.config(text=f"Floating PnL: +${floating_pnl_total:.2f}", foreground="green")
        else:
            self.lbl_floating.config(text=f"Floating PnL: -${abs(floating_pnl_total):.2f}", foreground="red")
            
        self.lbl_open_pos.config(text=f"Open Positions: {len(open_trades)}")
        
        # Update chart cursor line
        self.vline.set_xdata([t, t])
        self.canvas.draw_idle()
        
        # Update slider without triggering callback
        self._updating_slider = True
        self.slider.set(self.current_idx)
        self._updating_slider = False

def load_data_for_timeframe(tf_val, start_date, end_date):
    data_path = PROCESSED_DIR / f'features_{tf_val}_full.parquet'
    if not data_path.exists():
        return None
        
    df = pd.read_parquet(data_path)
    df = df.sort_values('timestamp')
    if start_date: df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date: df = df[df['timestamp'] <= pd.to_datetime(end_date)]
    return df

def main():
    parser = argparse.ArgumentParser(description="GUI Backtest for Multi-Portfolio")
    parser.add_argument('--config', type=str, default="ml/test_portfolios.json", help='Path to JSON configuration file')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    
    # Global overrides
    parser.add_argument('--capital', type=float, help='Override capital for ALL portfolios')
    parser.add_argument('--leverage', type=float, help='Override leverage for ALL portfolios')
    parser.add_argument('--threshold', type=float, help='Override entry confidence threshold for ALL portfolios')
    parser.add_argument('--max-pos', '--max-positions', dest='max_positions', type=int, help='Override max open positions for ALL portfolios')
    parser.add_argument('--use-scanner', action='store_true', default=None, help='Enable SmartScanner')
    parser.add_argument('--no-scanner', action='store_false', dest='use_scanner', help='Disable SmartScanner')
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"Config file {args.config} not found.")
        return
        
    with open(args.config, 'r') as f:
        portfolios = json.load(f)
        
    if not isinstance(portfolios, list):
        print("Error: Config must be a JSON array.")
        return
        
    print(f"🚀 Running Multi-Portfolio Backtest GUI with {len(portfolios)} strategies...")
    
    # Apply global overrides
    for port in portfolios:
        if args.capital is not None: port['capital'] = args.capital
        if args.leverage is not None: port['leverage'] = args.leverage
        if args.threshold is not None: port['threshold'] = args.threshold
        if args.max_positions is not None: port['max_positions'] = args.max_positions
        if args.use_scanner is not None: port['use_scanner'] = args.use_scanner

    results = {}
    total_capital = 0.0
    all_trades = []
    
    # Collect price histories across all symbols & timeframes
    # Dict mapping symbol -> {timestamp: price}
    symbols_data = {}

    for i, port in enumerate(portfolios):
        name = port.get('name', f"Strategy_{i+1}")
        total_capital += port.get('capital', 100)
        tf_val = port.get('timeframe', '4h')
        
        # Load dataset to extract prices for UI
        df = load_data_for_timeframe(tf_val, args.start, args.end)
        if df is not None:
             for symbol in df['symbol'].unique():
                 if symbol not in symbols_data:
                     symbols_data[symbol] = {}
                 symbol_df = df[df['symbol'] == symbol]
                 for idx, row in symbol_df.iterrows():
                     symbols_data[symbol][row['timestamp']] = row['close']
        
        # Run Backtest
        res = run_portfolio(port, args.start, args.end)
        if res:
            results[name] = res
            for t in res.trades:
                trade_copy = copy.deepcopy(t)
                trade_copy.strategy = name
                all_trades.append(trade_copy)
                
    if not results:
        print("No successful runs. Exiting.")
        return
        
    print("🔄 Aggregating Portfolio Data for UI...")
    
    # Merge Equity Curves
    all_timestamps = set()
    for name, res in results.items():
        if res.timestamps:
            all_timestamps.update(res.timestamps)
            
    all_timestamps = sorted(list(all_timestamps))
    if not all_timestamps:
        print("No timestamp data collected.")
        return
        
    df_equity = pd.DataFrame(index=all_timestamps)
    
    for name, res in results.items():
        if len(res.timestamps) == 0:
            df_equity[name] = res.config.initial_capital
            continue
            
        s = pd.Series(res.equity_curve, index=res.timestamps)
        s = s[~s.index.duplicated(keep='last')]
        s = s.reindex(all_timestamps).ffill().bfill()
        df_equity[name] = s
        
    df_equity['Total'] = df_equity.sum(axis=1)
    
    all_trades.sort(key=lambda t: t.entry_time)

    print("Launch UI...")
    app = MultiPortfolioGUI(results, all_timestamps, df_equity, total_capital, all_trades, symbols_data)
    app.mainloop()

if __name__ == '__main__':
    main()
