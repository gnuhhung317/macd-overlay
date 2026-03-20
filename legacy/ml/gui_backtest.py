#!/usr/bin/env python3
"""
GUI Backtester for 3-Stage ML System.
Runs the backtest and provides a Tkinter UI to step through it bar-by-bar.
"""
import argparse
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import threading
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import get_timeframe_config
from backtest_3stage import ThreeStageBacktester, BacktestConfig, DATA_DIR, PROCESSED_DIR

class BacktestGUI(tk.Tk):
    def __init__(self, df, result, config, symbols_data):
        super().__init__()
        self.title("3-Stage ML Backtest - Bar by Bar")
        self.geometry("1200x1000")
        
        self.df = df
        self.result = result
        self.config = config
        self.symbols_data = symbols_data
        
        self.timestamps = sorted(df['timestamp'].unique())
        self.current_idx = 0
        self.max_idx = len(self.timestamps) - 1
        
        self.is_playing = False
        self.play_speed_ms = 100
        
        self.equity_curve = self._compute_full_equity_curve()
        
        self._build_ui()
        self._update_ui_for_current_bar()
        
    def _compute_full_equity_curve(self):
        """Pre-compute the MTM equity curve across all timestamps for the chart."""
        equity_data = []
        capital = self.config.initial_capital
        open_positions = []
        
        # This is a simplified MTM calculation for the chart context
        # It won't perfectly match the tick-by-tick but closely approximates the bar-close equity
        for t in self.timestamps:
            # Add new trades
            new_trades = [tr for tr in self.result.trades if tr.entry_time == t]
            open_positions.extend(new_trades)
            
            # Remove closed trades and update capital with realized PnL
            closed_trades = [tr for tr in open_positions if tr.exit_time and tr.exit_time == t]
            for tr in closed_trades:
                capital += tr.pnl
                open_positions.remove(tr)
                
            # Calculate floating PnL
            floating_pnl = 0
            for tr in open_positions:
                curr_price = self.symbols_data.get(tr.symbol, {}).get(t)
                if curr_price is None:
                    curr_price = tr.entry_price
                if tr.direction == 'LONG':
                    pnl_pct = (curr_price - tr.entry_price) / tr.entry_price
                else:
                    pnl_pct = (tr.entry_price - curr_price) / tr.entry_price
                # Simplified fee approximation
                fee_est = tr.position_size * self.config.fee_rate * 2
                floating_pnl += (tr.position_size * pnl_pct) - fee_est
                
            equity_data.append(capital + floating_pnl)
            
        return equity_data
        
    def _build_ui(self):
        # --- Top Info Frame ---
        info_frame = ttk.LabelFrame(self, text="Portfolio State", padding=10)
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
        
        # --- Chart Frame (NEW) ---
        chart_frame = ttk.LabelFrame(self, text="Equity Curve", padding=10)
        chart_frame.pack(fill="x", expand=False, padx=10, pady=5)
        
        self.fig = Figure(figsize=(10, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Plot the equity curve
        self.ax.plot(self.timestamps, self.equity_curve, color="blue", linewidth=1.5)
        self.ax.set_ylabel("Equity ($)")
        self.ax.grid(True, alpha=0.3)
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
        table_frame = ttk.LabelFrame(self, text="Open Positions", padding=10)
        table_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        columns = ("symbol", "direction", "entry_time", "entry_price", "current_price", "size", "pnl")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        
        self.tree.heading("symbol", text="Symbol")
        self.tree.heading("direction", text="Direction")
        self.tree.heading("entry_time", text="Entry Time")
        self.tree.heading("entry_price", text="Entry Price")
        self.tree.heading("current_price", text="Current Price")
        self.tree.heading("size", text="Size ($)")
        self.tree.heading("pnl", text="Floating PnL ($)")
        
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
            
        # event.xdata is a matplotlib date float. Convert back to closest timestamp index
        try:
            from matplotlib.dates import num2date # Usually not needed if index is datetime
        except ImportError:
            pass
            
        clicked_time = matplotlib.dates.num2date(event.xdata).replace(tzinfo=None)
        
        # Find closest index
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
        # Look up price from pre-processed dictionary holding (symbol, time) mapping
        try:
            return self.symbols_data[symbol][t]
        except KeyError:
            return None
            
    def _update_ui_for_current_bar(self):
        t = self.timestamps[self.current_idx]
        
        # Find trades
        closed_trades = [tr for tr in self.result.trades if tr.exit_time and tr.exit_time <= t]
        
        # Open trades: entered at or before t, and either not exited yet, or exited AFTER t
        open_trades = [tr for tr in self.result.trades if tr.entry_time <= t and (not tr.exit_time or tr.exit_time > t)]
        
        # Calculate Realized
        realized_pnl = sum(tr.pnl for tr in closed_trades)
        capital = self.config.initial_capital + realized_pnl
        
        # Calculate Floating and Open Positions Table
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
                
            # Approximate floating PnL (ignoring dynamic exit fees for now, keeping it simple or using a fixed fee)
            # The backtester takes fee_rate on entry size, and fee_rate on exit size.
            fee_est = tr.position_size * self.config.fee_rate * 2
            float_pnl = (tr.position_size * pnl_pct) - fee_est
            
            floating_pnl_total += float_pnl
            
            tag = "profit" if float_pnl >= 0 else "loss"
            self.tree.insert("", "end", values=(
                tr.symbol,
                tr.direction,
                tr.entry_time.strftime("%Y-%m-%d %H:%M"),
                f"{tr.entry_price:.5f}",
                f"{curr_price:.5f}",
                f"${tr.position_size:.2f}",
                f"${float_pnl:.2f}"
            ), tags=(tag,))
            
        mtm_equity = capital + (floating_pnl_total if self.config.margin_mode == 'CROSS' else 0)
        # Note: In Isolated mode, equity technically doesn't include floating PnL until closure, but usually traders want to see MTM equity anyway.
        # We will show true MTM equity.
        mtm_equity = capital + floating_pnl_total
            
        # Update Labels
        self.lbl_time.config(text=f"Time: {t.strftime('%Y-%m-%d %H:%M:%S')}")
        self.lbl_equity.config(text=f"Equity: ${mtm_equity:.2f}")
        
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

def main():
    parser = argparse.ArgumentParser(description="GUI Backtest for 3-Stage ML System")
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage multiplier')
    parser.add_argument('--margin-mode', type=str, default='ISOLATED', choices=['ISOLATED', 'CROSS'])
    parser.add_argument('--tf', '--timeframe', type=str, default='4h', dest='timeframe', help='Timeframe (e.g., 1h, 4h, 1d)')
    parser.add_argument('--capital', type=float, default=100.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade')
    parser.add_argument('--max-pos', '--max-positions', type=int, default=10, dest='max_positions', help='Max open positions')
    parser.add_argument('--threshold', type=float, default=0.65, help='Entry confidence threshold')
    parser.add_argument('--use-scanner', action='store_true', help='Enable SmartScanner Entry Zone filtering')
    parser.add_argument('--live-mode', action='store_true', help='Calculate risk based on available balance (mimics Livebot)')
    args = parser.parse_args()
    
    # Configure backtest
    tf_val = args.timeframe
    max_pos_val = args.max_positions
    
    tf_config = get_timeframe_config(tf_val)
    config = BacktestConfig(
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        leverage=args.leverage,
        margin_mode=args.margin_mode,
        timeframe=tf_val,
        max_bars=tf_config.max_bars,
        max_open_trades=max_pos_val,
        entry_threshold=args.threshold,
        use_scanner_filter=args.use_scanner,
        use_available_balance_for_risk=args.live_mode,
        start_date=args.start,
        end_date=args.end
    )
    
    print("Loading data...")
    data_path = PROCESSED_DIR / f'features_{tf_val}_full.parquet'
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found.")
        return
        
    df = pd.read_parquet(data_path)
    df = df.sort_values('timestamp')
    
    # Date filtering
    if args.start:
        df = df[df['timestamp'] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df['timestamp'] <= pd.to_datetime(args.end)]
        
    if df.empty:
        print("Error: No data available in the specified date range.")
        return
        
    print(f"Running backtest from {df['timestamp'].min()} to {df['timestamp'].max()}...")
    backtester = ThreeStageBacktester(config)
    result = backtester.run_backtest(df, verbose=False)
    
    print(f"Backtest complete. {len(result.trades)} trades executed.")
    print("Preparing UI data structures...")
    
    # Pre-calculate prices for fast UI updates
    # Dictionary mapping symbol -> {timestamp: close_price}
    symbols_data = {}
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        symbols_data[symbol] = dict(zip(symbol_df['timestamp'], symbol_df['close']))
    
    print("Launching GUI...")
    app = BacktestGUI(df, result, config, symbols_data)
    app.mainloop()

if __name__ == "__main__":
    main()
