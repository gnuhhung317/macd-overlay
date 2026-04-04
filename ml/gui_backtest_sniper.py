#!/usr/bin/env python3
"""
GUI Backtester for Quant-Refined ML Sniper System.
Runs the backtest and provides a Tkinter UI to step through it bar-by-bar.
"""
import argparse
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from backtest_sniper import (
    BacktestConfig,
    _apply_profile_to_config,
    _default_auto018_profile_path,
    _sanitize_output_tag,
    run_backtest_with_config,
)


def _resolve_input_path(raw_value: str, search_roots: list[Path], label: str) -> str:
    raw = Path(raw_value)
    if raw.exists():
        return str(raw)

    for root in search_roots:
        candidate = root / raw
        if candidate.exists():
            print(f"Resolved {label}: {raw_value} -> {candidate}")
            return str(candidate)

    raise FileNotFoundError(
        f"{label} not found: {raw_value}. "
        f"Provide full path or place file under one of: {', '.join(str(x) for x in search_roots)}"
    )

class SniperBacktestGUI(tk.Tk):
    def __init__(self, trades, price_db, config):
        super().__init__()
        self.title("Sniper Model Backtest - Bar by Bar")
        self.geometry("1300x1000")
        
        self.trades = trades
        # Align price_db: set timestamp as index and use 'close' for lookups
        self.price_db = {
            sym: df.set_index('timestamp')['close'] 
            for sym, df in price_db.items() 
            if 'timestamp' in df.columns
        }
        self.config = config
        
        # Get all unique timestamps from all trades
        all_ts = set()
        for t in self.trades:
            if t.entry_time: all_ts.add(t.entry_time)
            if t.exit_time: all_ts.add(t.exit_time)
            
        self.timestamps = sorted(list(all_ts))
        if not self.timestamps:
            print("No trades found to display.")
            self.destroy()
            return
            
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
        realized_capital = self.config.initial_capital
        open_positions = []
        
        for t in self.timestamps:
            # Replicate portfoilo logic
            # Close trades
            closed = [tr for tr in open_positions if tr.exit_time and tr.exit_time <= t]
            for tr in closed:
                realized_capital += tr.pnl_usd
                open_positions.remove(tr)
                
            # Add new trades
            new_trades = [tr for tr in self.trades if tr.entry_time and tr.entry_time == t]
            open_positions.extend(new_trades)
            
            # Calculate floating PnL
            floating_pnl = 0
            for tr in open_positions:
                curr_price = self._get_current_price(tr.symbol, t)
                if curr_price is None:
                    curr_price = tr.entry_price
                if tr.type == 'LONG':
                    pnl_pct = (curr_price - tr.entry_price) / tr.entry_price
                else:
                    pnl_pct = (tr.entry_price - curr_price) / tr.entry_price
                    
                fee_est = tr.pos_size_usd * self.config.fee_rate * 2
                floating_pnl += (tr.pos_size_usd * pnl_pct) - fee_est
                
            equity_data.append(realized_capital + floating_pnl)
            
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
        
        self.ax.plot(self.timestamps, self.equity_curve, color="blue", linewidth=1.5)
        self.ax.set_ylabel("Equity ($)")
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.vline = self.ax.axvline(x=self.timestamps[0], color='red', linestyle='--', linewidth=2)
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
        
        self.tree.column("symbol", width=120, anchor="center")
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
            clicked_time = num2date(event.xdata).replace(tzinfo=None)
            closest_idx = min(range(len(self.timestamps)), key=lambda i: abs(self.timestamps[i] - clicked_time))
            if closest_idx != self.current_idx:
                self.current_idx = closest_idx
                self._update_ui_for_current_bar()
        except:
            pass
        
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
        if symbol in self.price_db and t in self.price_db[symbol].index:
            return self.price_db[symbol].loc[t]
        
        # If timestamp perfectly matches, return it. If not, fallback to nearest before
        if symbol in self.price_db:
            series = self.price_db[symbol]
            past_prices = series[series.index <= t]
            if not past_prices.empty:
                return past_prices.iloc[-1]
        return None
            
    def _update_ui_for_current_bar(self):
        t = self.timestamps[self.current_idx]
        
        closed_trades = [tr for tr in self.trades if tr.exit_time and tr.exit_time <= t]
        open_trades = [tr for tr in self.trades if tr.entry_time and tr.entry_time <= t and (not tr.exit_time or tr.exit_time > t)]
        
        realized_pnl = sum(tr.pnl_usd for tr in closed_trades)
        capital = self.config.initial_capital + realized_pnl
        
        floating_pnl_total = 0.0
        
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for tr in open_trades:
            curr_price = self._get_current_price(tr.symbol, t)
            if curr_price is None:
                curr_price = tr.entry_price
                
            if tr.type == 'LONG':
                pnl_pct = (curr_price - tr.entry_price) / tr.entry_price
            else:
                pnl_pct = (tr.entry_price - curr_price) / tr.entry_price
                
            fee_est = tr.pos_size_usd * self.config.fee_rate * 2
            float_pnl = (tr.pos_size_usd * pnl_pct) - fee_est
            
            floating_pnl_total += float_pnl
            
            tag = "profit" if float_pnl >= 0 else "loss"
            self.tree.insert("", "end", values=(
                tr.symbol,
                tr.type,
                tr.entry_time.strftime("%Y-%m-%d %H:%M"),
                f"{tr.entry_price:.5f}",
                f"{curr_price:.5f}",
                f"${tr.pos_size_usd:.2f}",
                f"${float_pnl:.2f}"
            ), tags=(tag,))
            
        mtm_equity = capital + floating_pnl_total
            
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
        
        self.vline.set_xdata([t, t])
        self.canvas.draw_idle()
        
        self._updating_slider = True
        self.slider.set(self.current_idx)
        self._updating_slider = False

def main():
    parser = argparse.ArgumentParser(description="GUI Backtest for Sniper Model")
    parser.add_argument('--start', type=str, default='2025-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--leverage', type=float, default=10.0, help='Leverage multiplier')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange data to use (binance or bitget)')
    parser.add_argument('--capital', type=float, default=100.0, help='Initial capital')
    parser.add_argument('--risk', type=float, default=0.005, help='Risk per trade')
    parser.add_argument('--max-pos', '--max-positions', dest='max_pos', type=int, default=3, help='Max concurrent positions')
    parser.add_argument('--max-files', type=int, default=60, help='Limit number of symbols scanned (0 = all)')
    parser.add_argument('--equity-mode', choices=['event', 'mtm', 'both'], default='both')
    parser.add_argument('--output-tag', type=str, default='')

    parser.add_argument('--profile-path', type=str, default=None, help='Path to profile JSON (p3_edge_research experiment format)')
    parser.add_argument('--profile-name', type=str, default=None, help='Experiment name in profile JSON')
    parser.add_argument('--use-auto018-profile', action='store_true', help='Load auto_018_live profile defaults')

    parser.add_argument('--selector-artifact-path', type=str, default=None, help='Path to pre-trained selector artifact (.joblib)')
    parser.add_argument('--use-research-model-selection', action='store_true', help='Use selector model pipeline (loads artifact if provided)')

    parser.add_argument('--research-compatible', action='store_true', help='Shortcut for fair comparison against run_research')
    parser.add_argument('--no-selection-debug-checks', dest='selection_debug_checks', action='store_false', help='Disable selector debug checks')
    parser.set_defaults(selection_debug_checks=True)
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parent.parent

    profile_path = args.profile_path
    if args.use_auto018_profile and profile_path is None:
        profile_path = str(_default_auto018_profile_path())
    if profile_path:
        profile_path = _resolve_input_path(
            profile_path,
            search_roots=[
                base_dir,
                base_dir / 'ml' / 'p3_edge_research' / 'experiments',
            ],
            label='Profile',
        )

    selector_artifact_path = args.selector_artifact_path
    if selector_artifact_path:
        selector_artifact_path = _resolve_input_path(
            selector_artifact_path,
            search_roots=[
                base_dir,
                base_dir / 'output' / 'selector_artifacts',
            ],
            label='Selector artifact',
        )

    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        leverage=args.leverage,
        exchange=args.exchange,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        max_open_trades=args.max_pos,
        max_files=int(args.max_files),
        equity_mode=args.equity_mode,
        output_tag=_sanitize_output_tag(args.output_tag),
        use_research_model_selection=bool(args.use_research_model_selection),
        selector_artifact_path=selector_artifact_path,
        selection_debug_checks=bool(args.selection_debug_checks),
    )

    if args.research_compatible:
        config.universe_mode = 'research'
        config.selection_mode = 'research'
        config.enforce_symbol_lock = False
        if config.min_stop_distance <= 0.0:
            config.min_stop_distance = 0.005

    if profile_path:
        profile_info = _apply_profile_to_config(config, Path(profile_path), args.profile_name)
        print(
            'Loaded profile '
            f"{profile_info['profile_name']} from {profile_info['profile_path']}"
        )
        if not config.output_tag:
            config.output_tag = _sanitize_output_tag(profile_info['profile_name'])
    
    print("Running sniper backtest first to gather UI data...")
    potential_signals, price_db, trades, equity_curve = run_backtest_with_config(config)
    
    if not trades:
        print("No valid trades executed in the backtest timeframe.")
        return
        
    print("Launching GUI...")
    app = SniperBacktestGUI(trades, price_db, config)
    app.mainloop()

if __name__ == "__main__":
    main()
