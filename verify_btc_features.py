import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class QuantAnalyzer:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        # Convert times
        self.df['entry_time'] = pd.to_datetime(self.df['entry_time'])
        self.df['exit_time'] = pd.to_datetime(self.df['exit_time'])
        self.df['hour_of_day'] = self.df['entry_time'].dt.hour
        self.df['day_of_week'] = self.df['entry_time'].dt.dayofweek
        
        # Filter only executed trades
        self.executed = self.df[self.df['result'].isin(['WIN', 'LOSS', 'TIMEOUT'])].copy()

    def print_core_metrics(self):
        wins = self.executed[self.executed['result'] == 'WIN']
        losses = self.executed[self.executed['result'] == 'LOSS']
        
        win_rate = len(wins) / len(self.executed) if len(self.executed) > 0 else 0
        avg_win = wins['pnl_usd'].mean() if not wins.empty else 0
        avg_loss = abs(losses['pnl_usd'].mean()) if not losses.empty else 0
        
        gross_profit = wins['pnl_usd'].sum()
        gross_loss = abs(losses['pnl_usd'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        print(f"=== CORE EDGE METRICS ===")
        print(f"Total Trades: {len(self.executed)}")
        print(f"Win Rate:     {win_rate * 100:.2f}%")
        print(f"Profit Factor:{profit_factor:.2f}")
        print(f"Expectancy:   ${expectancy:.2f} per trade")
        print(f"Avg Reward/Risk: {avg_win / avg_loss if avg_loss > 0 else 0:.2f}")
        print("=========================\n")

    def analyze_mfe_mae(self):
        """
        Analyzes Maximum Favorable/Adverse Excursion to see if TP/SL are optimal.
        """
        print("=== MFE / MAE EFFICIENCY ===")
        # How much further did winning trades go beyond entry?
        wins = self.executed[self.executed['result'] == 'WIN']
        losses = self.executed[self.executed['result'] == 'LOSS']
        
        print(f"Avg MFE on Wins: {wins['mfe_atr'].mean():.2f} ATR (Are we cutting winners early?)")
        print(f"Avg MAE on Wins: {wins['mae_atr'].mean():.2f} ATR (How much heat do we take before winning?)")
        print(f"Avg MFE on Losses: {losses['mfe_atr'].mean():.2f} ATR (Did they go in profit before stopping out?)")
        
        # Plotting
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.executed, x='mae_atr', y='mfe_atr', hue='result', style='type', palette={'WIN':'green', 'LOSS':'red', 'TIMEOUT':'gray'})
        plt.title('Trade Efficiency: MFE vs MAE (in ATR)')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('Maximum Adverse Excursion (Heat Taken)')
        plt.ylabel('Maximum Favorable Excursion (Potential Profit)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def analyze_signal_decay(self):
        """
        Does waiting for a limit order reduce the edge?
        """
        print("\n=== SIGNAL DECAY (Wait Bars) ===")
        decay_stats = self.executed.groupby('wait_bars').agg(
            trades=('result', 'count'),
            win_rate=('result', lambda x: (x == 'WIN').mean() * 100),
            avg_pnl=('pnl_usd', 'mean')
        ).round(2)
        print(decay_stats)

    def analyze_regimes(self):
        """
        Performance by Long/Short and Time of Day.
        """
        print("\n=== DIRECTIONAL & TIME BIAS ===")
        dir_stats = self.executed.groupby('type').agg(
            trades=('result', 'count'),
            win_rate=('result', lambda x: (x == 'WIN').mean() * 100),
            pf=('pnl_usd', lambda x: x[x>0].sum() / abs(x[x<0].sum()) if abs(x[x<0].sum())>0 else np.inf)
        ).round(2)
        print("Directional Performance:")
        print(dir_stats)

if __name__ == "__main__":
    # Point this to your generated CSV
    csv_file = "ml/backtest_results_quant_sniper.csv" 
    
    try:
        analyzer = QuantAnalyzer(csv_file)
        analyzer.print_core_metrics()
        analyzer.analyze_signal_decay()
        analyzer.analyze_regimes()
        analyzer.analyze_mfe_mae() # This will block execution until the plot is closed
    except FileNotFoundError:
        print(f"Could not find {csv_file}. Ensure the path is correct.")