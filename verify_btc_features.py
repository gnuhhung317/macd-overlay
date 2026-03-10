import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_market_neutral_backtest(csv_path, top_n=5, horizon=7, fee=0.004):
    """
    Backtest Market Neutral:
    - Long Top N Alpha coins.
    - Short BTC (Benchmark) để triệt tiêu rủi ro thị trường.
    - Phí giao dịch tính cho cả 2 vị thế (0.4% tổng cộng).
    """
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Lấy dữ liệu BTC làm benchmark để Short
    # Giả sử bạn đã merge btc_future_return vào CSV, nếu chưa hãy dùng 'future_return' của BTCUSDT
    btc_returns = df[df['symbol'] == 'BTCUSDT'][['timestamp', 'future_return']].rename(columns={'future_return': 'btc_return'})
    df = df.merge(btc_returns, on='timestamp', how='left')
    
    all_dates = sorted(df['timestamp'].unique())
    trade_dates = all_dates[::horizon]
    
    results = []
    
    for date in trade_dates:
        day_df = df[df['timestamp'] == date]
        if day_df.empty: continue
            
        # Chọn Top N để Long
        top_assets = day_df.nlargest(top_n, 'pred_ens')
        
        if len(top_assets) > 0:
            long_ret = top_assets['future_return'].mean()
            # Lấy lợi nhuận BTC cùng kỳ để Short
            short_ret = day_df['btc_return'].iloc[0] if not pd.isna(day_df['btc_return'].iloc[0]) else 0
            
            # Lợi nhuận Neutral = (Lợi nhuận Long - Lợi nhuận Short) / 2 (do chia đôi vốn)
            # Trừ phí giao dịch cho cả 2 vị thế vào/ra
            net_ret = (long_ret - short_ret) - fee
            
            results.append({'timestamp': date, 'return': net_ret})
    
    res_df = pd.DataFrame(results)
    res_df['cum_return'] = (1 + res_df['return']).cumprod()
    
    # Tính toán Metrics
    sharpe = np.sqrt(365/horizon) * res_df['return'].mean() / (res_df['return'].std() + 1e-9)
    rolling_max = res_df['cum_return'].cummax()
    drawdown = (res_df['cum_return'] - rolling_max) / rolling_max
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    plt.plot(res_df['timestamp'], res_df['cum_return'], label='Market Neutral (Alpha Only)', color='#9b59b6')
    plt.fill_between(res_df['timestamp'], drawdown, alpha=0.2, color='red')
    plt.title('Hệ thống Market Neutral: Tách biệt Alpha khỏi sóng thị trường', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"✅ KẾT QUẢ MARKET NEUTRAL:")
    print(f"💰 Lợi nhuận cuối cùng: {res_df['cum_return'].iloc[-1]:.2f}x tài khoản")
    print(f"📊 Chỉ số Sharpe: {sharpe:.2f}")
    print(f"📉 Max Drawdown: {drawdown.min():.2%}")
    return res_df
# Cách dùng:
results = run_market_neutral_backtest('oos_predictions.csv')