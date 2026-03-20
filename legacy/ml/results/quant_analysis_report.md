# Quantitative Strategy Analysis

## 1. Return/Risk Ratio (Calmar Proxy)
> Ratio of Average Return to Maximum Drawdown. Higher is better.

| Signal_Tier       |   12h |   1d |   8h |
|:------------------|------:|-----:|-----:|
| All Signals       |  7.93 | 1.83 | 3.63 |
| Balanced (>=0.66) |  4.62 | 5.93 | 3.4  |
| Elite (1.0)       |  1.49 | 2.24 | 1.53 |
| Weak (>=0.33)     |  6.08 | 3.89 | 3.63 |

## 2. Profit Probability (% of Profitable Windows)
> Consistency metric: How often does this config survive a 90-day period with >0% return?

| Signal_Tier       |   12h |    1d |    8h |
|:------------------|------:|------:|------:|
| All Signals       | 86.27 | 82.35 | 88.24 |
| Balanced (>=0.66) | 88.24 | 96.08 | 94.12 |
| Elite (1.0)       | 80.39 | 76.47 | 62.75 |
| Weak (>=0.33)     | 90.2  | 88.24 | 88.24 |

## 3. Return Volatility (Standard Deviation)
> Represents the variance in returns across different market regimes. Lower means more stable.

| Signal_Tier       |     12h |     1d |     8h |
|:------------------|--------:|-------:|-------:|
| All Signals       | 1268.32 | 168.23 | 454.35 |
| Balanced (>=0.66) |  464.8  | 140.23 | 194.38 |
| Elite (1.0)       |  123.86 |  62.19 | 122.71 |
| Weak (>=0.33)     | 1187.69 | 208.91 | 482.08 |

## 4. Median Drawdown (%) Across Timeframes
> Typical risk profile experienced in a window.

| Signal_Tier       |   12h |    1d |    8h |
|:------------------|------:|------:|------:|
| All Signals       | 33.14 | 21.81 | 23.91 |
| Balanced (>=0.66) | 23.4  |  2.49 | 14.11 |
| Elite (1.0)       | 13.43 |  0    |  0    |
| Weak (>=0.33)     | 27.85 | 14.02 | 21.44 |

## 5. Full Master Data Table

| Timeframe   | Signal_Tier       |   Total_Windows |   Avg_Return |   Median_Return |   Return_Std |   Worst_DD |   Median_DD |   Avg_DD |   Avg_WinRate |   Avg_Trades |   Profitable_Count |   Profit_Probability |   Calmar_Proxy |
|:------------|:------------------|----------------:|-------------:|----------------:|-------------:|-----------:|------------:|---------:|--------------:|-------------:|-------------------:|---------------------:|---------------:|
| 12h         | All Signals       |              51 |       479.66 |          100.28 |      1268.32 |      60.51 |       33.14 |    32.3  |       65.8319 |     31.4314  |                 44 |                86.27 |           7.93 |
| 12h         | Balanced (>=0.66) |              51 |       285.53 |           89.11 |       464.8  |      61.75 |       23.4  |    26.49 |       69.1745 |     25       |                 45 |                88.24 |           4.62 |
| 12h         | Elite (1.0)       |              51 |        65.67 |           37.03 |       123.86 |      43.96 |       13.43 |    13.21 |       72.0926 |      8.52941 |                 41 |                80.39 |           1.49 |
| 12h         | Weak (>=0.33)     |              51 |       443.84 |          115.44 |      1187.69 |      72.97 |       27.85 |    30.14 |       66.4573 |     29.3922  |                 46 |                90.2  |           6.08 |
| 1d          | All Signals       |              51 |       144.7  |          104.77 |       168.23 |      79.27 |       21.81 |    24.36 |       71.1412 |     15.1569  |                 42 |                82.35 |           1.83 |
| 1d          | Balanced (>=0.66) |              51 |       118.69 |           77.12 |       140.23 |      20    |        2.49 |     5.87 |       82.4359 |      8       |                 49 |                96.08 |           5.93 |
| 1d          | Elite (1.0)       |              51 |        50.08 |           19.1  |        62.19 |      22.33 |        0    |     2.46 |       68.9618 |      3.64706 |                 39 |                76.47 |           2.24 |
| 1d          | Weak (>=0.33)     |              51 |       173.04 |          114.48 |       208.91 |      44.5  |       14.02 |    15.65 |       73.2176 |     13.098   |                 45 |                88.24 |           3.89 |
| 8h          | All Signals       |              51 |       326.45 |          174.17 |       454.35 |      89.84 |       23.91 |    27.18 |       68.1074 |     36.3725  |                 45 |                88.24 |           3.63 |
| 8h          | Balanced (>=0.66) |              51 |       149.44 |           73.81 |       194.38 |      43.93 |       14.11 |    16.3  |       75.0442 |     17.3922  |                 48 |                94.12 |           3.4  |
| 8h          | Elite (1.0)       |              51 |        51.45 |            9.9  |       122.71 |      33.61 |        0    |     6.08 |       64.257  |      5.19608 |                 32 |                62.75 |           1.53 |
| 8h          | Weak (>=0.33)     |              51 |       310.71 |          168.02 |       482.08 |      85.62 |       21.44 |    24.34 |       68.7667 |     31.6667  |                 45 |                88.24 |           3.63 |

