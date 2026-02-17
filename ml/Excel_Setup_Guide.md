# Excel Trading Journal Setup Guide

## Template Structure

### Columns Explanation:
- **Entry_Date/Time**: Khi vào lệnh
- **Symbol**: Coin trading (BTCUSDT, ETHUSDT...)  
- **Timeframe**: 1h, 4h, 8h, 12h, 1d
- **Direction**: LONG hoặc SHORT
- **Entry_Price**: Giá vào lệnh
- **SL_Price/TP_Price**: Stop Loss và Take Profit prices
- **Position_Size**: Size position tính bằng USD
- **ML_Confidence**: Confidence từ ML model (0-1)
- **Status**: OPEN, CLOSED, TIMEOUT
- **Timeout_Date/Time**: Tự động tính theo timeframe
- **Exit details**: Khi close trade
- **PnL**: Tự động tính
- **Notes**: Ghi chú cá nhân

## Excel Formulas

### 1. Timeout Date Calculation (Column L):
```excel
=SWITCH(D2,
    "1h", A2 + (10/24),
    "4h", A2 + (40/24),  
    "8h", A2 + (80/24),
    "12h", A2 + (120/24),
    "1d", A2 + 10,
    A2 + 10)
```

### 2. Timeout Time Calculation (Column M):
```excel
=SWITCH(D2,
    "1h", B2 + TIME(10,0,0),
    "4h", B2 + TIME(16,0,0),
    "8h", B2,
    "12h", B2,
    "1d", B2,
    B2)
```

### 3. PnL USD Calculation (Column Q):
```excel
=IF(E2="LONG",
    (P2-F2)/F2*I2,
    (F2-P2)/F2*I2)
```

### 4. PnL Percentage Calculation (Column R):
```excel
=IF(E2="LONG",
    (P2-F2)/F2*100,
    (F2-P2)/F2*100)
```

### 5. Status Alert (Conditional Formatting):
- **OPEN**: Màu xanh
- **OVERDUE**: Màu đỏ (khi past timeout)
- **CLOSED**: Màu xám

**Formula for Overdue:**
```excel
=AND(K2="OPEN", TODAY()+TIME(HOUR(NOW()),MINUTE(NOW()),0) > L2+M2)
```

## Dashboard Summary Formulas

### Active Trades Count:
```excel
=COUNTIF(K:K,"OPEN")
```

### Overdue Trades:
```excel
=SUMPRODUCT((K:K="OPEN")*(TODAY()+TIME(HOUR(NOW()),MINUTE(NOW()),0) > L:L+M:M))
```

### Today's PnL:
```excel
=SUMIFS(Q:Q,N:N,TODAY(),K:K,"CLOSED")
```

### Win Rate:
```excel
=COUNTIFS(K:K,"CLOSED",Q:Q,">0")/COUNTIF(K:K,"CLOSED")*100
```

### Average Trade PnL:
```excel
=AVERAGEIF(K:K,"CLOSED",Q:Q)
```

## Risk Management Alerts

### Position Size vs Balance (add in new column):
```excel
=IF(I2 > 1000, "⚠️ Large Position", 
   IF(I2 > 500, "⚡ Medium Risk", "✅ Safe"))
```

### Confidence Level Alert:
```excel
=IF(J2 >= 0.8, "🔥 High Confidence",
   IF(J2 >= 0.65, "👍 Good Signal", 
   "⚠️ Low Confidence"))
```

## Usage Workflow:

1. **New Trade**: Fill Entry columns A-J
2. **Timeout auto-calculates**: Check L-M columns  
3. **Monitor**: Sort by Status, check overdue trades
4. **Close Trade**: Fill Exit columns N-S
5. **Review**: Check PnL and update Notes

## Tips:

- **Backup daily**: Save copy với tên Trading_Journal_YYYYMMDD.xlsx
- **Color coding**: Dùng conditional formatting cho visual alerts
- **Pivot Tables**: Tạo monthly/weekly performance reports
- **Charts**: Plot equity curve từ cumulative PnL

## Advanced Features:

### Monthly Performance Sheet:
```excel
=SUMIFS('Journal'!Q:Q,'Journal'!N:N,">="&DATE(YEAR(TODAY()),MONTH(TODAY()),1),'Journal'!N:N,"<"&DATE(YEAR(TODAY()),MONTH(TODAY())+1,1))
```

### Best/Worst Trade:
```excel
=MAX(Q:Q)  // Best
=MIN(Q:Q)  // Worst
```

### Drawdown Tracking:
```excel
=MAX($Q$2:Q2)-Q2  // Running drawdown
```

File template CSV đã tạo ở: `Trading_Journal_Template.csv`
Import vào Excel và setup formulas theo hướng dẫn trên!