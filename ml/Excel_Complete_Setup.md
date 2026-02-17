# 📊 Excel Trading Journal - Complete Setup Guide

## 🎯 File Created: `Trading_Journal_Template.xlsx`

### 4 Sheets được tạo:
1. **Trade_Journal** - Main trading log
2. **Dashboard** - Performance metrics  
3. **Formulas_Reference** - Formula examples
4. **Risk_Management** - Risk monitoring

## 🎨 Conditional Formatting Setup

### Trade_Journal Sheet:

#### 1. Status Column (K) - Color Coding:
- **OPEN** → 🟢 Green fill, dark green text
  - Select K:K → Home → Conditional Formatting → New Rule
  - Formula: `=$K1="OPEN"`
  - Format: Green fill (#D4F6D4), dark green text

- **CLOSED** → ⚪ Light gray fill  
  - Formula: `=$K1="CLOSED"`
  - Format: Light gray fill (#F0F0F0)

- **TIMEOUT** → 🟡 Yellow fill
  - Formula: `=$K1="TIMEOUT"`  
  - Format: Yellow fill (#FFF2CC)

- **OVERDUE** → 🔴 Red fill (past timeout)
  - Formula: `=AND($K1="OPEN",TODAY()>$L1)`
  - Format: Red fill (#FFE6E6), red text

#### 2. PnL Column (Q) - Profit/Loss Colors:
- **Profit** → 🟢 Green text
  - Formula: `=$Q1>0`
  - Format: Green text (#008000), bold

- **Loss** → 🔴 Red text
  - Formula: `=$Q1<0`  
  - Format: Red text (#FF0000), bold

#### 3. Confidence Column (J) - Confidence Levels:
- **High (≥0.8)** → 🟢 Green background
  - Formula: `=$J1>=0.8`
  - Format: Green fill

- **Medium (0.65-0.8)** → 🟡 Yellow background
  - Formula: `=AND($J1>=0.65,$J1<0.8)`
  - Format: Yellow fill

- **Low (<0.65)** → 🔴 Red background
  - Formula: `=$J1<0.65`
  - Format: Red fill

## 📋 Data Validation Setup

### 1. Timeframe Column (D):
- Select D:D → Data → Data Validation
- Allow: List
- Source: `1h,4h,8h,12h,1d`

### 2. Direction Column (E):
- Allow: List  
- Source: `LONG,SHORT`

### 3. Status Column (K):
- Allow: List
- Source: `OPEN,CLOSED,TIMEOUT`

### 4. Exit_Reason Column (S):
- Allow: List
- Source: `TP_HIT,SL_HIT,MANUAL_CLOSE,TIMEOUT,LIQUIDATED`

## 🧮 Essential Formulas for Trade_Journal

### Column L (Timeout_Date) - Auto calculation:
```excel
=SWITCH(D2,
    "1h", A2+(10/24),
    "4h", A2+(40/24),
    "8h", A2+(80/24), 
    "12h", A2+(120/24),
    "1d", A2+10,
    A2+10)
```

### Column Q (PnL_USD) - Auto calculation:
```excel
=IF(AND(P2<>"",F2<>""),
    IF(E2="LONG",(P2-F2)/F2*I2,(F2-P2)/F2*I2),
    "")
```

### Column R (PnL_Percent):
```excel
=IF(AND(P2<>"",F2<>""),
    IF(E2="LONG",(P2-F2)/F2*100,(F2-P2)/F2*100),
    "")
```

## 📈 Dashboard Sheet - Key Metrics

### Already includes formulas for:
- Total/Active/Closed trades count
- Win rate percentage
- Total PnL and average trade
- Best/worst trade tracking
- Weekly/monthly performance

## ⚡ Quick Setup Steps:

1. **Open file** → `Trading_Journal_Template.xlsx`

2. **Apply conditional formatting** (follow guide above)

3. **Set up data validation** for dropdown lists

4. **Test formulas** by adding a new trade entry

5. **Customize colors** to your preference

6. **Add charts** (optional):
   - Insert → Chart → Line chart for equity curve
   - Use Dashboard metrics for summary charts

## 💡 Usage Tips:

### Daily Workflow:
1. **New Trade**: Fill columns A-J only
2. **Monitor**: Check Dashboard for overdue trades  
3. **Close Trade**: Fill columns N-S
4. **Review**: Check PnL and update notes

### Risk Management:
- Check Risk_Management sheet daily
- Monitor position sizes and exposure
- Set alerts for breaches

### Performance Analysis:
- Use Dashboard for quick overview
- Filter by timeframe/symbol for detailed analysis
- Export data periodically for backup

## 🔧 Advanced Features:

### 1. Equity Curve Chart:
- Create running total of PnL_USD
- Plot as line chart over time

### 2. Pivot Tables:
- Analyze performance by Symbol, Timeframe, Direction
- Monthly/weekly breakdowns

### 3. Macros (optional):
- Auto-refresh dashboard
- Quick trade entry forms
- Backup automation

## 📱 Mobile Access:
- Save to OneDrive/Google Drive for mobile access
- Use Excel mobile app to quickly check trades
- Set up phone notifications for overdue alerts

Your complete Excel trading journal is ready! 🚀