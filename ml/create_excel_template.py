#!/usr/bin/env python3
"""
Create Excel Trading Journal Template with formulas and formatting
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_excel_trading_journal():
    """Create complete Excel trading journal with formulas and formatting."""
    
    # Sample data
    sample_data = {
        'Entry_Date': [
            datetime(2026, 2, 13),
            datetime(2026, 2, 12), 
            datetime(2026, 2, 11),
            datetime(2026, 2, 10)
        ],
        'Entry_Time': [
            '14:30',
            '08:00',
            '16:00', 
            '12:00'
        ],
        'Symbol': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT'],
        'Timeframe': ['4h', '1d', '8h', '12h'],
        'Direction': ['LONG', 'SHORT', 'LONG', 'SHORT'],
        'Entry_Price': [42500, 2100, 0.45, 95.5],
        'SL_Price': [41650, 2142, 0.441, 97.3],
        'TP_Price': [44200, 2016, 0.468, 91.4],
        'Position_Size': [1000, 800, 500, 1200],
        'ML_Confidence': [0.78, 0.82, 0.71, 0.85],
        'Status': ['OPEN', 'CLOSED', 'CLOSED', 'TIMEOUT'],
        'Timeout_Date': [
            datetime(2026, 2, 15),
            datetime(2026, 2, 22),
            datetime(2026, 2, 15),
            datetime(2026, 2, 15)
        ],
        'Timeout_Time': ['06:30', '08:00', '00:00', '12:00'],
        'Exit_Date': [None, datetime(2026, 2, 15), datetime(2026, 2, 13), datetime(2026, 2, 15)],
        'Exit_Time': [None, '12:30', '08:00', '12:00'],
        'Exit_Price': [None, 2016, 0.441, 93.2],
        'PnL_USD': [None, 64, -9, 27.6],
        'PnL_Percent': [None, 4.0, -2.0, 2.3],
        'Exit_Reason': [None, 'TP_HIT', 'SL_HIT', 'TIMEOUT'],
        'Notes': [
            'MACD xoay tăng + RSI oversold',
            'Perfect entry setup', 
            'False breakout',
            'Held too long but still profit'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Create Excel file with multiple sheets
    output_file = Path(__file__).parent / 'Trading_Journal_Template.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main trading journal
        df.to_excel(writer, sheet_name='Trade_Journal', index=False)
        
        # Dashboard sheet
        dashboard_data = {
            'Metric': [
                'Total Trades', 'Active Trades', 'Closed Trades', 'Overdue Trades',
                'Win Rate (%)', 'Total PnL ($)', 'Avg Trade ($)', 'Best Trade ($)',
                'Worst Trade ($)', 'This Week PnL ($)', 'This Month PnL ($)'
            ],
            'Value': [
                '=COUNTA(Trade_Journal!A:A)-1',
                '=COUNTIF(Trade_Journal!K:K,"OPEN")',
                '=COUNTIF(Trade_Journal!K:K,"CLOSED")',
                '=SUMPRODUCT((Trade_Journal!K:K="OPEN")*(TODAY()>Trade_Journal!L:L))',
                '=IFERROR(COUNTIFS(Trade_Journal!K:K,"CLOSED",Trade_Journal!Q:Q,">0")/COUNTIF(Trade_Journal!K:K,"CLOSED")*100,0)',
                '=SUMIF(Trade_Journal!K:K,"CLOSED",Trade_Journal!Q:Q)',
                '=IFERROR(AVERAGEIF(Trade_Journal!K:K,"CLOSED",Trade_Journal!Q:Q),0)',
                '=IFERROR(MAXIFS(Trade_Journal!Q:Q,Trade_Journal!K:K,"CLOSED"),0)',
                '=IFERROR(MINIFS(Trade_Journal!Q:Q,Trade_Journal!K:K,"CLOSED"),0)',
                '=SUMIFS(Trade_Journal!Q:Q,Trade_Journal!N:N,">="&TODAY()-7,Trade_Journal!K:K,"CLOSED")',
                '=SUMIFS(Trade_Journal!Q:Q,Trade_Journal!N:N,">="&EOMONTH(TODAY(),-1)+1,Trade_Journal!K:K,"CLOSED")'
            ],
            'Formula_Description': [
                'Count all trades', 'Open positions count', 'Completed trades',
                'Trades past timeout', 'Percentage of winning trades',
                'Total profit/loss', 'Average per trade', 'Best single trade',
                'Worst single trade', 'Last 7 days PnL', 'Current month PnL'
            ]
        }
        
        dashboard_df = pd.DataFrame(dashboard_data)
        dashboard_df.to_excel(writer, sheet_name='Dashboard', index=False)
        
        # Formulas sheet for reference
        formulas_data = {
            'Column': [
                'L (Timeout_Date)', 'M (Timeout_Time)', 'Q (PnL_USD)', 
                'R (PnL_Percent)', 'Risk_Alert', 'Confidence_Alert'
            ],
            'Formula': [
                '=SWITCH(D2,"1h",A2+(10/24),"4h",A2+(40/24),"8h",A2+(80/24),"12h",A2+(120/24),"1d",A2+10,A2+10)',
                '=SWITCH(D2,"1h",TIME(10,0,0),"4h",TIME(16,0,0),"8h",TIME(0,0,0),"12h",TIME(0,0,0),"1d",TIME(0,0,0),TIME(0,0,0))',
                '=IF(E2="LONG",(P2-F2)/F2*I2,(F2-P2)/F2*I2)',
                '=IF(E2="LONG",(P2-F2)/F2*100,(F2-P2)/F2*100)',
                '=IF(I2>1000,"⚠️ Large",IF(I2>500,"⚡ Medium","✅ Safe"))',
                '=IF(J2>=0.8,"🔥 High",IF(J2>=0.65,"👍 Good","⚠️ Low"))'
            ],
            'Description': [
                'Auto calculate timeout date based on timeframe',
                'Calculate timeout time (for intraday timeframes)',
                'Calculate PnL in USD (+ for profit, - for loss)',
                'Calculate PnL percentage',
                'Risk level based on position size',
                'Confidence level indicator'
            ]
        }
        
        formulas_df = pd.DataFrame(formulas_data)
        formulas_df.to_excel(writer, sheet_name='Formulas_Reference', index=False)
        
        # Risk Management sheet
        risk_data = {
            'Risk_Rule': [
                'Max Position Size', 'Max Risk per Trade', 'Max Daily Loss', 
                'Max Open Trades', 'Min Confidence', 'Max Drawdown Alert'
            ],
            'Limit': [1500, 100, 200, 5, 0.60, -500],
            'Current': [
                '=MAX(Trade_Journal!I:I)',
                '=MAX(Trade_Journal!Q:Q)', 
                '=SUMIFS(Trade_Journal!Q:Q,Trade_Journal!N:N,TODAY())',
                '=COUNTIF(Trade_Journal!K:K,"OPEN")',
                '=MIN(Trade_Journal!J:J)',
                '=MIN(Trade_Journal!Q:Q)'
            ],
            'Status': [
                '=IF(C2>B2,"❌ BREACH","✅ OK")',
                '=IF(C3>B3,"❌ BREACH","✅ OK")',
                '=IF(C4<-ABS(B4),"❌ BREACH","✅ OK")',
                '=IF(C5>B5,"❌ BREACH","✅ OK")',
                '=IF(C6<B6,"❌ BREACH","✅ OK")',
                '=IF(C7<B7,"❌ BREACH","✅ OK")'
            ]
        }
        
        risk_df = pd.DataFrame(risk_data)
        risk_df.to_excel(writer, sheet_name='Risk_Management', index=False)
    
    print(f"✅ Created Excel trading journal: {output_file}")
    print("\n📊 Sheets created:")
    print("   1. Trade_Journal - Main trading log")
    print("   2. Dashboard - Performance metrics")
    print("   3. Formulas_Reference - Formula examples")
    print("   4. Risk_Management - Risk monitoring")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Open {output_file.name} in Excel")
    print(f"   2. Apply conditional formatting for Status column")
    print(f"   3. Set up charts for equity curve")
    print(f"   4. Add data validation for dropdowns")
    
    return output_file

if __name__ == '__main__':
    create_excel_trading_journal()