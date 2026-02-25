import ccxt
def test_bitget_future():
    print("Khởi tạo sàn Bitget (mảng Future/Swap) qua CCXT...")
    
    # # 1. Khởi tạo sàn và cấu hình mặc định là Future (Swap)
    # exchange = ccxt.bitget({
    #     'enableRateLimit': True,
    #     'options': {
    #         'defaultType': 'swap', # <-- ĐẶT MẶC ĐỊNH LÀ FUTURE
    #     },
    #     # Điền API Keys của bạn vào đây (cần thiết cho đặt lệnh, xem số dư/vị thế future):
    #     # 'apiKey': 'YOUR_API_KEY',
    #     # 'secret': 'YOUR_API_SECRET',
    #     # 'password': 'YOUR_API_PASSWORD', # Bitget yêu cầu passphrase
    # })
    
    exchange = ccxt.bitget({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap', # <-- ĐẶT MẶC ĐỊNH LÀ FUTURE
        },
        # Điền API Keys của bạn vào đây (cần thiết cho đặt lệnh, xem số dư/vị thế future):
        'apiKey': 'x',
        'secret': 'x',
        'password': 'x', # Bitget yêu cầu passphrase
    })

    try:
        # --- TEST PUBLIC API (FUTURE) ---
        print("\n[1] Đang tải danh sách các thị trường (Markets)...")
        exchange.load_markets()
        
        # Lọc ra các thị trường Future (swap)
        swap_markets = [m for m in exchange.markets.values() if m['swap']]
        print(f"Đã tải {len(swap_markets)} thị trường Future từ Bitget.")
        
        # Chọn cặp BTC/USDT quy định dưới dạng Future của CCXT (ký quỹ bằng USDT)
        # Thường là: 'BTC/USDT:USDT'
        symbol = 'BTC/USDT:USDT' 
        
        if symbol not in exchange.markets:
            print(f"Không tìm thấy {symbol}, fallback về 'BTC/USDT'...")
            symbol = 'BTC/USDT'
            
        print(f"\n[2] Đang lấy giá hiện tại (Ticker) cho hợp đồng {symbol}...")
        ticker = exchange.fetch_ticker(symbol)
        print(f"Giá hiện tại của {symbol}: {ticker['last']} USDT")
        
        print(f"\n[3] Đang lấy thông tin Orderbook (Sổ lệnh) cho {symbol}...")
        orderbook = exchange.fetch_order_book(symbol, limit=5)
        print("Bid tốt nhất (Giá mua cao nhất):", orderbook['bids'][0][0] if orderbook['bids'] else 'N/A')
        print("Ask tốt nhất (Giá bán thấp nhất):", orderbook['asks'][0][0] if orderbook['asks'] else 'N/A')
        # --- TEST PRIVATE API (Yêu cầu API Key) ---
        if exchange.apiKey:
            print("\n[4] Có API Key, đang lấy thông tin Balance ví Future...")
            # Nhờ option 'defaultType': 'swap', hàm fetch_balance() sẽ query ví Future
            balance = exchange.fetch_balance()
            
            usdt_balance = balance.get('USDT', {})
            print("Tổng số dư USDT (Future):", usdt_balance.get('total', 0.0))
            print("Số dư USDT khả dụng (Future):", usdt_balance.get('free', 0.0))
            
            print(f"\n[5] Đang kiểm tra các vị thế (Positions) đang mở cho {symbol}...")
            positions = exchange.fetch_positions([symbol])
            
            has_position = False
            for pos in positions:
                contracts = float(pos.get('contracts', 0) or 0)
                if contracts > 0:
                    has_position = True
                    print(f"-> Vị thế: {pos['side'].upper()} | Vào giá: {pos.get('entryPrice')} | Qty: {contracts} | PnL: {pos.get('unrealizedPnl')}")
            
            if not has_position:
                print("-> Không có vị thế nào đang mở cho cặp này.")
        else:
            print("\n[!] Bỏ qua test Private API (bao gồm Ví và Vị thế) vì chưa cấu hình key.")
    except Exception as e:
        print(f"\n[LỖI]: {e}")
if __name__ == "__main__":
    test_bitget_future()
