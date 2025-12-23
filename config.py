"""
File cấu hình - Chỉnh sửa các thông số tại đây
"""

# ==================== BINANCE API ====================
# Để trống nếu chỉ sử dụng public data
BINANCE_API_KEY = "PolvgaP1tWf4nxT4x7Nr41uopKLS5Hc4MHZmmRFDRwpmW3ZTI3uuYJpZYf1zmQjp"
BINANCE_API_SECRET = "secret FUn7xluIXkMYKs4hLSDSg5M05TWgmme0wiuP5q5jsm2T8m6QR09SpkJmvRt3LHJA"

# ==================== TELEGRAM ====================
# Lấy từ @BotFather
TELEGRAM_BOT_TOKEN = "8484997609:AAHb_L8wO0WjtKRioas0USfhqHOXW_zlFQ0"

# Lấy từ https://api.telegram.org/bot<TOKEN>/getUpdates
TELEGRAM_CHAT_ID = "6465176588"

# ==================== TRADING PARAMETERS ====================
# Symbol để theo dõi
SYMBOL = "BTCUSDT"

# Khung thời gian: 1m, 5m, 15m, 30m, 1h, 4h, 1d
INTERVAL = "1h"

# Khoảng thời gian phân tích (cho historical analysis)
# Hỗ trợ tự động fetch batch, không giới hạn số nến
# Có thể dùng: '1 year ago UTC', '2024-01-01', etc.
START_DATE = "1 year ago UTC"
END_DATE = "now UTC"

# Khoảng thời gian check (giây) khi monitor real-time
CHECK_INTERVAL = 60

# ==================== MACD PARAMETERS ====================
FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9
