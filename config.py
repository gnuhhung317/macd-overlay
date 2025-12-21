"""
File cấu hình - Chỉnh sửa các thông số tại đây
"""

# ==================== BINANCE API ====================
# Để trống nếu chỉ sử dụng public data
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""

# ==================== TELEGRAM ====================
# Lấy từ @BotFather
TELEGRAM_BOT_TOKEN = ""

# Lấy từ https://api.telegram.org/bot<TOKEN>/getUpdates
TELEGRAM_CHAT_ID = ""

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
