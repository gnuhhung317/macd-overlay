import requests
from datetime import datetime


class TelegramNotifier:
    """
    Module xử lý gửi thông báo qua Telegram
    """
    
    def __init__(self, token, chat_id):
        """
        Khởi tạo Telegram notifier
        
        Args:
            token (str): Telegram bot token
            chat_id (str): Telegram chat ID
        """
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.token}"
        
    def send_message(self, message, parse_mode='HTML'):
        """
        Gửi tin nhắn qua Telegram
        
        Args:
            message (str): Nội dung tin nhắn
            parse_mode (str): HTML hoặc Markdown
            
        Returns:
            bool: True nếu gửi thành công
        """
        url = f"{self.api_url}/sendMessage"
        
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode
        }
        
        try:
            response = requests.post(url, data=payload)
            if response.status_code == 200:
                print("✓ Đã gửi tin nhắn Telegram")
                return True
            else:
                print(f"✗ Lỗi gửi Telegram: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Exception khi gửi Telegram: {e}")
            return False
    
    def send_crossover_alert(self, crossover, symbol):
        """
        Gửi thông báo crossover
        
        Args:
            crossover (dict): Thông tin crossover
            symbol (str): Symbol (e.g., BTCUSDT)
        """
        message = self.format_crossover_message(crossover, symbol)
        return self.send_message(message)
    
    def format_crossover_message(self, crossover, symbol):
        """
        Format tin nhắn crossover cho Telegram
        
        Args:
            crossover (dict): Thông tin crossover
            symbol (str): Trading symbol
            
        Returns:
            str: Tin nhắn đã format
        """
        emoji = "🟢" if crossover['type'] == 'BULLISH' else "🔴"
        signal_type = "Tín hiệu MUA" if crossover['type'] == 'BULLISH' else "Tín hiệu BÁN"
        
        message = f"""
{emoji} <b>MACD Crossover - {symbol}</b>

📊 Loại: <b>{signal_type}</b>
💰 Giá: <b>${crossover['price']:.2f}</b>
📅 Thời gian: {crossover['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

📈 MACD: {crossover['macd']:.4f}
📉 Signal: {crossover['signal']:.4f}
📊 Histogram: {crossover['macd'] - crossover['signal']:.4f}
        """
        
        return message.strip()
    
    def send_summary(self, summary_data):
        """
        Gửi báo cáo tổng hợp
        
        Args:
            summary_data (dict): Dữ liệu tổng hợp
        """
        message = f"""
📊 <b>Báo Cáo MACD</b>

Symbol: {summary_data.get('symbol', 'N/A')}
Khung thời gian: {summary_data.get('interval', 'N/A')}
Số nến: {summary_data.get('candles', 0)}

🔍 Crossovers tìm thấy:
• Bullish: {summary_data.get('bullish', 0)}
• Bearish: {summary_data.get('bearish', 0)}
• Tổng: {summary_data.get('total', 0)}

📅 Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        return self.send_message(message.strip())
    
    def test_connection(self):
        """
        Test kết nối Telegram
        
        Returns:
            bool: True nếu kết nối OK
        """
        print("Đang test kết nối Telegram...")
        test_message = "🤖 Test kết nối thành công!"
        return self.send_message(test_message)
