import requests
from datetime import datetime
from pathlib import Path



# --- Constants (From ML Analysis Report) ---
AVG_MAE_STATS = {
    '4h': 0.035,   # ~3.5%
    '8h': 0.045,   # ~4.5%
    '12h': 0.055,  # ~5.5%
    '1d': 0.065    # ~6.5%
}

def format_price(price: float) -> str:
    """Smart price formatting - show enough decimals for low-price coins."""
    if price is None or price == 0:
        return "$0"
    if price >= 1000:
        return f"${price:,.0f}"
    elif price >= 1:
        return f"${price:.2f}"
    elif price >= 0.01:
        return f"${price:.4f}"
    elif price >= 0.0001:
        return f"${price:.6f}"
    else:
        return f"${price:.8f}"
        

class TelegramNotifier:
    """
    Module xử lý gửi thông báo qua Telegram
    Tích hợp ML predictions cho entry, SL, TP
    """
    
    def __init__(self, token, chat_id, ml_system=None):
        """
        Khởi tạo Telegram notifier
        
        Args:
            token (str): Telegram bot token
            chat_id (str): Telegram chat ID
            ml_system: Optional ML system for predictions
        """
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.token}"
        self.ml_system = ml_system
        self.entry_threshold = 0.5  # Minimum confidence to send alert
        
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
    

    def send_crossover_alert(self, crossover, symbol, interval=None, features_df=None):
        """
        Gửi thông báo crossover với ML predictions
        Args:
            crossover (dict): Thông tin crossover
            symbol (str): Symbol (e.g., BTCUSDT)
            interval (str, optional): Khung thời gian (e.g., 1d, 1h)
            features_df (pd.DataFrame, optional): Features for ML prediction
        Returns:
            bool: True nếu gửi thành công, False nếu bị filter hoặc lỗi
        """
        ml_prediction = None
        
        # Check if ml_prediction was already passed in crossover dict
        if 'ml_prediction' in crossover:
            ml_prediction = crossover['ml_prediction']
        # Otherwise, try to get prediction from features_df
        elif self.ml_system is not None and features_df is not None:
            try:
                ml_prediction = self.ml_system.predict(features_df)
            except Exception as e:
                print(f"⚠️ ML prediction error: {e}")
                ml_prediction = None
        
        # Filter by confidence threshold (only if we have ml_prediction)
        if ml_prediction is not None and ml_prediction.get('entry_confidence', 1.0) < self.entry_threshold:
            print(f"⚠️ Signal filtered: confidence {ml_prediction['entry_confidence']:.1%} < {self.entry_threshold:.0%}")
            return False
        
        message = self.format_crossover_message(crossover, symbol, interval, ml_prediction)
        return self.send_message(message)

    def format_crossover_message(self, crossover, symbol, interval=None, ml_prediction=None):
        """
        Format tin nhắn crossover cho Telegram với ML predictions
        Args:
            crossover (dict): Thông tin crossover
            symbol (str): Trading symbol
            interval (str, optional): Khung thời gian
            ml_prediction (dict, optional): ML prediction results
        Returns:
            str: Tin nhắn đã format
        """
        is_bullish = crossover['type'] == 'BULLISH'
        emoji = "🟢" if is_bullish else "🔴"
        signal_type = "Tín hiệu MUA" if is_bullish else "Tín hiệu BÁN"
        interval_str = f" | Khung: <b>{interval}</b>" if interval else ""
        
        price = crossover['price']
        price_str = format_price(price)
        
        message = f"""
{emoji} <b>MACD Crossover - {symbol}{interval_str}</b>

📊 Loại: <b>{signal_type}</b>
💰 Giá: <b>{price_str}</b>
📅 Thời gian: {crossover['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

📈 MACD: {crossover['macd']:.4f}
📉 Signal: {crossover['signal']:.4f}
📊 Histogram: {crossover['macd'] - crossover['signal']:.4f}"""

        # Add ML predictions if available
        if ml_prediction is not None:
            confidence = ml_prediction.get('entry_confidence', 0)
            # Support both old keys (sl_pct/tp_pct) and new keys (sl_percent/tp_percent)
            sl_pct = ml_prediction.get('sl_percent', ml_prediction.get('sl_pct', 0.02))
            tp_pct = ml_prediction.get('tp_percent', ml_prediction.get('tp_pct', 0.04))
            # Convert from percentage to decimal if needed (>1 means it's percentage like 7.9)
            if sl_pct > 1:
                sl_pct = sl_pct / 100
            if tp_pct > 1:
                tp_pct = tp_pct / 100
            # Calculate risk/reward ratio
            rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 2.0
            
            # --- Entry Zone Calculation ---
            mae_stat = AVG_MAE_STATS.get(interval, 0.04) # Default 4%
            
            if is_bullish:
                zone_top = price
                zone_bottom = price * (1 - mae_stat)
                zone_str = f"{format_price(zone_bottom)} - {format_price(zone_top)}"
            else:
                zone_bottom = price
                zone_top = price * (1 + mae_stat)
                zone_str = f"{format_price(zone_bottom)} - {format_price(zone_top)}" # Price is bottom of short zone? No, short sell near top. 
                # For SHORT: Ideal entry is HIGH. So Zone is [Price, Price + MAE].
                # Wait, earlier I said "Discount Entry" for Short is getting a higher price.
                # So for Short, Zone is actually [Price, Price + MAE].
                zone_str = f"{format_price(zone_bottom)} - {format_price(zone_top)}"

            
            # Calculate recommended entry, SL, and TP
            # We favor prices already calculated by the engine (InferenceEngine)
            # but fall back to manual calculation if they are missing
            recommended_entry = ml_prediction.get('limit_price', ml_prediction.get('entry_price', price))
            sl_price = ml_prediction.get('sl_price')
            tp_price = ml_prediction.get('tp_price')
            
            # Manual calculation fallback if engine didn't provide prices
            if sl_price is None or tp_price is None:
                # Calculate recommended entry (adjust if SL is far)
                # If SL > 5%, suggest limit order below current price
                entry_adjustment = 0
                if sl_pct > 0.05:
                    # Adjust entry by (SL - 3%) to get better RR
                    entry_adjustment = (sl_pct - 0.03) * 0.5
                
                if is_bullish:
                    recommended_entry = price * (1 - entry_adjustment)
                    sl_price = recommended_entry * (1 - sl_pct)
                    tp_price = recommended_entry * (1 + tp_pct)
                else:
                    recommended_entry = price * (1 + entry_adjustment)
                    sl_price = recommended_entry * (1 + sl_pct)
                    tp_price = recommended_entry * (1 - tp_pct)
            
            # Entry adjustment percentage for display
            entry_adjustment_pct = abs(recommended_entry / price - 1)
            
            # Trailing Trigger (Breakeven)
            if is_bullish:
                trailing_trigger = recommended_entry * (1 + tp_pct * 0.5)
            else:
                trailing_trigger = recommended_entry * (1 - tp_pct * 0.5)
            
            # Confidence emoji
            if confidence >= 0.7:
                conf_emoji = "🔥"
                conf_text = "Cao"
            elif confidence >= 0.5:
                conf_emoji = "✅"
                conf_text = "Trung bình"
            else:
                conf_emoji = "⚠️"
                conf_text = "Thấp"
            
            # RR quality
            if rr_ratio >= 2.5:
                rr_emoji = "🎯"
            elif rr_ratio >= 1.5:
                rr_emoji = "👍"
            else:
                rr_emoji = "⚖️"
            
            message += f"""

━━━━━━ <b>🤖 ML Recommendation</b> ━━━━━━

{conf_emoji} Độ tin cậy: <b>{confidence:.1%}</b> ({conf_text})
{rr_emoji} Risk/Reward: <b>1:{rr_ratio:.1f}</b>

💎 <b>Vùng Mua Gom</b> (Entry Zone):
   <b>{zone_str}</b>

💵 Entry Signal: <b>{format_price(recommended_entry)}</b>"""
            
            message += f" (limit -{entry_adjustment_pct*100:.1f}%)" if entry_adjustment_pct > 0 else ""
            
            message += f"""
🛑 Stop Loss: <b>{format_price(sl_price)}</b> ({sl_pct*100:.1f}%)
🎯 Take Profit: <b>{format_price(tp_price)}</b> ({tp_pct*100:.1f}%)

📌 <b>Trailing SL:</b>
• Khi giá đạt {format_price(trailing_trigger)} (+{tp_pct*50:.1f}%)
• Dời SL lên breakeven {format_price(recommended_entry)}"""
        else:
            # No ML prediction available - show warning
            message += """

━━━━━━ <b>⚠️ ML Prediction</b> ━━━━━━

❌ <b>ML không khả dụng cho symbol này</b>
• Không đủ dữ liệu lịch sử
• Hoặc lỗi tính toán features

📌 <i>Khuyến nghị: Chờ tín hiệu có ML support</i>"""

        return message.strip()
    
    def set_ml_system(self, ml_system):
        """Set ML system for predictions."""
        self.ml_system = ml_system
    
    def set_entry_threshold(self, threshold: float):
        """Set minimum confidence threshold for alerts."""
        self.entry_threshold = threshold
    
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
