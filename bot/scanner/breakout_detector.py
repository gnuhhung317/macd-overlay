import pandas as pd
import numpy as np
from src.utils.logger import logger
from src.scanner.breakout_insights import BreakoutInsights
from src.scanner.early_detector import EarlyWarningDetector

class BreakoutDetector:
    def __init__(self, data_fetcher, config):
        self.data_fetcher = data_fetcher
        self.config = config
        
        # Lấy tham số từ config
        self.price_period = config.price_period
        self.volume_period = config.volume_period
        self.sma_len = config.sma_len
        self.kline_limit = config.kline_limit
        self.pre_breakout_threshold = config.pre_breakout_threshold
        
        # Pre-filter settings (can be configured later)
        self.enable_prefilter = getattr(config, 'enable_prefilter', True)
        self.min_adx = getattr(config, 'min_adx', 20)
        self.min_volume_percentile = getattr(config, 'min_volume_percentile', 70)
        self.min_bb_squeeze_percentile = getattr(config, 'min_bb_squeeze_percentile', 30)

        # Initialize analyzers
        self.insights = BreakoutInsights(config)
        self.early_detector = EarlyWarningDetector(config)

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index) for trend strength."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']

            # Calculate +DM and -DM
            high_diff = high.diff()
            low_diff = -low.diff()

            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

            # Calculate TR (True Range)
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Smooth with Wilder's method
            atr = tr.ewm(alpha=1/period, adjust=False).mean()
            plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
            minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

            # Calculate DX and ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(alpha=1/period, adjust=False).mean()

            return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        except Exception as e:
            logger.debug(f"Error calculating ADX: {e}")
            return 0.0

    def _volume_percentile(self, df: pd.DataFrame, window: int = 60) -> float:
        """Calculate volume percentile in recent window."""
        try:
            recent_volume = df['volume'].tail(window)
            current_vol = df['volume'].iloc[-1]

            if len(recent_volume) < 5:
                return 0.0

            percentile = (recent_volume < current_vol).sum() / len(recent_volume) * 100
            return float(percentile)
        except Exception as e:
            logger.debug(f"Error calculating volume percentile: {e}")
            return 0.0

    def _check_consolidation(self, df: pd.DataFrame, bb_period: int = 20) -> tuple:
        """Check if price is in consolidation (Bollinger Band squeeze)."""
        try:
            close = df['close']

            # Calculate Bollinger Bands
            sma = close.rolling(bb_period).mean()
            std = close.rolling(bb_period).std()
            upper = sma + (2 * std)
            lower = sma - (2 * std)

            # BB Width as % of price
            bb_width = ((upper - lower) / sma * 100).dropna()

            if len(bb_width) < 60:
                return False, 50.0

            # Current BB width
            current_width = bb_width.iloc[-1]

            # BB width percentile (lower = tighter squeeze)
            recent_widths = bb_width.tail(60)
            width_percentile = (recent_widths < current_width).sum() / len(recent_widths) * 100

            # Squeeze = width in bottom 30% of recent range
            is_squeeze = width_percentile <= self.min_bb_squeeze_percentile

            return is_squeeze, float(width_percentile)
        except Exception as e:
            logger.debug(f"Error checking consolidation: {e}")
            return False, 50.0

    def _pre_filter_quality(self, df: pd.DataFrame) -> tuple:
        """Pre-filter to check if setup quality is good enough."""
        if not self.enable_prefilter:
            return True, "prefilter_disabled", {}

        metrics = {}

        try:
            # 1. ADX check (trend strength)
            adx = self._calculate_adx(df)
            metrics['adx'] = adx

            if adx < self.min_adx:
                return False, f"weak_trend_adx_{adx:.1f}", metrics

            # 2. Volume check
            vol_percentile = self._volume_percentile(df)
            metrics['volume_percentile'] = vol_percentile

            if vol_percentile < self.min_volume_percentile:
                return False, f"weak_volume_p{vol_percentile:.0f}", metrics

            # 3. Consolidation check (optional - not strict)
            is_squeeze, width_pct = self._check_consolidation(df)
            metrics['bb_squeeze'] = is_squeeze
            metrics['bb_width_percentile'] = width_pct

            return True, "passed", metrics

        except Exception as e:
            logger.debug(f"Error in pre-filter: {e}")
            return True, "prefilter_error", metrics  # Allow through on error

    def detect_breakout(self, symbol, timeframe, btc_df=None, df=None):
        """Phát hiện breakout"""
        try:
            # Use pre-fetched data if available (for backtesting)
            if df is not None:
                pass
            else:
                ohlcv = self.data_fetcher.fetch_ohlcv(symbol, timeframe, limit=self.kline_limit)

                if not ohlcv or len(ohlcv) < max(self.price_period, self.volume_period, self.sma_len) + 2:
                    logger.debug(f"Not enough data for {symbol} on {timeframe}")
                    return None

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Tính các chỉ báo (SHIFT 1 nến)
            df['price_highest_prev'] = df['high'].rolling(window=self.price_period).max().shift(1)
            df['price_lowest_prev'] = df['low'].rolling(window=self.price_period).min().shift(1)
            df['volume_highest_prev'] = df['volume'].rolling(window=self.volume_period).max().shift(1)
            df['sma'] = df['close'].rolling(window=self.sma_len).mean().shift(1)

            last_row = df.iloc[-1]

            if pd.isna(last_row['price_highest_prev']) or pd.isna(last_row['price_lowest_prev']) or \
               pd.isna(last_row['volume_highest_prev']) or pd.isna(last_row['sma']):
                return None

            filter_passed, filter_reason, filter_metrics = self._pre_filter_quality(df)

            if not filter_passed:
                return None

            close_now = last_row['close']
            volume_now = last_row['volume']
            ph_prev = last_row['price_highest_prev']
            pl_prev = last_row['price_lowest_prev']
            vh_prev = last_row['volume_highest_prev']
            sma_now = last_row['sma']

            try:
                quote_volume_now = float(close_now) * float(volume_now)
            except Exception:
                quote_volume_now = 0.0
            min_breakout_quote = getattr(self.config, 'breakout_min_volume_usdt', 0)

            has_volume_confirm = volume_now > vh_prev and (min_breakout_quote <= 0 or quote_volume_now >= min_breakout_quote)

            vol_ma20 = df['volume'].rolling(20).mean().iloc[-1]
            vol_surge_ratio = volume_now / vol_ma20 if vol_ma20 > 0 else 0

            vol_60_candles = df['volume'].tail(60)
            vol_percentile = (vol_60_candles < volume_now).sum() / len(vol_60_candles) * 100 if len(vol_60_candles) > 0 else 0

            has_moderate_volume = vol_surge_ratio >= 1.2 and vol_percentile >= 60

            distance_to_high = ((ph_prev - close_now) / close_now) if close_now > 0 else 999
            distance_to_low = ((close_now - pl_prev) / close_now) if close_now > 0 else 999

            if close_now > sma_now and has_volume_confirm:
                insights = self.insights.analyze_breakout(df, {'signal': 'LONG', 'price_ref': ph_prev}, btc_df=btc_df, multi_tf_score=0)
                return {
                    'signal': 'LONG', 'symbol': symbol, 'timeframe': timeframe, 'close': close_now,
                    'price_ref': ph_prev, 'volume': volume_now, 'volume_ref': vh_prev, 'sma': sma_now,
                    'timestamp': int(last_row['timestamp']), 'quote_volume': quote_volume_now, 'insights': insights
                }

            elif (distance_to_high <= self.pre_breakout_threshold and close_now <= ph_prev and close_now > sma_now and not has_volume_confirm and has_moderate_volume):
                distance_pct = distance_to_high * 100
                return {
                    'signal': 'PRE_LONG', 'symbol': symbol, 'timeframe': timeframe, 'close': close_now,
                    'price_ref': ph_prev, 'volume': volume_now, 'volume_ref': vh_prev, 'sma': sma_now,
                    'timestamp': int(last_row['timestamp']), 'quote_volume': quote_volume_now,
                    'distance_pct': distance_pct, 'vol_surge_ratio': vol_surge_ratio, 'vol_percentile': vol_percentile
                }

            elif close_now < sma_now and has_volume_confirm:
                insights = self.insights.analyze_breakout(df, {'signal': 'SHORT', 'price_ref': pl_prev}, btc_df=btc_df, multi_tf_score=0)
                return {
                    'signal': 'SHORT', 'symbol': symbol, 'timeframe': timeframe, 'close': close_now,
                    'price_ref': pl_prev, 'volume': volume_now, 'volume_ref': vh_prev, 'sma': sma_now,
                    'timestamp': int(last_row['timestamp']), 'quote_volume': quote_volume_now, 'insights': insights
                }

            elif (distance_to_low <= self.pre_breakout_threshold and close_now >= pl_prev and close_now < sma_now and not has_volume_confirm and has_moderate_volume):
                distance_pct = distance_to_low * 100
                return {
                    'signal': 'PRE_SHORT', 'symbol': symbol, 'timeframe': timeframe, 'close': close_now,
                    'price_ref': pl_prev, 'volume': volume_now, 'volume_ref': vh_prev, 'sma': sma_now,
                    'timestamp': int(last_row['timestamp']), 'quote_volume': quote_volume_now,
                    'distance_pct': distance_pct, 'vol_surge_ratio': vol_surge_ratio, 'vol_percentile': vol_percentile
                }
            else:
                enable_early = getattr(self.config, 'enable_early_warning', True)
                if enable_early:
                    early_warning = self.early_detector.detect_early_signals(df, symbol, timeframe, current_signal=None)
                    if early_warning:
                        return early_warning

            return None

        except Exception as e:
            logger.error(f"Error detecting breakout for {symbol} on {timeframe}: {e}")
            return None

    def analyze_symbols(self, symbols, timeframe, btc_df=None):
        """Quét tất cả symbols trên 1 timeframe"""
        breakout_signals = []
        for symbol in symbols:
            signal = self.detect_breakout(symbol, timeframe, btc_df=btc_df)
            if signal:
                breakout_signals.append(signal)
        return breakout_signals

    def run(self, symbols, timeframes, btc_data_by_tf=None):
        """Quét tất cả symbols trên tất cả timeframes"""
        all_signals = {}
        for timeframe in timeframes:
            logger.info(f"Scanning timeframe: {timeframe}")
            btc_df = btc_data_by_tf.get(timeframe) if btc_data_by_tf else None
            all_signals[timeframe] = self.analyze_symbols(symbols, timeframe, btc_df=btc_df)
            logger.info(f"Found {len(all_signals[timeframe])} breakouts on {timeframe}")
