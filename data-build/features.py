"""
Feature Engineering Module for Trading Data
Implements 116+ advanced technical features for market analysis
Based on features.json.data specification
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import warnings
import traceback

warnings.filterwarnings('ignore')

class FeatureBuilder:
    """Build technical and fundamental features for trading signals"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.features_dict = {}
    
    def build_all(self) -> pd.DataFrame:
        """Build features with Dynamic Error Tracing"""
        
        # Danh sách toàn bộ các hàm bạn muốn chạy
        methods_to_run = [
            'build_volatility_features', 'build_trend_features', 
            'build_mean_reversion_features', 'build_volume_features',
            'build_momentum_features', 'build_market_structure_features',
            'build_funding_features', 'build_candle_pattern_features',
            'build_risk_features', 'build_liquidity_features',
            'build_advanced_composite_features', 'build_ohlc_candle_features',
            'build_returns_volatility_features', 'build_advanced_momentum_features',
            'build_session_temporal_features', 'build_advanced_ratio_features',
            'build_open_interest_features', 'build_long_short_features',
            'build_additional_bb_features', 'build_advanced_volatility_features',
            'build_volatility_weighted_features', 'build_cumulative_features',
            'build_additional_funding_features', 'build_squeeze_and_depletion_features',
            'build_price_extension_features', 'build_ranking_features',
            'build_leverage_tension_features', 'build_relative_performance_features',
            'build_trend_state_features', 'build_speculative_features',
            'build_multi_period_rsi', 'build_multi_period_atr',
            'build_multi_period_ema', 'build_multi_period_sma',
            'build_multi_period_bollinger', 'build_multi_period_roc',
            'build_multi_period_cci', 'build_multi_period_volume',
            'build_multi_period_volatility', 'build_multi_parameter_oscillators'
        ]
        
        print("Bắt đầu quét và xây dựng Features...")
        
        for method_name in methods_to_run:
            # Kiểm tra xem hàm có thực sự tồn tại trong class chưa (phòng hờ bạn comment code)
            if hasattr(self, method_name):
                func = getattr(self, method_name)
                try:
                    # Thực thi hàm
                    func()
                except Exception as e:
                    # Bắt trúng tim đen lỗi ndarray
                    if "'numpy.ndarray' object has no attribute 'rolling'" in str(e):
                        print("\n" + "!"*60)
                        print(f" BẮT ĐƯỢC TỘI PHẠM TẠI HÀM: {method_name}")
                        print("!"*60)
                        print("Nguyên nhân: Có một phép tính dùng np.where() hoặc chia 2 cột")
                        print("bị biến thành numpy array, sau đó lại bị gắn .rolling() vào đằng sau.")
                        print(f"Chi tiết lỗi: {str(e)}")
                        print("!"*60)
                        
                        # In traceback đầy đủ để bạn biết dòng số mấy
                        traceback.print_exc()
                        
                        # Dừng chương trình ngay lập tức
                        raise SystemExit(f"Hệ thống dừng khẩn cấp để sửa hàm {method_name}.")
                    else:
                        # Nếu là lỗi khác, vẫn báo tên hàm để dễ debug
                        print(f"\n[LỖI CHƯA XÁC ĐỊNH] tại hàm {method_name}: {str(e)}")
                        traceback.print_exc()
                        raise SystemExit("Dừng chương trình.")
        
        # Add all features to dataframe sau khi đã pass qua hết các hàm
        print(f"Đã chạy thành công {len(methods_to_run)} hàm. Đang gom dữ liệu...")
        for feature_name, feature_values in self.features_dict.items():
            self.df[feature_name] = feature_values
            
        return self.df
    
    # ==================== VOLATILITY FEATURES ====================
    def build_volatility_features(self):
        """Build volatility-related features"""
        df = self.df
        
        # ATR (Average True Range)
        df['tr_14'] = self._true_range()
        df['atr_14'] = df['tr_14'].rolling(14).mean()
        df['atr_7'] = self._true_range().rolling(7).mean()
        df['atr_21'] = df['tr_14'].rolling(21).mean()
        df['atr_pct_14'] = df['atr_14'] / df['close']
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(20).mean()
        df['std_20'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        self.features_dict['bb_width_zscore_20'] = (
            (df['bb_width'] - df['bb_width'].rolling(20).mean()) / 
            (df['bb_width'].rolling(20).std() + 1e-9)
        )
        
        # Historical Volatility
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        self.features_dict['volatility_20'] = df['log_returns'].rolling(20).std()
        self.features_dict['volatility_50'] = df['log_returns'].rolling(50).std()
        
        # Parkinson Volatility
        self.features_dict['parkinson_vol'] = (
            np.log(df['high'] / df['low']).rolling(20).mean() / (2 * np.sqrt(np.log(4)))
        )
        
        # Garman-Klass Volatility
        hl_ratio = np.log(df['high'] / df['low'])
        co_ratio = np.log(df['close'] / df['open'])
        self.features_dict['garman_klass_vol'] = np.sqrt(
            0.5 * hl_ratio**2 - (2*np.log(2) - 1) * co_ratio**2
        ).rolling(20).mean()
        
        # Yang-Zhang Volatility (sophisticated)
        log_ho = np.log(df['high'] / df['open'])
        log_lo = np.log(df['low'] / df['open'])
        log_co = np.log(df['close'] / df['open'])
        log_oc_prev = np.log(df['open'] / df['close'].shift(1))
        
        s_o = log_oc_prev.rolling(20).var()
        s_c = log_co.rolling(20).var()
        s_rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(20).mean()
        k = 0.34 / (1.34 + 21 / 19)
        vol = np.sqrt(s_o + k * s_c + (1 - k) * s_rs)
        self.features_dict['yang_zhang_vol_zscore'] = (
            (vol - vol.rolling(20).mean()) / (vol.rolling(20).std() + 1e-9)
        )
        
        # Vol Ratio Alpha
        self.features_dict['vol_ratio_alpha'] = (
            df['returns'].rolling(5).std() / (df['returns'].rolling(20).std() + 1e-9)
        )
        self.features_dict['volume_zscore'] = (
            (df['volume'] - df['volume'].rolling(20).mean()) / 
            (df['volume'].rolling(20).std() + 1e-9)
        )
    
    # ==================== TREND FEATURES ====================
    def build_trend_features(self):
        """Build trend-related features"""
        df = self.df
        
        # EMA
        df['ema_7'] = df['close'].ewm(span=7).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        # Price to EMA
        self.features_dict['dist_to_ema_50'] = df['close'] - df['ema_50']
        self.features_dict['dist_to_ema_50_pct'] = (
            (df['close'] - df['ema_50']) / df['ema_50']
        )
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Hurst Exponent (Trend Persistence)
        log_ret = np.log(df['close'] / df['close'].shift(1))
        rs = (
            (log_ret.rolling(20).max() - log_ret.rolling(20).min()) / 
            (log_ret.rolling(20).std() + 1e-9)
        )
        self.features_dict['hurst_deviation'] = (
            (np.log(rs + 1e-9) / np.log(20)) - 0.5
        )
        
        # ADX (Average Directional Index)
        self.features_dict['adx_14'] = self._calculate_adx(14)
    
    # ==================== MEAN REVERSION FEATURES ====================
    def build_mean_reversion_features(self):
        """Build mean reversion and tension features"""
        df = self.df
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = self._calculate_rsi(14)
        df['rsi_7'] = self._calculate_rsi(7)
        
        # Stochastic RSI
        rsi = self._calculate_rsi(14)
        self.features_dict['stoch_rsi_14'] = (
            (rsi - rsi.rolling(14).min()) / 
            (rsi.rolling(14).max() - rsi.rolling(14).min() + 1e-9)
        )
        
        # Distance to EMA with shadow pressure
        df['upper_shadow_pct'] = (df['high'] - df.loc[:, ['open', 'close']].max(axis=1)) / df['close']
        df['lower_shadow_pct'] = (df.loc[:, ['open', 'close']].min(axis=1) - df['low']) / df['close']
        df['body_size_pct'] = np.abs(df['close'] - df['open']) / df['close']
        df['body_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        
        # Mean Reversion Tension Score
        self.features_dict['mean_reversion_tension_score'] = (
            self.features_dict['dist_to_ema_50_pct'] * 
            (df['volume'] / df['volume'].rolling(20).mean()) * 
            (df['upper_shadow_pct'] - df['lower_shadow_pct'])
        )
        
        # Price position in channel
        self.features_dict['position_in_bb'] = (
            (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
        )
    
    # ==================== VOLUME FEATURES ====================
    def build_volume_features(self, mfi_window=14, vfi_window=20, fve_window=20):
        """Build volume-related features using pure Pandas to avoid ndarray errors"""
        df = self.df
        
        # 1. Basic Volume Metrics & Z-Score (Mở rộng)
        df['volume_sma'] = df['volume'].rolling(mfi_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)
        
        # Feature mới: Độ đột biến của khối lượng
        vol_std = df['volume'].rolling(mfi_window).std().replace(0, np.nan)
        self.features_dict['volume_zscore'] = (df['volume'] - df['volume_sma']) / vol_std
        
        # 2. Price-Volume Trend (PVT)
        self.features_dict['pvt'] = (df['close'].pct_change() * df['volume']).cumsum()
        
        # 3. On Balance Volume (OBV) - Fixed
        # np.sign bọc qua df['close'].diff() tự động trả về Pandas Series
        direction = np.sign(df['close'].diff()).fillna(0)
        df['obv'] = (direction * df['volume']).cumsum()
        self.features_dict['obv'] = df['obv']
        
        # Feature mới: Tỉ lệ OBV so với SMA của nó (đo lường đà gom hàng)
        self.features_dict['obv_sma_ratio'] = df['obv'] / df['obv'].rolling(20).mean().replace(0, np.nan)
        
        # 4. Money Flow Index (MFI) - Fixed
        money_flow = df['close'] * df['volume']
        close_diff = df['close'].diff()
        
        # Dùng .where của Pandas để giữ nguyên pd.Series
        positive_flow = money_flow.where(close_diff > 0, 0.0)
        negative_flow = money_flow.where(close_diff < 0, 0.0)
        
        pos_flow_sum = positive_flow.rolling(mfi_window).sum()
        neg_flow_sum = negative_flow.rolling(mfi_window).sum().replace(0, np.nan)
        
        money_ratio = pos_flow_sum / neg_flow_sum
        self.features_dict[f'mfi_{mfi_window}'] = (100 - (100 / (1 + money_ratio))).fillna(100)
        
        # 5. Volume Flow Indicator (VFI) - Fixed
        typ = (df['high'] + df['low'] + df['close']) / 3.0
        typ_diff = typ.diff()
        
        # Tương tự OBV, nhân trực tiếp vector với sign
        mf = df['volume'] * np.sign(typ_diff).fillna(0)
        vfi_denom = df['volume'].rolling(vfi_window).mean().replace(0, np.nan)
        self.features_dict[f'vfi_{vfi_window}'] = mf.rolling(vfi_window).sum() / vfi_denom
        
        # 6. Finite Volume Elements (FVE) - Fixed
        hl_avg = (df['high'] + df['low']) / 2.0
        # Dùng .where: Nếu typ > hl_avg thì giữ df['volume'], ngược lại thành -df['volume']
        fve_mf = df['volume'].where(typ > hl_avg, -df['volume'])
        
        fve_denom = df['volume'].rolling(fve_window).sum().replace(0, np.nan)
        self.features_dict[f'fve_{fve_window}'] = fve_mf.rolling(fve_window).sum() / fve_denom
        
        # 7. Cumulative Volume
        self.features_dict['cumulative_volume'] = df['volume'].cumsum()
        
        # 8. VWAP & VWAP Distance (Feature Định lượng cực mạnh)
        cum_pv = (typ * df['volume']).cumsum()
        cum_v = df['volume'].cumsum().replace(0, np.nan)
        df['vwap'] = cum_pv / cum_v
        
        # Khoảng cách từ giá hiện tại đến VWAP: Giúp ML nhận diện "Quá xa bờ" -> Dễ đảo chiều
        self.features_dict['vwap_distance_pct'] = (df['close'] - df['vwap']) / df['vwap']
    
    # ==================== MOMENTUM FEATURES ====================
    def build_momentum_features(self):
        """Build momentum-related features"""
        df = self.df
        
        # Rate of Change
        self.features_dict['roc_12'] = (
            (df['close'] - df['close'].shift(12)) / df['close'].shift(12)
        )
        self.features_dict['roc_24'] = (
            (df['close'] - df['close'].shift(24)) / df['close'].shift(24)
        )
        
        # Momentum
        self.features_dict['momentum_10'] = df['close'] - df['close'].shift(10)
        self.features_dict['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # Efficiency Thrust Index
        self.features_dict['efficiency_thrust_index'] = (
            (df['returns'].rolling(10).sum() / (df['atr_14'].rolling(10).sum() + 1e-9)) * 
            df['volume_ratio']
        )
        
        # CCI (Commodity Channel Index)
        typ = (df['high'] + df['low'] + df['close']) / 3
        sma_typ = typ.rolling(20).mean()
        mad = (typ - sma_typ).rolling(20).apply(lambda x: np.abs(x).mean())
        self.features_dict['cci_20'] = (typ - sma_typ) / (0.015 * mad + 1e-9)
    
    # ==================== MARKET STRUCTURE FEATURES ====================
    def build_market_structure_features(self):
        """Build market structure and fractal features"""
        df = self.df
        
        # Fractal Dimension Index
        diffs = (df['close'] - df['close'].shift(1)).abs()
        path_l = diffs.rolling(20).sum()
        range_l = df['close'].rolling(20).max() - df['close'].rolling(20).min()
        self.features_dict['fdi_20'] = (
            1.0 + (np.log(path_l / (range_l + 1e-9) + 1e-9)) / np.log(20)
        )
        
        # Skewness
        self.features_dict['skewness_20d'] = df['close'].pct_change().rolling(20).skew()
        
        # Kurtosis
        self.features_dict['kurtosis_20d'] = df['close'].pct_change().rolling(20).kurt()
        
        # Tail Regime Stress (combining multiple indicators)
        self.features_dict['tail_regime_stress_score'] = (
            self.features_dict['skewness_20d'] * 
            self.features_dict['yang_zhang_vol_zscore'] * 
            (self.features_dict['hurst_deviation'] + 0.5)
        )
        
        # Structural VFI Efficiency
        self.features_dict['structural_vfi_efficiency'] = (
            self.features_dict['vfi_20'] * (2.0 - self.features_dict['fdi_20'])
        )
    
    # ==================== FUNDING FEATURES ====================
    def build_funding_features(self):
        """Build funding rate related features"""
        df = self.df
        
        # Ensure fundingRate exists
        if 'fundingRate' not in df.columns:
            df['fundingRate'] = 0
        
        funding = df['fundingRate']
        
        # Basic Funding Metrics
        self.features_dict['funding_zscore'] = (
            (funding - funding.rolling(48).mean()) / 
            (funding.rolling(48).std() + 1e-9)
        )
        self.features_dict['funding_ma_24'] = funding.rolling(24).mean()
        
        # Shadow Funding Asymmetry Index
        body_size = np.abs(df['close'] - df['open']) / df['close']
        shadow_diff = df['upper_shadow_pct'] - df['lower_shadow_pct']
        self.features_dict['shadow_funding_asymmetry_index'] = (
            (shadow_diff / (body_size + 0.001)) * np.sign(funding) * 
            (np.abs(funding)**1.2) * (df['volume'] / df['volume'].rolling(20).mean())
        )
        self.features_dict['shadow_funding_asymmetry_index'] = (
            (self.features_dict['shadow_funding_asymmetry_index'] - 
             self.features_dict['shadow_funding_asymmetry_index'].rolling(120).mean()) / 
            (self.features_dict['shadow_funding_asymmetry_index'].rolling(120).std() + 1e-9)
        )
        
        # Funding Velocity Volatility Spread
        funding_vel = funding.diff() / (funding.rolling(24).std() + 1e-9)
        self.features_dict['funding_velocity_volatility_spread_zscore'] = (
            (funding_vel * df['atr_pct_14'] * (df['volume'] / df['volume'].rolling(24).mean())) -
            (funding_vel * df['atr_pct_14'] * (df['volume'] / df['volume'].rolling(24).mean())).rolling(168).mean()
        ) / ((funding_vel * df['atr_pct_14'] * (df['volume'] / df['volume'].rolling(24).mean())).rolling(168).std() + 1e-9)
        
        # Trapped Liquidity Funding Oscillator
        metric = (df['close'] - df['open']) * df['volume'] * np.sign(funding.shift(1)) * (np.abs(funding.shift(1))**1.5)
        self.features_dict['trapped_liquidity_funding_oscillator'] = (
            (metric - metric.rolling(144).mean()) / (metric.rolling(144).std() + 1e-9)
        )
        
        # Funding Convexity Volume Multiplier
        funding_centered = funding - funding.rolling(48).mean()
        metric = (funding_centered**3) * np.log(df['volume'] / df['volume'].rolling(48).mean() + 1e-9) / (df['close'].rolling(48).std() + 1e-9)
        self.features_dict['funding_convexity_volume_multiplier_z_score'] = (
            (metric - metric.rolling(48).mean()) / (metric.rolling(48).std() + 1e-9)
        )
        
        # Synthetic Liquidation Delta
        metric = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)) * np.log(df['volume'] + 1) * np.sign(funding) * np.exp(np.abs(funding))
        self.features_dict['synthetic_liquidation_delta_z_score'] = (
            (metric - metric.rolling(48).mean()) / (metric.rolling(48).std() + 1e-9)
        )
    
    # ==================== CANDLE PATTERN FEATURES ====================
    def build_candle_pattern_features(self):
        """Build candle pattern features"""
        df = self.df
        
        # Candle metrics
        df['hl_range'] = df['high'] - df['low']
        df['oc_range'] = np.abs(df['close'] - df['open'])
        df['hl_range_pct'] = df['hl_range'] / df['close']
        
        # Doji pattern
        self.features_dict['doji_score'] = (
            df['oc_range'] / (df['hl_range'] + 1e-9)
        )
        
        # Candle strength
        self.features_dict['candle_strength'] = (
            df['oc_range'] / (df['hl_range'] + 1e-9)
        )
        
        # Wick ratios
        self.features_dict['upper_wick_ratio'] = df['upper_shadow_pct'] / (df['hl_range_pct'] + 1e-9)
        self.features_dict['lower_wick_ratio'] = df['lower_shadow_pct'] / (df['hl_range_pct'] + 1e-9)
    
    # ==================== RISK FEATURES ====================
    def build_risk_features(self):
        """Build risk-related features"""
        df = self.df
        
        # Drawdown
        cummax = df['close'].cummax()
        self.features_dict['drawdown'] = (df['close'] - cummax) / cummax
        self.features_dict['drawdown_pct'] = self.features_dict['drawdown'] * 100
        
        # Value at Risk (VaR) - 5th percentile
        self.features_dict['var_95'] = df['returns'].rolling(20).quantile(0.05)
        
        # Conditional VaR
        self.features_dict['cvar_95'] = df['returns'].rolling(20).apply(
            lambda x: x[x <= x.quantile(0.05)].mean()
        )
        
        # Volatility of Volatility
        self.features_dict['vol_of_vol'] = (
            self.features_dict['volatility_20'].rolling(20).std()
        )
    
    # ==================== LIQUIDITY FEATURES ====================
    def build_liquidity_features(self):
        """Build liquidity-related features"""
        df = self.df
        
        # Amihud Illiquidity
        illiquidity = np.abs(df['returns']) / (df['volume'] * df['close'] + 1e-9)
        self.features_dict['amihud_zscore'] = (
            (illiquidity - illiquidity.rolling(20).mean()) / 
            (illiquidity.rolling(20).std() + 1e-9)
        )
        
        # Corwin-Schultz Spread
        beta = (np.log(df['high'] / df['low'])**2).rolling(2).sum()
        gamma = (np.log(df['high'].rolling(2).max() / df['low'].rolling(2).min()))**2
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        self.features_dict['corwin_schultz_pct'] = (
            2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        )
        
        # Bid-Ask Spread Estimate
        self.features_dict['ba_spread'] = df['high'] - df['low']
        self.features_dict['ba_spread_pct'] = self.features_dict['ba_spread'] / df['close']
    
    # ==================== ADVANCED COMPOSITE FEATURES ====================
    def build_advanced_composite_features(self):
        """Build advanced composite features"""
        df = self.df
        
        # Volatility Expansion Intensity
        vol_ratio_alpha = self.features_dict.get('vol_ratio_alpha')
        if vol_ratio_alpha is None:
            vol_ratio_alpha = df['atr_7'] / (df['atr_21'] + 1e-9)
            self.features_dict['vol_ratio_alpha'] = vol_ratio_alpha
        
        self.features_dict['volatility_expansion_intensity'] = (
            ((vol_ratio_alpha - vol_ratio_alpha.rolling(20).mean()) / 
             (vol_ratio_alpha.rolling(20).std() + 1e-9)) * df['volume_zscore']
        )
        
        # Volatility Momentum Tension Flux
        self.features_dict['volatility_momentum_tension_flux'] = (
            ((df['close'] - df['ema_21']) / (df['atr_14'] + 1e-9)) * 
            (df['atr_7'] / (df['atr_21'] + 1e-9)) * 
            (df['volume'] / (df['volume_sma_14'] + 1e-9))
        )
        
        # Volume Weighted Fractal Efficiency
        self.features_dict['volume_weighted_fractal_efficiency'] = (
            ((df['close'] - df['close'].shift(10)).abs() / 
             (df['high'].rolling(10).max() - df['low'].rolling(10).min() + 1e-9)) * 
            (df['volume'] / (df['volume'].rolling(20).mean() + 1e-9))
        )
        
        # Cross Modal Funding Squeeze Momentum (if funding available)
        if 'fundingRate' in df.columns and df['fundingRate'].notna().sum() > 0:
            clv = (
                df['volume'] * np.sign(df['fundingRate']) * 
                ((df['close'] - df['low']) - (df['high'] - df['close'])) / 
                (df['high'] - df['low'] + 1e-8)
            )
            clv_vol = clv.rolling(24).mean()
            metric = clv_vol * np.exp(np.abs(df['fundingRate']) / (df['fundingRate'].rolling(168).std() + 1e-8))
            self.features_dict['cross_modal_funding_squeeze_momentum'] = (
                (metric - metric.rolling(168).mean()) / (metric.rolling(168).std() + 1e-9)
            )
    
    # ==================== ADDITIONAL COMPREHENSIVE FEATURES ====================
    def build_ohlc_candle_features(self):
        """Build additional OHLC and candle pattern features"""
        df = self.df
        
        # Candle features
        df['intraday_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
        df['true_range_pct'] = (
            pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1) / df['close'].shift(1)
        )
        self.features_dict['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        self.features_dict['lower_shadow_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        self.features_dict['upper_shadow_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        self.features_dict['body_size_pct'] = (df['close'] - df['open']).abs() / df['close']
        self.features_dict['intraday_momentum'] = df['intraday_momentum']
        self.features_dict['true_range_pct'] = df['true_range_pct']
    
    def build_returns_volatility_features(self):
        """Build returns and volatility calculation features"""
        df = self.df
        
        # Returns and volatility
        self.features_dict['returns_std_7'] = df['close'].pct_change().rolling(7).std()
        self.features_dict['returns_std_14'] = df['close'].pct_change().rolling(14).std()
        self.features_dict['returns_std_21'] = df['close'].pct_change().rolling(21).std()
        self.features_dict['returns_zscore_20d'] = (
            (df['returns'] - df['returns'].rolling(20).mean()) / 
            (df['returns'].rolling(20).std() + 1e-9)
        )
    
    def build_advanced_momentum_features(self):
        """Build advanced momentum features"""
        df = self.df
        
        # TRIX (Triple EMA)
        ema1 = df['close'].ewm(span=15, adjust=False).mean()
        ema2 = ema1.ewm(span=15, adjust=False).mean()
        ema3 = ema2.ewm(span=15, adjust=False).mean()
        self.features_dict['trix_pct'] = ema3.pct_change() * 100
        
        # Williams %R
        hh = df['high'].rolling(14).max()
        ll = df['low'].rolling(14).min()
        self.features_dict['williams_r_normalized'] = ((hh - df['close']) / (hh - ll + 1e-9)) * -1
        
        # CCI Normalized
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad + 1e-9)
        self.features_dict['cci_normalized'] = cci / 100
        
        # ROC
        self.features_dict['roc_14'] = df['close'].pct_change(periods=14)
        
        # PPO (Percentage Price Oscillator)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        self.features_dict['ppo'] = ((ema12 - ema26) / (ema26 + 1e-9)) * 100
        
        # MACD variants
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        self.features_dict['macd_histogram_pct'] = (macd - signal) / df['close']
        self.features_dict['macd_normalized'] = macd / df['close']
        
        # RSI Normalized
        self.features_dict['rsi_normalized'] = ((df['rsi_14'] - 50) / 50)
    
    def build_session_temporal_features(self):
        """Build session and temporal features"""
        df = self.df
        
        # Session indicators
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            day_of_week = df.index.dayofweek
        else:
            if 'timestamp' in df.columns:
                hour = pd.to_datetime(df['timestamp']).dt.hour
                day_of_week = pd.to_datetime(df['timestamp']).dt.dayofweek
            else:
                # Default to zeros if no time info
                hour = pd.Series(0, index=df.index)
                day_of_week = pd.Series(0, index=df.index)
        
        self.features_dict['is_weekend'] = (day_of_week >= 5).astype(int)
        self.features_dict['is_american_session'] = ((hour >= 16) & (hour < 24)).astype(int)
        self.features_dict['is_european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
        self.features_dict['is_asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
        
        # Circular encoding of day and hour
        self.features_dict['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        self.features_dict['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        self.features_dict['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        self.features_dict['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    
    def build_advanced_ratio_features(self):
        """Build advanced ratio and interaction features"""
        df = self.df
        
        # Risk-adjusted metrics
        self.features_dict['sharpe_7d'] = (
            df['close'].pct_change(7) / (df['returns'].rolling(7).std() + 1e-9)
        )
        self.features_dict['sharpe_14d'] = (
            df['close'].pct_change(14) / (df['returns'].rolling(14).std() + 1e-9)
        )
        self.features_dict['momentum_volatility_ratio'] = (
            df['close'].pct_change(7) / (df['returns'].rolling(14).std() + 1e-9)
        )
        self.features_dict['risk_adjusted_momentum'] = (
            df['close'].pct_change(14) / (df['atr_pct_14'] + 1e-9)
        )
        
        # Trend efficiency
        self.features_dict['normalized_trend_efficiency_index'] = (
            ((df['close'] - df['close'].shift(14)) / 
             (df['close'].diff().abs().rolling(14).sum() + 1e-9)) * 
            (df['adx_14'] / 25.0)
        )
        
        # Volume absorption
        self.features_dict['volume_absorption_index'] = (
            (df['volume'] / (df['volume_sma_14'] + 1e-9)) * 
            (1 - df['intraday_momentum'].abs())
        )
        
        # Relative absorption
        self.features_dict['relative_absorption_ratio'] = (
            (df['volume'] / (df['volume_sma_14'] + 1e-9)) / 
            ((df['high'] - df['low'] + 1e-9) / (df['atr_14'] + 1e-9))
        )
        
        # Volume thrust
        self.features_dict['volume_thrust_efficiency'] = (
            (df['log_returns'] * df['volume_zscore']) / 
            (df['volatility_20'] + 1e-9)
        )
        
        # Volume correlation
        if len(df) > 20:
            self.features_dict['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'].pct_change())
        else:
            self.features_dict['volume_price_correlation'] = 0
    
    def build_open_interest_features(self, window_short=1, window_long=24, atr_window=14):
        """
        Build Open Interest features dynamically.
        FIXED: Using native Pandas .replace() to avoid ndarray type collision.
        """
        df = self.df
        
        if 'sum_open_interest' not in df.columns or df['sum_open_interest'].isna().all():
            return
            
        oi = df['sum_open_interest']
        
        # 1. FLOWS
        oi_change_short = oi.pct_change(window_short)
        oi_change_long = oi.pct_change(window_long)
        
        self.features_dict[f'oi_change_{window_short}_pct'] = oi_change_short
        self.features_dict[f'oi_change_{window_long}_pct'] = oi_change_long
        self.features_dict['oi_velocity'] = oi_change_short.diff(window_short)
        self.features_dict['oi_acceleration'] = self.features_dict['oi_velocity'].diff(window_short)
        
        # 2. PRICE CONTEXT
        if 'close' in df.columns:
            price_change = df['close'].pct_change(window_short)
            self.features_dict['oi_price_regime'] = np.sign(oi_change_short) * np.sign(price_change) * np.abs(oi_change_short)
        
        # 3. ANOMALIES (Đã fix lỗi ndarray)
        # Thay vì np.where, dùng pd.Series.replace() để giữ nguyên type là Series
        vol_safe = df['volume'].replace(0, np.nan) 
        oi_vol_ratio = oi / vol_safe
        self.features_dict['oi_to_volume_ratio'] = oi_vol_ratio
        
        # Lúc này oi_vol_ratio chắc chắn là Series, gọi rolling() thoải mái
        rolling_ratio = oi_vol_ratio.rolling(window_long)
        ratio_mean = rolling_ratio.mean()
        ratio_std = rolling_ratio.std().replace(0, np.nan)
        self.features_dict['oi_to_volume_ratio_zscore'] = (oi_vol_ratio - ratio_mean) / ratio_std
        
        # 4. CONVICTION
        atr_col = f'atr_pct_{atr_window}'
        if 'volume_ratio' in df.columns and atr_col in df.columns:
            # Fix tương tự cho ATR
            atr_safe = df[atr_col].replace(0, np.nan)
            self.features_dict['oi_volume_conviction_ratio'] = (oi_change_short * df['volume_ratio']) / atr_safe
            
        # 5. EXHAUSTION
        if 'fundingRate' in df.columns and df['fundingRate'].notna().sum() > 0:
            funding = df['fundingRate']
            rolling_fund = funding.rolling(window_long)
            fund_zscore = (funding - rolling_fund.mean()) / rolling_fund.std().replace(0, np.nan)
            
            self.features_dict['oi_funding_interaction'] = oi_change_long * fund_zscore
            
    def build_long_short_features(self, funding_windows=[12, 24, 48, 96], bb_windows=[10, 20, 40, 80]):
        """
        Build Long/Short ratio features dynamically across multiple time windows.
        Captures micro-structure shifts from short-term to long-term trends.
        """
        df = self.df
        
        if 'top_ls_ratio' not in df.columns or 'global_ls_ratio' not in df.columns:
            return
            
        top_ls = df['top_ls_ratio']
        global_ls = df['global_ls_ratio']
        
        # 1. BASE IMBALANCE (Độc lập với Time-window)
        ls_imbalance = top_ls - global_ls
        self.features_dict['ls_imbalance'] = ls_imbalance
        self.features_dict['ls_imbalance_velocity'] = ls_imbalance.diff()
        
        # 2. ĐA CHU KỲ CHO CẤU TRÚC GIÁ (Bollinger Bands & Z-Scores)
        for bb_w in bb_windows:
            # Z-Score Normalization theo từng khung thời gian
            top_ls_roll = top_ls.rolling(window=bb_w)
            global_ls_roll = global_ls.rolling(window=bb_w)
            
            self.features_dict[f'top_ls_ratio_zscore_{bb_w}'] = (top_ls - top_ls_roll.mean()) / top_ls_roll.std().replace(0, np.nan)
            self.features_dict[f'global_ls_ratio_zscore_{bb_w}'] = (global_ls - global_ls_roll.mean()) / global_ls_roll.std().replace(0, np.nan)
            
            # Whale Mean Reversion Bias đa chu kỳ
            if 'close' in df.columns and 'volume_zscore' in self.features_dict:
                bb_sma = df['close'].rolling(window=bb_w).mean()
                bb_std = df['close'].rolling(window=bb_w).std().replace(0, np.nan)
                
                # Tính %B
                bb_lower = bb_sma - 2 * bb_std
                bb_pos = (df['close'] - bb_lower) / (4 * bb_std)
                
                self.features_dict[f'whale_mean_reversion_bias_{bb_w}'] = (
                    ls_imbalance * (0.5 - bb_pos) * (1 + self.features_dict['volume_zscore'])
                )

        # 3. ĐA CHU KỲ CHO FUNDING ALIGNMENT
        if 'fundingRate' in df.columns and df['fundingRate'].notna().any():
            funding = df['fundingRate']
            for fund_w in funding_windows:
                fund_roll = funding.rolling(window=fund_w)
                fund_zscore = (funding - fund_roll.mean()) / fund_roll.std().replace(0, np.nan)
                
                # Áp lực của cá mập so với sự quá nhiệt của Funding từng khung
                self.features_dict[f'ls_funding_alignment_{fund_w}'] = ls_imbalance * fund_zscore
    
    def build_additional_bb_features(self):
        """Build additional Bollinger Band features"""
        df = self.df
        
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        bb_upper = sma + 2*std
        bb_lower = sma - 2*std
        
        # BB position and distance
        self.features_dict['bb_position_20'] = (
            (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-9)
        )
        self.features_dict['bb_distance_upper_pct_20'] = (
            (df['close'] - bb_upper) / bb_upper
        )
        self.features_dict['bb_distance_lower_pct_20'] = (
            (df['close'] - bb_lower) / bb_lower
        )
        self.features_dict['bb_width_pct_20'] = (
            (bb_upper - bb_lower) / sma
        )
        
        # BB squeeze
        bb_width = bb_upper - bb_lower
        self.features_dict['bb_squeeze_20'] = (
            (bb_width < bb_width.rolling(20).quantile(0.1)).astype(int)
        )
    
    def build_advanced_volatility_features(self):
        """Build advanced volatility regime features"""
        df = self.df
        
        tr = self._true_range()
        atr = tr.rolling(14).mean()
        atr_pct = atr / df['close']
        
        # Volatility percentiles and regimes
        self.features_dict['volatility_percentile_14_30'] = (
            atr_pct.rolling(30).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        )
        self.features_dict['volatility_regime_14_30'] = (
            (atr_pct > atr_pct.rolling(30).mean()).astype(int)
        )
        
        # Asymmetric volatility
        returns_up = df['returns'].clip(lower=0).rolling(20).std()
        returns_down = df['returns'].clip(upper=0).abs().rolling(20).std()
        returns_std = df['returns'].rolling(20).std()
        self.features_dict['asymmetric_volatility_index'] = (
            (returns_up - returns_down) / (returns_std + 1e-9)
        )
        
        # Momentum conviction
        rsi_norm = (df['rsi_14'] - 50) / 50
        squeeze_ratio = df['bb_width'] / df['bb_width'].rolling(20).mean()
        self.features_dict['momentum_conviction_index'] = (
            rsi_norm * (df['volume'] / (df['volume_sma_14'] + 1e-9)) * 
            (1 - squeeze_ratio / (squeeze_ratio + 1e-9))
        )
    
    def build_volatility_weighted_features(self):
        """Build volatility-weighted features"""
        df = self.df
        
        # Shadow imbalance weighted by volatility
        vol_ratio = df['atr_7'] / (df['atr_21'] + 1e-9)
        self.features_dict['volatility_weighted_shadow_imbalance'] = (
            ((df['lower_shadow_pct'] - df['upper_shadow_pct']) / 
             (df['body_size_pct'] + 0.001)) * vol_ratio
        )
    
    def build_cumulative_features(self):
        """Build cumulative and normalized volume features"""
        df = self.df
        
        # OBV Normalized
        obv = (df['volume'] * np.sign(df['close'].diff()).fillna(0)).cumsum()
        obv_std = obv.rolling(30).std()
        self.features_dict['obv_normalized'] = obv / (obv_std + 1e-9)
        
        # Volume percentile
        self.features_dict['volume_percentile_30d'] = df['volume'].rolling(30).rank(pct=True)
        
        # CMF (Chaikin Money Flow)
        mf_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-9)
        mf_vol = mf_mult * df['volume']
        self.features_dict['cmf_20'] = mf_vol.rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)
    
    def build_additional_funding_features(self):
        """Build additional sophisticated funding features"""
        df = self.df
        
        if 'fundingRate' not in df.columns or df['fundingRate'].isna().all():
            return
        
        funding = df['fundingRate']
        
        # Funding extremes and percentile
        self.features_dict['funding_extreme_long'] = (
            (funding > funding.rolling(30).quantile(0.9)).astype(int)
        )
        self.features_dict['funding_extreme_short'] = (
            (funding < funding.rolling(30).quantile(0.1)).astype(int)
        )
        self.features_dict['funding_percentile'] = funding.rolling(30).rank(pct=True)
        self.features_dict['funding_zscore'] = (
            (funding - funding.rolling(14).mean()) / 
            (funding.rolling(14).std() + 1e-9)
        )
        self.features_dict['funding_change_zscore'] = (
            (funding.diff() - funding.diff().rolling(7).mean()) / 
            (funding.diff().rolling(7).std() + 1e-9)
        )
    
    def build_squeeze_and_depletion_features(self):
        """Build squeeze and volume depletion features"""
        df = self.df
        
        # Squeeze ratio (BB width vs ATR)
        bb_width = df['bb_width']
        bb_width_sma = bb_width.rolling(20).mean()
        self.features_dict['squeeze_ratio'] = bb_width / (bb_width_sma + 1e-9)
        
        # Volume depletion
        vol_sma = df['volume'].rolling(20).mean()
        self.features_dict['vol_depletion'] = df['volume'] / (vol_sma + 1e-9)
        
        # Pre-ignition score
        self.features_dict['pre_ignition_score'] = (
            (1 - self.features_dict['squeeze_ratio'] / 
             (self.features_dict['squeeze_ratio'] + 1e-9)) + 
            (1 - self.features_dict['vol_depletion'] / 
             (self.features_dict['vol_depletion'] + 1e-9))
        )
    
    def build_price_extension_features(self):
        """Build price extension and distribution features"""
        df = self.df
        
        # Weekend volatility exhaustion
        dist_ema_21_pct = (df['close'] - df['ema_21']) / df['ema_21']
        vol_ratio = df['atr_7'] / (df['atr_21'] + 1e-9)
        is_weekend = self.features_dict.get('is_weekend', 0)
        
        self.features_dict['weekend_volatility_exhaustion_ratio'] = (
            (dist_ema_21_pct.abs() * is_weekend) / (vol_ratio + 1e-9)
        )
        
        # Session momentum efficiency
        is_amer = self.features_dict.get('is_american_session', 0)
        is_euro = self.features_dict.get('is_european_session', 0)
        roc_14 = self.features_dict.get('roc_14', df['close'].pct_change(14))
        
        self.features_dict['session_momentum_efficiency_index'] = (
            (roc_14 * df['volume_ratio']) * 
            (1.0 + 0.5 * is_amer + 0.25 * is_euro)
        )
        
        # Distance to EMA50 normalized
        self.features_dict['dist_to_ema50_atr'] = (
            (df['close'] - df['ema_50']) / (df['atr_14'] + 1e-9)
        )
    
    def build_ranking_features(self):
        """Build cross-sectional ranking features"""
        df = self.df
        
        # These would require cross-sectional data (multiple symbols)
        # For now, create placeholders
        self.features_dict['rsi_rank_pct'] = pd.Series(0.5, index=df.index)
        self.features_dict['oi_growth_rank_pct'] = pd.Series(0.5, index=df.index)
        self.features_dict['volatility_rank_pct'] = pd.Series(0.5, index=df.index)
        self.features_dict['momentum_rank_pct'] = pd.Series(0.5, index=df.index)
    
    def build_leverage_tension_features(self):
        """Build leverage and PnL tension features"""
        df = self.df
        
        if 'fundingRate' not in df.columns or df['fundingRate'].isna().all():
            return
        
        funding = df['fundingRate']
        dist_ema_21_pct = (df['close'] - df['ema_21']) / (df['ema_21'] + 1e-9)
        
        # Leverage PnL tension
        self.features_dict['leverage_pnl_tension_index'] = (
            (funding / (funding.rolling(24).std() + 1e-12)) * 
            (dist_ema_21_pct / (df['atr_pct_14'] + 1e-9))
        )
        
        # Normalized funding momentum shock
        self.features_dict['normalized_funding_momentum_shock'] = (
            (funding.diff(1) / (funding.rolling(24).std() + 1e-12)) / 
            (df['atr_pct_14'] + 1e-9)
        )
    
    def build_relative_performance_features(self):
        """Build relative momentum and quality features"""
        df = self.df
        
        # Relative momentum quality
        momentumrank = df['close'].pct_change(7)
        volatrank = df['returns'].rolling(14).std()
        
        self.features_dict['relative_momentum_quality'] = (
            ((momentumrank.rank(pct=True) - volatrank.rank(pct=True)) * 
             (df['adx_14'] / 100.0))
        )
    
    def build_trend_state_features(self):
        """Build trend state and acceleration features"""
        df = self.df
        
        # Trend state
        self.features_dict['trend_state'] = np.where(
            df['adx_14'] < 20,
            0,
            np.where(df['close'] > df['sma_50'], 1, -1)
        )
        
        # Volume acceleration
        self.features_dict['vol_acceleration'] = (
            df['volume'].diff().diff() / (df['volume_sma_14'] + 1e-9)
        )
    
    def build_speculative_features(self):
        """Build speculative positioning features"""
        df = self.df
        
        # Speculative congestion (requires ls_imbalance)
        ls_imbal = self.features_dict.get('ls_imbalance', 0)
        oi_vol_zscore = self.features_dict.get('oi_to_volume_ratio_zscore', 0)
        funding_zscore = self.features_dict.get('funding_zscore', 0)
        squeeze_ratio = self.features_dict.get('squeeze_ratio', 1)
        
        self.features_dict['speculative_congestion_index'] = (
            (oi_vol_zscore * funding_zscore) / (squeeze_ratio + 1e-9)
        )
        
        # Volume-funding divergence
        if 'fundingRate' in df.columns and df['fundingRate'].notna().sum() > 0:
            self.features_dict['volume_funding_divergence'] = (
                df['volume_zscore'] * np.sign(df['fundingRate'])
            )
    
    # ==================== MULTI-PERIOD & MULTI-PARAMETER FEATURES ====================
    def build_multi_period_rsi(self):
        """Build RSI with multiple periods: 7, 14, 21, 28"""
        df = self.df
        periods = [7, 14, 21, 28]
        for period in periods:
            self.features_dict[f'rsi_{period}'] = self._calculate_rsi(period)
    
    def build_multi_period_atr(self):
        """Build ATR with multiple periods: 7, 14, 21, 28"""
        df = self.df
        tr = self._true_range()
        periods = [7, 14, 21, 28]
        for period in periods:
            atr = tr.rolling(period).mean()
            self.features_dict[f'atr_{period}'] = atr
            self.features_dict[f'atr_pct_{period}'] = atr / df['close']
    
    def build_multi_period_ema(self):
        """Build EMA with multiple periods: 7, 12, 21, 50, 100, 200"""
        df = self.df
        periods = [7, 12, 21, 50, 100, 200]
        for period in periods:
            self.features_dict[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    def build_multi_period_sma(self):
        """Build SMA with multiple periods: 10, 20, 50, 100, 200"""
        df = self.df
        periods = [10, 20, 50, 100, 200]
        for period in periods:
            self.features_dict[f'sma_{period}'] = df['close'].rolling(period).mean()
    
    def build_multi_period_bollinger(self):
        """Build Bollinger Bands with multiple periods: 20, 30, 50"""
        df = self.df
        periods = [20, 30, 50]
        for period in periods:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            self.features_dict[f'bb_upper_{period}'] = sma + 2 * std
            self.features_dict[f'bb_lower_{period}'] = sma - 2 * std
            self.features_dict[f'bb_mid_{period}'] = sma
            self.features_dict[f'bb_width_{period}'] = 4 * std
            self.features_dict[f'bb_position_{period}'] = (
                (df['close'] - (sma - 2*std)) / (4*std + 1e-9)
            )
    
    def build_multi_period_roc(self):
        """Build Rate of Change with multiple periods: 7, 12, 24"""
        df = self.df
        periods = [7, 12, 24]
        for period in periods:
            self.features_dict[f'roc_{period}'] = df['close'].pct_change(period)
    
    def build_multi_period_cci(self):
        """Build CCI with multiple periods: 14, 20, 30"""
        df = self.df
        periods = [14, 20, 30]
        for period in periods:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
            self.features_dict[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-9)
    
    def build_multi_period_volume(self):
        """Build volume-based features with multiple periods: 7, 14, 20, 30"""
        df = self.df
        periods = [7, 14, 20, 30]
        for period in periods:
            vol_ma = df['volume'].rolling(period).mean()
            self.features_dict[f'volume_ma_{period}'] = vol_ma
            self.features_dict[f'volume_ratio_{period}'] = df['volume'] / (vol_ma + 1e-9)
            self.features_dict[f'volume_zscore_{period}'] = (
                (df['volume'] - vol_ma) / (df['volume'].rolling(period).std() + 1e-9)
            )
    
    def build_multi_period_volatility(self):
        """Build realized volatility with multiple periods: 7, 14, 21, 30"""
        df = self.df
        returns = df['close'].pct_change()
        periods = [7, 14, 21, 30]
        for period in periods:
            self.features_dict[f'volatility_{period}'] = returns.rolling(period).std()
            self.features_dict[f'volatility_{period}_zscore'] = (
                (returns.rolling(period).std() - returns.rolling(period).std().rolling(period).mean()) /
                (returns.rolling(period).std().rolling(period).std() + 1e-9)
            )
    
    def build_multi_parameter_oscillators(self):
        """Build MACD, Stochastic with multiple parameter combinations"""
        df = self.df
        
        # MACD with different fast/slow combinations
        macd_params = [(12, 26, 9), (5, 35, 5), (10, 20, 9)]
        for idx, (fast, slow, signal) in enumerate(macd_params):
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            self.features_dict[f'macd_{fast}_{slow}_{signal}'] = macd
            self.features_dict[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            self.features_dict[f'macd_hist_{fast}_{slow}_{signal}'] = macd - macd_signal
        
        # Stochastic RSI with different lookback periods
        for period in [7, 14, 21]:
            rsi = self._calculate_rsi(period)
            rsi_low = rsi.rolling(period).min()
            rsi_high = rsi.rolling(period).max()
            self.features_dict[f'stoch_rsi_{period}'] = (
                (rsi - rsi_low) / (rsi_high - rsi_low + 1e-9)
            )
    
    # ==================== HELPER METHODS ====================
    def _true_range(self) -> pd.Series:
        """Calculate True Range"""
        df = self.df
        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift(1))
        tr3 = np.abs(df['low'] - df['close'].shift(1))
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def _calculate_rsi(self, period: int) -> pd.Series:
        """Calculate RSI"""
        df = self.df
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, period: int) -> pd.Series:
        """Calculate ADX"""
        df = self.df
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr = self._true_range()
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(period).mean()
        
        return adx


def create_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to create all market features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns: timestamp, open, high, low, close, volume,
        sum_open_interest, top_ls_ratio, global_ls_ratio, fundingRate
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with all features added
    """
    builder = FeatureBuilder(df)
    return builder.build_all()

