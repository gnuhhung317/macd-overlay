import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
import traceback

warnings.filterwarnings('ignore')

class FeatureBuilder:
    """Build technical and fundamental features for trading signals (Fully Optimized)"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        btc_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        debug: bool = False
    ):
        self.df = df
        self._input_columns = list(self.df.columns)
        self.features_dict = {}
        self.verbose = verbose
        self.debug = debug
        self.btc_df = btc_df
        
        # Caches
        self._tr_cache = None
        self._rsi_cache = {}
        self._adx_cache = {}
        self._ema_cache = {}
        self._rolling_cache = {}
        self._series_refs = {}
        
        # Precomputed safe series to avoid duplicate .replace(0, np.nan)
        self._close_safe = self.df['close'].replace(0, np.nan)
        if 'volume' in self.df.columns:
            self._volume_safe = self.df['volume'].replace(0, np.nan)
        self._open_safe = self.df['open'].replace(0, np.nan)
        self._low_safe = self.df['low'].replace(0, np.nan)
        self._hl_range_safe = (self.df['high'] - self.df['low']).replace(0, np.nan)
    def _rolling(self, series: pd.Series, window: int, func: str, min_periods: Optional[int] = None) -> pd.Series:
        series_id = id(series)
        key = (series_id, window, func, min_periods)
        if key in self._rolling_cache:
            return self._rolling_cache[key]
        
        # Keep series alive to prevent id() reuse by Python's garbage collector
        self._series_refs[series_id] = series
        
        roll = series.rolling(window, min_periods=min_periods)
        if func == 'mean':
            result = roll.mean()
        elif func == 'std':
            result = roll.std()
        elif func == 'min':
            result = roll.min()
        elif func == 'max':
            result = roll.max()
        elif func == 'sum':
            result = roll.sum()
        elif func == 'var':
            result = roll.var()
        else:
            raise ValueError(f"Unsupported func: {func}")
            
        self._rolling_cache[key] = result
        return result

    def get_raw_columns(self) -> List[str]:
        """Trả về danh sách các cột raw (không biến đổi) để bề mặt logic sử dụng lại lúc cần."""
        raw_cols = ['open', 'high', 'low', 'close', 'volume', 'sum_open_interest']
        return [col for col in raw_cols if col in self.df.columns]

    def get_non_stationary_columns(self) -> List[str]:
        """Trả về danh sách các cột gốc / price-dependent không đạt chuẩn dừng (stationary) để có thể drop trước khi feed vào ML models."""
        non_stationary = [
            'open', 'high', 'low', 'close', 'volume', 'sum_open_interest',
            'timestamp', 'symbol', 'datetime', 'time',
            # Giá trị tuyệt đối của MACD, Bollinger (nếu có để lại base không chuẩn hóa)
            'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_mid', 'bb_width',
            'vwap', 'pvt', 'obv', 'cumulative_volume',
            # 'top_ls_ratio', 'global_ls_ratio' # Base value đôi khi không stationary
        ]
        
        # Thêm dynamic các cột sma_, ema_ dựa trên pattern
        for col in self.df.columns:
            if col.startswith(('sma_', 'ema_', 'bb_upper_', 'bb_lower_', 'bb_mid_', 'atr_')) and not col.endswith(('_pct', '_zscore', '_alpha')):
                non_stationary.append(col)
                
        return list(set(col for col in non_stationary if col in self.df.columns))

    def _make_features_stationary(self):
        """
        Biến đổi các tính năng không dừng (non-stationary) thành dừng (stationary).
        Được gọi sau cùng nhằm đảm bảo không ảnh hưởng đến các logic tính toán trước đó 
        cần giá trị tuyệt đối.
        """
        df = self.df
        
        # 1. Các feature bám theo giá trị tuyệt đối của giá (Moving Averages, Bollinger Bands, VWAP)
        # Cách xử lý: Chuyển thành % khoảng cách so với giá close hiện tại (distance/percentage from close)
        price_trackers = [
            'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_7', 'ema_12', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
            'vwap',
            'bb_upper', 'bb_lower', 'bb_mid',
            'bb_upper_20', 'bb_lower_20', 'bb_mid_20',
            'bb_upper_30', 'bb_lower_30', 'bb_mid_30',
            'bb_upper_50', 'bb_lower_50', 'bb_mid_50'
        ]
        
        for col in price_trackers:
            if col in df.columns:
                df[col] = (df[col] - df['close']) / self._close_safe
                
        # 2. Các feature tăng dần/xu hướng theo thời gian (Cumulative)
        # Cách xử lý: Lấy sai phân (difference) để đưa về dạng dao động dừng
        cumulative_cols = ['pvt', 'obv', 'cumulative_volume']
        
        for col in cumulative_cols:
            if col in df.columns:
                df[col] = df[col].diff()
    
    def build_all(self) -> pd.DataFrame:
        """Build features with Dynamic Error Tracing"""
        
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
            'build_btc_context_features',
            'build_trend_state_features', 'build_speculative_features',
            'build_multi_period_rsi', 'build_multi_period_atr',
            'build_multi_period_ema', 'build_multi_period_sma',
            'build_multi_period_bollinger', 'build_multi_period_roc',
            'build_multi_period_cci', 'build_multi_period_volume',
            'build_multi_period_volatility', 'build_multi_parameter_oscillators', 'build_smart_money_mechanics'
        ]
        
        if self.verbose:
            print("Bắt đầu quét và xây dựng Features (Quant Standard)...")
        
        for method_name in methods_to_run:
            func = getattr(self, method_name, None)
            if func is None:
                continue

            if self.debug:
                try:
                    func()
                except Exception:
                    print("\n" + "!" * 60)
                    print(f" BẮT ĐƯỢC TỘI PHẠM TẠI HÀM: {method_name}")
                    print("!" * 60)
                    traceback.print_exc()
                    raise SystemExit(f"Hệ thống dừng khẩn cấp để sửa hàm {method_name}.")
            else:
                func()
        
        if self.verbose:
            print(f"Đã chạy thành công {len(methods_to_run)} hàm. Đang gom dữ liệu...")
        for feature_name, feature_values in self.features_dict.items():
            self.df[feature_name] = feature_values

        # Make features stationary (must be called before shifting to preserve metrics correctly)
        self._make_features_stationary()

        # Enforce causal snapshot: entry at T only sees engineered features from T-1.
        self._apply_entry_t_minus_1_lag()
            
        return self.df
    
    # ==================== VOLATILITY FEATURES ====================
    def build_volatility_features(self):
        df = self.df
        
        tr = self._true_range()
        df['tr_14'] = tr
        df['atr_14'] = self._rolling(df['tr_14'], 14, 'mean')
        df['atr_7'] = self._rolling(tr, 7, 'mean')
        df['atr_21'] = self._rolling(df['tr_14'], 21, 'mean')
        df['atr_pct_14'] = df['atr_14'] / self._close_safe
        
        df['sma_20'] = self._rolling(df['close'], 20, 'mean')
        df['std_20'] = self._rolling(df['close'], 20, 'std')
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        self.features_dict['bb_width_zscore_20'] = (
            (df['bb_width'] - self._rolling(df['bb_width'], 20, 'mean')) / 
            self._rolling(df['bb_width'], 20, 'std').replace(0, np.nan)
        )
        
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        self.features_dict['volatility_20'] = self._rolling(df['log_returns'], 20, 'std')
        self.features_dict['volatility_50'] = self._rolling(df['log_returns'], 50, 'std')
        
        self.features_dict['parkinson_vol'] = (
            np.log(df['high'] / self._low_safe).rolling(20).mean() / (2 * np.sqrt(np.log(4)))
        )
        
        hl_ratio = np.log(df['high'] / self._low_safe)
        co_ratio = np.log(df['close'] / self._open_safe)
        self.features_dict['garman_klass_vol'] = np.sqrt(
            np.clip(0.5 * hl_ratio**2 - (2*np.log(2) - 1) * co_ratio**2, 0, None)
        ).rolling(20).mean()
        
        log_ho = np.log(df['high'] / self._open_safe)
        log_lo = np.log(df['low'] / self._open_safe)
        log_co = np.log(df['close'] / self._open_safe)
        log_oc_prev = np.log(df['open'] / df['close'].shift(1).replace(0, np.nan))
        
        s_o = self._rolling(log_oc_prev, 20, 'var')
        s_c = self._rolling(log_co, 20, 'var')
        s_rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(20).mean()
        k = 0.34 / (1.34 + 21 / 19)
        vol = np.sqrt(np.clip(s_o + k * s_c + (1 - k) * s_rs, 0, None))
        self.features_dict['yang_zhang_vol_zscore'] = (
            (vol - self._rolling(vol, 20, 'mean')) / self._rolling(vol, 20, 'std').replace(0, np.nan)
        )
        
        self.features_dict['vol_ratio_alpha'] = (
            self._rolling(df['returns'], 5, 'std') / self._rolling(df['returns'], 20, 'std').replace(0, np.nan)
        )
    
    # ==================== TREND FEATURES ====================
    def build_trend_features(self):
        df = self.df
        
        df['ema_7'] = self._ema(7)
        df['ema_21'] = self._ema(21)
        df['ema_50'] = self._ema(50)
        df['ema_200'] = self._ema(200)
        
        self.features_dict['dist_to_ema_50'] = df['close'] - df['ema_50']
        self.features_dict['dist_to_ema_50_pct'] = (df['close'] - df['ema_50']) / df['ema_50'].replace(0, np.nan)
        
        ema12 = self._ema(12)
        ema26 = self._ema(26)
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        log_ret = df['log_returns']
        rs = (
            (self._rolling(log_ret, 20, 'max') - self._rolling(log_ret, 20, 'min')) / 
            self._rolling(log_ret, 20, 'std').replace(0, np.nan)
        )
        self.features_dict['hurst_deviation'] = (np.log(rs.replace(0, np.nan)) / np.log(20)) - 0.5
        self.features_dict['adx_14'] = self._calculate_adx(14)
    
    # ==================== MEAN REVERSION FEATURES ====================
    def build_mean_reversion_features(self):
        df = self.df
        
        df['rsi_14'] = self._calculate_rsi(14)
        df['rsi_7'] = self._calculate_rsi(7)
        
        rsi = df['rsi_14']
        self.features_dict['stoch_rsi_14'] = (
            (rsi - self._rolling(rsi, 14, 'min')) / 
            (self._rolling(rsi, 14, 'max') - self._rolling(rsi, 14, 'min')).replace(0, np.nan)
        )
        
        df['upper_shadow_pct'] = (df['high'] - df.loc[:, ['open', 'close']].max(axis=1)) / self._close_safe
        df['lower_shadow_pct'] = (df.loc[:, ['open', 'close']].min(axis=1) - df['low']) / self._close_safe
        df['body_size_pct'] = np.abs(df['close'] - df['open']) / self._close_safe
        df['body_position'] = (df['close'] - df['low']) / self._hl_range_safe
        
        vol_sma_20 = self._rolling(df['volume'], 20, 'mean').replace(0, np.nan)
        self.features_dict['mean_reversion_tension_score'] = (
            self.features_dict['dist_to_ema_50_pct'] * (df['volume'] / vol_sma_20) * (df['upper_shadow_pct'] - df['lower_shadow_pct'])
        )
        
        self.features_dict['position_in_bb'] = (
            (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
        )
    
    # ==================== VOLUME FEATURES ====================
    def build_volume_features(self, mfi_window=14, vfi_window=20, fve_window=20):
        df = self.df
        
        df['volume_sma'] = self._rolling(df['volume'], mfi_window, 'mean')
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)

        # Stationary transforms for raw volume level.
        volume_non_negative = df['volume'].clip(lower=0)
        volume_log = np.log1p(volume_non_negative)
        self.features_dict['volume_log_diff_1'] = volume_log.diff(1)
        self.features_dict['volume_log_diff_24'] = volume_log.diff(24)
        self.features_dict['volume_pct_change_1'] = df['volume'].pct_change(1)
        self.features_dict['volume_pct_change_24'] = df['volume'].pct_change(24)
        vol_log_roll_48 = volume_log.rolling(48, min_periods=self._adaptive_min_periods(48))
        self.features_dict['volume_detrended_48'] = volume_log - vol_log_roll_48.mean()
        
        vol_std = self._rolling(df['volume'], mfi_window, 'std').replace(0, np.nan)
        self.features_dict['volume_zscore'] = (df['volume'] - df['volume_sma']) / vol_std
        
        pvt = (df['close'].pct_change() * df['volume']).cumsum()
        self.features_dict['pvt'] = pvt
        self.features_dict['pvt_delta'] = pvt.diff()
        self.features_dict['pvt_zscore_252'] = self._rolling_zscore(
            pvt,
            window=252,
            min_periods=self._adaptive_min_periods(252)
        )
        
        direction = np.sign(df['close'].diff()).fillna(0)
        obv = (direction * df['volume']).cumsum()
        df['obv'] = obv
        self.features_dict['obv'] = obv
        self.features_dict['obv_sma_ratio'] = df['obv'] / self._rolling(df['obv'], 20, 'mean').replace(0, np.nan)
        self.features_dict['obv_delta'] = obv.diff()
        self.features_dict['obv_zscore_252'] = self._rolling_zscore(
            obv,
            window=252,
            min_periods=self._adaptive_min_periods(252)
        )
        
        money_flow = df['close'] * df['volume']
        close_diff = df['close'].diff()
        
        positive_flow = money_flow.where(close_diff > 0, 0.0)
        negative_flow = money_flow.where(close_diff < 0, 0.0)
        
        pos_flow_sum = self._rolling(positive_flow, mfi_window, 'sum')
        neg_flow_sum = self._rolling(negative_flow, mfi_window, 'sum').replace(0, np.nan)
        
        money_ratio = pos_flow_sum / neg_flow_sum
        self.features_dict[f'mfi_{mfi_window}'] = (100 - (100 / (1 + money_ratio))).fillna(100)
        
        typ = (df['high'] + df['low'] + df['close']) / 3.0
        typ_diff = typ.diff()
        mf = df['volume'] * np.sign(typ_diff).fillna(0)
        vfi_denom = self._rolling(df['volume'], vfi_window, 'mean').replace(0, np.nan)
        self.features_dict[f'vfi_{vfi_window}'] = self._rolling(mf, vfi_window, 'sum') / vfi_denom
        
        hl_avg = (df['high'] + df['low']) / 2.0
        fve_mf = df['volume'].where(typ > hl_avg, -df['volume'])
        fve_denom = self._rolling(df['volume'], fve_window, 'sum').replace(0, np.nan)
        self.features_dict[f'fve_{fve_window}'] = self._rolling(fve_mf, fve_window, 'sum') / fve_denom
        
        cumulative_volume = df['volume'].cumsum()
        self.features_dict['cumulative_volume'] = cumulative_volume
        self.features_dict['cumulative_volume_delta'] = cumulative_volume.diff()
        self.features_dict['cumulative_volume_zscore_252'] = self._rolling_zscore(
            cumulative_volume,
            window=252,
            min_periods=self._adaptive_min_periods(252)
        )
        
        cum_pv = (typ * df['volume']).cumsum()
        cum_v = df['volume'].cumsum().replace(0, np.nan)
        df['vwap'] = cum_pv / cum_v
        vwap_distance = (df['close'] - df['vwap']) / df['vwap'].replace(0, np.nan)
        self.features_dict['vwap_distance_pct'] = vwap_distance
        self.features_dict['vwap_distance_zscore_120'] = self._rolling_zscore(
            vwap_distance,
            window=120,
            min_periods=self._adaptive_min_periods(120)
        )
    
    # ==================== MOMENTUM FEATURES ====================
    def build_momentum_features(self):
        df = self.df
        
        self.features_dict['roc_12'] = (df['close'] - df['close'].shift(12)) / df['close'].shift(12).replace(0, np.nan)
        self.features_dict['roc_24'] = (df['close'] - df['close'].shift(24)) / df['close'].shift(24).replace(0, np.nan)
        
        self.features_dict['momentum_10'] = df['close'] - df['close'].shift(10)
        self.features_dict['momentum_20'] = df['close'] - df['close'].shift(20)
        
        self.features_dict['efficiency_thrust_index'] = (
            (self._rolling(df['returns'], 10, 'sum') / self._rolling(df['atr_14'], 10, 'sum').replace(0, np.nan)) * df.get('volume_ratio', 1)
        )
        
        typ = (df['high'] + df['low'] + df['close']) / 3
        sma_typ = self._rolling(typ, 20, 'mean')
        mad = self._rolling_mad_fast(typ, 20)
        self.features_dict['cci_20'] = (typ - sma_typ) / (0.015 * mad).replace(0, np.nan)
    
    # ==================== MARKET STRUCTURE FEATURES ====================
    def build_market_structure_features(self):
        df = self.df
        
        diffs = (df['close'] - df['close'].shift(1)).abs()
        path_l = self._rolling(diffs, 20, 'sum')
        range_l = (self._rolling(df['close'], 20, 'max') - self._rolling(df['close'], 20, 'min')).replace(0, np.nan)
        
        # log1p an toàn
        self.features_dict['fdi_20'] = 1.0 + (np.log(path_l / range_l)) / np.log(20)
        
        self.features_dict['skewness_20d'] = df['close'].pct_change().rolling(20).skew()
        self.features_dict['kurtosis_20d'] = df['close'].pct_change().rolling(20).kurt()
        
        yz_vol = self.features_dict.get('yang_zhang_vol_zscore', self._rolling(df['close'], 20, 'std'))
        hurst = self.features_dict.get('hurst_deviation', 0)
        
        self.features_dict['tail_regime_stress_score'] = (
            self.features_dict['skewness_20d'] * yz_vol * (hurst + 0.5)
        )
        
        vfi = self.features_dict.get('vfi_20', 0)
        fdi = self.features_dict.get('fdi_20', 1.5)
        self.features_dict['structural_vfi_efficiency'] = vfi * (2.0 - fdi)
    
    # ==================== FUNDING FEATURES ====================
    def build_funding_features(self):
        df = self.df
        if 'fundingRate' not in df.columns:
            df['fundingRate'] = 0
            
        funding = df['fundingRate'].ffill()

        roll_48 = funding.rolling(48, min_periods=self._adaptive_min_periods(48))
        roll_24 = funding.rolling(24, min_periods=self._adaptive_min_periods(24))
        fund_std_48 = roll_48.std().replace(0, np.nan)
        self.features_dict['funding_zscore'] = (funding - roll_48.mean()) / fund_std_48
        self.features_dict['funding_ma_24'] = roll_24.mean()
        
        body_size = (np.abs(df['close'] - df['open']) / self._close_safe).replace(0, np.nan)
        shadow_diff = df.get('upper_shadow_pct', 0) - df.get('lower_shadow_pct', 0)
        vol_ratio = df['volume'] / self._rolling(df['volume'], 20, 'mean').replace(0, np.nan)
        
        sfai = (shadow_diff / body_size) * np.sign(funding) * (np.abs(funding)**1.2) * vol_ratio
        sfai_roll = sfai.rolling(120, min_periods=self._adaptive_min_periods(120))
        self.features_dict['shadow_funding_asymmetry_index'] = (
            (sfai - sfai_roll.mean()) / sfai_roll.std().replace(0, np.nan)
        )
        
        funding_vel = funding.diff() / roll_24.std().replace(0, np.nan)
        vol_roll_24 = df['volume'].rolling(24, min_periods=self._adaptive_min_periods(24)).mean().replace(0, np.nan)
        vol_spread = funding_vel * df.get('atr_pct_14', 1) * (df['volume'] / vol_roll_24)
        vol_spread_roll = vol_spread.rolling(168, min_periods=self._adaptive_min_periods(168))
        self.features_dict['funding_velocity_volatility_spread_zscore'] = (
            (vol_spread - vol_spread_roll.mean()) / vol_spread_roll.std().replace(0, np.nan)
        )
        
        tlfo_metric = (df['close'] - df['open']) * df['volume'] * np.sign(funding.shift(1)) * (np.abs(funding.shift(1))**1.5)
        tlfo_roll = tlfo_metric.rolling(144, min_periods=self._adaptive_min_periods(144))
        self.features_dict['trapped_liquidity_funding_oscillator'] = (
            (tlfo_metric - tlfo_roll.mean()) / tlfo_roll.std().replace(0, np.nan)
        )
        
        vol_roll_48 = df['volume'].rolling(48, min_periods=self._adaptive_min_periods(48)).mean().replace(0, np.nan)
        close_std_48 = df['close'].rolling(48, min_periods=self._adaptive_min_periods(48)).std().replace(0, np.nan)
        funding_centered = funding - roll_48.mean()
        fcvm_metric = (funding_centered**3) * np.log1p(df['volume'] / vol_roll_48) / close_std_48
        fcvm_roll = fcvm_metric.rolling(48, min_periods=self._adaptive_min_periods(48))
        self.features_dict['funding_convexity_volume_multiplier_z_score'] = (
            (fcvm_metric - fcvm_roll.mean()) / fcvm_roll.std().replace(0, np.nan)
        )
        
        hl_range = self._hl_range_safe
        sld_metric = ((df['close'] - df['open']) / hl_range) * np.log1p(df['volume']) * np.sign(funding) * np.exp(np.clip(np.abs(funding), None, 10))
        sld_roll = sld_metric.rolling(48, min_periods=self._adaptive_min_periods(48))
        self.features_dict['synthetic_liquidation_delta_z_score'] = (
            (sld_metric - sld_roll.mean()) / sld_roll.std().replace(0, np.nan)
        )
    
    # ==================== CANDLE PATTERN FEATURES ====================
    def build_candle_pattern_features(self):
        df = self.df
        
        df['hl_range'] = df['high'] - df['low']
        df['oc_range'] = np.abs(df['close'] - df['open'])
        df['hl_range_pct'] = df['hl_range'] / self._close_safe
        
        hl_safe = df['hl_range'].replace(0, np.nan)
        self.features_dict['doji_score'] = df['oc_range'] / hl_safe
        self.features_dict['candle_strength'] = df['oc_range'] / hl_safe
        
        hl_pct_safe = df['hl_range_pct'].replace(0, np.nan)
        self.features_dict['upper_wick_ratio'] = df.get('upper_shadow_pct', 0) / hl_pct_safe
        self.features_dict['lower_wick_ratio'] = df.get('lower_shadow_pct', 0) / hl_pct_safe
    
    # ==================== RISK FEATURES ====================
    def build_risk_features(self):
        df = self.df
        
        cummax = df['close'].cummax()
        self.features_dict['drawdown'] = (df['close'] - cummax) / cummax.replace(0, np.nan)
        self.features_dict['drawdown_pct'] = self.features_dict['drawdown'] * 100
        
        self.features_dict['var_95'] = df['returns'].rolling(20).quantile(0.05)
        
        def cvar_np(x):
            q = np.nanquantile(x, 0.05)
            tail = x[x <= q]
            return np.nanmean(tail) if tail.size else np.nan

        self.features_dict['cvar_95'] = df['returns'].rolling(20).apply(cvar_np, raw=True)
        
        vol_20 = self.features_dict.get('volatility_20', self._rolling(df['returns'], 20, 'std'))
        self.features_dict['vol_of_vol'] = self._rolling(vol_20, 20, 'std')
    
    # ==================== LIQUIDITY FEATURES ====================
    def build_liquidity_features(self):
        df = self.df
        
        illiquidity = np.abs(df['returns']) / (df['volume'] * df['close']).replace(0, np.nan)
        self.features_dict['amihud_zscore'] = (
            (illiquidity - self._rolling(illiquidity, 20, 'mean')) / self._rolling(illiquidity, 20, 'std').replace(0, np.nan)
        )
        
        beta = (np.log(df['high'] / self._low_safe)**2).rolling(2).sum()
        gamma = (np.log(self._rolling(df['high'], 2, 'max') / self._rolling(df['low'], 2, 'min').replace(0, np.nan)))**2
        
        den = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / den - np.sqrt(gamma / den)
        
        self.features_dict['corwin_schultz_pct'] = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        self.features_dict['ba_spread'] = df['high'] - df['low']
        self.features_dict['ba_spread_pct'] = self.features_dict['ba_spread'] / self._close_safe
    
    # ==================== ADVANCED COMPOSITE FEATURES ====================
    def build_advanced_composite_features(self):
        df = self.df
        f_dict = self.features_dict 
        
        atr_7 = f_dict.get('atr_7', df.get('atr_7'))
        atr_21 = f_dict.get('atr_21', df.get('atr_21'))
        
        if atr_7 is not None and atr_21 is not None:
            vol_ratio_alpha = f_dict.get('vol_ratio_alpha')
            if vol_ratio_alpha is None:
                vol_ratio_alpha = atr_7 / atr_21.replace(0, np.nan)
                f_dict['vol_ratio_alpha'] = vol_ratio_alpha
            
            vol_z = f_dict.get('volume_zscore')
            if vol_z is not None:
                vol_ratio_roll = vol_ratio_alpha.rolling(20)
                vol_ratio_zscore = (vol_ratio_alpha - vol_ratio_roll.mean()) / vol_ratio_roll.std().replace(0, np.nan)
                f_dict['volatility_expansion_intensity'] = vol_ratio_zscore * vol_z
        
        atr_14 = f_dict.get('atr_14', df.get('atr_14'))
        ema_21 = f_dict.get('ema_21', df.get('ema_21'))
        vol_sma = f_dict.get('volume_sma', df.get('volume_sma')) 
        
        if all(v is not None for v in [atr_14, ema_21, atr_7, atr_21, vol_sma]):
            f_dict['volatility_momentum_tension_flux'] = (
                ((df['close'] - ema_21) / atr_14.replace(0, np.nan)) * (atr_7 / atr_21.replace(0, np.nan)) * (df['volume'] / vol_sma.replace(0, np.nan))
            )
        
        high_roll_max = self._rolling(df['high'], 10, 'max')
        low_roll_min = self._rolling(df['low'], 10, 'min')
        price_range = (high_roll_max - low_roll_min).replace(0, np.nan)
        vol_mean_20 = self._rolling(df['volume'], 20, 'mean').replace(0, np.nan)
        
        f_dict['volume_weighted_fractal_efficiency'] = (
            (df['close'].diff(10).abs() / price_range) * (df['volume'] / vol_mean_20)
        )
        
        if 'fundingRate' in df.columns and df['fundingRate'].notna().any():
            funding = df['fundingRate']
            fund_sign = pd.Series(np.sign(funding), index=df.index)
            hl_range = self._hl_range_safe
            clv_num = (df['close'] - df['low']) - (df['high'] - df['close'])
            clv = df['volume'] * fund_sign * (clv_num / hl_range)
            clv_vol = self._rolling(clv, 24, 'mean')
            fund_std_168 = self._rolling(funding, 168, 'std').replace(0, np.nan)
            fund_exp_term = np.clip(np.abs(funding) / fund_std_168, a_min=None, a_max=10)
            metric = clv_vol * np.exp(fund_exp_term)
            metric_roll = metric.rolling(168)
            f_dict['cross_modal_funding_squeeze_momentum'] = (
                (metric - metric_roll.mean()) / metric_roll.std().replace(0, np.nan)
            )
            
    # ==================== ADDITIONAL COMPREHENSIVE FEATURES ====================
    def build_ohlc_candle_features(self):
        df = self.df
        
        df['intraday_momentum'] = (df['close'] - df['open']) / self._hl_range_safe
        df['true_range_pct'] = (
            pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift(1)).abs(),
                (df['low'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1) / df['close'].shift(1).replace(0, np.nan)
        )
        self.features_dict['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        self.features_dict['lower_shadow_pct'] = (df[['open', 'close']].min(axis=1) - df['low']) / self._close_safe
        self.features_dict['upper_shadow_pct'] = (df['high'] - df[['open', 'close']].max(axis=1)) / self._close_safe
        self.features_dict['body_size_pct'] = (df['close'] - df['open']).abs() / self._close_safe
        self.features_dict['intraday_momentum'] = df['intraday_momentum']
        self.features_dict['true_range_pct'] = df['true_range_pct']
    
    def build_returns_volatility_features(self):
        df = self.df
        
        self.features_dict['returns_std_7'] = df['close'].pct_change().rolling(7).std()
        self.features_dict['returns_std_14'] = df['close'].pct_change().rolling(14).std()
        self.features_dict['returns_std_21'] = df['close'].pct_change().rolling(21).std()
        self.features_dict['returns_zscore_20d'] = (
            (df['returns'] - self._rolling(df['returns'], 20, 'mean')) / 
            self._rolling(df['returns'], 20, 'std').replace(0, np.nan)
        )
    
    def build_advanced_momentum_features(self):
        df = self.df
        
        ema1 = df['close'].ewm(span=15, adjust=False).mean()
        ema2 = ema1.ewm(span=15, adjust=False).mean()
        ema3 = ema2.ewm(span=15, adjust=False).mean()
        self.features_dict['trix_pct'] = ema3.pct_change() * 100
        
        hh = self._rolling(df['high'], 14, 'max')
        ll = self._rolling(df['low'], 14, 'min')
        self.features_dict['williams_r_normalized'] = ((hh - df['close']) / (hh - ll).replace(0, np.nan)) * -1
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = self._rolling(tp, 20, 'mean')
        mad = self._rolling_mad_fast(tp, 20)
        cci = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)
        self.features_dict['cci_normalized'] = cci / 100
        
        self.features_dict['roc_14'] = df['close'].pct_change(periods=14)
        
        ema12 = self._ema(12)
        ema26 = self._ema(26)
        self.features_dict['ppo'] = ((ema12 - ema26) / ema26.replace(0, np.nan)) * 100
        
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        self.features_dict['macd_histogram_pct'] = (macd - signal) / self._close_safe
        self.features_dict['macd_normalized'] = macd / self._close_safe
        
        rsi_base = df['rsi_14'] if 'rsi_14' in df.columns else self._calculate_rsi(14)
        self.features_dict['rsi_normalized'] = ((rsi_base - 50) / 50)
    
    def build_session_temporal_features(self):
        df = self.df
        
        if isinstance(df.index, pd.DatetimeIndex):
            hour = df.index.hour
            day_of_week = df.index.dayofweek
        else:
            if 'timestamp' in df.columns:
                hour = pd.to_datetime(df['timestamp']).dt.hour
                day_of_week = pd.to_datetime(df['timestamp']).dt.dayofweek
            else:
                hour = pd.Series(0, index=df.index)
                day_of_week = pd.Series(0, index=df.index)
        
        self.features_dict['is_weekend'] = (day_of_week >= 5).astype(int)
        self.features_dict['is_american_session'] = ((hour >= 16) & (hour < 24)).astype(int)
        self.features_dict['is_european_session'] = ((hour >= 8) & (hour < 16)).astype(int)
        self.features_dict['is_asian_session'] = ((hour >= 0) & (hour < 8)).astype(int)
        
        self.features_dict['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        self.features_dict['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        self.features_dict['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        self.features_dict['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    
    def build_advanced_ratio_features(self):
        df = self.df
        f_dict = self.features_dict
        
        ret_std_7 = self._rolling(df['returns'], 7, 'std').replace(0, np.nan)
        ret_std_14 = self._rolling(df['returns'], 14, 'std').replace(0, np.nan)
        atr_pct_14 = df.get('atr_pct_14', df.get('atr_14') / df['close']).replace(0, np.nan)
        
        f_dict['sharpe_7d'] = df['close'].pct_change(7) / ret_std_7
        f_dict['sharpe_14d'] = df['close'].pct_change(14) / ret_std_14
        f_dict['momentum_volatility_ratio'] = df['close'].pct_change(7) / ret_std_14
        f_dict['risk_adjusted_momentum'] = df['close'].pct_change(14) / atr_pct_14
        
        trend_denom = df['close'].diff().abs().rolling(14).sum().replace(0, np.nan)
        adx = f_dict['adx_14'] if 'adx_14' in f_dict else self._calculate_adx(14)
        f_dict['normalized_trend_efficiency_index'] = ((df['close'] - df['close'].shift(14)) / trend_denom) * (adx / 25.0)
        
        vol_sma = f_dict.get('volume_sma', self._rolling(df['volume'], 14, 'mean')).replace(0, np.nan)
        intraday_mom = f_dict.get('intraday_momentum', (df['close']-df['open'])/(df['high']-df['low']).replace(0,np.nan))
        
        f_dict['volume_absorption_index'] = (df['volume'] / vol_sma) * (1 - intraday_mom.abs())
        
        hl_atr_ratio = (df['high'] - df['low']) / df.get('atr_14', 1).replace(0, np.nan)
        f_dict['relative_absorption_ratio'] = (df['volume'] / vol_sma) / hl_atr_ratio.replace(0, np.nan)
        
        vol_z = f_dict.get('volume_zscore', 0)
        volat_20 = f_dict.get('volatility_20', self._rolling(df['returns'], 20, 'std')).replace(0, np.nan)
        f_dict['volume_thrust_efficiency'] = (df.get('log_returns', 0) * vol_z) / volat_20
        
        if len(df) > 20:
            f_dict['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'].pct_change())
        else:
            f_dict['volume_price_correlation'] = 0
    
    def build_open_interest_features(self, window_short=1, window_long=24, atr_window=14):
        df = self.df
        if 'sum_open_interest' not in df.columns or df['sum_open_interest'].isna().all(): return
            
        oi = df['sum_open_interest']

        # Stationary transforms for open-interest level.
        oi_non_negative = oi.clip(lower=0)
        oi_log = np.log1p(oi_non_negative)
        self.features_dict['oi_log_diff_1'] = oi_log.diff(1)
        self.features_dict['oi_log_diff_24'] = oi_log.diff(24)
        oi_log_roll_48 = oi_log.rolling(48, min_periods=self._adaptive_min_periods(48))
        self.features_dict['oi_detrended_48'] = oi_log - oi_log_roll_48.mean()
        
        oi_change_short = oi.pct_change(window_short)
        oi_change_long = oi.pct_change(window_long)
        
        self.features_dict[f'oi_change_{window_short}_pct'] = oi_change_short
        self.features_dict[f'oi_change_{window_long}_pct'] = oi_change_long
        if window_short == 1:
            self.features_dict['oi_change_1h'] = oi_change_short
        if window_long == 24:
            self.features_dict['oi_change_24h'] = oi_change_long
        self.features_dict['oi_velocity'] = oi_change_short.diff(window_short)
        self.features_dict['oi_acceleration'] = self.features_dict['oi_velocity'].diff(window_short)
        
        if 'close' in df.columns:
            price_change = df['close'].pct_change(window_short)
            self.features_dict['oi_price_regime'] = np.sign(oi_change_short) * np.sign(price_change) * np.abs(oi_change_short)
        
        vol_safe = self._volume_safe 
        oi_vol_ratio = oi / vol_safe
        self.features_dict['oi_to_volume_ratio'] = oi_vol_ratio
        
        rolling_ratio = oi_vol_ratio.rolling(window_long, min_periods=self._adaptive_min_periods(window_long))
        self.features_dict['oi_to_volume_ratio_zscore'] = (oi_vol_ratio - rolling_ratio.mean()) / rolling_ratio.std().replace(0, np.nan)
        
        atr_col = f'atr_pct_{atr_window}'
        if 'volume_ratio' in df.columns and atr_col in df.columns:
            atr_safe = df[atr_col].replace(0, np.nan)
            self.features_dict['oi_volume_conviction_ratio'] = (oi_change_short * df['volume_ratio']) / atr_safe
            
        if 'fundingRate' in df.columns and df['fundingRate'].notna().sum() > 0:
            funding = df['fundingRate'].ffill()
            rolling_fund = funding.rolling(window_long, min_periods=self._adaptive_min_periods(window_long))
            fund_zscore = (funding - rolling_fund.mean()) / rolling_fund.std().replace(0, np.nan)
            self.features_dict['oi_funding_interaction'] = oi_change_long * fund_zscore
            
    def build_long_short_features(self, funding_windows=None, bb_windows=None):
        df = self.df
        if 'top_ls_ratio' not in df.columns or 'global_ls_ratio' not in df.columns: return
        if funding_windows is None:
            funding_windows = [12, 24, 48, 96]
        if bb_windows is None:
            bb_windows = [10, 20, 40, 80]
            
        top_ls = df['top_ls_ratio'].ffill()
        global_ls = df['global_ls_ratio'].ffill()
        
        ls_imbalance = top_ls - global_ls
        self.features_dict['ls_imbalance'] = ls_imbalance
        self.features_dict['ls_imbalance_velocity'] = ls_imbalance.diff()
        
        for bb_w in bb_windows:
            min_p = self._adaptive_min_periods(bb_w)
            top_ls_roll = top_ls.rolling(window=bb_w, min_periods=min_p)
            global_ls_roll = global_ls.rolling(window=bb_w, min_periods=min_p)
            
            self.features_dict[f'top_ls_ratio_zscore_{bb_w}'] = (top_ls - top_ls_roll.mean()) / top_ls_roll.std().replace(0, np.nan)
            self.features_dict[f'global_ls_ratio_zscore_{bb_w}'] = (global_ls - global_ls_roll.mean()) / global_ls_roll.std().replace(0, np.nan)
            
            if 'close' in df.columns and 'volume_zscore' in self.features_dict:
                bb_sma = df['close'].rolling(window=bb_w).mean()
                bb_std = df['close'].rolling(window=bb_w).std().replace(0, np.nan)
                bb_lower = bb_sma - 2 * bb_std
                bb_pos = (df['close'] - bb_lower) / (4 * bb_std).replace(0, np.nan)
                self.features_dict[f'whale_mean_reversion_bias_{bb_w}'] = ls_imbalance * (0.5 - bb_pos) * (1 + self.features_dict['volume_zscore'])

        if 'fundingRate' in df.columns and df['fundingRate'].notna().any():
            funding = df['fundingRate'].ffill()
            for fund_w in funding_windows:
                fund_roll = funding.rolling(window=fund_w, min_periods=self._adaptive_min_periods(fund_w))
                fund_zscore = (funding - fund_roll.mean()) / fund_roll.std().replace(0, np.nan)
                self.features_dict[f'ls_funding_alignment_{fund_w}'] = ls_imbalance * fund_zscore
    
    def build_additional_bb_features(self):
        df = self.df
        
        sma = self._rolling(df['close'], 20, 'mean')
        std = self._rolling(df['close'], 20, 'std')
        bb_upper = sma + 2*std
        bb_lower = sma - 2*std
        
        self.features_dict['bb_position_20'] = (df['close'] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        self.features_dict['bb_distance_upper_pct_20'] = (df['close'] - bb_upper) / bb_upper.replace(0, np.nan)
        self.features_dict['bb_distance_lower_pct_20'] = (df['close'] - bb_lower) / bb_lower.replace(0, np.nan)
        self.features_dict['bb_width_pct_20'] = (bb_upper - bb_lower) / sma.replace(0, np.nan)
        
        bb_width = bb_upper - bb_lower
        self.features_dict['bb_squeeze_20'] = (bb_width < bb_width.rolling(20).quantile(0.1)).astype(int)
    
    def build_advanced_volatility_features(self):
        df = self.df
        
        if 'atr_pct_14' in df.columns:
            atr_pct = df['atr_pct_14']
        else:
            atr_pct = self._true_range().rolling(14).mean() / self._close_safe
        
        self.features_dict['volatility_percentile_14_30'] = atr_pct.rolling(30).rank(pct=True)
        self.features_dict['volatility_regime_14_30'] = (atr_pct > self._rolling(atr_pct, 30, 'mean')).astype(int)
        
        returns_up = df['returns'].clip(lower=0).rolling(20).std()
        returns_down = df['returns'].clip(upper=0).abs().rolling(20).std()
        returns_std = self._rolling(df['returns'], 20, 'std').replace(0, np.nan)
        self.features_dict['asymmetric_volatility_index'] = (returns_up - returns_down) / returns_std
        
        rsi_source = df['rsi_14'] if 'rsi_14' in df.columns else self._calculate_rsi(14)
        rsi_norm = (rsi_source - 50) / 50
        bb_width = df.get('bb_width', (self._rolling(df['close'], 20, 'mean') + 2*self._rolling(df['close'], 20, 'std')) - (self._rolling(df['close'], 20, 'mean') - 2*self._rolling(df['close'], 20, 'std')))
        squeeze_ratio = bb_width / self._rolling(bb_width, 20, 'mean').replace(0, np.nan)
        squeeze_compression = 1.0 / (1.0 + squeeze_ratio.abs())
        
        self.features_dict['momentum_conviction_index'] = (
            rsi_norm * (df['volume'] / df.get('volume_sma', self._rolling(df['volume'], 14, 'mean')).replace(0, np.nan)) * squeeze_compression
        )
    
    def build_volatility_weighted_features(self):
        df = self.df
        f_dict = self.features_dict
        
        atr_7 = f_dict.get('atr_7', df.get('atr_7', 1))
        atr_21 = f_dict.get('atr_21', df.get('atr_21', 1)).replace(0, np.nan)
        vol_ratio = atr_7 / atr_21
        
        body_size = f_dict.get('body_size_pct', np.abs(df['close']-df['open'])/df['close'].replace(0,np.nan)).replace(0, np.nan)
        lower_shadow = f_dict.get('lower_shadow_pct', 0)
        upper_shadow = f_dict.get('upper_shadow_pct', 0)
        
        self.features_dict['volatility_weighted_shadow_imbalance'] = ((lower_shadow - upper_shadow) / body_size) * vol_ratio
    
    def build_cumulative_features(self):
        df = self.df
        
        direction = np.sign(df['close'].diff()).fillna(0)
        obv = (df['volume'] * direction).cumsum()
        obv_std = self._rolling(obv, 30, 'std').replace(0, np.nan)
        self.features_dict['obv_normalized'] = obv / obv_std
        
        self.features_dict['volume_percentile_30d'] = df['volume'].rolling(30).rank(pct=True)
        
        hl_range = self._hl_range_safe
        mf_mult = ((df['close'] - df['low']) - (df['high'] - df['close'])) / hl_range
        mf_vol = mf_mult * df['volume']
        self.features_dict['cmf_20'] = self._rolling(mf_vol, 20, 'sum') / self._rolling(df['volume'], 20, 'sum').replace(0, np.nan)
    
    def build_additional_funding_features(self):
        df = self.df
        if 'fundingRate' not in df.columns or df['fundingRate'].isna().all(): return
        
        funding = df['fundingRate'].ffill()
        roll_30 = funding.rolling(30, min_periods=self._adaptive_min_periods(30))
        roll_14 = funding.rolling(14, min_periods=self._adaptive_min_periods(14))
        fund_diff = funding.diff()
        roll_diff_7 = fund_diff.rolling(7, min_periods=self._adaptive_min_periods(7))
        
        self.features_dict['funding_extreme_long'] = (funding > roll_30.quantile(0.9)).astype(int)
        self.features_dict['funding_extreme_short'] = (funding < roll_30.quantile(0.1)).astype(int)
        self.features_dict['funding_percentile'] = roll_30.rank(pct=True)
        self.features_dict['funding_zscore'] = (funding - roll_14.mean()) / roll_14.std().replace(0, np.nan)
        self.features_dict['funding_change_zscore'] = (fund_diff - roll_diff_7.mean()) / roll_diff_7.std().replace(0, np.nan)
    
    def build_squeeze_and_depletion_features(self):
        df = self.df
        
        bb_width = df.get('bb_width', 0)
        bb_width_sma = self._rolling(bb_width, 20, 'mean').replace(0, np.nan)
        self.features_dict['squeeze_ratio'] = bb_width / bb_width_sma
        
        vol_sma = self._rolling(df['volume'], 20, 'mean').replace(0, np.nan)
        self.features_dict['vol_depletion'] = df['volume'] / vol_sma
        
        squeeze_compression = 1.0 / (1.0 + self.features_dict['squeeze_ratio'].abs())
        volume_dryup = 1.0 / (1.0 + self.features_dict['vol_depletion'].abs())
        self.features_dict['pre_ignition_score'] = squeeze_compression + volume_dryup
    
    def build_price_extension_features(self):
        df = self.df
        f_dict = self.features_dict
        
        ema_21 = (df['ema_21'] if 'ema_21' in df.columns else self._ema(21)).replace(0, np.nan)
        dist_ema_21_pct = (df['close'] - ema_21) / ema_21
        
        atr_21 = (df['atr_21'] if 'atr_21' in df.columns else self._true_range().rolling(21).mean()).replace(0, np.nan)
        vol_ratio = f_dict.get('vol_ratio_alpha', df.get('atr_7', 1) / atr_21).replace(0, np.nan)
        is_weekend = f_dict.get('is_weekend', pd.Series(0, index=df.index))
        
        self.features_dict['weekend_volatility_exhaustion_ratio'] = (dist_ema_21_pct.abs() * is_weekend) / vol_ratio
        
        is_amer = f_dict.get('is_american_session', 0)
        is_euro = f_dict.get('is_european_session', 0)
        roc_14 = f_dict.get('roc_14', df['close'].pct_change(14))
        vol_ratio_df = df.get('volume_ratio', 1)
        
        self.features_dict['session_momentum_efficiency_index'] = (roc_14 * vol_ratio_df) * (1.0 + 0.5 * is_amer + 0.25 * is_euro)
        
        atr_14 = (df['atr_14'] if 'atr_14' in df.columns else self._true_range().rolling(14).mean()).replace(0, np.nan)
        ema_50 = df['ema_50'] if 'ema_50' in df.columns else self._ema(50)
        self.features_dict['dist_to_ema50_atr'] = (df['close'] - ema_50) / atr_14
    
    def build_ranking_features(self):
        df = self.df
        self.features_dict['rsi_rank_pct'] = pd.Series(0.5, index=df.index)
        self.features_dict['oi_growth_rank_pct'] = pd.Series(0.5, index=df.index)
        self.features_dict['volatility_rank_pct'] = pd.Series(0.5, index=df.index)
        self.features_dict['momentum_rank_pct'] = pd.Series(0.5, index=df.index)
    
    def build_leverage_tension_features(self):
        df = self.df
        if 'fundingRate' not in df.columns or df['fundingRate'].isna().all(): return
        
        funding = df['fundingRate'].ffill()
        ema_21 = (df['ema_21'] if 'ema_21' in df.columns else self._ema(21)).replace(0, np.nan)
        dist_ema_21_pct = (df['close'] - ema_21) / ema_21
        atr_pct_14 = df.get('atr_pct_14', 1).replace(0, np.nan)
        
        fund_std = funding.rolling(24, min_periods=self._adaptive_min_periods(24)).std().replace(0, np.nan)
        self.features_dict['leverage_pnl_tension_index'] = (funding / fund_std) * (dist_ema_21_pct / atr_pct_14)
        self.features_dict['normalized_funding_momentum_shock'] = (funding.diff(1) / fund_std) / atr_pct_14
    
    def build_relative_performance_features(self):
        df = self.df
        momentumrank = df['close'].pct_change(7)
        volatrank = self._rolling(df['returns'], 14, 'std')
        adx = self.features_dict['adx_14'] if 'adx_14' in self.features_dict else self._calculate_adx(14)
        
        self.features_dict['relative_momentum_quality'] = ((momentumrank.rank(pct=True) - volatrank.rank(pct=True)) * (adx / 100.0))

    def build_btc_context_features(self):
        df = self.df

        own_log_returns = df.get('log_returns', np.log1p(df['close'].pct_change()))
        btc_ref = self._resolve_btc_reference_frame()

        if btc_ref is None or 'close' not in btc_ref.columns:
            zeros = pd.Series(0.0, index=df.index)
            self.features_dict['btc_is_bull_regime'] = pd.Series(0, index=df.index)
            self.features_dict['btc_trend_strength'] = pd.Series(0, index=df.index)
            self.features_dict['rs_vs_btc'] = zeros
            self.features_dict['rs_vs_btc_sma7'] = zeros
            self.features_dict['btc_corr_24'] = zeros
            self.features_dict['btc_beta_48'] = zeros
            self.features_dict['idiosyncratic_return'] = zeros
            return

        btc_close = btc_ref['close']
        btc_log_returns = btc_ref.get('log_returns', np.log1p(btc_close.pct_change()))
        btc_log_returns = btc_log_returns.replace([np.inf, -np.inf], np.nan)

        btc_ema_200 = btc_close.ewm(span=200, adjust=False).mean()
        btc_adx_14 = btc_ref.get('adx_14')
        if btc_adx_14 is None and {'high', 'low', 'close'}.issubset(set(btc_ref.columns)):
            btc_adx_14 = self._calculate_adx_from_ohlc(
                high=btc_ref['high'],
                low=btc_ref['low'],
                close=btc_ref['close'],
                period=14
            )

        if btc_adx_14 is None:
            btc_adx_14 = (
                btc_log_returns.abs().rolling(14).mean() /
                btc_log_returns.abs().rolling(56).mean().replace(0, np.nan)
            ) * 25.0

        rs_vs_btc = own_log_returns - btc_log_returns
        corr_window = 24
        beta_window = 48

        self.features_dict['btc_is_bull_regime'] = (btc_close > btc_ema_200).astype(int)
        self.features_dict['btc_trend_strength'] = (btc_adx_14 > 25).astype(int)
        self.features_dict['rs_vs_btc'] = rs_vs_btc
        self.features_dict['rs_vs_btc_sma7'] = self._rolling(rs_vs_btc, 7, 'mean')

        self.features_dict['btc_corr_24'] = own_log_returns.rolling(
            corr_window,
            min_periods=self._adaptive_min_periods(corr_window)
        ).corr(btc_log_returns)

        cov = own_log_returns.rolling(
            beta_window,
            min_periods=self._adaptive_min_periods(beta_window)
        ).cov(btc_log_returns)
        btc_var = btc_log_returns.rolling(
            beta_window,
            min_periods=self._adaptive_min_periods(beta_window)
        ).var().replace(0, np.nan)
        beta = cov / btc_var

        self.features_dict['btc_beta_48'] = beta
        self.features_dict['idiosyncratic_return'] = own_log_returns - (beta * btc_log_returns)
    
    def build_trend_state_features(self):
        df = self.df
        adx = self.features_dict['adx_14'] if 'adx_14' in self.features_dict else self._calculate_adx(14)
        sma_50 = df.get('sma_50', self._rolling(df['close'], 50, 'mean'))
        
        trend_state = pd.Series(-1, index=df.index)
        trend_state[df['close'] > sma_50] = 1
        trend_state[adx < 20] = 0
        self.features_dict['trend_state'] = trend_state
        
        vol_sma_14 = df.get('volume_sma', self._rolling(df['volume'], 14, 'mean')).replace(0, np.nan)
        self.features_dict['vol_acceleration'] = df['volume'].diff().diff() / vol_sma_14
    
    def build_speculative_features(self):
        df = self.df
        f_dict = self.features_dict
        
        oi_vol_zscore = f_dict.get('oi_to_volume_ratio_zscore', pd.Series(0.0, index=df.index)).fillna(0)
        funding_zscore = f_dict.get('funding_zscore', pd.Series(0.0, index=df.index)).fillna(0)
        squeeze_ratio = f_dict.get('squeeze_ratio', pd.Series(1.0, index=df.index)).replace(0, np.nan)
        
        self.features_dict['speculative_congestion_index'] = (oi_vol_zscore * funding_zscore) / squeeze_ratio
        
        if 'fundingRate' in df.columns and df['fundingRate'].notna().any():
            vol_zscore = f_dict.get('volume_zscore', 0)
            self.features_dict['volume_funding_divergence'] = vol_zscore * np.sign(df['fundingRate'])
    
    # ==================== MULTI-PERIOD & MULTI-PARAMETER FEATURES ====================
    def build_multi_period_rsi(self):
        periods = [7, 14, 21, 28]
        for period in periods:
            self.features_dict[f'rsi_{period}'] = self._calculate_rsi(period)
    
    def build_multi_period_atr(self):
        df = self.df
        tr = self._true_range()
        periods = [7, 14, 21, 28]
        for period in periods:
            atr = self._rolling(tr, period, 'mean')
            self.features_dict[f'atr_{period}'] = atr
            self.features_dict[f'atr_pct_{period}'] = atr / self._close_safe
    
    def build_multi_period_ema(self):
        periods = [7, 12, 21, 50, 100, 200]
        for period in periods:
            self.features_dict[f'ema_{period}'] = self._ema(period)
    
    def build_multi_period_sma(self):
        df = self.df
        periods = [10, 20, 50, 100, 200]
        for period in periods:
            self.features_dict[f'sma_{period}'] = self._rolling(df['close'], period, 'mean')
    
    def build_multi_period_bollinger(self):
        df = self.df
        periods = [20, 30, 50]
        for period in periods:
            sma = self._rolling(df['close'], period, 'mean')
            std = self._rolling(df['close'], period, 'std')
            self.features_dict[f'bb_upper_{period}'] = sma + 2 * std
            self.features_dict[f'bb_lower_{period}'] = sma - 2 * std
            self.features_dict[f'bb_mid_{period}'] = sma
            self.features_dict[f'bb_width_{period}'] = 4 * std
            self.features_dict[f'bb_position_{period}'] = (df['close'] - (sma - 2*std)) / (4*std).replace(0, np.nan)
    # ==================== SMART MONEY & MECHANICS FEATURES ====================
    def build_smart_money_mechanics(self):
        df = self.df
        f_dict = self.features_dict
        import numpy as np
        atr_14 = f_dict.get('atr_14', self._true_range().rolling(14).mean()).replace(0, np.nan)
        bull_fvg_gap = df['low'] - df['high'].shift(2)
        bear_fvg_gap = df['low'].shift(2) - df['high']
        bull_fvg = np.where(bull_fvg_gap > 0, bull_fvg_gap, 0)
        bear_fvg = np.where(bear_fvg_gap > 0, bear_fvg_gap, 0)
        f_dict['fvg_imbalance_score'] = (bull_fvg - bear_fvg) / atr_14
        body = np.abs(df['close'] - df['open'])
        hl_range = self._hl_range_safe
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        if 'volume_zscore' in f_dict:
            vol_z = f_dict['volume_zscore']
        else:
            vol_sma = self._rolling(df['volume'], 20, 'mean').replace(0, np.nan)
            vol_z = (df['volume'] - vol_sma) / self._rolling(df['volume'], 20, 'std').replace(0, np.nan)
        is_bull_sweep = (lower_shadow > 2 * body) & (lower_shadow > 0.5 * hl_range) & (vol_z > 0.5)
        is_bear_sweep = (upper_shadow > 2 * body) & (upper_shadow > 0.5 * hl_range) & (vol_z > 0.5)
        f_dict['liquidity_sweep_score'] = np.where(is_bull_sweep, 1, np.where(is_bear_sweep, -1, 0)) * vol_z
        if 'sum_open_interest' in df.columns:
            oi_diff = df['sum_open_interest'].diff()
            vol_safe = self._volume_safe
            f_dict['oi_vol_divergence_ratio'] = oi_diff / vol_safe

        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            hours = timestamps.dt.hour
        else:
            if getattr(df.index, 'tz', None) is None:
                try:
                    dt_index = pd.to_datetime(df.index, utc=True)
                    hours = dt_index.hour
                except Exception:
                    hours = np.zeros(len(df))
            else:
                hours = df.index.tz_convert('UTC').hour

        hours_to_func = 8 - (hours % 8)
        hours_to_func = np.where(hours_to_func == 8, 0, hours_to_func)
        f_dict['hours_to_funding_trap'] = hours_to_func

    def build_multi_period_roc(self):
        df = self.df
        periods = [7, 12, 24]
        for period in periods:
            self.features_dict[f'roc_{period}'] = df['close'].pct_change(period)
    
    def build_multi_period_cci(self):
        df = self.df
        tp = (df['high'] + df['low'] + df['close']) / 3
        periods = [14, 20, 30]
        for period in periods:
            sma_tp = self._rolling(tp, period, 'mean')
            mad = self._rolling_mad_fast(tp, period)
            self.features_dict[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)
    
    def build_multi_period_volume(self):
        df = self.df
        periods = [7, 14, 20, 30]
        for period in periods:
            vol_ma = self._rolling(df['volume'], period, 'mean')
            self.features_dict[f'volume_ma_{period}'] = vol_ma
            self.features_dict[f'volume_ratio_{period}'] = df['volume'] / vol_ma.replace(0, np.nan)
            self.features_dict[f'volume_zscore_{period}'] = (df['volume'] - vol_ma) / self._rolling(df['volume'], period, 'std').replace(0, np.nan)
    
    def build_multi_period_volatility(self):
        df = self.df
        returns = df['close'].pct_change()
        periods = [7, 14, 21, 30]
        for period in periods:
            std_roll = self._rolling(returns, period, 'std')
            self.features_dict[f'volatility_{period}'] = std_roll
            self.features_dict[f'volatility_{period}_zscore'] = (
                (std_roll - self._rolling(std_roll, period, 'mean')) / self._rolling(std_roll, period, 'std').replace(0, np.nan)
            )
    
    def build_multi_parameter_oscillators(self):
        macd_params = [(12, 26, 9), (5, 35, 5), (10, 20, 9)]
        for fast, slow, signal in macd_params:
            ema_fast = self._ema(fast)
            ema_slow = self._ema(slow)
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            self.features_dict[f'macd_{fast}_{slow}_{signal}'] = macd
            self.features_dict[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            self.features_dict[f'macd_hist_{fast}_{slow}_{signal}'] = macd - macd_signal
        
        for period in [7, 14, 21]:
            rsi_key = f'rsi_{period}'
            rsi = self.features_dict[rsi_key] if rsi_key in self.features_dict else self._calculate_rsi(period)
            rsi_low = self._rolling(rsi, period, 'min')
            rsi_high = self._rolling(rsi, period, 'max')
            self.features_dict[f'stoch_rsi_{period}'] = (rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
    
    # ==================== HELPER METHODS ====================
    def _true_range(self) -> pd.Series:
        if self._tr_cache is None:
            df = self.df
            tr1 = df['high'] - df['low']
            tr2 = np.abs(df['high'] - df['close'].shift(1))
            tr3 = np.abs(df['low'] - df['close'].shift(1))
            self._tr_cache = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return self._tr_cache
    
    def _calculate_rsi(self, period: int) -> pd.Series:
        cached = self._rsi_cache.get(period)
        if cached is not None:
            return cached

        df = self.df
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        self._rsi_cache[period] = rsi
        return rsi
    
    def _calculate_adx(self, period: int) -> pd.Series:
        cached = self._adx_cache.get(period)
        if cached is not None:
            return cached

        df = self.df
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        # Native Pandas để tránh ndarray Error
        plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=df.index)
        minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=df.index)
        
        tr = self._true_range()
        atr = self._rolling(tr, period, 'mean').replace(0, np.nan)
        
        plus_di = 100 * self._rolling(plus_dm, period, 'mean') / atr
        minus_di = 100 * self._rolling(minus_dm, period, 'mean') / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        adx = self._rolling(dx, period, 'mean')
        self._adx_cache[period] = adx
        return adx

    def _ema(self, span: int) -> pd.Series:
        cached = self._ema_cache.get(span)
        if cached is not None:
            return cached
        ema = self.df['close'].ewm(span=span, adjust=False).mean()
        self._ema_cache[span] = ema
        return ema

    @staticmethod
    def _adaptive_min_periods(window: int, ratio: float = 0.35, min_floor: int = 3) -> int:
        return max(min_floor, int(window * ratio))

    def _rolling_mad_fast(self, series: pd.Series, window: int) -> pd.Series:
        # approx MAD
        return self._rolling(series, window, 'std') * 0.79788

    @staticmethod
    def _rolling_zscore(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
        if min_periods is None:
            min_periods = window
        roll = series.rolling(window=window, min_periods=min_periods)
        mean = roll.mean()
        std = roll.std().replace(0, np.nan)
        return (series - mean) / std

    @staticmethod
    def _normalize_timestamp_series(ts: pd.Series) -> pd.Series:
        normalized = pd.to_datetime(ts, errors='coerce', utc=True)
        if isinstance(normalized, pd.Series):
            return normalized.dt.tz_localize(None)
        return pd.Series(normalized.tz_localize(None))

    def _resolve_btc_reference_frame(self) -> Optional[pd.DataFrame]:
        df = self.df
        ref = pd.DataFrame(index=df.index)

        prefixed = {
            'open': 'btc_open',
            'high': 'btc_high',
            'low': 'btc_low',
            'close': 'btc_close',
            'log_returns': 'btc_log_returns',
            'adx_14': 'btc_adx_14'
        }

        for key, col in prefixed.items():
            if col in df.columns:
                ref[key] = df[col]

        if 'close' in ref.columns and ref['close'].notna().any():
            return ref

        source = self.btc_df
        if source is None and 'symbol' in df.columns:
            symbol_upper = df['symbol'].astype(str).str.upper()
            btc_mask = symbol_upper.isin({'BTC', 'BTCUSDT', 'BTCUSDT_USDT', 'BTC/USDT', 'XBTUSDT'})
            if btc_mask.any():
                source = df.loc[btc_mask].copy()

        if source is None or source.empty or 'close' not in source.columns:
            return None

        base_ts = self._extract_timestamp_series(df)
        if base_ts is None:
            return None

        src_ts = self._extract_timestamp_series(source)
        if src_ts is None:
            return None

        source = source.copy()
        source['__ts'] = src_ts
        source = source.dropna(subset=['__ts']).sort_values('__ts').drop_duplicates('__ts', keep='last')
        if source.empty:
            return None

        source_map = source.set_index('__ts')
        for col in ['open', 'high', 'low', 'close', 'log_returns', 'adx_14']:
            source_col = col
            if source_col not in source_map.columns and col == 'adx_14' and 'adx' in source_map.columns:
                source_col = 'adx'
            if source_col in source_map.columns:
                ref[col] = base_ts.map(source_map[source_col]).ffill()

        if 'close' not in ref.columns or ref['close'].isna().all():
            return None

        return ref

    def _extract_timestamp_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if 'timestamp' in df.columns:
            return self._normalize_timestamp_series(df['timestamp'])

        if isinstance(df.index, pd.DatetimeIndex):
            return self._normalize_timestamp_series(pd.Series(df.index, index=df.index))

        return None

    def _calculate_adx_from_ohlc(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0), index=high.index)
        minus_dm = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0), index=high.index)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = self._rolling(tr, period, 'mean').replace(0, np.nan)
        plus_di = 100 * self._rolling(plus_dm, period, 'mean') / atr
        minus_di = 100 * self._rolling(minus_dm, period, 'mean') / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return self._rolling(dx, period, 'mean')

    def _apply_entry_t_minus_1_lag(self):
        engineered_cols = [c for c in self.df.columns if c not in self._input_columns]
        if not engineered_cols:
            return

        if 'symbol' in self.df.columns:
            self.df[engineered_cols] = self.df.groupby('symbol', group_keys=False)[engineered_cols].shift(1)
            return

        self.df[engineered_cols] = self.df[engineered_cols].shift(1)


_GLOBAL_BTC_DF = None

def create_market_features(df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
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
    global _GLOBAL_BTC_DF
    
    # Auto-load BTC dataframe if not provided and current df is not BTC
    if btc_df is None:
        is_btc = False
        if 'symbol' in df.columns and len(df) > 0:
            sym = str(df['symbol'].iloc[0]).upper()
            if 'BTC' in sym:
                is_btc = True
                
        if not is_btc:
            if _GLOBAL_BTC_DF is None:
                try:
                    from pathlib import Path
                    btc_path = Path('data/ohlcv/BTC_USDT.parquet')
                    if btc_path.exists():
                        _GLOBAL_BTC_DF = pd.read_parquet(btc_path)
                except Exception as e:
                    print(f"Could not auto-load BTC reference data: {e}")
            
            btc_df = _GLOBAL_BTC_DF

    builder = FeatureBuilder(df, btc_df=btc_df)
    return builder.build_all()


    