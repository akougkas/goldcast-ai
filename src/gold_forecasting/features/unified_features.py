"""
Unified feature engineering module for gold price forecasting.
Supports both CPU and GPU pipelines with comprehensive feature generation.
"""


import numpy as np
import pandas as pd
import structlog

try:
    import cudf
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cudf = None

logger = structlog.get_logger(__name__)

class UnifiedFeatureEngineer:
    """
    Unified feature engineering class that generates all required features
    for both CPU and GPU training pipelines.
    """

    def __init__(self, use_gpu: bool = None):
        """
        Initialize the feature engineer.

        Args:
            use_gpu: Whether to use GPU acceleration. If None, auto-detect.
        """
        if use_gpu is None:
            self.use_gpu = HAS_GPU
        else:
            self.use_gpu = use_gpu and HAS_GPU

        if self.use_gpu and not HAS_GPU:
            logger.warning("GPU requested but RAPIDS not available, falling back to CPU")
            self.use_gpu = False

        logger.info(f"FeatureEngineer initialized with {'GPU' if self.use_gpu else 'CPU'} acceleration")

    def create_lagged_features(self, data: pd.DataFrame, columns: list[str],
                              lags: list[int]) -> pd.DataFrame:
        """
        Create lagged features for specified columns.
        
        Args:
            data: Input DataFrame
            columns: List of columns to create lags for
            lags: List of lag values
            
        Returns:
            DataFrame with lagged features
        """
        result = data.copy()

        for col in columns:
            if col not in data.columns:
                logger.warning(f"Column {col} not found in data, skipping")
                continue

            for lag in lags:
                feature_name = f'{col}_lag_{lag}'
                if self.use_gpu and isinstance(data, pd.DataFrame):
                    # Use cupy for GPU acceleration
                    col_values = cp.asarray(data[col].values)
                    lagged = cp.concatenate([cp.full(lag, cp.nan), col_values[:-lag]])
                    result[feature_name] = cp.asnumpy(lagged)
                else:
                    # Standard pandas shift
                    result[feature_name] = result[col].shift(lag)

        return result

    def create_rolling_statistics(self, data: pd.DataFrame, columns: list[str],
                                 windows: list[int]) -> pd.DataFrame:
        """
        Create comprehensive rolling statistics.
        
        Args:
            data: Input DataFrame
            columns: List of columns to create rolling stats for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling statistics
        """
        result = data.copy()

        for col in columns:
            if col not in data.columns:
                logger.warning(f"Column {col} not found in data, skipping")
                continue

            for window in windows:
                # Basic rolling statistics
                result[f'{col}_rolling_mean_{window}'] = data[col].rolling(window, min_periods=1).mean()
                result[f'{col}_rolling_std_{window}'] = data[col].rolling(window, min_periods=1).std()
                result[f'{col}_rolling_min_{window}'] = data[col].rolling(window, min_periods=1).min()
                result[f'{col}_rolling_max_{window}'] = data[col].rolling(window, min_periods=1).max()

                # Robust statistics
                result[f'{col}_rolling_median_{window}'] = data[col].rolling(window, min_periods=1).median()
                result[f'{col}_rolling_iqr_{window}'] = (
                    data[col].rolling(window, min_periods=1).quantile(0.75) -
                    data[col].rolling(window, min_periods=1).quantile(0.25)
                )

                # Higher moments for financial data
                result[f'{col}_rolling_skew_{window}'] = data[col].rolling(window, min_periods=1).skew()
                result[f'{col}_rolling_kurt_{window}'] = data[col].rolling(window, min_periods=1).kurt()

                # Volatility measures
                result[f'{col}_rolling_volatility_{window}'] = data[col].rolling(window, min_periods=1).std() * np.sqrt(window)

        return result

    def create_technical_indicators(self, data: pd.DataFrame, price_col: str,
                                   windows: list[int]) -> pd.DataFrame:
        """
        Create technical indicators for financial analysis.
        
        Args:
            data: Input DataFrame
            price_col: Name of the price column
            windows: List of window sizes for indicators
            
        Returns:
            DataFrame with technical indicators
        """
        result = data.copy()

        if price_col not in data.columns:
            logger.warning(f"Price column {price_col} not found, skipping technical indicators")
            return result

        prices = data[price_col]

        for window in windows:
            # Simple Moving Average
            sma = prices.rolling(window, min_periods=1).mean()
            result[f'{price_col}_sma_{window}'] = sma

            # Exponential Moving Average
            result[f'{price_col}_ema_{window}'] = prices.ewm(span=window, min_periods=1).mean()

            # Bollinger Bands
            rolling_mean = sma
            rolling_std = prices.rolling(window, min_periods=1).std()
            result[f'{price_col}_bb_upper_{window}'] = rolling_mean + (rolling_std * 2)
            result[f'{price_col}_bb_lower_{window}'] = rolling_mean - (rolling_std * 2)
            result[f'{price_col}_bb_width_{window}'] = result[f'{price_col}_bb_upper_{window}'] - result[f'{price_col}_bb_lower_{window}']

            # RSI (Relative Strength Index)
            rsi_min_window = 14
            if window >= rsi_min_window:  # RSI typically uses 14 periods
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                rs = gain / loss
                result[f'{price_col}_rsi_{window}'] = 100 - (100 / (1 + rs))

            # Price momentum
            result[f'{price_col}_momentum_{window}'] = prices.pct_change(window)

            # Price position within rolling range
            rolling_min = prices.rolling(window, min_periods=1).min()
            rolling_max = prices.rolling(window, min_periods=1).max()
            result[f'{price_col}_position_{window}'] = (prices - rolling_min) / (rolling_max - rolling_min + 1e-8)

        return result

    def create_returns_and_volatility(self, data: pd.DataFrame, price_col: str,
                                     windows: list[int]) -> pd.DataFrame:
        """
        Create returns and volatility measures.
        
        Args:
            data: Input DataFrame
            price_col: Name of the price column
            windows: List of window sizes
            
        Returns:
            DataFrame with returns and volatility features
        """
        result = data.copy()

        if price_col not in data.columns:
            logger.warning(f"Price column {price_col} not found, skipping returns/volatility")
            return result

        # Calculate returns
        returns = data[price_col].pct_change(fill_method=None)
        result[f'{price_col}_return'] = returns

        # Log returns
        result[f'{price_col}_log_return'] = np.log(data[price_col] / data[price_col].shift(1))

        for window in windows:
            # Volatility clustering
            result[f'{price_col}_vol_cluster_{window}'] = returns.rolling(window, min_periods=1).std()

            # Downside risk (volatility of negative returns)
            negative_returns = returns.where(returns < 0, 0)
            result[f'{price_col}_downside_risk_{window}'] = negative_returns.rolling(window, min_periods=1).std()

            # Upside potential (volatility of positive returns)
            positive_returns = returns.where(returns > 0, 0)
            result[f'{price_col}_upside_potential_{window}'] = positive_returns.rolling(window, min_periods=1).std()

            # Value at Risk approximation (5th percentile of returns)
            result[f'{price_col}_var_5_{window}'] = returns.rolling(window, min_periods=1).quantile(0.05)

            # Conditional Value at Risk (mean of returns below VaR)
            result[f'{price_col}_cvar_5_{window}'] = returns.rolling(window, min_periods=1).apply(
                lambda x: x[x <= x.quantile(0.05)].mean() if len(x[x <= x.quantile(0.05)]) > 0 else 0,
                raw=False
            )

            # Maximum drawdown
            cumulative_returns = (1 + returns).rolling(window, min_periods=1).apply(lambda x: x.prod(), raw=False)
            rolling_max = cumulative_returns.rolling(window, min_periods=1).max()
            result[f'{price_col}_max_drawdown_{window}'] = (cumulative_returns - rolling_max) / rolling_max

        return result

    def create_interaction_features(self, data: pd.DataFrame, price_col: str,
                                   other_cols: list[str]) -> pd.DataFrame:
        """
        Create interaction features between price and other economic indicators.
        
        Args:
            data: Input DataFrame
            price_col: Name of the price column
            other_cols: List of other columns to create interactions with
            
        Returns:
            DataFrame with interaction features
        """
        result = data.copy()

        if price_col not in data.columns:
            logger.warning(f"Price column {price_col} not found, skipping interactions")
            return result

        for col in other_cols:
            if col not in data.columns:
                logger.warning(f"Column {col} not found, skipping interaction")
                continue

            # Multiplicative interaction
            result[f'{price_col}_x_{col}'] = data[price_col] * data[col]

            # Ratio interaction
            result[f'{price_col}_ratio_{col}'] = data[price_col] / (data[col] + 1e-8)

            # Difference interaction
            result[f'{price_col}_diff_{col}'] = data[price_col] - data[col]

            # Correlation-based interaction (rolling correlation)
            for window in [21, 63]:
                corr = data[price_col].rolling(window, min_periods=1).corr(data[col])
                result[f'{price_col}_corr_{col}_{window}'] = corr

        return result

    def create_calendar_features(self, data: pd.DataFrame, date_col: str = 'time') -> pd.DataFrame:
        """
        Create calendar-based features.
        
        Args:
            data: Input DataFrame
            date_col: Name of the date column
            
        Returns:
            DataFrame with calendar features
        """
        result = data.copy()

        if date_col not in data.columns:
            logger.warning(f"Date column {date_col} not found, skipping calendar features")
            return result

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            result[date_col] = pd.to_datetime(data[date_col])

        dt = result[date_col]

        # Basic calendar features
        result['year'] = dt.dt.year
        result['month'] = dt.dt.month
        result['day'] = dt.dt.day
        result['day_of_week'] = dt.dt.dayofweek
        result['day_of_year'] = dt.dt.dayofyear
        result['week_of_year'] = dt.dt.isocalendar().week
        result['quarter'] = dt.dt.quarter

        # Cyclical encoding for periodic features
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_year_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365.25)
        result['day_of_year_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365.25)

        # Financial calendar features
        result['is_month_start'] = dt.dt.is_month_start.astype(int)
        result['is_month_end'] = dt.dt.is_month_end.astype(int)
        result['is_quarter_start'] = dt.dt.is_quarter_start.astype(int)
        result['is_quarter_end'] = dt.dt.is_quarter_end.astype(int)
        result['is_year_start'] = dt.dt.is_year_start.astype(int)
        result['is_year_end'] = dt.dt.is_year_end.astype(int)

        return result

    def create_target_variable(self, data: pd.DataFrame, price_col: str,
                              target_type: str = 'return', horizon: int = 1) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            data: Input DataFrame
            price_col: Name of the price column
            target_type: Type of target ('return', 'price', 'log_return')
            horizon: Prediction horizon (periods ahead)
            
        Returns:
            DataFrame with target variable
        """
        result = data.copy()

        if price_col not in data.columns:
            logger.warning(f"Price column {price_col} not found, cannot create target")
            return result

        prices = data[price_col]

        if target_type == 'return':
            # Next period return
            result['target'] = prices.pct_change(horizon).shift(-horizon)
        elif target_type == 'log_return':
            # Next period log return
            result['target'] = np.log(prices / prices.shift(horizon)).shift(-horizon)
        elif target_type == 'price':
            # Next period price
            result['target'] = prices.shift(-horizon)
        else:
            logger.error(f"Unknown target type: {target_type}")
            return result

        return result

    def generate_features_for_window(self, data: pd.DataFrame, target_col: str,
                                   window_size: int, **kwargs) -> pd.DataFrame:
        """
        Generate all features for a specific lookback window.
        
        Args:
            data: Input DataFrame with time series data
            target_col: Name of the target column (e.g., 'Gold_Price_USD')
            window_size: Lookback window size
            **kwargs: Additional arguments for feature engineering
            
        Returns:
            DataFrame with all generated features
        """
        logger.info(f"Generating features for {window_size}-day window")

        # Start with a copy of the data
        result = data.copy()

        # Determine which columns to use for feature engineering
        feature_columns = [col for col in data.columns if col not in ['time', 'date']]
        other_columns = [col for col in feature_columns if col != target_col]

        # Define lag and window configurations based on the lookback window
        lags = [lag for lag in [1, 5, 10, 21] if lag <= window_size]
        rolling_windows = [w for w in [5, 10, 21, 63] if w <= window_size]

        # 1. Create lagged features
        result = self.create_lagged_features(result, feature_columns, lags)

        # 2. Create rolling statistics
        result = self.create_rolling_statistics(result, feature_columns, rolling_windows)

        # 3. Create technical indicators
        result = self.create_technical_indicators(result, target_col, rolling_windows)

        # 4. Create returns and volatility measures
        result = self.create_returns_and_volatility(result, target_col, rolling_windows)

        # 5. Create interaction features with other economic indicators
        if other_columns:
            result = self.create_interaction_features(result, target_col, other_columns)

        # 6. Create calendar features
        if 'time' in data.columns:
            result = self.create_calendar_features(result, 'time')

        # 7. Create target variable
        target_type = kwargs.get('target_type', 'return')
        horizon = kwargs.get('horizon', 1)
        result = self.create_target_variable(result, target_col, target_type, horizon)

        # 8. Drop rows with NaN values
        initial_rows = len(result)
        result = result.dropna()
        dropped_rows = initial_rows - len(result)

        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with NaN values")

        # 9. Ensure we have sufficient data
        if len(result) < window_size * 2:
            logger.warning(f"Insufficient data after feature engineering: {len(result)} < {window_size * 2}")

        logger.info(f"Generated {len(result.columns)} features for {len(result)} samples")

        return result

    def generate_features_for_all_windows(self, data: pd.DataFrame, target_col: str,
                                        windows: list[int] = [21, 63, 126],
                                        **kwargs) -> dict[str, pd.DataFrame]:
        """
        Generate features for multiple lookback windows.
        
        Args:
            data: Input DataFrame with time series data
            target_col: Name of the target column
            windows: List of lookback window sizes
            **kwargs: Additional arguments for feature engineering
            
        Returns:
            Dictionary mapping window size to DataFrame with features
        """
        results = {}

        for window in windows:
            try:
                windowed_data = self.generate_features_for_window(
                    data, target_col, window, **kwargs
                )

                if len(windowed_data) >= window * 2:
                    results[str(window)] = windowed_data
                    logger.info(f"Successfully generated features for {window}-day window")
                else:
                    logger.warning(f"Skipping {window}-day window due to insufficient data")

            except Exception as e:
                logger.error(f"Error generating features for {window}-day window: {e}")
                continue

        return results

# Convenience functions for backward compatibility
def create_rolling_features(data: pd.DataFrame, target_col: str,
                          lookback_windows: list[int] = [21, 63, 126]) -> dict[str, pd.DataFrame]:
    """
    Legacy function for CPU training compatibility.
    """
    engineer = UnifiedFeatureEngineer(use_gpu=False)
    return engineer.generate_features_for_all_windows(data, target_col, lookback_windows)

def create_enhanced_features_gpu(data: pd.DataFrame, target_col: str,
                               lookback_windows: list[int] = [21, 63, 126]) -> dict[str, pd.DataFrame]:
    """
    Legacy function for GPU training compatibility.
    """
    engineer = UnifiedFeatureEngineer(use_gpu=True)
    return engineer.generate_features_for_all_windows(data, target_col, lookback_windows)

# Main interface function
def create_features(data: pd.DataFrame, target_col: str,
                   lookback_windows: list[int] = [21, 63, 126],
                   use_gpu: bool = None, **kwargs) -> dict[str, pd.DataFrame]:
    """
    Main interface for feature engineering.
    
    Args:
        data: Input DataFrame with time series data
        target_col: Name of the target column
        lookback_windows: List of lookback window sizes
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional arguments for feature engineering

    Returns:
        Dictionary mapping window size to DataFrame with features
    """
    engineer = UnifiedFeatureEngineer(use_gpu=use_gpu)
    return engineer.generate_features_for_all_windows(
        data, target_col, lookback_windows, **kwargs
    )

