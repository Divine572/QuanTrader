"""
Market regime identification module.

This module provides functions to identify different market regimes,
such as bullish, bearish, or ranging markets.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from numba import njit

from .trend import sma, ema, kama


def sma_market_regime(
    df: pd.DataFrame,
    col: str = 'close',
    fast_window: int = 20,
    slow_window: int = 50,
    normalized: bool = False
) -> pd.Series:
    """
    Determine market regime based on SMA crossover.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    fast_window : int, default 20
        Window size for the fast moving average.
    slow_window : int, default 50
        Window size for the slow moving average.
    normalized : bool, default False
        If True, return normalized difference between fast and slow SMA.
        If False, return categorical values: 1 (bullish), -1 (bearish), 0 (neutral).
        
    Returns
    -------
    pd.Series
        Series containing the market regime values.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import quantrader.features as qf
    >>> df = pd.DataFrame({'close': [1, 2, 3, 4, 5] * 20})
    >>> qf.market_regime.sma_market_regime(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if not (isinstance(fast_window, int) and fast_window > 0 and
            isinstance(slow_window, int) and slow_window > 0):
        raise ValueError("Window sizes must be positive integers")
    
    if fast_window >= slow_window:
        raise ValueError("Fast window must be smaller than slow window")
    
    # Calculate fast and slow SMAs
    fast_sma = sma(df, col=col, window_size=fast_window)
    slow_sma = sma(df, col=col, window_size=slow_window)
    
    # Calculate difference
    sma_diff = fast_sma - slow_sma
    
    if normalized:
        # Normalize the difference by current price
        return sma_diff / df[col]
    else:
        # Categorical signals: 1 (bullish), -1 (bearish), 0 (neutral)
        regime = pd.Series(0, index=df.index)
        regime[sma_diff > 0] = 1
        regime[sma_diff < 0] = -1
        regime.name = f"SMA_Regime_{fast_window}_{slow_window}"
        return regime


def kama_market_regime(
    df: pd.DataFrame,
    col: str = 'close',
    er_window: int = 10,
    fast_sc: int = 2,
    slow_sc: int = 30,
    normalized: bool = False
) -> pd.Series:
    """
    Determine market regime based on KAMA slope.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    er_window : int, default 10
        Window size for the efficiency ratio.
    fast_sc : int, default 2
        Fast smoothing constant for KAMA.
    slow_sc : int, default 30
        Slow smoothing constant for KAMA.
    normalized : bool, default False
        If True, return normalized KAMA slope.
        If False, return categorical values: 1 (bullish), -1 (bearish), 0 (neutral).
        
    Returns
    -------
    pd.Series
        Series containing the market regime values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate KAMA
    k = kama(df, col=col, l1=er_window, l2=fast_sc, l3=slow_sc)
    
    # Calculate KAMA slope (using 5-period difference)
    slope = k - k.shift(5)
    
    if normalized:
        # Normalize the slope by current KAMA value
        return slope / k
    else:
        # Categorical signals: 1 (bullish), -1 (bearish), 0 (neutral)
        regime = pd.Series(0, index=df.index)
        
        # Define thresholds for slopes (adjust based on your asset)
        regime[slope > 0.001 * k] = 1  # Bullish when slope is clearly positive
        regime[slope < -0.001 * k] = -1  # Bearish when slope is clearly negative
        
        regime.name = f"KAMA_Regime_{er_window}_{fast_sc}_{slow_sc}"
        return regime


def rsi_market_regime(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 14,
    overbought: int = 70,
    oversold: int = 30
) -> pd.Series:
    """
    Determine market regime based on RSI values.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 14
        Window size for RSI calculation.
    overbought : int, default 70
        Threshold for overbought condition.
    oversold : int, default 30
        Threshold for oversold condition.
        
    Returns
    -------
    pd.Series
        Series containing the market regime values:
        2 (strongly bullish), 1 (bullish), 0 (neutral), 
        -1 (bearish), -2 (strongly bearish).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate price changes
    delta = df[col].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=window_size).mean()
    avg_loss = loss.rolling(window=window_size).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Determine regime based on RSI
    regime = pd.Series(0, index=df.index)
    
    # Overbought
    regime[rsi > overbought] = 2
    
    # Bullish
    regime[(rsi > 50) & (rsi <= overbought)] = 1
    
    # Bearish
    regime[(rsi < 50) & (rsi >= oversold)] = -1
    
    # Oversold
    regime[rsi < oversold] = -2
    
    regime.name = f"RSI_Regime_{window_size}"
    return regime


def bollinger_market_regime(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20,
    num_std: float = 2.0
) -> pd.Series:
    """
    Determine market regime based on Bollinger Bands position.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for moving average calculation.
    num_std : float, default 2.0
        Number of standard deviations for the bands.
        
    Returns
    -------
    pd.Series
        Series containing the market regime values:
        2 (strongly bullish), 1 (bullish), 0 (neutral), 
        -1 (bearish), -2 (strongly bearish).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate middle band (SMA)
    middle_band = df[col].rolling(window=window_size).mean()
    
    # Calculate standard deviation
    std = df[col].rolling(window=window_size).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    # Calculate %B (percentage position within the bands)
    percent_b = (df[col] - lower_band) / (upper_band - lower_band)
    
    # Determine regime based on %B
    regime = pd.Series(0, index=df.index)
    
    # Strongly bullish: above upper band
    regime[percent_b > 1] = 2
    
    # Bullish: above middle band but below upper band
    regime[(percent_b > 0.5) & (percent_b <= 1)] = 1
    
    # Bearish: below middle band but above lower band
    regime[(percent_b >= 0) & (percent_b < 0.5)] = -1
    
    # Strongly bearish: below lower band
    regime[percent_b < 0] = -2
    
    regime.name = f"BB_Regime_{window_size}"
    return regime


def volatility_regime(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20,
    high_threshold: float = 0.2,
    low_threshold: float = 0.1,
    annualize: bool = True,
    trading_periods: int = 252
) -> pd.Series:
    """
    Determine volatility regime (high, normal, low).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for volatility calculation.
    high_threshold : float, default 0.2
        Annualized volatility threshold for high volatility regime.
    low_threshold : float, default 0.1
        Annualized volatility threshold for low volatility regime.
    annualize : bool, default True
        Whether to annualize the volatility.
    trading_periods : int, default 252
        Number of trading periods in a year.
        
    Returns
    -------
    pd.Series
        Series containing the volatility regime:
        1 (high volatility), 0 (normal volatility), -1 (low volatility).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate log returns
    returns = np.log(df[col] / df[col].shift(1))
    
    # Calculate rolling standard deviation
    volatility = returns.rolling(window=window_size).std()
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(trading_periods)
    
    # Determine volatility regime
    regime = pd.Series(0, index=df.index)
    
    # High volatility
    regime[volatility > high_threshold] = 1
    
    # Low volatility
    regime[volatility < low_threshold] = -1
    
    regime.name = f"Volatility_Regime_{window_size}"
    return regime


def multi_timeframe_regime(
    df: pd.DataFrame,
    col: str = 'close',
    short_window: int = 10,
    medium_window: int = 50,
    long_window: int = 200
) -> pd.Series:
    """
    Determine market regime based on multiple timeframe trend analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    short_window : int, default 10
        Window size for short-term moving average.
    medium_window : int, default 50
        Window size for medium-term moving average.
    long_window : int, default 200
        Window size for long-term moving average.
        
    Returns
    -------
    pd.Series
        Series containing the market regime:
        3 (strongly bullish), 2 (bullish), 1 (slightly bullish), 
        0 (neutral), -1 (slightly bearish), -2 (bearish), -3 (strongly bearish).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate moving averages
    short_ma = df[col].rolling(window=short_window).mean()
    medium_ma = df[col].rolling(window=medium_window).mean()
    long_ma = df[col].rolling(window=long_window).mean()
    
    # Initialize regime
    regime = pd.Series(0, index=df.index)
    
    # Score based on relative positions of MAs and price
    # Price > Short MA
    regime[df[col] > short_ma] += 1
    
    # Price < Short MA
    regime[df[col] < short_ma] -= 1
    
    # Short MA > Medium MA
    regime[short_ma > medium_ma] += 1
    
    # Short MA < Medium MA
    regime[short_ma < medium_ma] -= 1
    
    # Medium MA > Long MA
    regime[medium_ma > long_ma] += 1
    
    # Medium MA < Long MA
    regime[medium_ma < long_ma] -= 1
    
    # Clip to range [-3, 3]
    regime = regime.clip(-3, 3)
    
    regime.name = f"MTF_Regime_{short_window}_{medium_window}_{long_window}"
    return regime


def hurst_market_regime(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 100,
    threshold_trending: float = 0.6,
    threshold_mean_reverting: float = 0.4
) -> pd.Series:
    """
    Determine market regime based on Hurst exponent.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 100
        Window size for Hurst exponent calculation.
    threshold_trending : float, default 0.6
        Hurst exponent threshold for trending market.
    threshold_mean_reverting : float, default 0.4
        Hurst exponent threshold for mean reverting market.
        
    Returns
    -------
    pd.Series
        Series containing the market regime:
        1 (trending), 0 (random walk), -1 (mean reverting).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Initialize regime
    regime = pd.Series(0, index=df.index)
    
    # Need at least window_size data points to calculate Hurst exponent
    if len(df) < window_size:
        return regime
    
    def calculate_hurst(prices, lag_range=20):
        """Calculate Hurst exponent for a price series."""
        lags = range(2, lag_range)
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]  # Hurst exponent is the slope
    
    # Calculate Hurst exponent for each window
    for i in range(window_size, len(df)):
        window_data = df[col].iloc[i-window_size:i].values
        try:
            hurst_value = calculate_hurst(window_data)
            
            # Trending market (Hurst > 0.6)
            if hurst_value > threshold_trending:
                regime.iloc[i] = 1
            # Mean-reverting market (Hurst < 0.4)
            elif hurst_value < threshold_mean_reverting:
                regime.iloc[i] = -1
            # Random walk (0.4 <= Hurst <= 0.6)
            else:
                regime.iloc[i] = 0
        except:
            # In case of calculation error, keep as 0 (random walk)
            regime.iloc[i] = 0
    
    regime.name = f"Hurst_Regime_{window_size}"
    return regime