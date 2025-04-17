"""
Trend indicators module.

This module provides functions to calculate various trend indicators
such as moving averages and linear regression analysis.
"""

import numpy as np
import pandas as pd
from numba import njit
from typing import Union, Optional


def sma(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20
) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for the moving average.
        
    Returns
    -------
    pd.Series
        Series containing the SMA values.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import quantrader.features as qf
    >>> df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
    >>> qf.trend.sma(df, window_size=3)
    0    NaN
    1    NaN
    2    2.0
    3    3.0
    4    4.0
    Name: SMA_3, dtype: float64
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer")
    
    result = df[col].rolling(window=window_size).mean()
    result.name = f"SMA_{window_size}"
    return result


def ema(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20,
    adjust: bool = True
) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for the moving average.
    adjust : bool, default True
        Adjust the weights to account for the imbalance in the beginning.
        
    Returns
    -------
    pd.Series
        Series containing the EMA values.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import quantrader.features as qf
    >>> df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
    >>> qf.trend.ema(df, window_size=3)
    0    NaN
    1    NaN
    2    2.000000
    3    3.000000
    4    4.000000
    Name: EMA_3, dtype: float64
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer")
    
    alpha = 2 / (window_size + 1)
    result = df[col].ewm(alpha=alpha, adjust=adjust).mean()
    result.name = f"EMA_{window_size}"
    return result


@njit
def _calc_efficiency_ratio(prices: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calculate Efficiency Ratio for KAMA calculation.
    
    Parameters
    ----------
    prices : np.ndarray
        Array of prices.
    window_size : int
        Window size for calculation.
        
    Returns
    -------
    np.ndarray
        Array of efficiency ratios.
    """
    n = len(prices)
    er = np.zeros(n)
    
    for i in range(window_size, n):
        # Direction: current - window_size periods ago
        direction = abs(prices[i] - prices[i - window_size])
        
        # Volatility: sum of period to period price changes
        volatility = 0.0
        for j in range(i - window_size + 1, i + 1):
            volatility += abs(prices[j] - prices[j - 1])
        
        # Efficiency Ratio
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0.0
            
    return er


def kama(
    df: pd.DataFrame,
    col: str = 'close',
    l1: int = 10,
    l2: int = 2,
    l3: int = 30
) -> pd.Series:
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    l1 : int, default 10
        Efficiency Ratio period.
    l2 : int, default 2
        Fastest EMA period.
    l3 : int, default 30
        Slowest EMA period.
        
    Returns
    -------
    pd.Series
        Series containing the KAMA values.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import quantrader.features as qf
    >>> df = pd.DataFrame({'close': np.arange(1, 21)})
    >>> qf.trend.kama(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if not (isinstance(l1, int) and l1 > 0 and
            isinstance(l2, int) and l2 > 0 and
            isinstance(l3, int) and l3 > 0):
        raise ValueError("All periods must be positive integers")
    
    prices = df[col].to_numpy()
    n = len(prices)
    
    # Calculate Efficiency Ratio
    er = _calc_efficiency_ratio(prices, l1)
    
    # Calculate Smoothing Constant from fastest and slowest EMA periods
    sc_fast = 2.0 / (l2 + 1.0)
    sc_slow = 2.0 / (l3 + 1.0)
    
    # Calculate Smoothing Constant based on Efficiency Ratio
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    
    # Calculate KAMA - Initialize with first available price
    kama_values = np.zeros(n)
    start_index = l1
    
    if start_index >= n:
        return pd.Series(np.nan, index=df.index)
    
    kama_values[start_index] = prices[start_index]
    
    # Calculate KAMA recursively
    for i in range(start_index + 1, n):
        kama_values[i] = kama_values[i-1] + sc[i] * (prices[i] - kama_values[i-1])
    
    # Convert to pandas Series
    result = pd.Series(kama_values, index=df.index)
    result.name = f"KAMA_{l1}_{l2}_{l3}"
    
    return result


def linear_regression_slope(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20
) -> pd.Series:
    """
    Calculate linear regression slope over a rolling window.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for the linear regression.
        
    Returns
    -------
    pd.Series
        Series containing the slope values.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import quantrader.features as qf
    >>> df = pd.DataFrame({'close': np.arange(1, 21)})
    >>> qf.trend.linear_regression_slope(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer")
    
    # X values (0 to window_size-1)
    x = np.arange(window_size)
    
    # Calculate mean of x
    x_mean = np.mean(x)
    
    # Calculate sum of (x - x_mean)^2
    x_diff_squared_sum = np.sum((x - x_mean) ** 2)
    
    def _calculate_slope(y):
        if len(y) < window_size:
            return np.nan
        
        # Calculate mean of y
        y_mean = np.mean(y)
        
        # Calculate sum of (x - x_mean) * (y - y_mean)
        xy_diff_sum = np.sum((x - x_mean) * (y - y_mean))
        
        # Calculate slope
        return xy_diff_sum / x_diff_squared_sum
    
    # Apply rolling calculation
    result = df[col].rolling(window=window_size).apply(_calculate_slope, raw=True)
    result.name = f"LR_Slope_{window_size}"
    
    return result


def hull_moving_average(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20
) -> pd.Series:
    """
    Calculate Hull Moving Average (HMA).
    
    The Hull Moving Average reduces lag and improves smoothing,
    calculated as: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for the moving average.
        
    Returns
    -------
    pd.Series
        Series containing the HMA values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer")
    
    # Calculate half period WMA
    half_window = window_size // 2
    wma_half = df[col].rolling(window=half_window).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    
    # Calculate full period WMA
    wma_full = df[col].rolling(window=window_size).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    
    # Calculate 2 * half period WMA - full period WMA
    raw_hma = 2 * wma_half - wma_full
    
    # Take WMA of the result with sqrt(window_size) period
    sqrt_window = int(np.sqrt(window_size))
    hma = raw_hma.rolling(window=sqrt_window).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)),
        raw=True
    )
    
    hma.name = f"HMA_{window_size}"
    return hma


def rsi(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 14
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 14
        Window size for the RSI calculation.
        
    Returns
    -------
    pd.Series
        Series containing the RSI values (0-100).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer")
    
    # Calculate price changes
    price_diff = df[col].diff()
    
    # Separate gains and losses
    gains = price_diff.where(price_diff > 0, 0)
    losses = -price_diff.where(price_diff < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window_size).mean()
    avg_losses = losses.rolling(window=window_size).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    rsi.name = f"RSI_{window_size}"
    return rsi