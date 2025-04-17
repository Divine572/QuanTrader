"""
Mathematical transformations module.

This module provides functions for mathematical transformations of price data,
such as derivatives, log returns, autocorrelation, Hurst exponent, etc.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
from numba import njit


def log_returns(
    df: pd.DataFrame,
    col: str = 'close',
    periods: int = 1
) -> pd.Series:
    """
    Calculate logarithmic returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    periods : int, default 1
        Number of periods to calculate returns over.
        
    Returns
    -------
    pd.Series
        Series containing the log returns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate log returns
    log_ret = np.log(df[col] / df[col].shift(periods))
    log_ret.name = f"Log_Returns_{periods}"
    return log_ret


def pct_returns(
    df: pd.DataFrame,
    col: str = 'close',
    periods: int = 1
) -> pd.Series:
    """
    Calculate percentage returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    periods : int, default 1
        Number of periods to calculate returns over.
        
    Returns
    -------
    pd.Series
        Series containing the percentage returns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate percentage returns
    pct_ret = df[col].pct_change(periods=periods)
    pct_ret.name = f"Pct_Returns_{periods}"
    return pct_ret


def rate_of_change(
    df: pd.DataFrame,
    col: str = 'close',
    periods: int = 1
) -> pd.Series:
    """
    Calculate Rate of Change (ROC).
    
    ROC = (Current Price / Price n periods ago - 1) * 100
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    periods : int, default 1
        Number of periods to calculate ROC over.
        
    Returns
    -------
    pd.Series
        Series containing the ROC values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate ROC
    roc = (df[col] / df[col].shift(periods) - 1) * 100
    roc.name = f"ROC_{periods}"
    return roc


def momentum(
    df: pd.DataFrame,
    col: str = 'close',
    periods: int = 14
) -> pd.Series:
    """
    Calculate momentum.
    
    Momentum = Current Price - Price n periods ago
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    periods : int, default 14
        Number of periods to calculate momentum over.
        
    Returns
    -------
    pd.Series
        Series containing the momentum values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate momentum
    mom = df[col] - df[col].shift(periods)
    mom.name = f"Momentum_{periods}"
    return mom


def autocorrelation(
    df: pd.DataFrame,
    col: str = 'close',
    lag: int = 1,
    window_size: int = 20
) -> pd.Series:
    """
    Calculate rolling autocorrelation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    lag : int, default 1
        Lag for autocorrelation calculation.
    window_size : int, default 20
        Window size for rolling calculation.
        
    Returns
    -------
    pd.Series
        Series containing the autocorrelation values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate returns first
    returns = df[col].pct_change()
    
    # Calculate rolling autocorrelation
    autocorr = returns.rolling(window=window_size).apply(
        lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else np.nan,
        raw=False
    )
    
    autocorr.name = f"Autocorr_{lag}_{window_size}"
    return autocorr


def hurst(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 100,
    max_lag: int = 20
) -> pd.Series:
    """
    Calculate rolling Hurst exponent.
    
    The Hurst exponent measures the long-term memory of a time series.
    H > 0.5 indicates trending, H < 0.5 indicates mean-reversion,
    and H = 0.5 indicates a random walk.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 100
        Window size for rolling calculation.
    max_lag : int, default 20
        Maximum lag for R/S calculation.
        
    Returns
    -------
    pd.Series
        Series containing the Hurst exponent values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    def calculate_hurst(price_array, max_lag=max_lag):
        """Calculate Hurst exponent for a single window."""
        # Convert to returns if we have price data
        returns = np.diff(np.log(price_array))
        if len(returns) < max_lag:
            return np.nan
        
        # Calculate lags from 2 to max_lag
        lags = range(2, max_lag)
        
        # Calculate variance of the differenced series
        tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
        
        # Regression
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        
        # Hurst exponent is the slope
        return poly[0]
    
    # Initialize Hurst series
    hurst_values = pd.Series(np.nan, index=df.index)
    
    # Calculate Hurst for each window
    for i in range(window_size, len(df)):
        window_data = df[col].iloc[i-window_size:i].values
        try:
            hurst_values.iloc[i] = calculate_hurst(window_data)
        except:
            hurst_values.iloc[i] = np.nan
    
    hurst_values.name = f"Hurst_{window_size}"
    return hurst_values


def zscore(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20
) -> pd.Series:
    """
    Calculate rolling Z-score.
    
    Z-score = (Value - Mean) / Standard Deviation
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for rolling calculation.
        
    Returns
    -------
    pd.Series
        Series containing the Z-score values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate rolling mean and standard deviation
    rolling_mean = df[col].rolling(window=window_size).mean()
    rolling_std = df[col].rolling(window=window_size).std()
    
    # Calculate Z-score
    zscore = (df[col] - rolling_mean) / rolling_std
    
    zscore.name = f"ZScore_{window_size}"
    return zscore


def normalization(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20,
    method: str = 'minmax'
) -> pd.Series:
    """
    Calculate rolling normalization.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for rolling calculation.
    method : str, default 'minmax'
        Normalization method: 'minmax' for MinMax scaling (0-1),
        'zscore' for Z-score normalization.
        
    Returns
    -------
    pd.Series
        Series containing the normalized values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if method.lower() not in ['minmax', 'zscore']:
        raise ValueError("Method must be 'minmax' or 'zscore'")
    
    if method.lower() == 'minmax':
        # Calculate rolling min and max
        rolling_min = df[col].rolling(window=window_size).min()
        rolling_max = df[col].rolling(window=window_size).max()
        
        # Calculate MinMax scaling (0-1)
        norm = (df[col] - rolling_min) / (rolling_max - rolling_min)
        norm.name = f"MinMax_{window_size}"
        
    else:  # 'zscore'
        norm = zscore(df, col, window_size)
        norm.name = f"ZScore_{window_size}"
    
    return norm


def exponential_smoothing(
    df: pd.DataFrame,
    col: str = 'close',
    alpha: float = 0.2
) -> pd.Series:
    """
    Calculate exponential smoothing.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    alpha : float, default 0.2
        Smoothing factor (0 < alpha < 1).
        
    Returns
    -------
    pd.Series
        Series containing the exponentially smoothed values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")
    
    # Calculate exponential smoothing
    smoothed = df[col].ewm(alpha=alpha, adjust=False).mean()
    
    smoothed.name = f"ExpSmooth_{alpha}"
    return smoothed