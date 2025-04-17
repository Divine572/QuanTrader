"""
Volatility estimators module.

This module provides functions to calculate various volatility estimators
such as close-to-close, Parkinson, Rogers-Satchell, and Yang-Zhang volatility.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from numba import njit


def close_to_close_volatility(
    df: pd.DataFrame,
    window_size: int = 20,
    annualize: bool = True,
    trading_periods: int = 252
) -> pd.Series:
    """
    Calculate close-to-close volatility estimator.
    
    This is the traditional volatility measure based on log returns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'close' column.
    window_size : int, default 20
        Number of periods to calculate volatility over.
    annualize : bool, default True
        Whether to annualize the volatility.
    trading_periods : int, default 252
        Number of trading periods in a year.
        
    Returns
    -------
    pd.Series
        Series containing the volatility values.
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")
    
    # Calculate log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate standard deviation of log returns
    volatility = log_returns.rolling(window=window_size).std()
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(trading_periods)
    
    volatility.name = f"Close-to-Close_Volatility_{window_size}"
    return volatility


def parkinson_volatility(
    df: pd.DataFrame,
    window_size: int = 20,
    annualize: bool = True,
    trading_periods: int = 252
) -> pd.Series:
    """
    Calculate Parkinson volatility estimator.
    
    This estimator uses the high and low prices to capture intraday volatility.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'high' and 'low' columns.
    window_size : int, default 20
        Number of periods to calculate volatility over.
    annualize : bool, default True
        Whether to annualize the volatility.
    trading_periods : int, default 252
        Number of trading periods in a year.
        
    Returns
    -------
    pd.Series
        Series containing the volatility values.
    """
    if 'high' not in df.columns or 'low' not in df.columns:
        raise ValueError("DataFrame must contain 'high' and 'low' columns")
    
    # Calculate log high-low range squared
    hl_squared = (np.log(df['high'] / df['low'])**2) / (4 * np.log(2))
    
    # Calculate rolling mean of hl_squared
    parkinsons = np.sqrt(hl_squared.rolling(window=window_size).mean())
    
    # Annualize if requested
    if annualize:
        parkinsons = parkinsons * np.sqrt(trading_periods)
    
    parkinsons.name = f"Parkinson_Volatility_{window_size}"
    return parkinsons


def rogers_satchell_volatility(
    df: pd.DataFrame,
    window_size: int = 20,
    annualize: bool = True,
    trading_periods: int = 252
) -> pd.Series:
    """
    Calculate Rogers-Satchell volatility estimator.
    
    This estimator is more accurate in trending markets.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', and 'close' columns.
    window_size : int, default 20
        Number of periods to calculate volatility over.
    annualize : bool, default True
        Whether to annualize the volatility.
    trading_periods : int, default 252
        Number of trading periods in a year.
        
    Returns
    -------
    pd.Series
        Series containing the volatility values.
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate log price differences
    log_ho = np.log(df['high'] / df['open'])
    log_lo = np.log(df['low'] / df['open'])
    log_co = np.log(df['close'] / df['open'])
    
    # Rogers-Satchell term
    rs_term = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    # Calculate rolling mean
    rs_vol = np.sqrt(rs_term.rolling(window=window_size).mean())
    
    # Annualize if requested
    if annualize:
        rs_vol = rs_vol * np.sqrt(trading_periods)
    
    rs_vol.name = f"Rogers_Satchell_Volatility_{window_size}"
    return rs_vol


def yang_zhang_volatility(
    df: pd.DataFrame,
    window_size: int = 20,
    annualize: bool = True,
    trading_periods: int = 252,
    k: float = 0.34
) -> pd.Series:
    """
    Calculate Yang-Zhang volatility estimator.
    
    This estimator is designed to be robust to both opening jumps and drift.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', and 'close' columns.
    window_size : int, default 20
        Number of periods to calculate volatility over.
    annualize : bool, default True
        Whether to annualize the volatility.
    trading_periods : int, default 252
        Number of trading periods in a year.
    k : float, default 0.34
        Parameter that controls the weighting of overnight volatility vs. day volatility.
        
    Returns
    -------
    pd.Series
        Series containing the volatility values.
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate returns for overnight (close to open) and intraday (open to close)
    log_co = np.log(df['close'] / df['open'])
    log_oc = np.log(df['open'] / df['close'].shift(1))
    
    # Calculate overnight volatility component (close to open)
    overnight_vol = log_oc.rolling(window=window_size).var()
    
    # Calculate Rogers-Satchell intraday volatility
    log_ho = np.log(df['high'] / df['open'])
    log_lo = np.log(df['low'] / df['open'])
    rs_term = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    intraday_vol = rs_term.rolling(window=window_size).mean()
    
    # Combine using Yang-Zhang formula
    yang_zhang = np.sqrt(overnight_vol + k * intraday_vol)
    
    # Annualize if requested
    if annualize:
        yang_zhang = yang_zhang * np.sqrt(trading_periods)
    
    yang_zhang.name = f"Yang_Zhang_Volatility_{window_size}"
    return yang_zhang


def garch_volatility(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20,
    alpha: float = 0.1, 
    beta: float = 0.8,
    annualize: bool = True,
    trading_periods: int = 252
) -> pd.Series:
    """
    Calculate GARCH(1,1) volatility estimator.
    
    This implements a simple GARCH(1,1) model for volatility estimation.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for initial volatility calculation.
    alpha : float, default 0.1
        GARCH parameter for return innovation.
    beta : float, default 0.8
        GARCH parameter for previous volatility (alpha + beta < 1).
    annualize : bool, default True
        Whether to annualize the volatility.
    trading_periods : int, default 252
        Number of trading periods in a year.
        
    Returns
    -------
    pd.Series
        Series containing the GARCH volatility estimates.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    if alpha + beta >= 1:
        raise ValueError("alpha + beta must be less than 1 for stationarity")
    
    # Calculate log returns
    returns = np.log(df[col] / df[col].shift(1))
    returns = returns.fillna(0)
    
    # Initialize with simple volatility estimate
    vol = returns.rolling(window=window_size).std().fillna(returns.std())
    
    # Calculate long-run average variance (omega)
    omega = (1 - alpha - beta) * returns.var()
    
    # Initialize GARCH series
    garch = pd.Series(index=df.index, dtype=float)
    garch.iloc[0] = vol.iloc[0]
    
    # Calculate GARCH recursively
    for i in range(1, len(df)):
        # GARCH(1,1) formula: σ²ₜ = ω + α*rₜ₋₁² + β*σ²ₜ₋₁
        garch.iloc[i] = np.sqrt(omega + alpha * returns.iloc[i-1]**2 + beta * garch.iloc[i-1]**2)
    
    # Annualize if requested
    if annualize:
        garch = garch * np.sqrt(trading_periods)
    
    garch.name = f"GARCH_Volatility_{window_size}"
    return garch


def average_true_range(
    df: pd.DataFrame,
    window_size: int = 14,
    method: str = 'sma'
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    ATR is a measure of volatility that accounts for gaps.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'high', 'low', and 'close' columns.
    window_size : int, default 14
        Number of periods to calculate ATR over.
    method : str, default 'sma'
        Method to use for averaging: 'sma' for simple moving average,
        'ema' for exponential moving average.
        
    Returns
    -------
    pd.Series
        Series containing the ATR values.
    """
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate true range
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate ATR based on method
    if method.lower() == 'sma':
        atr = true_range.rolling(window=window_size).mean()
    elif method.lower() == 'ema':
        atr = true_range.ewm(span=window_size, adjust=False).mean()
    else:
        raise ValueError("Method must be 'sma' or 'ema'")
    
    atr.name = f"ATR_{window_size}"
    return atr


def bollinger_bands(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
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
    Tuple[pd.Series, pd.Series, pd.Series]
        Tuple containing (upper_band, middle_band, lower_band) series.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate middle band (SMA)
    middle_band = df[col].rolling(window=window_size).mean()
    middle_band.name = f"BB_Middle_{window_size}"
    
    # Calculate standard deviation
    std = df[col].rolling(window=window_size).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    upper_band.name = f"BB_Upper_{window_size}"
    
    lower_band = middle_band - (std * num_std)
    lower_band.name = f"BB_Lower_{window_size}"
    
    return upper_band, middle_band, lower_band


def keltner_channels(
    df: pd.DataFrame,
    window_size: int = 20,
    atr_window: int = 14,
    atr_multiplier: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channels.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'high', 'low', and 'close' columns.
    window_size : int, default 20
        Window size for the EMA calculation.
    atr_window : int, default 14
        Window size for the ATR calculation.
    atr_multiplier : float, default 2.0
        Multiplier for the ATR.
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        Tuple containing (upper_channel, middle_channel, lower_channel) series.
    """
    required_cols = ['high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate middle channel (EMA of close)
    middle_channel = df['close'].ewm(span=window_size, adjust=False).mean()
    middle_channel.name = f"KC_Middle_{window_size}"
    
    # Calculate ATR
    atr = average_true_range(df, window_size=atr_window, method='ema')
    
    # Calculate upper and lower channels
    upper_channel = middle_channel + (atr * atr_multiplier)
    upper_channel.name = f"KC_Upper_{window_size}"
    
    lower_channel = middle_channel - (atr * atr_multiplier)
    lower_channel.name = f"KC_Lower_{window_size}"
    
    return upper_channel, middle_channel, lower_channel


def historical_volatility(
    df: pd.DataFrame,
    col: str = 'close',
    window_size: int = 20,
    annualize: bool = True,
    trading_periods: int = 252
) -> pd.Series:
    """
    Calculate historical volatility.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data.
    col : str, default 'close'
        Column name for the price series.
    window_size : int, default 20
        Window size for volatility calculation.
    annualize : bool, default True
        Whether to annualize the volatility.
    trading_periods : int, default 252
        Number of trading periods in a year.
        
    Returns
    -------
    pd.Series
        Series containing the historical volatility values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Calculate log returns
    log_returns = np.log(df[col] / df[col].shift(1))
    
    # Calculate volatility
    volatility = log_returns.rolling(window=window_size).std()
    
    # Annualize if requested
    if annualize:
        volatility = volatility * np.sqrt(trading_periods)
    
    volatility.name = f"Historical_Volatility_{window_size}"
    return volatility