"""
Candlestick analysis module.

This module provides functions for analyzing candlestick patterns and characteristics
such as direction, filling, amplitude, and spread.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List



def candle_direction(
    df: pd.DataFrame
) -> pd.Series:
    """
    Determine the direction of each candle.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open' and 'close' columns.
        
    Returns
    -------
    pd.Series
        Series containing the candle direction:
        1 for bullish (close > open), -1 for bearish (close < open), 0 for doji (close = open).
    """
    required_cols = ['open', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Determine direction
    direction = pd.Series(0, index=df.index)
    direction[df['close'] > df['open']] = 1
    direction[df['close'] < df['open']] = -1
    
    direction.name = "Candle_Direction"
    return direction


def candle_filling(
    df: pd.DataFrame,
    normalize: bool = False
) -> pd.Series:
    """
    Calculate the filling of each candle (body size relative to total range).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
    normalize : bool, default False
        If True, normalize the filling to [0, 1] range.
        If False, return the ratio of body size to total range.
        
    Returns
    -------
    pd.Series
        Series containing the candle filling values.
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate body size (absolute difference between open and close)
    body_size = abs(df['close'] - df['open'])
    
    # Calculate total range (high - low)
    total_range = df['high'] - df['low']
    
    # Calculate filling ratio
    filling = body_size / total_range.replace(0, np.nan)
    
    # Handle zero range candles
    filling.fillna(0, inplace=True)
    
    # Normalize if requested
    if normalize:
        filling = filling / filling.max()
    
    filling.name = "Candle_Filling"
    return filling


def candle_amplitude(
    df: pd.DataFrame,
    normalize: bool = False,
    window_size: int = 20
) -> pd.Series:
    """
    Calculate the amplitude of each candle (total range relative to average).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'high', 'low' columns.
    normalize : bool, default False
        If True, normalize the amplitude to the rolling window.
        If False, return the absolute range.
    window_size : int, default 20
        Window size for normalization (only used if normalize=True).
        
    Returns
    -------
    pd.Series
        Series containing the candle amplitude values.
    """
    required_cols = ['high', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate total range
    amplitude = df['high'] - df['low']
    
    # Normalize if requested
    if normalize:
        # Calculate rolling average range
        avg_amplitude = amplitude.rolling(window=window_size).mean()
        
        # Normalize relative to rolling average
        amplitude = amplitude / avg_amplitude
    
    amplitude.name = "Candle_Amplitude"
    return amplitude


def candle_spread(
    df: pd.DataFrame,
    ema_window: int = 20
) -> pd.Series:
    """
    Calculate the spread of each candle from its EMA.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'close' column.
    ema_window : int, default 20
        Window size for the EMA calculation.
        
    Returns
    -------
    pd.Series
        Series containing the candle spread values.
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column")
    
    # Calculate EMA
    ema = df['close'].ewm(span=ema_window, adjust=False).mean()
    
    # Calculate spread (distance from close to EMA)
    spread = (df['close'] - ema) / ema * 100
    
    spread.name = f"Candle_Spread_EMA{ema_window}"
    return spread


def candle_information(
    df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Extract main candlestick information.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        Tuple containing (direction, filling, amplitude) series.
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate direction
    direction = candle_direction(df)
    
    # Calculate filling
    filling = candle_filling(df)
    
    # Calculate amplitude
    amplitude = candle_amplitude(df)
    
    return direction, filling, amplitude


def doji_pattern(
    df: pd.DataFrame,
    threshold: float = 0.1
) -> pd.Series:
    """
    Identify doji candlestick patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
    threshold : float, default 0.1
        Maximum body-to-range ratio to consider as a doji.
        
    Returns
    -------
    pd.Series
        Series with boolean values indicating where doji patterns occur.
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate filling
    filling = candle_filling(df)
    
    # Identify doji pattern (small body relative to range)
    doji = filling <= threshold
    
    doji.name = "Doji_Pattern"
    return doji


def hammer_pattern(
    df: pd.DataFrame,
    body_threshold: float = 0.3,
    shadow_threshold: float = 0.7
) -> pd.Series:
    """
    Identify hammer and hanging man candlestick patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
    body_threshold : float, default 0.3
        Maximum body-to-range ratio for a hammer.
    shadow_threshold : float, default 0.7
        Minimum lower shadow-to-range ratio for a hammer.
        
    Returns
    -------
    pd.Series
        Series with values indicating pattern type:
        1 for hammer (bullish reversal at bottom), -1 for hanging man (bearish reversal at top),
        0 for no pattern.
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Calculate body size
    body_size = abs(df['close'] - df['open'])
    
    # Calculate total range
    total_range = df['high'] - df['low']
    
    # Calculate the body-to-range ratio
    body_ratio = body_size / total_range.replace(0, np.nan)
    body_ratio.fillna(0, inplace=True)
    
    # Calculate upper shadow
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    
    # Calculate lower shadow
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    
    # Calculate shadow ratios
    upper_ratio = upper_shadow / total_range.replace(0, np.nan)
    upper_ratio.fillna(0, inplace=True)
    
    lower_ratio = lower_shadow / total_range.replace(0, np.nan)
    lower_ratio.fillna(0, inplace=True)
    
    # Initialize pattern series
    pattern = pd.Series(0, index=df.index)
    
    # Identify hammer pattern conditions:
    # 1. Small body
    # 2. Long lower shadow
    # 3. Small or nonexistent upper shadow
    hammer_condition = (
        (body_ratio <= body_threshold) &
        (lower_ratio >= shadow_threshold) &
        (upper_ratio <= 1 - shadow_threshold - body_threshold)
    )
    
    # Calculate 20-day trend
    trend = df['close'].rolling(window=20).mean().diff().fillna(0)
    
    # Hammer (bullish reversal pattern at bottom)
    pattern[hammer_condition & (trend < 0)] = 1
    
    # Hanging man (bearish reversal pattern at top)
    pattern[hammer_condition & (trend > 0)] = -1
    
    pattern.name = "Hammer_Pattern"
    return pattern


def engulfing_pattern(
    df: pd.DataFrame
) -> pd.Series:
    """
    Identify bullish and bearish engulfing candlestick patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
        
    Returns
    -------
    pd.Series
        Series with values indicating pattern type:
        1 for bullish engulfing, -1 for bearish engulfing, 0 for no pattern.
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Initialize pattern series
    pattern = pd.Series(0, index=df.index)
    
    # Need at least 2 candles to identify engulfing pattern
    if len(df) < 2:
        return pattern
    
    # Previous candle's open and close
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    
    # Current candle's open and close
    curr_open = df['open']
    curr_close = df['close']
    
    # Bullish engulfing pattern:
    # 1. Previous candle is bearish (close < open)
    # 2. Current candle is bullish (close > open)
    # 3. Current candle's body completely engulfs previous candle's body
    bullish_engulfing = (
        (prev_close < prev_open) &
        (curr_close > curr_open) &
        (curr_open <= prev_close) &
        (curr_close >= prev_open)
    )
    
    # Bearish engulfing pattern:
    # 1. Previous candle is bullish (close > open)
    # 2. Current candle is bearish (close < open)
    # 3. Current candle's body completely engulfs previous candle's body
    bearish_engulfing = (
        (prev_close > prev_open) &
        (curr_close < curr_open) &
        (curr_open >= prev_close) &
        (curr_close <= prev_open)
    )
    
    # Set pattern values
    pattern[bullish_engulfing] = 1
    pattern[bearish_engulfing] = -1
    
    pattern.name = "Engulfing_Pattern"
    return pattern