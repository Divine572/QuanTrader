"""
Smart Money Concepts (SMC) module.

This module provides functions to calculate various SMC indicators and patterns
such as liquidity levels, order blocks, fair value gaps, and market structure.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, List
import warnings


def order_blocks(
    df: pd.DataFrame,
    window_size: int = 5,
    threshold_pct: float = 0.5,
    bull_block: bool = True,
    bear_block: bool = True
) -> pd.DataFrame:
    """
    Identify bullish and bearish order blocks.
    
    Order blocks are areas of significant imbalance where big players (smart money)
    enter or exit the market, often leading to subsequent price movements.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
    window_size : int, default 5
        Number of candles to look back to identify impulse moves.
    threshold_pct : float, default 0.5
        Minimum percentage move required to identify an impulse move.
    bull_block : bool, default True
        Whether to identify bullish order blocks.
    bear_block : bool, default True
        Whether to identify bearish order blocks.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'bull_ob': Bullish order block identification (1 or 0)
        - 'bear_ob': Bearish order block identification (1 or 0)
        - 'bull_ob_high': Upper boundary of bullish order block
        - 'bull_ob_low': Lower boundary of bullish order block
        - 'bear_ob_high': Upper boundary of bearish order block
        - 'bear_ob_low': Lower boundary of bearish order block
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()
    
    # Initialize order block columns
    result['bull_ob'] = 0
    result['bear_ob'] = 0
    result['bull_ob_high'] = np.nan
    result['bull_ob_low'] = np.nan
    result['bear_ob_high'] = np.nan
    result['bear_ob_low'] = np.nan
    
    # Calculate percentage change for impulse move identification
    result['pct_change'] = result['close'].pct_change(periods=1) * 100
    
    # Look for bullish order blocks (bearish candle before bullish impulse)
    if bull_block:
        for i in range(window_size, len(result)):
            # Check for bullish impulse (strong up move)
            if result['pct_change'].iloc[i] > threshold_pct:
                # Look back for the last bearish candle (potential bullish order block)
                for j in range(i-1, max(0, i-window_size), -1):
                    if result['close'].iloc[j] < result['open'].iloc[j]:
                        # Mark the bearish candle as a bullish order block
                        result.loc[result.index[j], 'bull_ob'] = 1
                        result.loc[result.index[j], 'bull_ob_high'] = result['high'].iloc[j]
                        result.loc[result.index[j], 'bull_ob_low'] = result['low'].iloc[j]
                        break
    
    # Look for bearish order blocks (bullish candle before bearish impulse)
    if bear_block:
        for i in range(window_size, len(result)):
            # Check for bearish impulse (strong down move)
            if result['pct_change'].iloc[i] < -threshold_pct:
                # Look back for the last bullish candle (potential bearish order block)
                for j in range(i-1, max(0, i-window_size), -1):
                    if result['close'].iloc[j] > result['open'].iloc[j]:
                        # Mark the bullish candle as a bearish order block
                        result.loc[result.index[j], 'bear_ob'] = 1
                        result.loc[result.index[j], 'bear_ob_high'] = result['high'].iloc[j]
                        result.loc[result.index[j], 'bear_ob_low'] = result['low'].iloc[j]
                        break
    
    # Drop the temporary column
    result = result.drop(columns=['pct_change'])
    
    return result


def fair_value_gaps(
    df: pd.DataFrame,
    min_gap_pct: float = 0.1
) -> pd.DataFrame:
    """
    Identify Fair Value Gaps (FVGs).
    
    Fair Value Gaps are areas on a chart where price has moved rapidly,
    creating an imbalance that is often revisited later.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
    min_gap_pct : float, default 0.1
        Minimum percentage gap required to identify an FVG.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'bull_fvg': Bullish FVG (up gap)
        - 'bear_fvg': Bearish FVG (down gap)
        - 'bull_fvg_low': Lower boundary of bullish FVG
        - 'bull_fvg_high': Upper boundary of bullish FVG
        - 'bear_fvg_low': Lower boundary of bearish FVG
        - 'bear_fvg_high': Upper boundary of bearish FVG
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()
    
    # Initialize FVG columns
    result['bull_fvg'] = 0
    result['bear_fvg'] = 0
    result['bull_fvg_low'] = np.nan
    result['bull_fvg_high'] = np.nan
    result['bear_fvg_low'] = np.nan
    result['bear_fvg_high'] = np.nan
    
    # Need at least 3 candles to identify an FVG
    if len(result) < 3:
        return result
    
    # Look for bullish FVG (gap up)
    for i in range(2, len(result)):
        # Bullish FVG: current candle's low > previous candle's high
        if result['low'].iloc[i] > result['high'].iloc[i-2]:
            # Calculate gap percentage
            gap_pct = (result['low'].iloc[i] - result['high'].iloc[i-2]) / result['high'].iloc[i-2]
            
            if gap_pct >= min_gap_pct:
                result.loc[result.index[i], 'bull_fvg'] = 1
                result.loc[result.index[i], 'bull_fvg_low'] = result['high'].iloc[i-2]
                result.loc[result.index[i], 'bull_fvg_high'] = result['low'].iloc[i]
    
    # Look for bearish FVG (gap down)
    for i in range(2, len(result)):
        # Bearish FVG: current candle's high < previous candle's low
        if result['high'].iloc[i] < result['low'].iloc[i-2]:
            # Calculate gap percentage
            gap_pct = (result['low'].iloc[i-2] - result['high'].iloc[i]) / result['low'].iloc[i-2]
            
            if gap_pct >= min_gap_pct:
                result.loc[result.index[i], 'bear_fvg'] = 1
                result.loc[result.index[i], 'bear_fvg_low'] = result['high'].iloc[i]
                result.loc[result.index[i], 'bear_fvg_high'] = result['low'].iloc[i-2]
    
    return result


def liquidity_levels(
    df: pd.DataFrame,
    window_size: int = 10,
    threshold: int = 2,
    swing_threshold_pct: float = 0.5
) -> pd.DataFrame:
    """
    Identify liquidity levels (equal highs, equal lows, swing highs/lows).
    
    Liquidity levels are areas where stop losses are clustered, making them
    attractive targets for smart money.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'high', 'low' columns.
    window_size : int, default 10
        Number of candles to consider when looking for swing points.
    threshold : int, default 2
        Minimum number of similar highs/lows needed to establish a liquidity level.
    swing_threshold_pct : float, default 0.5
        Percentage threshold for identifying swing highs/lows.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'equal_highs': Equal highs liquidity level
        - 'equal_lows': Equal lows liquidity level
        - 'swing_high': Swing high identification
        - 'swing_low': Swing low identification
        - 'eq_high_level': Price level of equal highs
        - 'eq_low_level': Price level of equal lows
    """
    required_cols = ['high', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()
    
    # Initialize liquidity level columns
    result['equal_highs'] = 0
    result['equal_lows'] = 0
    result['swing_high'] = 0
    result['swing_low'] = 0
    result['eq_high_level'] = np.nan
    result['eq_low_level'] = np.nan
    
    # Need at least window_size candles to identify swing points
    if len(result) < window_size:
        return result
    
    # Identify swing highs and lows
    for i in range(window_size, len(result) - window_size):
        # Check if this is a swing high
        if all(result['high'].iloc[i] > result['high'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(result['high'].iloc[i] > result['high'].iloc[i+j] for j in range(1, window_size+1)):
            result.loc[result.index[i], 'swing_high'] = 1
        
        # Check if this is a swing low
        if all(result['low'].iloc[i] < result['low'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(result['low'].iloc[i] < result['low'].iloc[i+j] for j in range(1, window_size+1)):
            result.loc[result.index[i], 'swing_low'] = 1
    
    # Find equal highs (within a small tolerance)
    tolerance_high = result['high'].mean() * 0.0005  # 0.05% tolerance
    
    for i in range(window_size, len(result)):
        # Look for similar highs in the past
        high_val = result['high'].iloc[i]
        similar_highs = sum(abs(result['high'].iloc[i-window_size:i] - high_val) < tolerance_high)
        
        if similar_highs >= threshold:
            result.loc[result.index[i], 'equal_highs'] = 1
            result.loc[result.index[i], 'eq_high_level'] = high_val
    
    # Find equal lows (within a small tolerance)
    tolerance_low = result['low'].mean() * 0.0005  # 0.05% tolerance
    
    for i in range(window_size, len(result)):
        # Look for similar lows in the past
        low_val = result['low'].iloc[i]
        similar_lows = sum(abs(result['low'].iloc[i-window_size:i] - low_val) < tolerance_low)
        
        if similar_lows >= threshold:
            result.loc[result.index[i], 'equal_lows'] = 1
            result.loc[result.index[i], 'eq_low_level'] = low_val
    
    return result


def market_structure(
    df: pd.DataFrame,
    window_size: int = 5
) -> pd.DataFrame:
    """
    Identify market structure shifts (higher highs/lows, lower highs/lows).
    
    Market structure helps understand trend direction and potential reversal points.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'high', 'low' columns.
    window_size : int, default 5
        Window size for identifying significant swing points.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'hh': Higher high
        - 'll': Lower low
        - 'hl': Higher low
        - 'lh': Lower high
        - 'structure': Current market structure (uptrend, downtrend, consolidation)
    """
    required_cols = ['high', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()
    
    # Initialize market structure columns
    result['hh'] = 0  # Higher high
    result['ll'] = 0  # Lower low
    result['hl'] = 0  # Higher low
    result['lh'] = 0  # Lower high
    result['structure'] = 'undefined'
    
    # Not enough data to analyze market structure
    if len(result) < 2 * window_size:
        return result
    
    # Find significant swing highs and lows
    result['swing_high'] = 0
    result['swing_low'] = 0
    
    for i in range(window_size, len(result) - window_size):
        # Significant swing high
        if all(result['high'].iloc[i] > result['high'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(result['high'].iloc[i] > result['high'].iloc[i+j] for j in range(1, window_size+1)):
            result.loc[result.index[i], 'swing_high'] = 1
        
        # Significant swing low
        if all(result['low'].iloc[i] < result['low'].iloc[i-j] for j in range(1, window_size+1)) and \
           all(result['low'].iloc[i] < result['low'].iloc[i+j] for j in range(1, window_size+1)):
            result.loc[result.index[i], 'swing_low'] = 1
    
    # Analyze market structure by comparing swing points
    prev_swing_high_idx = None
    prev_swing_low_idx = None
    
    for i in range(len(result)):
        # Process swing highs
        if result['swing_high'].iloc[i] == 1:
            if prev_swing_high_idx is not None:
                # Higher high
                if result['high'].iloc[i] > result['high'].iloc[prev_swing_high_idx]:
                    result.loc[result.index[i], 'hh'] = 1
                # Lower high
                else:
                    result.loc[result.index[i], 'lh'] = 1
            prev_swing_high_idx = i
        
        # Process swing lows
        if result['swing_low'].iloc[i] == 1:
            if prev_swing_low_idx is not None:
                # Higher low
                if result['low'].iloc[i] > result['low'].iloc[prev_swing_low_idx]:
                    result.loc[result.index[i], 'hl'] = 1
                # Lower low
                else:
                    result.loc[result.index[i], 'll'] = 1
            prev_swing_low_idx = i
        
        # Determine current market structure
        if i >= 2 * window_size:
            recent_window = result.iloc[max(0, i-10*window_size):i+1]
            hh_count = recent_window['hh'].sum()
            hl_count = recent_window['hl'].sum()
            lh_count = recent_window['lh'].sum()
            ll_count = recent_window['ll'].sum()
            
            # Uptrend: higher highs and higher lows
            if hh_count > 0 and hl_count > 0 and hh_count + hl_count > lh_count + ll_count:
                result.loc[result.index[i], 'structure'] = 'uptrend'
            # Downtrend: lower highs and lower lows
            elif lh_count > 0 and ll_count > 0 and lh_count + ll_count > hh_count + hl_count:
                result.loc[result.index[i], 'structure'] = 'downtrend'
            # Consolidation: mixed signals
            else:
                result.loc[result.index[i], 'structure'] = 'consolidation'
    
    # Clean up temporary columns
    result = result.drop(columns=['swing_high', 'swing_low'])
    
    return result


def breaker_blocks(
    df: pd.DataFrame,
    window_size: int = 5,
    threshold_pct: float = 0.3
) -> pd.DataFrame:
    """
    Identify breaker blocks.
    
    Breaker blocks are a specific type of order block that form after price
    breaks a swing point and returns to it, often reversing at the level.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
    window_size : int, default 5
        Number of candles to look back to identify swing points.
    threshold_pct : float, default 0.3
        Minimum percentage move required to identify a significant move.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'bull_breaker': Bullish breaker block
        - 'bear_breaker': Bearish breaker block
        - 'bull_breaker_low': Lower boundary of bullish breaker block
        - 'bull_breaker_high': Upper boundary of bullish breaker block
        - 'bear_breaker_low': Lower boundary of bearish breaker block
        - 'bear_breaker_high': Upper boundary of bearish breaker block
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()
    
    # Calculate liquidity levels first to get swing highs and lows
    liq = liquidity_levels(result, window_size)
    
    # Create swing point columns
    result['swing_high'] = liq['swing_high']
    result['swing_low'] = liq['swing_low']
    
    # Initialize breaker block columns
    result['bull_breaker'] = 0
    result['bear_breaker'] = 0
    result['bull_breaker_low'] = np.nan
    result['bull_breaker_high'] = np.nan
    result['bear_breaker_low'] = np.nan
    result['bear_breaker_high'] = np.nan
    
    # Find bullish breaker blocks
    # (Price breaks below a swing low, then returns to that level)
    for i in range(window_size * 2, len(result)):
        # Look for swing lows in the past
        for j in range(i - window_size, i - 1):
            if result['swing_low'].iloc[j]:
                swing_low_price = result['low'].iloc[j]
                
                # Check if price broke below the swing low
                broke_below = False
                break_index = None
                for k in range(j + 1, i):
                    if result['low'].iloc[k] < swing_low_price * (1 - threshold_pct / 100):
                        broke_below = True
                        break_index = k
                        break
                
                # Check if price returned to the swing low level
                if broke_below and abs(result['low'].iloc[i] - swing_low_price) / swing_low_price < 0.005:
                    result.loc[result.index[i], 'bull_breaker'] = 1
                    result.loc[result.index[i], 'bull_breaker_low'] = min(result['low'].iloc[break_index], result['low'].iloc[i])
                    result.loc[result.index[i], 'bull_breaker_high'] = max(result['high'].iloc[break_index], result['high'].iloc[i])
                    break
    
    # Find bearish breaker blocks
    # (Price breaks above a swing high, then returns to that level)
    for i in range(window_size * 2, len(result)):
        # Look for swing highs in the past
        for j in range(i - window_size, i - 1):
            if result['swing_high'].iloc[j]:
                swing_high_price = result['high'].iloc[j]
                
                # Check if price broke above the swing high
                broke_above = False
                break_index = None
                for k in range(j + 1, i):
                    if result['high'].iloc[k] > swing_high_price * (1 + threshold_pct / 100):
                        broke_above = True
                        break_index = k
                        break
                
                # Check if price returned to the swing high level
                if broke_above and abs(result['high'].iloc[i] - swing_high_price) / swing_high_price < 0.005:
                    result.loc[result.index[i], 'bear_breaker'] = 1
                    result.loc[result.index[i], 'bear_breaker_low'] = min(result['low'].iloc[break_index], result['low'].iloc[i])
                    result.loc[result.index[i], 'bear_breaker_high'] = max(result['high'].iloc[break_index], result['high'].iloc[i])
                    break
    
    # Clean up temporary columns
    result = result.drop(columns=['swing_high', 'swing_low'])
    
    return result


def mitigation(
    df: pd.DataFrame,
    liquidity_window: int = 10,
    threshold_pct: float = 0.1
) -> pd.DataFrame:
    """
    Identify liquidity mitigation events (sweeps).
    
    Liquidity mitigation occurs when price sweeps beyond a liquidity level
    and then reverses, often indicating smart money activity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
    liquidity_window : int, default 10
        Number of candles to look back to identify liquidity levels.
    threshold_pct : float, default 0.1
        Percentage threshold for identifying a sweep beyond the liquidity level.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'high_sweep': High liquidity sweep
        - 'low_sweep': Low liquidity sweep
        - 'high_sweep_level': Price level of high sweep
        - 'low_sweep_level': Price level of low sweep
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()
    
    # Find liquidity levels first
    liq_levels = liquidity_levels(result, liquidity_window)
    
    # Initialize mitigation columns
    result['high_sweep'] = 0
    result['low_sweep'] = 0
    result['high_sweep_level'] = np.nan
    result['low_sweep_level'] = np.nan
    
    # Find high sweeps (price goes above equal highs and then reverses down)
    for i in range(liquidity_window, len(result) - 1):
        if liq_levels['equal_highs'].iloc[i] == 1:
            # Get the equal high level
            high_level = liq_levels['eq_high_level'].iloc[i]
            
            # Look for a sweep in the next few candles
            for j in range(i + 1, min(i + 5, len(result) - 1)):
                # Price goes above the high level
                if result['high'].iloc[j] > high_level * (1 + threshold_pct / 100):
                    # And then reverses down
                    if result['close'].iloc[j + 1] < result['open'].iloc[j + 1]:
                        result.loc[result.index[j], 'high_sweep'] = 1
                        result.loc[result.index[j], 'high_sweep_level'] = high_level
                        break
    
    # Find low sweeps (price goes below equal lows and then reverses up)
    for i in range(liquidity_window, len(result) - 1):
        if liq_levels['equal_lows'].iloc[i] == 1:
            # Get the equal low level
            low_level = liq_levels['eq_low_level'].iloc[i]
            
            # Look for a sweep in the next few candles
            for j in range(i + 1, min(i + 5, len(result) - 1)):
                # Price goes below the low level
                if result['low'].iloc[j] < low_level * (1 - threshold_pct / 100):
                    # And then reverses up
                    if result['close'].iloc[j + 1] > result['open'].iloc[j + 1]:
                        result.loc[result.index[j], 'low_sweep'] = 1
                        result.loc[result.index[j], 'low_sweep_level'] = low_level
                        break
    
    return result


def premium_discount(
    df: pd.DataFrame,
    vwap_period: int = 1,
    threshold_pct: float = 0.5
) -> pd.DataFrame:
    """
    Identify premium and discount zones based on VWAP.
    
    Premium is when price is trading above VWAP, discount is when price is trading below VWAP.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close', 'volume' columns.
    vwap_period : int, default 1
        Number of days to calculate VWAP over.
    threshold_pct : float, default 0.5
        Percentage threshold for identifying significant premium/discount.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'vwap': Volume Weighted Average Price
        - 'premium': Premium zone indicator (1 = significant premium)
        - 'discount': Discount zone indicator (1 = significant discount)
        - 'premium_pct': Percentage above VWAP
        - 'discount_pct': Percentage below VWAP
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()
    
    # Initialize columns
    result['vwap'] = np.nan
    result['premium'] = 0
    result['discount'] = 0
    result['premium_pct'] = np.nan
    result['discount_pct'] = np.nan
    
    # Calculate typical price
    result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
    
    # Calculate cumulative values for VWAP
    result['tp_x_vol'] = result['typical_price'] * result['volume']
    
    # Calculate VWAP for each period
    # Reset cumulative values at the start of each period
    if 'timestamp' in result.index.names:
        # If timestamp is the index, use it to determine periods
        result['date'] = result.index.date
        period_groups = result.groupby('date')
    else:
        # Otherwise divide into equal periods
        period_size = len(result) // vwap_period
        result['period'] = np.repeat(range(vwap_period), period_size)[:len(result)]
        period_groups = result.groupby('period')
    
    # Calculate VWAP for each period
    for period, group in period_groups:
        cumulative_tp_x_vol = group['tp_x_vol'].cumsum()
        cumulative_vol = group['volume'].cumsum()
        
        # Calculate VWAP
        vwap = cumulative_tp_x_vol / cumulative_vol
        
        # Update the VWAP column
        result.loc[group.index, 'vwap'] = vwap
    
    # Calculate premium/discount percentage
    result['premium_pct'] = (result['close'] - result['vwap']) / result['vwap'] * 100
    result['discount_pct'] = (result['vwap'] - result['close']) / result['vwap'] * 100
    
    # Identify premium and discount zones
    result.loc[result['premium_pct'] > threshold_pct, 'premium'] = 1
    result.loc[result['discount_pct'] > threshold_pct, 'discount'] = 1
    
    # Clean up temporary columns
    result = result.drop(columns=['typical_price', 'tp_x_vol'])
    if 'date' in result.columns:
        result = result.drop(columns=['date'])
    if 'period' in result.columns:
        result = result.drop(columns=['period'])
    
    return result


def choch_pattern(
    df: pd.DataFrame,
    window_size: int = 10
) -> pd.DataFrame:
    """
    Identify Change of Character (CHoCH) patterns.
    
    Change of Character is a pattern where price breaks a significant swing point,
    indicating a potential trend change.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'high', 'low' columns.
    window_size : int, default 10
        Window size for identifying significant swing points.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'bull_choch': Bullish Change of Character (1 or 0)
        - 'bear_choch': Bearish Change of Character (1 or 0)
        - 'bull_choch_level': Price level of bullish CHoCH
        - 'bear_choch_level': Price level of bearish CHoCH
    """
    required_cols = ['high', 'low']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()
    
    # Get market structure
    ms = market_structure(result, window_size)
    
    # Initialize CHoCH columns
    result['bull_choch'] = 0
    result['bear_choch'] = 0
    result['bull_choch_level'] = np.nan
    result['bear_choch_level'] = np.nan
    
    # Find swing highs and lows
    swing_highs = [i for i, x in enumerate(ms['swing_high'].values) if x == 1]
    swing_lows = [i for i, x in enumerate(ms['swing_low'].values) if x == 1]
    
    # Loop through swing highs to find bearish CHoCH
    for i in range(1, len(swing_highs)):
        current_idx = swing_highs[i]
        prev_idx = swing_highs[i-1]
        
        # Bearish CHoCH: Lower High after a Higher High
        if ms['lh'].iloc[current_idx] == 1 and ms['hh'].iloc[prev_idx] == 1:
            result.loc[result.index[current_idx], 'bear_choch'] = 1
            result.loc[result.index[current_idx], 'bear_choch_level'] = result['high'].iloc[current_idx]
    
    # Loop through swing lows to find bullish CHoCH
    for i in range(1, len(swing_lows)):
        current_idx = swing_lows[i]
        prev_idx = swing_lows[i-1]
        
        # Bullish CHoCH: Higher Low after a Lower Low
        if ms['hl'].iloc[current_idx] == 1 and ms['ll'].iloc[prev_idx] == 1:
            result.loc[result.index[current_idx], 'bull_choch'] = 1
            result.loc[result.index[current_idx], 'bull_choch_level'] = result['low'].iloc[current_idx]
    
    return result


def imbalance(
    df: pd.DataFrame,
    min_imbalance_pct: float = 0.2
) -> pd.DataFrame:
    """
    Identify imbalances in the market.
    
    Imbalances occur when there is a gap between candles, indicating potential institutional activity.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data with 'open', 'high', 'low', 'close' columns.
    min_imbalance_pct : float, default 0.2
        Minimum percentage gap required to identify an imbalance.
        
    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional columns:
        - 'up_imbalance': Upward imbalance (1 or 0)
        - 'down_imbalance': Downward imbalance (1 or 0)
        - 'up_imbalance_low': Lower boundary of upward imbalance
        - 'up_imbalance_high': Upper boundary of upward imbalance
        - 'down_imbalance_low': Lower boundary of downward imbalance
        - 'down_imbalance_high': Upper boundary of downward imbalance
    """
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")
    
    # Create a copy of the input DataFrame to avoid modifying the original
    result = df.copy()
    
    # Initialize imbalance columns
    result['up_imbalance'] = 0
    result['down_imbalance'] = 0
    result['up_imbalance_low'] = np.nan
    result['up_imbalance_high'] = np.nan
    result['down_imbalance_low'] = np.nan
    result['down_imbalance_high'] = np.nan
    
    # Need at least 2 candles to identify imbalances
    if len(result) < 2:
        return result
    
    # Identify upward imbalance (current candle's low > previous candle's high)
    for i in range(1, len(result)):
        if result['low'].iloc[i] > result['high'].iloc[i-1]:
            # Calculate imbalance percentage
            imbalance_pct = (result['low'].iloc[i] - result['high'].iloc[i-1]) / result['high'].iloc[i-1]
            
            if imbalance_pct >= min_imbalance_pct / 100:
                result.loc[result.index[i], 'up_imbalance'] = 1
                result.loc[result.index[i], 'up_imbalance_low'] = result['high'].iloc[i-1]
                result.loc[result.index[i], 'up_imbalance_high'] = result['low'].iloc[i]
    
    # Identify downward imbalance (current candle's high < previous candle's low)
    for i in range(1, len(result)):
        if result['high'].iloc[i] < result['low'].iloc[i-1]:
            # Calculate imbalance percentage
            imbalance_pct = (result['low'].iloc[i-1] - result['high'].iloc[i]) / result['low'].iloc[i-1]
            
            if imbalance_pct >= min_imbalance_pct / 100:
                result.loc[result.index[i], 'down_imbalance'] = 1
                result.loc[result.index[i], 'down_imbalance_low'] = result['high'].iloc[i]
                result.loc[result.index[i], 'down_imbalance_high'] = result['low'].iloc[i-1]
    
    return result