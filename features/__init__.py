"""
Feature engineering module for quantitative trading.

This module provides functions to calculate various technical indicators and 
features used in quantitative trading strategies.
"""

# Import submodules
from . import trend
from . import volatility
from . import candle
from . import math
from . import market_regime
from . import smc

# Import main functions directly for easier access
from .trend import sma, ema, kama, linear_regression_slope, hull_moving_average, rsi
from .volatility import close_to_close_volatility, parkinson_volatility, rogers_satchell_volatility, yang_zhang_volatility, garch_volatility, average_true_range, bollinger_bands, keltner_channels, historical_volatility
from .candle import candle_direction, candle_filling, candle_amplitude, candle_spread, candle_information, doji_pattern, hammer_pattern, engulfing_pattern
from .math import log_returns, pct_returns, rate_of_change, momentum, autocorrelation, hurst, zscore, normalization, exponential_smoothing
from .market_regime import sma_market_regime, kama_market_regime, rsi_market_regime, bollinger_market_regime, volatility_regime, multi_timeframe_regime, hurst_market_regime
from .smc import order_blocks, fair_value_gaps, liquidity_levels, market_structure, breaker_blocks, mitigation, premium_discount, choch_pattern, imbalance

