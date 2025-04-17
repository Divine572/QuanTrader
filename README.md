# QuanTrader

<p align="center">
  <img src="/Quantrader.png" alt="QuanTrader Logo" width="200"/>
</p>

<p align="center">
  <b>A comprehensive Python library for quantitative trading research and implementation</b>
</p>

<p align="center">
  <a href="#installation"><strong>Installation</strong></a> ·
  <a href="#key-features"><strong>Key Features</strong></a> ·
  <a href="#getting-started"><strong>Getting Started</strong></a> ·
  <a href="#modules"><strong>Modules</strong></a> ·
  <a href="#examples"><strong>Examples</strong></a> ·
  <a href="#tutorials"><strong>Tutorials</strong></a> ·
  <a href="#project-structure"><strong>Project Structure</strong></a> ·
  <a href="#contributing"><strong>Contributing</strong></a> ·
  <a href="#license"><strong>License</strong></a>
</p>

---

## Overview

QuanTrader is an open-source Python library designed for quantitative traders and researchers, offering powerful tools for every stage of the trading strategy development lifecycle. From feature engineering to backtesting, QuanTrader provides optimized functions and intuitive APIs to simplify your workflow.

Whether you're a beginner exploring algorithmic trading or an experienced quantitative analyst, QuanTrader offers the tools you need to develop, test, and refine your trading strategies with professional-grade functionality.

## Installation

Install QuanTrader using pip:

```bash
pip install quantrader
```

For the development version, install directly from GitHub:

```bash
pip install git+https://github.com/Divine572/quantrader.git
```

## Key Features

- **⚡ Optimized Performance**: Built with vectorized operations and Numba acceleration
- **🔧 Modular Design**: Mix and match components to create custom trading systems
- **📊 Comprehensive Tools**: Feature engineering, target engineering, model building, and backtesting
- **🔍 Research-Focused**: Designed for both rapid prototyping and production-ready strategies
- **📈 ML Integration**: Seamless workflow with scikit-learn, TensorFlow, and PyTorch
- **📚 Well-Documented**: Extensive documentation with practical examples and tutorials
- **💰 Smart Money Concepts**: Identification of key SMC patterns (liquidity, order blocks, FVGs)

## Getting Started

Import the necessary modules:

```python
# Feature engineering tools
import quantrader.features as qf

# Target engineering tools
import quantrader.targets as qt

# Feature selection utilities
import quantrader.selection as qs

# Model building and evaluation
import quantrader.models as qm

# Backtesting framework
import quantrader.backtest as qb

# Risk management tools
import quantrader.risk as qr
```

Basic usage example:

```python
import pandas as pd
import quantrader.features as qf

# Load your OHLCV data
df = pd.read_csv('your_data.csv')

# Calculate technical indicators
df['sma_20'] = qf.trend.sma(df, col='close', window_size=20)
df['volatility'] = qf.volatility.parkinson_volatility(df, window_size=30)
df['regime'] = qf.market_regime.kama_market_regime(df, col='close')

# Create a target variable
import quantrader.targets as qt
df['target'] = qt.directional.future_returns_sign(df, window_size=10)

# Print the result
print(df.tail())
```

## Modules

### Features Engineering (`quantrader.features`)

- **Trend**: Moving averages (SMA, EMA, KAMA), linear regression slope
- **Volatility**: Multiple estimators (Close-to-close, Parkinson, Rogers-Satchell, Yang-Zhang)
- **Candle**: Candlestick analysis (direction, filling, amplitude, spread)
- **Math**: Mathematical transformations (derivatives, log returns, autocorrelation, Hurst exponent)
- **Market Regime**: Identify market phases (bullish, bearish, ranging)
- **Smart Money Concepts**: Order blocks, fair value gaps, breaker blocks, liquidity levels, market structure

### Target Engineering (`quantrader.targets`)

- **Directional**: Trend prediction labels (future return sign, direction)
- **Magnitude**: Continuous targets (future returns, volatility prediction)
- **Event-Based**: Turning point detection (peaks and valleys)
- **Quantile-Based**: Multi-class labeling based on return distributions

### Feature Selection (`quantrader.selection`)

- **Correlation**: Analysis of feature relationships
- **Information Value**: Metrics for feature importance
- **Elimination**: Methods for removing redundant features
- **Importance**: Extraction of significant features from models

### Model Building (`quantrader.models`)

- **Cross-Validation**: Time-series specific validation frameworks
- **Optimization**: Parameter tuning for trading models
- **Metrics**: Performance evaluation specific to trading
- **Persistence**: Model saving and loading utilities

### Backtesting (`quantrader.backtest`)

- **Signals**: Converting predictions to trade signals
- **Position Sizing**: Dynamic position size calculation
- **Execution**: Trade simulation with various order types
- **Reporting**: Comprehensive performance analysis

### Risk Management (`quantrader.risk`)

- **Position**: Optimal position sizing strategies
- **Drawdown**: Methods to control maximum drawdown
- **Portfolio**: Multi-asset optimization techniques
- **Value at Risk**: VaR and conditional VaR calculations

## Examples

### Calculate Multiple Technical Indicators

```python
import pandas as pd
import quantrader.features as qf

# Load data
df = pd.read_csv('data.csv')

# Calculate technical indicators
df['sma_20'] = qf.trend.sma(df, col='close', window_size=20)
df['sma_50'] = qf.trend.sma(df, col='close', window_size=50)
df['kama'] = qf.trend.kama(df, col='close', l1=10, l2=2, l3=30)
df['volatility'] = qf.volatility.yang_zhang_volatility(df, window_size=20)
df['hurst'] = qf.math.hurst(df, col='close', window_size=100)
df['candle_way'], df['filling'], df['amplitude'] = qf.candle.candle_information(df)

print(df.tail())
```

### Create a Trading Target

```python
import pandas as pd
import quantrader.features as qf
import quantrader.targets as qt

# Load data
df = pd.read_csv('data.csv')

# Create feature
df['sma_cross'] = qf.market_regime.sma_market_regime(df, 'close', 
                                                    fast_window=20, 
                                                    slow_window=50)

# Create target
df['future_ret'] = qt.magnitude.future_returns(df, window_size=10)
df['direction'] = qt.directional.future_returns_sign(df, window_size=10)
df['peaks_valleys'] = qt.event_based.detect_peaks_valleys(df)

# Create quantile-based labels
df['quantile_labels'], upper_q, lower_q = qt.directional.quantile_label(
    df, col='future_ret', upper_quantile_level=0.67, return_thresholds=True
)

print(df[['close', 'sma_cross', 'future_ret', 'direction', 'quantile_labels']].tail())
print(f"Upper quantile threshold: {upper_q}, Lower quantile threshold: {lower_q}")
```

### Smart Money Concepts Analysis

```python
import pandas as pd
import quantrader.features as qf

# Load data
df = pd.read_csv('data.csv')

# Identify order blocks
ob_df = qf.smc.order_blocks(df, window_size=5, threshold_pct=0.5)

# Find fair value gaps
fvg_df = qf.smc.fair_value_gaps(df, min_gap_pct=0.1)

# Identify liquidity levels
liq_df = qf.smc.liquidity_levels(df, window_size=10)

# Analyze market structure
ms_df = qf.smc.market_structure(df, window_size=5)

# Print the results
print("Order Blocks:")
print(ob_df[['close', 'bull_ob', 'bear_ob']].tail())

print("\nFair Value Gaps:")
print(fvg_df[['close', 'bull_fvg', 'bear_fvg']].tail())

print("\nMarket Structure:")
print(ms_df[['close', 'structure']].tail())
```

## Tutorials

We provide Jupyter notebook tutorials covering various aspects of quantitative trading:

1. **[Getting Started with QuanTrader](tutorials/01_getting_started.ipynb)** - Basic setup and workflow
2. **[Feature Engineering Fundamentals](tutorials/02_feature_engineering.ipynb)** - Creating effective technical indicators
3. **[Target Engineering for Trading Models](tutorials/03_target_engineering.ipynb)** - Designing prediction targets
4. **[Feature Selection Techniques](tutorials/04_feature_selection.ipynb)** - Identifying relevant features
5. **[Building Effective Trading Models](tutorials/05_model_building.ipynb)** - Model training and evaluation
6. **[Backtesting Trading Strategies](tutorials/06_backtesting.ipynb)** - Testing strategy performance
7. **[Risk Management Principles](tutorials/07_risk_management.ipynb)** - Protecting your trading capital
8. **[Smart Money Concepts](tutorials/08_smart_money_concepts.ipynb)** - Understanding and implementing SMC trading

## Project Structure

```
quantrader/
│
├── __init__.py                # Package initialization
├── features/                  # Feature engineering module
│   ├── __init__.py
│   ├── trend.py               # Trend indicators (SMA, EMA, KAMA)
│   ├── volatility.py          # Volatility estimators
│   ├── candle.py              # Candlestick analysis
│   ├── math.py                # Mathematical transformations
│   ├── market_regime.py       # Market phase identification
│   └── smc.py                 # Smart Money Concepts
│
├── targets/                   # Target engineering module
│   ├── __init__.py
│   ├── directional.py         # Trend prediction labels
│   ├── magnitude.py           # Continuous targets
│   ├── event_based.py         # Turning point detection
│   └── quantile_based.py      # Multi-class labeling
│
├── selection/                 # Feature selection module
│   ├── __init__.py
│   ├── correlation.py         # Correlation analysis
│   ├── information_value.py   # Information value metrics
│   ├── elimination.py         # Feature elimination
│   └── importance.py          # Feature importance
│
├── models/                    # Model building module
│   ├── __init__.py
│   ├── cross_validation.py    # Time-series cross-validation
│   ├── optimization.py        # Parameter tuning
│   ├── metrics.py             # Performance metrics
│   └── persistence.py         # Model saving/loading
│
├── backtest/                  # Backtesting module
│   ├── __init__.py
│   ├── signals.py             # Signal generation
│   ├── position_sizing.py     # Position size calculation
│   ├── execution.py           # Trade simulation
│   └── reporting.py           # Performance analysis
│
├── risk/                      # Risk management module
│   ├── __init__.py
│   ├── position.py            # Position sizing strategies
│   ├── drawdown.py            # Drawdown control
│   ├── portfolio.py           # Multi-asset optimization
│   └── value_at_risk.py       # VaR calculations
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── validation.py          # Input validation
│   └── performance.py         # Performance optimization
│
├── tutorials/                 # Tutorial notebooks
│   ├── 01_getting_started.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_target_engineering.ipynb
│   ├── 04_feature_selection.ipynb
│   ├── 05_model_building.ipynb
│   ├── 06_backtesting.ipynb
│   ├── 07_risk_management.ipynb
│   └── 08_smart_money_concepts.ipynb
```

## Contributing

We welcome contributions to QuanTrader! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

QuanTrader is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Inspired by various open-source quant libraries, including Quantreo
- Special thanks to all contributors who have dedicated their time and expertise

---

<p align="center">
  Made with ❤️ for quant traders everywhere
</p>