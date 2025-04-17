from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="quantrader",
    version="0.1.0",
    author="dsage",
    author_email="divinef837@gmail.com",
    description="A comprehensive Python library for quantitative trading research and implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Divine572/quantrader",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "scipy>=1.5.0",
        "scikit-learn>=0.23.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "numba>=0.50.0",
        "statsmodels>=0.12.0",
        "tqdm>=4.50.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "flake8>=3.8.0",
            "black>=20.8b1",
            "isort>=5.7.0",
            "mypy>=0.800",
        ],
        "ml": [
            "tensorflow>=2.4.0",
            "torch>=1.7.0",
            "xgboost>=1.3.0",
            "lightgbm>=3.1.0",
        ],
    },
)