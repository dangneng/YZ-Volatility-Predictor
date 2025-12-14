# Volatility Predictor

A machine learning system for predicting next-day Yang-Zhang volatility using ensemble methods, GARCH modeling, and historical daily market data.

## Overview

This project implements a volatility forecasting system that combines:
- **Yang-Zhang volatility estimation** - Accounts for overnight gaps and intraday range (Outperforms basic daily estimators eg. Close-to-Close, Parkinson, Garman-Klass, Rogers-Satchell)
- **GARCH(1,1) modeling** - Captures volatility clustering and memory
- **VIX feature inclusion** - Captures market's macro effect on individual stocks
- **Stacking ensemble** - Combines XGBoost, Lasso, and K-Nearest Neighbours regressors
- **Web interface** - FastAPI-powered UI for real-time predictions

## Features

- **Ensemble learning** that outperforms singular models
- **Next-day volatility predictions** for any US stock ticker
- **Web interface** with responsive layout
- **Model interpretability** with contribution weights
- **Fast training** with parallel processing
- **Robust error handling** with comprehensive validation

## Architecture

### Model Pipeline

```
Market Data (OHLC + VIX)
         ↓
  Feature Engineering (Default 2 lags)
  - Lagged OHLC
  - Lagged YZ volatility
  - Lagged returns
  - GARCH predicted conditional volatility
         ↓
   Stacking Ensemble
   ├─ XGBoost Regressor
   ├─ LassoCV (L1) Regressor
   └─ K-Nearest Neighbors Regressor
         ↓
   Ridge Meta-Learner
         ↓
  Volatility Prediction
```

### Base Models

| Model | Purpose | Key Parameters |
|-------|---------|----------------|
| **XGBoost** | Captures non-linear patterns and feature interactions | n_estimators = 100, lr = 0.15, depth = 4, subsample = 0.8, colsample_bytree = 0.8 |
| **Lasso** | Linear baseline with feature selection | alphas = [1.0, 0.1, 0.01], max_iter = 10000, selection = random |
| **KNN** | Local pattern recognition | Default neighbors |
| **Ridge** | Meta-learner for optimal combination without feature selection | alpha=1.0 |

## Installation

### Prerequisites

- Python 3.11+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/dangneng/YZ-Volatility-Predictor
cd YZ-Volatility-Predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn yfinance scikit-learn xgboost pandas numpy arch pandas-market-calendars joblib pydantic
```

### Dependencies

```txt
fastapi>=0.104.0
uvicorn>=0.24.0
yfinance>=0.2.30
scikit-learn>=1.3.0
xgboost>=2.0.0
arch>=6.2.0
pandas>=2.1.0
numpy>=1.24.0
pandas-market-calendars>=4.3.0
joblib>=1.3.0
pydantic>=2.0.0
```

## Quick Start

### Live Demo

**Live App:** [https://yz-volatility-predictor.onrender.com](https://yz-volatility-predictor.onrender.com)

### Running the Web Interface

```bash
python main.py
```

Then open your browser to:
```
http://localhost:8000
```

### Using the API

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "ticker": "AAPL",               # desired ticker
        "target_date": "2025-12-15"     # predicted date
    }
)

result = response.json()
print(f"Predicted volatility: {result['predicted_volatility']*100:.2f}%")
print(f"Confidence interval: {result['confidence_interval']}")
```

### Helper Function Usage

```python
from helper import preprocess, evaluate, model, symbols

# Single prediction
scaler_Y, train_X, train_Y, test_X = preprocess(
    symbol = "AAPL",
    target_date = "2025-12-15",
    years = 2,
    lag = 2
)

model.fit(train_X, train_Y.values.ravel())
prediction = model.predict(test_X)

# Backtesting
mse, variance = evaluate(
    symbols = symbols,          # List of stock tickers to use
    start = "2020-01-01",       # Start of testing timeframe (Includes training window)
    end = "2025-01-01",         # End of testing timeframe
    model = model,
    x = 1000,                   # Number of random samples (Random stock * date predictions)
    years = 2,                  # Length of training data in years (Floats accepted)
    lag = 2                     # Length of lagged features
)
```

## Features & Configuration

### Preprocessing Options

```python
preprocess(
    symbol = "AAPL",            # Stock ticker
    target_date = "2025-12-16", # Prediction date
    years = 2,                  # Historical data window (1-3 recommended)
    underlying_ar = False,      # Use AR(1) mean in GARCH (False -- Assume expected returns = 0)
    lag = 2                     # Number of lags (1-5 recommended)
)
```

### Optimized Model Hyperparameters

**XGBoost:**
```python
xgb_params = {
    "n_estimators": 100,
    "learning_rate": 0.15,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}
```

**Lasso:**
```python
lasso_params = {
    "alphas": [1.0, 0.1, 0.01],
    "cv": 5,
    "max_iter": 10000,
    "tol": 1e-4,
    "selection": "random"
}
```

## Performance

### Evaluation Metrics

- **MSE (Mean Squared Error)** - Overall prediction accuracy
- **Error Variance** - Prediction stability
- **Confidence Intervals** - Model uncertainty quantification

## Web Interface

The web interface provides:

- **Interactive input form** for ticker and date selection
- **Confidence intervals** with visual display
- **Model contributions** showing individual model weights
- **Responsive design** that works on mobile and desktop
- **Error handling** with user-friendly messages

### Screenshot Features

- Gradient background (slate-900 → blue-900)
- Glassmorphism cards with backdrop blur
- Smooth animations and transitions
- Accessible color contrast
- Professional typography

## API Reference

### POST `/predict`

**Request:**
```json
{
  "ticker": "AAPL",
  "target_date": "2025-12-16"
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "target_date": "2025-12-16",
  "predicted_volatility": 0.0234,
  "confidence_interval": {
    "lower": 0.0198,
    "upper": 0.0270
  },
  "model_contributions": {
    "XGB": 0.52,
    "Lasso": 0.31,
    "KNN-R": 0.17
  },
  "model_type": "StackingRegressor (XGB + Lasso + KNN)",
  "timestamp": "2025-12-14T10:30:00"
}
```

### GET `/`

Serves the web UI interface.

## Project Structure

```
volatility-predictor/
├── main.py                 # FastAPI application & web UI
├── helper.py               # Core functions (preprocess, evaluate, models -- For backtesting etc.)
├── README.md               # This file
├── requirements.txt        # Dependencies
├── .gitignore              # Git ignore rules (optional)
└── notebooks/              # Jupyter notebooks for analysis (optional)
```

## Troubleshooting

### Common Issues

**"No data available for ticker"**
- Check ticker symbol is valid (e.g., AAPL not Apple)
- Ensure date is a valid trading day

**"Insufficient data for prediction"**
- Need at least 50 trading days of data
- Increase `years` parameter to 2 or 3
- Check if ticker has sufficient history

**Lasso convergence warnings**
- Increase `max_iter` to 20000
- Relax `tol` to 1e-3

**"attempt to get argmax of an empty sequence"**
- Data validation issue in preprocessing
- Check ticker exists and has data for date range
- Ensure target_date is not too far in past/future