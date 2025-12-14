from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from xgboost import XGBRegressor
import joblib
from typing import Optional, Dict
import uvicorn
import yfinance as yf  
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
# from statsmodels.tsa.arima.model import ARIMA   # ARMA
from arch import arch_model                     # GARCH
from xgboost import XGBRegressor

app = FastAPI(title="Volatility Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    ticker: str
    target_date: str

class PredictionResponse(BaseModel):
    ticker: str
    target_date: str
    predicted_volatility: float
    confidence_interval: Optional[Dict[str, float]]
    model_contributions: Optional[Dict[str, float]]
    model_type: str
    timestamp: str

model = None

def get_yz_vol(data):

    # length
    n = len(data)

    # lowercase
    data.rename(columns = {col : col.lower() for col in data.columns})
    
    # overnight variance
    r_open = np.log(data['open'] / data['close'].shift(1))
    sigma_o2 = r_open**2
    
    # close-to-close variance
    r_close = np.log(data['close'] / data['open'])
    sigma_c2 = r_close**2
    
    # Rogers-Satchell
    h = np.log(data['high'] / data['open'])
    l = np.log(data['low'] / data['open'])
    c = np.log(data['close'] / data['open'])
    rs = h * (h - c) + l * (l - c)
    
    # weighting factor
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    
    # yz variance
    yz_var = sigma_o2 + k * sigma_c2 + (1 - k) * rs
    
    # yz daily volatilty
    yz_vol = np.sqrt(yz_var)
    
    return yz_vol

def get_next_trading_day(date: pd.Timestamp, exchange: str = "NYSE"):
    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=date, end_date=date + pd.Timedelta(days=10)) # 10 days to be safe (long weekends)
    return schedule.index[1]

def get_previous_trading_day(date: pd.Timestamp, exchange: str = "NYSE"):
    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=date - pd.Timedelta(days=10), end_date=date - pd.Timedelta(days = 1)) # 10 days to be safe (long weekends)
    return schedule.index[-1]

def preprocess(symbol : str, target_date : str, years : int | float = 2, underlying_ar : bool = False, lag : int = 2):
    
    # retrieving data
    data = yf.download([symbol], start = pd.to_datetime(target_date) - pd.Timedelta(days = int(years * 365)), end = target_date, progress = False).droplevel(level = "Ticker", axis = 1)
    vix = yf.download(["^VIX"], start = pd.to_datetime(target_date) - pd.Timedelta(days = int(years * 365)), end = target_date, progress = False).droplevel(level = "Ticker", axis = 1)[["Close"]].rename({"Close" : "vix"}, axis = 1)
    if vix.empty:
        vix = yf.download(["^VIX"], start = get_next_trading_day(pd.to_datetime(target_date) - pd.Timedelta(days = 365)), end = target_date, progress = False).droplevel(level = "Ticker", axis = 1)[["Close"]].rename({"Close" : "vix"}, axis = 1)
    
    data.rename(columns = {col : col.lower() for col in data.columns}, inplace = True)

    data["returns"] = data["close"].pct_change()
    data["yz_vol"] = get_yz_vol(data)
    data = data.join(vix)

    next_trading_day = get_next_trading_day(data.index[-1])

    # garch model
    returns = data["returns"].dropna()

    if underlying_ar:
        garch = arch_model(returns, mean = "AR", lags = 1, vol = "GARCH", p = 1, q = 1)
    else:
        garch = arch_model(returns, mean = "Zero", vol = "GARCH", p = 1, q = 1)

    garch = garch.fit(disp = "off")

    # train preprocessing
    cond_vol = garch.conditional_volatility.reindex(data.index)

    # lagged features
    lagged_train_data = pd.DataFrame(index = data.index)
    lagged_test_data = pd.DataFrame(index = [next_trading_day])
    for i in range(1, lag + 1):
        buffer = data.rename(columns = {col : f"lag_{col}_{i}" for col in data.columns}).shift(i)
        lagged_train_data = lagged_train_data.join(buffer)

        buffer = pd.DataFrame(
            data.iloc[[-i]].values,
            columns = data.columns
        ).rename(columns = {col : f"lag_{col}_{i}" for col in data.columns}, index = {0 : next_trading_day})
        lagged_test_data = lagged_test_data.join(buffer)

    # training data
    train_data = data.join(cond_vol).join(lagged_train_data).drop(columns = [col for col in data.columns if col != "yz_vol"]).dropna()
    train_data.rename_axis("AAPL", axis = 1, inplace = True)

    # test data
    test_data = pd.DataFrame({
        "cond_vol": np.sqrt(garch.forecast(horizon = 1).variance.values)[0]
    }, index = [next_trading_day]).join(lagged_test_data).rename_axis("Date", axis = 0)
    test_data.rename_axis("AAPL", axis = 1, inplace = True)

    # scaling
    train_X = train_data.drop("yz_vol", axis = 1)
    train_Y = train_data[["yz_vol"]]

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    scaler_X.fit(train_X)
    scaler_Y.fit(train_Y)

    train_X = pd.DataFrame(scaler_X.transform(train_X), columns = train_X.columns, index = train_X.index)
    test_X = pd.DataFrame(scaler_X.transform(test_data), columns = train_X.columns, index = test_data.index)
    train_Y = pd.DataFrame(scaler_Y.transform(train_Y), columns = train_Y.columns, index = train_Y.index)
    
    return scaler_Y, train_X, train_Y, test_X

def inverse_pred(scaler, test_X, pred_Y):
    symbol = test_X.axes[1].name
    date = test_X.axes[0][0]
    true_data = yf.download([symbol], start = get_previous_trading_day(date), end = pd.to_datetime(date) + pd.Timedelta(days = 1), progress = False).droplevel(level = "Ticker", axis = 1)
    true_data.rename(columns = {col : col.lower() for col in true_data.columns}, inplace = True)
    true_yz_vol = get_yz_vol(true_data).values[-1]
    pred_yz_vol = scaler.inverse_transform(pred_Y.reshape(-1,1))[0]
    df = pd.DataFrame({
        "true_yz_vol" : true_yz_vol,
        "pred_yz_vol" : pred_yz_vol
    }, index = [date])
    return df

def train_model(X, y):
    """Train the stacking regressor"""
    xgb_params = {
        "n_estimators" : 100, 
        "learning_rate" : 0.15, 
        "max_depth" : 4,
        "subsample" : 0.8,
        "colsample_bytree" : 0.8,
        "eval_metric" : mean_squared_error
    }
    lasso_params = {
    "alphas" : [1.0, 0.1, 0.01], # strong, normal, weak (0.001 too weak, almost never converges)
    "cv" : 5,
    "max_iter" : 10000, # for convergence
    "tol" : 1e-4,
    "selection" : "random"
}
    
    estimators = [
        ("XGB", XGBRegressor(**xgb_params)),
        ("Lasso", LassoCV(**lasso_params)),
        ("KNN-R", KNeighborsRegressor())
    ]
    
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    model.fit(X, y)
    return model

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load('volatility_model.pkl')
        print("✓ Model loaded from disk")
    except:
        print("⚠ No saved model found. Will train on first prediction.")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI - No Node.js needed!"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Volatility Predictor</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            @keyframes spin { to { transform: rotate(360deg); } }
            .animate-spin { animation: spin 1s linear infinite; }
        </style>
    </head>
    <body class="bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 min-h-screen">
        <div id="app" class="p-6">
            <div class="max-w-4xl mx-auto">
                <!-- Header -->
                <div class="text-center mb-8">
                    <div class="flex items-center justify-center gap-3 mb-4">
                        <svg class="w-10 h-10 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                        </svg>
                        <h1 class="text-4xl font-bold text-white">Volatility Predictor</h1>
                    </div>
                    <p class="text-slate-300 text-lg">ML-powered Yang-Zhang volatility forecasting with GARCH and ensemble models</p>
                </div>

                <!-- Input Form -->
                <div class="bg-white/10 backdrop-blur-md rounded-2xl shadow-2xl p-8 mb-6 border border-white/20">
                    <div class="space-y-6">
                        <div>
                            <label class="block text-white font-semibold mb-2">Ticker Symbol</label>
                            <input type="text" id="ticker" placeholder="e.g., AAPL, TSLA, SPY" 
                                   class="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-400">
                        </div>
                        <div>
                            <label class="block text-white font-semibold mb-2">Target Date</label>
                            <input type="date" id="targetDate" 
                                   class="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white focus:outline-none focus:ring-2 focus:ring-blue-400">
                        </div>
                        <button onclick="predictVolatility()" id="predictBtn"
                                class="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold py-4 px-6 rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 shadow-lg">
                            Predict Volatility
                        </button>
                    </div>
                </div>

                <!-- Error Display -->
                <div id="errorDiv" class="hidden bg-red-500/20 border border-red-500/50 rounded-xl p-4 mb-6 backdrop-blur-sm">
                    <div class="flex items-start gap-3">
                        <svg class="w-6 h-6 text-red-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        <div>
                            <h3 class="text-red-200 font-semibold mb-1">Error</h3>
                            <p id="errorMsg" class="text-red-300"></p>
                        </div>
                    </div>
                </div>

                <!-- Results Display -->
                <div id="resultsDiv" class="hidden bg-white/10 backdrop-blur-md rounded-2xl shadow-2xl p-8 border border-white/20">
                    <h2 class="text-2xl font-bold text-white mb-6">Prediction Results</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="bg-blue-500/20 rounded-xl p-6 border border-blue-400/30">
                            <p class="text-blue-300 text-sm font-semibold mb-2">PREDICTED VOLATILITY</p>
                            <p id="predVol" class="text-4xl font-bold text-white">--</p>
                        </div>
                        <div class="bg-purple-500/20 rounded-xl p-6 border border-purple-400/30">
                            <p class="text-purple-300 text-sm font-semibold mb-2">CONFIDENCE INTERVAL</p>
                            <p id="confInt" class="text-white text-lg">--</p>
                        </div>
                        <div class="md:col-span-2 bg-slate-500/20 rounded-xl p-6 border border-slate-400/30">
                            <p class="text-slate-300 text-sm font-semibold mb-3">MODEL CONTRIBUTIONS</p>
                            <div id="modelContrib" class="space-y-2"></div>
                        </div>
                        <div class="md:col-span-2 bg-slate-500/20 rounded-xl p-6 border border-slate-400/30">
                            <p class="text-slate-300 text-sm font-semibold mb-2">DETAILS</p>
                            <div class="grid grid-cols-2 gap-4 text-white" id="details"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Set min date to today
            document.getElementById('targetDate').min = new Date().toISOString().split('T')[0];

            async function predictVolatility() {
                const ticker = document.getElementById('ticker').value.trim();
                const targetDate = document.getElementById('targetDate').value;
                const btn = document.getElementById('predictBtn');
                const errorDiv = document.getElementById('errorDiv');
                const resultsDiv = document.getElementById('resultsDiv');

                if (!ticker || !targetDate) {
                    showError('Please fill in all fields');
                    return;
                }

                // Show loading
                btn.disabled = true;
                btn.innerHTML = '<svg class="animate-spin h-5 w-5 mx-auto" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>';
                errorDiv.classList.add('hidden');
                resultsDiv.classList.add('hidden');

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ ticker: ticker.toUpperCase(), target_date: targetDate })
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Prediction failed');
                    }

                    const data = await response.json();
                    showResults(data);
                } catch (err) {
                    showError(err.message);
                } finally {
                    btn.disabled = false;
                    btn.innerHTML = 'Predict Volatility';
                }
            }

            function showError(msg) {
                document.getElementById('errorMsg').textContent = msg;
                document.getElementById('errorDiv').classList.remove('hidden');
            }

            function showResults(data) {
                document.getElementById('predVol').textContent = (data.predicted_volatility * 100).toFixed(2) + '%';
                
                if (data.confidence_interval) {
                    const ci = data.confidence_interval;
                    document.getElementById('confInt').textContent = 
                        `${(ci.lower * 100).toFixed(2)}% - ${(ci.upper * 100).toFixed(2)}%`;
                }

                if (data.model_contributions) {
                    const contrib = document.getElementById('modelContrib');
                    contrib.innerHTML = '';
                    for (const [model, weight] of Object.entries(data.model_contributions)) {
                        contrib.innerHTML += `
                            <div class="flex justify-between items-center">
                                <span class="text-white capitalize">${model}</span>
                                <span class="text-slate-300">${(weight * 100).toFixed(1)}</span>
                            </div>`;
                    }
                }

                document.getElementById('details').innerHTML = `
                    <div><p class="text-slate-400 text-sm">Ticker</p><p class="font-semibold">${data.ticker}</p></div>
                    <div><p class="text-slate-400 text-sm">Target Date</p><p class="font-semibold">${data.target_date}</p></div>
                    <div><p class="text-slate-400 text-sm">Model Type</p><p class="font-semibold">${data.model_type}</p></div>
                    <div><p class="text-slate-400 text-sm">Timestamp</p><p class="font-semibold">${new Date(data.timestamp).toLocaleString()}</p></div>
                `;

                document.getElementById('resultsDiv').classList.remove('hidden');
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResponse)
async def predict_volatility(request: PredictionRequest):
    """Predict volatility for a given ticker and date"""
    try:
        ticker = request.ticker.upper()
        target_date = request.target_date
        
        print(f"📊 Processing {ticker} for target date {target_date}...")
        
        # preprocess 
        scaler_Y, train_X, train_Y, test_X = preprocess(
            symbol = ticker,
            target_date = target_date,
            years = 2,
            underlying_ar = False,
            lag = 2
        )
        
        if len(train_X) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data for prediction")
        
        print(f"✓ Preprocessed {len(train_X)} rows of training data")
        
        # train model fresh every run
        print(f"🔧 Training model for {ticker}...")
        model = train_model(train_X, train_Y.values.ravel())
        print("✓ Model trained")
        
        # make prediction (scaled)
        pred_Y_scaled = model.predict(test_X.values)
        
        # inverse transform to get actual volatility
        pred_Y = scaler_Y.inverse_transform(pred_Y_scaled.reshape(-1, 1))[0][0]
        
        # get individual model predictions for confidence interval
        predictions = []
        for estimator_name, estimator in model.named_estimators_.items():
            pred_scaled = estimator.predict(test_X.values)[0]
            pred_actual = scaler_Y.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred_actual)
        
        std = np.std(predictions)
        ci_lower = max(0, pred_Y - 1.96 * std)
        ci_upper = pred_Y + 1.96 * std
        
        # model contributions
        try:
            ridge_coefs = model.final_estimator_.coef_
            
            total = np.abs(ridge_coefs).sum()
            if total > 0:
                contributions = {
                    name: float(np.abs(coef) / total)
                    for name, coef in zip(model.named_estimators_.keys(), ridge_coefs)
                }
            else:
                contributions = {name: 1.0 / len(model.named_estimators_) 
                                for name in model.named_estimators_.keys()}
        except Exception as e:
            print(f"Warning: Could not extract coefficients: {e}")
            contributions = {name: 1.0 / len(model.named_estimators_) 
                            for name in model.named_estimators_.keys()}
        
        print(f"✓ Prediction complete: {pred_Y*100:.2f}%")
        print(f"  Confidence Interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
        
        return PredictionResponse(
            ticker = ticker,
            target_date = target_date,
            predicted_volatility = float(pred_Y),
            confidence_interval = {"lower": float(ci_lower), "upper": float(ci_upper)},
            model_contributions = contributions,
            model_type = "StackingRegressor (XGB + Lasso + KNN)",
            timestamp = datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting Volatility Prediction Server...")
    print("📱 Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)