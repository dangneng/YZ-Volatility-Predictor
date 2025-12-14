import warnings
import yfinance as yf  
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import random
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from arch import arch_model
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

# symbols for testing purposes
symbols = ["AAPL", "NVDA", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NFLX", "ADBE", "INTC", "CRM", "JPM", "BAC", "GS", "C", "MS", "WMT", "PG", "KO", "PEP", "MCD", "NKE", "JNJ", "PFE", "MRK", "ABT", "LLY", "XOM", "CVX", "BA", "CAT", "GE"]

lasso_params = {
    "alphas" : [1.0, 0.1, 0.01], # strong, normal, weak (0.001 too weak, almost never converges)
    "cv" : 5,
    "max_iter" : 10000, # for convergence
    "tol" : 1e-4,
    "selection" : "random"
}
cv = 5
alphas = [0.1, 0.01, 0.001] # strong, normal, weak
max_iter = 5000
xgb_params = {
    "n_estimators" : 100, 
    "learning_rate" : 0.15, 
    "max_depth" : 4,
    "subsample" : 0.8,
    "colsample_bytree" : 0.8,
    "eval_metric" : mean_squared_error
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

# PREDICTING NEXT DAY YZ VOLATILTY

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

# EVALUATION

def evaluate_single(symbol, target_date, model, years, underlying_ar, lag):
    try:
        target_scaler, train_X, train_Y, test_X = preprocess(
            symbol, target_date, years=years, underlying_ar=underlying_ar, lag=lag
        )
        from sklearn.base import clone
        model_copy = clone(model)  # Important: clone the model
        model_copy.fit(train_X, train_Y.values.ravel())
        pred_Y_scaled = model_copy.predict(test_X)
        test_Y_df = inverse_pred(target_scaler, test_X, pred_Y_scaled)
        
        return (
            test_Y_df["true_yz_vol"].values[0],
            test_Y_df["pred_yz_vol"].values[0]
        )
    except Exception as e:
        return None

def evaluate(symbols : list, start : str, end : str, model, x = 1000, years = 2, underlying_ar = False, lag = 2, n_jobs = -1):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    dates = pd.date_range(start=start + pd.Timedelta(days=int(years * 365)), end=end)
    
    tasks = [(random.choice(symbols), random.choice(dates)) for _ in range(x)]
    
    print(f"Running {x} evaluations in parallel with {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(evaluate_single)(symbol, date, model, years, underlying_ar, lag)
        for symbol, date in tasks
    )
    
    results = [r for r in results if r is not None]
    
    if not results:
        return None, None
    
    true_yz_vol, pred_yz_vol = zip(*results)
    
    mse = mean_squared_error(true_yz_vol, pred_yz_vol)
    errors = np.array(true_yz_vol) - np.array(pred_yz_vol)
    error_variance = np.var(errors)
    
    print(f"Completed {len(results)}/{x} evaluations successfully")
    
    return mse, error_variance