import os
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from pandas import Timedelta

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# ================= MLflow =================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Stock_Price_Final_Production_V5")

os.makedirs("data", exist_ok=True)

# ================= Prometheus =================
REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter(
    "ml_requests_total", "Total hits", ["endpoint"], registry=REGISTRY
)
LATENCY = Histogram(
    "ml_prediction_latency_seconds", "Latency", registry=REGISTRY
)
MODEL_MSE = Gauge(
    "ml_model_mse", "Model Error", ["model_name"], registry=REGISTRY
)

# ================= App =================
app = FastAPI(title="Production ML Pipeline")

@app.get("/metrics")
def metrics():
    return PlainTextResponse(
        generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

# ================= Feature Engineering =================
def add_features(df):
    df = df.copy()

    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Close"].rolling(10).std()

    df["Return_1"] = df["Log_Return"].shift(1)
    df["Return_3"] = df["Log_Return"].shift(3)
    df["Return_5"] = df["Log_Return"].shift(5)

    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)

    df["Close_lag1"] = df["Close"].shift(1)
    df["Close_lag2"] = df["Close"].shift(2)

    return df.dropna()

FEATURES = [
    "Open", "High", "Low", "Volume",
    "MA_5", "MA_10", "MA_20",
    "Volatility", "Close_lag1", "Close_lag2",
    "Return_1", "Return_3", "Return_5",
    "Momentum_5"
]

# ================= Base Models =================
BASE_MODELS = {
    "XGBoost": XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=10
    ),
    "LinearReg": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ]),
}

# ================= Ensemble PyFunc =================
class EnsembleModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["weights"], "r") as f:
            self.weights = json.load(f)

        self.models = {
            name: mlflow.sklearn.load_model(
                f"models:/Stock_{name}/latest"
            )
            for name in self.weights
        }

    def predict(self, context, model_input):
        preds = np.zeros(len(model_input))
        for name, model in self.models.items():
            preds += self.weights[name] * model.predict(model_input)
        return preds

# ================= TRAIN =================
@app.post("/train")
async def train():
    REQUEST_COUNT.labels(endpoint="/train").inc()

    raw = yf.download(
        "RELIANCE.NS",
        start="2023-01-01",
        auto_adjust=False,
        group_by="column"
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()

    df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = add_features(df)
    df.to_csv("data/latest_stock.csv", index=False)

    if len(df) == 0:
        return {"status": "error", "message": "Insufficient data"}

    with open("data/last_close.json", "w") as f:
        json.dump({"last_close": float(df["Close"].iloc[-1])}, f)

    X = df[FEATURES]
    df["Log_Return_Next"] = np.log(df["Close"].shift(-1) / df["Close"])
    y = df["Log_Return_Next"].dropna().values.ravel()
    X = X.iloc[:-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False
    )

    rmse_scores = {}

    for name, model in BASE_MODELS.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)

            direction_acc = np.mean(
                np.sign(y_test) == np.sign(preds)
            )

            rmse_scores[name] = rmse
            MODEL_MSE.labels(model_name=name).set(mse)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("directional_accuracy", direction_acc)

            mlflow.sklearn.log_model(
                model,
                name="model",
                registered_model_name=f"Stock_{name}"
            )

    weights = {k: 1 / (v + 1e-8) for k, v in rmse_scores.items()}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    with open("ensemble_weights.json", "w") as f:
        json.dump(weights, f)

    with mlflow.start_run(run_name="Ensemble"):
        mlflow.pyfunc.log_model(
            name="ensemble_model",
            python_model=EnsembleModel(),
            artifacts={"weights": "ensemble_weights.json"},
            registered_model_name="Stock_Ensemble"
        )

    return {"status": "trained", "ensemble_weights": weights}

# ================= PREDICT =================
@app.post("/predict")
async def predict(payload: dict):
    start = time.time()
    REQUEST_COUNT.labels(endpoint="/predict").inc()

    model_name = payload.get("model_name", "XGBoost")
    feats = payload["features"]

    df = pd.read_csv("data/latest_stock.csv")
    df = add_features(df)

    row = df.iloc[-1:][FEATURES].copy()
    row["Open"] = feats["open"]
    row["High"] = feats["high"]
    row["Low"] = feats["low"]
    row["Volume"] = feats["volume"]

    if model_name == "Ensemble":
        model = mlflow.pyfunc.load_model(
            "models:/Stock_Ensemble/latest"
        )
    else:
        model = mlflow.sklearn.load_model(
            f"models:/Stock_{model_name}/latest"
        )

    predicted_return = model.predict(row)[0]

    with open("data/last_close.json") as f:
        last_close = json.load(f)["last_close"]

    final_price = last_close * np.exp(predicted_return)

    LATENCY.observe(time.time() - start)

    return {
        "model": model_name,
        "prediction": float(final_price),
        "predicted_return": float(predicted_return)
    }

# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
