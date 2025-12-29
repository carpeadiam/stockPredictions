import os
import time
import subprocess
import pandas as pd
import yfinance as yf
import mlflow
import mlflow.sklearn
import mlflow.shap
import shap
import lime
import lime.lime_tabular
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# --- EVIDENTLY (STRICT LEGACY) & FAIRLEARN ---
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
from fairlearn.metrics import MetricFrame

# --- CONFIGURATION ---
DB_PATH = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(DB_PATH)
mlflow.set_experiment("Stock_Price_Final_Production_V5")
os.makedirs("reports", exist_ok=True)
os.makedirs("data", exist_ok=True)

# --- PROMETHEUS SETUP ---
REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter("ml_requests_total", "Total hits", ["endpoint"], registry=REGISTRY)
LATENCY = Histogram("ml_prediction_latency_seconds", "Latency", registry=REGISTRY)
MODEL_MSE = Gauge("ml_model_mse", "Model Error", ["model_name"], registry=REGISTRY)

# FORCE INITIALIZATION: So Prometheus shows data immediately
for ep in ["/train", "/predict"]: REQUEST_COUNT.labels(endpoint=ep).inc(0)
for m in ["XGBoost", "RandomForest", "LinearReg"]: MODEL_MSE.labels(model_name=m).set(0)

app = FastAPI(title="Production ML Pipeline")

@app.get("/metrics")
def metrics():
    # FIX: Direct text output to prevent 404/Redirect bugs
    return PlainTextResponse(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

MODELS = {
    "XGBoost": XGBRegressor(),
    "RandomForest": RandomForestRegressor(),
    "LinearReg": LinearRegression()
}

def sanitize_columns(df):
    """Flattens multi-index columns and removes ticker suffixes."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).split()[0].replace("'", "").replace("(", "").replace(",", "") for c in df.columns]
    return df

# --- 1. TRAINING ---
@app.post("/train")
async def train():
    REQUEST_COUNT.labels(endpoint="/train").inc()
    raw_df = yf.download("RELIANCE.NS", start="2023-01-01").dropna()
    df = sanitize_columns(raw_df)[['Open', 'High', 'Low', 'Volume', 'Close']].copy()
    df.to_csv("data/latest_stock.csv", index=False)
    
    X = df[['Open', 'High', 'Low', 'Volume']].astype(float)
    y = df['Close'].shift(-1).ffill().astype(float).values.ravel() 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=f"{name}_Run"):
            model.fit(X_train, y_train)
            mse = mean_squared_error(y_test, model.predict(X_test))
            MODEL_MSE.labels(model_name=name).set(float(mse)) 
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, f"model_{name}", registered_model_name=f"Stock_{name}")

    return {"status": "Trained", "models": list(MODELS.keys())}

# --- 2. REPORTS (ALL MODELS) ---
@app.post("/generate-reports")
async def generate_reports():
    df = pd.read_csv("data/latest_stock.csv").apply(pd.to_numeric, errors='coerce').dropna()
    X = df[['Open', 'High', 'Low', 'Volume']].astype(float)
    
    # Evidently Drift Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=X.head(50), current_data=X.tail(50))
    report.save_html("reports/drift_report.html")

    # SHAP for ALL models
    for name, model in MODELS.items():
        with mlflow.start_run(run_name=f"SHAP_{name}"):
            def p_wrap(data):
                return model.predict(pd.DataFrame(data, columns=X.columns))
            mlflow.shap.log_explanation(p_wrap, X.tail(30))

    return {"status": "Reports Generated"}

# --- 3. PREDICT (FIXED) ---
@app.post("/predict")
async def predict(payload: dict):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    model_name = payload.get("model_name", "XGBoost")
    features = payload.get("features") 

    try:
        model = mlflow.sklearn.load_model(f"models:/Stock_{model_name}/latest")
        
        # FIX: Ensure it creates a DataFrame with the right orientation and names
        input_df = pd.DataFrame([features]) 
        input_df.columns = [str(c).capitalize() for c in input_df.columns]
        input_df = input_df[['Open', 'High', 'Low', 'Volume']].astype(float)
        
        prediction = model.predict(input_df)

        # LIME Local Explainability
        df_train = pd.read_csv("data/latest_stock.csv").head(100)
        explainer = lime.lime_tabular.LimeTabularExplainer(
            df_train[['Open', 'High', 'Low', 'Volume']].values,
            feature_names=['Open', 'High', 'Low', 'Volume'],
            mode='regression'
        )
        exp = explainer.explain_instance(input_df.values[0], model.predict)

        LATENCY.observe(time.time() - start_time)
        return {
            "model": model_name,
            "prediction": float(prediction[0]),
            "lime": exp.as_list()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)