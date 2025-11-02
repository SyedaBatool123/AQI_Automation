from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr
import hopsworks
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import shap
import os

app = FastAPI(title=" AQI Forecast API", version="1.0")


def load_model():
    project = hopsworks.login()
    mr = project.get_model_registry()
    model = mr.get_model("LightGBM_aqi_model", version=None)
    model_dir = model.download()
    pkl_files = [f for f in os.listdir(model_dir) if f.endswith((".pkl", ".joblib"))]
    model_path = os.path.join(model_dir, pkl_files[0])
    return joblib.load(model_path)

def get_latest_features():
    project = hopsworks.login()
    fs = project.get_feature_store()
    feature_group = fs.get_feature_group("air_quality_processed", version=2)
    df = feature_group.read()
    df = df.sort_values("time").tail(1)
    return df

def generate_future_features(df, days=3):
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, days + 1)]
    futures = []
    for date in future_dates:
        temp = df.copy()
        temp["timestamp"] = date
        temp["day"] = date.weekday()
        temp["hour"] = 12
        futures.append(temp)
    return pd.concat(futures, ignore_index=True)

def get_aqi_label(value):
    if value <= 50: return "Good"
    elif value <= 100: return "Moderate"
    elif value <= 150: return "Unhealthy (Sensitive)"
    elif value <= 200: return "Unhealthy"
    elif value <= 300: return "Very Unhealthy"
    else: return "Hazardous"


@app.get("/")
def root():
    return {"message": "🌫️ AQI Forecast API is running!"}

@app.get("/predict")
def predict(days: int = 3):
    """Predict AQI for next N days"""
    try:
        model = load_model()
        latest = get_latest_features()
        future = generate_future_features(latest, days)
        X_future = future.drop(columns=["timestamp", "time", "id"], errors="ignore")
        X_future = X_future.select_dtypes(include=["int64", "float64", "bool"]).copy()
        preds = model.predict(X_future)
        future["predicted_AQI"] = preds
        future["AQI_Label"] = future["predicted_AQI"].apply(get_aqi_label)

        response = future[["timestamp", "predicted_AQI", "AQI_Label"]].to_dict(orient="records")
        return JSONResponse(content={"forecast": response})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Gradio Dashboard 

def generate_dashboard():
    model = load_model()
    latest = get_latest_features()
    future = generate_future_features(latest)
    X_future = future.drop(columns=["timestamp", "time", "id"], errors="ignore")
    X_future = X_future.select_dtypes(include=["int64", "float64", "bool"]).copy()

    preds = model.predict(X_future)
    future["predicted_AQI"] = preds
    future["AQI_Label"] = future["predicted_AQI"].apply(get_aqi_label)

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(future["timestamp"], future["predicted_AQI"], marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted AQI")
    ax.set_title("Next 3 Days AQI Forecast")

    return future[["timestamp", "predicted_AQI", "AQI_Label"]], fig

with gr.Blocks(title="🌫️ AQI Forecast Dashboard") as dashboard:
    gr.Markdown("## 🌫️ AQI Forecast Dashboard")
    gr.Markdown("Displays next 3 days AQI forecast automatically.")
    table_output = gr.Dataframe(label="Predicted AQI Levels")
    plot_output = gr.Plot(label="Forecast Trend")
    refresh_btn = gr.Button(" Refresh Forecast")
    refresh_btn.click(fn=generate_dashboard, outputs=[table_output, plot_output])
    gr.Markdown("---")
    gr.Markdown("Dashboard powered by FastAPI + Gradio + Hopsworks")

app = gr.mount_gradio_app(app, dashboard, path="/dashboard")


