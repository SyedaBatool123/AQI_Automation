# src/dashboard_app.py
'''
import streamlit as st
import hopsworks
import pandas as pd
import numpy as np
import joblib
import os
import json
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import shap
import tempfile

st.set_page_config(page_title="Pearls AQI Predictor — Live Dashboard", layout="wide")

# -------------------------
# Config / constants
# -------------------------
FEATURE_GROUP_NAME = "air_quality_processed"
FEATURE_GROUP_VERSION = 2
LOCAL_MODELS_DIR = "models"
ARTIFACTS_INFO = "artifacts/best_model_info.json"
DROP_AQI_COLS = [
    "time", "id",
    "aqi_pm2_5", "aqi_pm10", "aqi_carbon_monoxide",
    "aqi_nitrogen_dioxide", "aqi_sulphur_dioxide", "aqi_ozone", "aqi"
]

# -------------------------
# Helpers
# -------------------------
@st.cache_resource(ttl=300)
def login_hopsworks():
    """Log in to Hopsworks project (caches connection)."""
    project = hopsworks.login()
    return project

def load_latest_model(project=None):
    """Try to load the latest model from Hopsworks model registry first,
    otherwise fallback to local models folder (loads latest timestamped file)."""
    # 1) Try artifacts metadata to get best model name
    best_name = None
    if os.path.exists(ARTIFACTS_INFO):
        try:
            with open(ARTIFACTS_INFO, "r") as f:
                info = json.load(f)
            best_name = info.get("best_model_name")
        except Exception:
            best_name = None

    # 2) If hopsworks is available, try registry
    if project is None:
        try:
            project = login_hopsworks()
        except Exception:
            project = None

    if project:
        try:
            mr = project.get_model_registry()
            # prefer model name from artifacts if available and expected suffix
            candidates = []
            if best_name:
                candidates.append(f"{best_name}_aqi_model")
            # also try common names saved earlier by your pipeline
            candidates += ["LightGBM_aqi_model", "RandomForest_aqi_model", "aqi_forecast_model"]
            for name in candidates:
                try:
                    m = mr.get_model(name, version=None)  # latest version
                    tmpdir = m.download()  # returns local dir
                    # find model file
                    for f in os.listdir(tmpdir):
                        if f.endswith((".pkl", ".joblib", ".h5")):
                            model_path = os.path.join(tmpdir, f)
                            model, scaler = _load_model_and_scaler_from_path(tmpdir)
                            st.info(f"Loaded model from Hopsworks registry: {name}")
                            return model, scaler
                except Exception:
                    continue
        except Exception as e:
            # fail quietly to fallback to local
            st.warning(f"Hopsworks model registry not reachable or no model found: {e}")

    # 3) Fallback: search local models folder for latest .pkl/.joblib/.h5
    if os.path.exists(LOCAL_MODELS_DIR):
        files = []
        for f in os.listdir(LOCAL_MODELS_DIR):
            if f.endswith((".pkl", ".joblib", ".h5")):
                files.append(os.path.join(LOCAL_MODELS_DIR, f))
        if files:
            files = sorted(files)
            model_path = files[-1]
            # try loading
            try:
                if model_path.endswith(".h5"):
                    from tensorflow.keras.models import load_model as keras_load
                    model = keras_load(model_path)
                    scaler = None
                else:
                    model = joblib.load(model_path)
                    # attempt to find scaler file with same timestamp
                    ts = os.path.basename(model_path).split("_")[-1].split(".")[0]
                    scaler_candidates = [f for f in os.listdir(LOCAL_MODELS_DIR) if f.startswith("scaler") and ts in f]
                    scaler = None
                    if scaler_candidates:
                        scaler = joblib.load(os.path.join(LOCAL_MODELS_DIR, scaler_candidates[-1]))
                st.info(f"Loaded local model: {os.path.basename(model_path)}")
                return model, scaler
            except Exception as e:
                st.error(f"Failed to load local model: {e}")

    raise RuntimeError("No model found. Ensure model exists in Hopsworks registry or local 'models/' folder.")

def _load_model_and_scaler_from_path(dirpath):
    """Helper to load model and scaler from a directory returned by Hopsworks download."""
    model = None
    scaler = None
    for f in os.listdir(dirpath):
        if f.endswith((".pkl", ".joblib")):
            model = joblib.load(os.path.join(dirpath, f))
        if "scaler" in f and f.endswith(".pkl"):
            try:
                scaler = joblib.load(os.path.join(dirpath, f))
            except Exception:
                scaler = None
    # h5 case
    if model is None:
        for f in os.listdir(dirpath):
            if f.endswith(".h5"):
                from tensorflow.keras.models import load_model as keras_load
                model = keras_load(os.path.join(dirpath, f))
    return model, scaler

@st.cache_data(ttl=120)
def fetch_latest_processed_row():
    """Fetch the latest processed features row from Hopsworks FG."""
    project = login_hopsworks()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    if "time" in df.columns:
        df = df.sort_values("time")
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    latest = df.tail(1).copy().reset_index(drop=True)
    return latest, df

def generate_future_features(latest_row, days=3):
    """Generate future rows based on latest_row by shifting the time and recomputing simple time features.
       Note: If you have more advanced derived features depending on raw weather/pollutant inputs, integrate the same logic here."""
    out_rows = []
    # determine timestamp column name
    time_col = None
    if "time" in latest_row.columns:
        time_col = "time"
    elif "timestamp" in latest_row.columns:
        time_col = "timestamp"

    base_time = pd.to_datetime(latest_row.iloc[0][time_col]) if time_col else pd.Timestamp.now()

    for d in range(1, days+1):
        r = latest_row.copy().iloc[0]
        new_time = base_time + pd.Timedelta(days=d)
        # set time/timestamp
        if time_col:
            r[time_col] = new_time
        else:
            r["timestamp"] = new_time
        # recompute simple time features (match your feature_engineering logic)
        r["hour"] = int(new_time.hour)
        r["day"] = int(new_time.day)
        r["month"] = int(new_time.month)
        r["weekday"] = int(new_time.weekday())
        r["is_weekend"] = int(r["weekday"] in [5,6])
        out_rows.append(r)
    future_df = pd.DataFrame(out_rows).reset_index(drop=True)
    return future_df

def prepare_model_input(df_rows, model):
    """Drop unwanted columns and align columns to the model's expected features.
       Returns X (numeric np.array or DataFrame), and feature_names used."""
    df = df_rows.copy()
    # drop columns we don't want to feed to model
    for c in DROP_AQI_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # convert datetimes etc
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("int64") // 10**9  # epoch seconds

    # keep only numeric and bool columns
    df_num = df.select_dtypes(include=["number", "bool"]).copy()

    # Align to model feature order if available
    feature_order = None
    if hasattr(model, "feature_name_"):
        feature_order = list(model.feature_name_)
    elif hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        try:
            feature_order = list(model.booster_.feature_name)
        except Exception:
            feature_order = None

    if feature_order:
        # use only intersection, preserve order
        feature_order = [f for f in feature_order if f in df_num.columns]
        df_num = df_num.reindex(columns=feature_order, fill_value=0)

    return df_num, df_num.columns.tolist()

def compute_shap_and_plots(model, X_df):
    """Compute shap values for X_df and return generated images (paths) for:
       - shap bar (global importance of the X_df rows combined)
       - shap waterfall per row (list)
    """
    plots = {}
    tmp = tempfile.mkdtemp(prefix="shap_")
    try:
        # Use TreeExplainer where possible for speed
        if ("sklearn" in str(type(model)).lower()) or ("lightgbm" in str(type(model)).lower()) or ("xgboost" in str(type(model)).lower()):
            explainer = shap.TreeExplainer(model, check_additivity=False)
            shap_values = explainer(X_df)
        else:
            # fallback to KernelExplainer (can be slow) using a small background sample
            background = X_df.sample(min(50, len(X_df)), random_state=0)
            explainer = shap.KernelExplainer(lambda x: model.predict(x), background)
            shap_values = explainer.shap_values(X_df, nsamples=100)

        # Summary / bar plot (mean abs)
        fig1 = plt.figure(figsize=(8, 4))
        try:
            shap.plots.bar(shap_values, show=False)
        except Exception:
            # produce simple importance if plotting wrapper fails
            mean_abs = np.abs(shap_values).mean(axis=0)
            idx_sorted = np.argsort(-mean_abs)
            names = np.array(X_df.columns)[idx_sorted]
            vals = mean_abs[idx_sorted]
            plt.barh(names[:20][::-1], vals[:20][::-1])
            plt.title("Mean |SHAP value| (approx)")
        plt.tight_layout()
        bar_path = os.path.join(tmp, "shap_bar.png")
        fig1.savefig(bar_path, dpi=200)
        plt.close(fig1)
        plots["bar"] = bar_path

        # waterfall per instance
        waterfall_paths = []
        for i in range(len(X_df)):
            try:
                fig_w = plt.figure(figsize=(8, 4))
                # shap supports waterfall for single sample
                shap.plots.waterfall(shap_values[i], show=False)
                wpath = os.path.join(tmp, f"shap_waterfall_{i}.png")
                plt.tight_layout()
                fig_w.savefig(wpath, dpi=200)
                plt.close(fig_w)
                waterfall_paths.append(wpath)
            except Exception:
                # if waterfall fails (older shap), create a simple bar of contributions for the instance
                vals = np.asarray(shap_values[i])
                names = X_df.columns
                idx_sorted = np.argsort(-np.abs(vals))
                fig_w = plt.figure(figsize=(8, 4))
                plt.barh(names[idx_sorted][:20][::-1], vals[idx_sorted][:20][::-1])
                plt.title(f"SHAP contributions (approx) - instance {i}")
                wpath = os.path.join(tmp, f"shap_waterfall_{i}_fallback.png")
                fig_w.savefig(wpath, dpi=200)
                plt.close(fig_w)
                waterfall_paths.append(wpath)

        plots["waterfalls"] = waterfall_paths
    except Exception as e:
        st.warning(f"SHAP computation failed: {e}")
        return {}
    return plots

def get_aqi_label(v):
    v = float(v)
    if v <= 50: return "Good"
    if v <= 100: return "Moderate"
    if v <= 150: return "Unhealthy (for Sensitive people)"
    if v <= 200: return "Unhealthy"
    if v <= 300: return "Very Unhealthy"
    return "Hazardous"

# -------------------------
# UI layout
# -------------------------
st.title("🌍 Pearls AQI Predictor — Live Dashboard")
st.markdown("Automatically uses the latest processed features + latest registered model to forecast AQI and explain predictions with SHAP.")

col1, col2 = st.columns([2, 1])

with col2:
    if st.button("Reload model & data"):
        st.experimental_rerun()
    st.write("Refresh to pick up retrained model or newly processed data.")

# -------------------------
# Main logic
# -------------------------
with st.spinner("Loading latest model and latest processed features..."):
    project = login_hopsworks()
    try:
        model, scaler = load_latest_model(project=project)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    latest_row, full_df = fetch_latest_processed_row()

st.success("Loaded latest model and features.")

# generate 3-day future features
future_df = generate_future_features(latest_row, days=3)

# prepare model input
X_future_df, feature_names = prepare_model_input(future_df, model)

# apply scaler if available
if 'scaler' in locals() and scaler is not None:
    try:
        X_future_np = scaler.transform(X_future_df)
        X_future = pd.DataFrame(X_future_np, columns=X_future_df.columns)
    except Exception as e:
        st.warning(f"Scaler transform failed, using raw features: {e}")
        X_future = X_future_df.copy()
else:
    X_future = X_future_df.copy()

# finalize numeric types
X_future = X_future.select_dtypes(include=["number", "bool"]).fillna(0)

# predict
try:
    if hasattr(model, "predict"):
        preds = model.predict(X_future)
    else:
        # keras model
        preds = model.predict(X_future.values).flatten()
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

future_df = future_df.reset_index(drop=True)
future_df["predicted_AQI"] = np.round(preds, 2)
future_df["AQI_Label"] = future_df["predicted_AQI"].apply(get_aqi_label)

# left column: forecast table + plots
with col1:
    st.subheader("3-day AQI Forecast (Auto)")
    st.dataframe(future_df[["time"] if "time" in future_df.columns else ["timestamp"] + ["predicted_AQI","AQI_Label"]].assign(predicted_AQI=future_df["predicted_AQI"], AQI_Label=future_df["AQI_Label"]).reset_index(drop=True), use_container_width=True)

    # small trend chart (last 7 days + forecast)
    st.subheader("AQI trend (recent + forecast)")
    hist = full_df.copy()
    if "time" in hist.columns:
        hist["time"] = pd.to_datetime(hist["time"])
        hist_plot = hist.set_index("time")[["aqi"]].tail(7*24)  # try last 7 days hourly if available
    elif "timestamp" in hist.columns:
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
        hist_plot = hist.set_index("timestamp")[["aqi"]].tail(7*24)
    else:
        hist_plot = hist[["aqi"]].tail(7)

    # prepare forecast series
    try:
        fc = future_df.copy()
        time_col = "time" if "time" in fc.columns else "timestamp"
        fc[time_col] = pd.to_datetime(fc[time_col])
        fc_plot = fc.set_index(time_col)[["predicted_AQI"]]
        combined = pd.concat([hist_plot, fc_plot])
        st.line_chart(combined)
    except Exception:
        st.line_chart(hist_plot)

    # alerts
    hazardous = future_df[future_df["predicted_AQI"] > 200]
    if not hazardous.empty:
        st.error("⚠️ Hazardous AQI expected in the forecast!")
    else:
        st.success("No hazardous AQI predicted in next 3 days.")

# right column: SHAP explanations
with col2:
    st.subheader("SHAP Explainability (dynamic per forecast day)")
    # compute SHAP only for tree/sklearn/xgboost/gbm models or small data
    plots = compute_shap_and_plots(model, X_future)
    if plots.get("bar"):
        st.image(plots["bar"], caption="SHAP: mean |value| importance", use_column_width=True)
    else:
        st.write("SHAP bar not available.")

    # show waterfall per day
    if plots.get("waterfalls"):
        st.markdown("### Per-day contributions")
        for i, p in enumerate(plots["waterfalls"]):
            st.markdown(f"**Day +{i+1} — Predicted AQI: {future_df.loc[i,'predicted_AQI']} ({future_df.loc[i,'AQI_Label']})**")
            st.image(p, use_column_width=True)
    else:
        st.write("SHAP waterfall images not available.")

st.markdown("---")
st.caption("Notes: SHAP uses TreeExplainer if model supports it (fast). For deep-learning models SHAP may be slower or skipped. Dashboard picks the latest model from Hopsworks registry or `models/` folder.")
'''



# 2nd code

# src/gradio_dashboard.py
"""
Gradio dashboard for Pearls AQI Predictor (Hopsworks-backed).
Modern Blocks UI: forecast table + chart + SHAP explainability (bar + per-day waterfalls).
"""




#                         bestttttttttttttttttttttttttttttttttttttttt


# src/gradio_dashboard.py
import os
import json
import tempfile
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import hopsworks
import shap
import plotly.express as px
import gradio as gr
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
FEATURE_GROUP_NAME = "air_quality_processed"
FEATURE_GROUP_VERSION = 2
LOCAL_MODELS_DIR = "models"
ARTIFACTS_INFO = "artifacts/best_model_info.json"
DROP_AQI_COLS = [
    "time", "timestamp", "id",
    "aqi_pm2_5", "aqi_pm10", "aqi_carbon_monoxide",
    "aqi_nitrogen_dioxide", "aqi_sulphur_dioxide", "aqi_ozone", "aqi"
]
FORECAST_DAYS = 3  # number of days (including today) to forecast

# -------------------------
# Hopsworks login (cached)
# -------------------------
_project_cache = {"project": None, "ts": 0}


def login_hopsworks():
    global _project_cache
    TTL = 300  # seconds
    if _project_cache["project"] is not None and (time.time() - _project_cache["ts"] < TTL):
        return _project_cache["project"]
    try:
        project = hopsworks.login()
        _project_cache = {"project": project, "ts": time.time()}
        return project
    except Exception as e:
        raise RuntimeError(f"Hopsworks login failed: {e}")


# -------------------------
# Model loading
# -------------------------
def _load_model_and_scaler_from_dir(dirpath):
    model = None
    scaler = None
    for f in os.listdir(dirpath):
        if f.endswith((".pkl", ".joblib")) and ("scaler" not in f):
            try:
                model = joblib.load(os.path.join(dirpath, f))
            except Exception:
                model = None
        if "scaler" in f and f.endswith(".pkl"):
            try:
                scaler = joblib.load(os.path.join(dirpath, f))
            except Exception:
                scaler = None
    if model is None:
        for f in os.listdir(dirpath):
            if f.endswith(".h5"):
                from tensorflow.keras.models import load_model as keras_load
                model = keras_load(os.path.join(dirpath, f))
    return model, scaler


def load_latest_model():
    best_name = None
    if os.path.exists(ARTIFACTS_INFO):
        try:
            with open(ARTIFACTS_INFO, "r") as f:
                info = json.load(f)
            best_name = info.get("best_model_name")
        except Exception:
            best_name = None

    # Try registry
    try:
        project = login_hopsworks()
        mr = project.get_model_registry()
        candidates = []
        if best_name:
            candidates.append(f"{best_name}_aqi_model")
        candidates += ["LightGBM_aqi_model", "RandomForest_aqi_model", "aqi_forecast_model"]
        for name in candidates:
            try:
                m = mr.get_model(name, version=None)
                tmpdir = m.download()
                model, scaler = _load_model_and_scaler_from_dir(tmpdir)
                if model is not None:
                    return model, scaler, f"registry:{name}"
            except Exception:
                continue
    except Exception:
        pass

    # fallback local
    if os.path.exists(LOCAL_MODELS_DIR):
        files = [os.path.join(LOCAL_MODELS_DIR, f) for f in os.listdir(LOCAL_MODELS_DIR)
                 if f.endswith((".pkl", ".joblib", ".h5"))]
        files = sorted(files)
        if files:
            model_path = files[-1]
            try:
                if model_path.endswith(".h5"):
                    from tensorflow.keras.models import load_model as keras_load
                    model = keras_load(model_path)
                    scaler = None
                else:
                    model = joblib.load(model_path)
                    base = os.path.basename(model_path)
                    ts = "".join([p for p in base.split("_") if p.isdigit()])
                    scaler_candidates = [f for f in os.listdir(LOCAL_MODELS_DIR) if "scaler" in f and ts in f]
                    scaler = None
                    if scaler_candidates:
                        scaler = joblib.load(os.path.join(LOCAL_MODELS_DIR, scaler_candidates[-1]))
                return model, scaler, f"local:{os.path.basename(model_path)}"
            except Exception as e:
                raise RuntimeError(f"Failed to load local model: {e}")
    raise RuntimeError("No model found in Hopsworks registry or local models/ folder.")


# -------------------------
# Feature fetching
# -------------------------
def fetch_latest_processed_row():
    project = login_hopsworks()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "time" in df.columns:
        df = df.sort_values("time")
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    latest = df.tail(1).copy().reset_index(drop=True)
    return latest, df


# -------------------------
# Forecast feature generator
# -------------------------
def generate_future_features(latest_row, days=FORECAST_DAYS):
    preferred_hour = None
    if "time" in latest_row.columns:
        try:
            preferred_hour = int(pd.to_datetime(latest_row.loc[0, "time"]).hour)
        except Exception:
            preferred_hour = None
    elif "timestamp" in latest_row.columns:
        try:
            preferred_hour = int(pd.to_datetime(latest_row.loc[0, "timestamp"]).hour)
        except Exception:
            preferred_hour = None

    base_time = pd.Timestamp.now(tz=None).replace(minute=0, second=0, microsecond=0)
    if preferred_hour is not None:
        base_time = base_time.replace(hour=preferred_hour)

    rows = []
    for d in range(days):
        r = latest_row.copy().iloc[0].copy()
        new_time = base_time + pd.Timedelta(days=d)
        if "time" in latest_row.columns:
            r["time"] = new_time
        else:
            r["timestamp"] = new_time
        r["hour"] = int(new_time.hour)
        r["day"] = int(new_time.day)
        r["month"] = int(new_time.month)
        r["weekday"] = int(new_time.weekday())
        r["is_weekend"] = int(r["weekday"] in [5, 6])
        rows.append(r)
    future_df = pd.DataFrame(rows).reset_index(drop=True)
    return future_df


# -------------------------
# Prepare model input & align features
# -------------------------
def prepare_model_input(df_rows, model):
    df = df_rows.copy()
    for c in DROP_AQI_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("int64") // 10**9

    df_num = df.select_dtypes(include=["number", "bool"]).copy().fillna(0)

    feature_order = None
    if hasattr(model, "feature_name_"):
        feature_order = list(model.feature_name_)
    else:
        try:
            if hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
                feature_order = list(model.booster_.feature_name)
        except Exception:
            feature_order = None

    if feature_order:
        feat_intersection = [f for f in feature_order if f in df_num.columns]
        df_num = df_num.reindex(columns=feat_intersection, fill_value=0)

    return df_num, df_num.columns.tolist()


# -------------------------
# SHAP computation & plots
# -------------------------
def compute_shap_and_return_paths(model, X_df):
    plots = {}
    tmp = tempfile.mkdtemp(prefix="shap_")
    try:
        shap_values = None
        explainer = None
        # Prefer shap.Explainer (general)
        try:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_df)
        except Exception:
            # Try TreeExplainer for tree models
            try:
                explainer = shap.TreeExplainer(model, check_additivity=False)
                shap_values = explainer(X_df)
            except Exception:
                # Fallback to KernelExplainer (slow) with small background
                bg = X_df.sample(min(50, len(X_df)), random_state=0)
                explainer = shap.KernelExplainer(lambda x: model.predict(x), bg)
                shap_values = explainer.shap_values(X_df, nsamples=100)

        # Bar plot
        fig = plt.figure(figsize=(8, 4))
        try:
            shap.plots.bar(shap_values, show=False)
        except Exception:
            arr = np.abs(shap_values.values) if hasattr(shap_values, "values") else np.abs(np.array(shap_values))
            mean_abs = arr.mean(axis=0)
            idx = np.argsort(-mean_abs)
            names = np.array(X_df.columns)[idx]
            vals = mean_abs[idx]
            plt.barh(names[:20][::-1], vals[:20][::-1])
            plt.title("Mean |SHAP value| (approx)")
        plt.tight_layout()
        bar_path = os.path.join(tmp, "shap_bar.png")
        fig.savefig(bar_path, dpi=200)
        plt.close(fig)
        plots["bar"] = bar_path

        # Waterfalls per instance
        waterfall_paths = []
        for i in range(len(X_df)):
            try:
                fig_w = plt.figure(figsize=(8, 4))
                shap.plots.waterfall(shap_values[i], show=False)
                wpath = os.path.join(tmp, f"shap_waterfall_{i}.png")
                plt.tight_layout()
                fig_w.savefig(wpath, dpi=200)
                plt.close(fig_w)
                waterfall_paths.append(wpath)
            except Exception:
                vals = np.asarray(shap_values.values[i]) if hasattr(shap_values, "values") else np.asarray(shap_values[i])
                names = X_df.columns
                idx_sorted = np.argsort(-np.abs(vals))
                fig_w = plt.figure(figsize=(8, 4))
                plt.barh(names[idx_sorted][:20][::-1], vals[idx_sorted][:20][::-1])
                plt.title(f"SHAP contributions (approx) - instance {i}")
                wpath = os.path.join(tmp, f"shap_waterfall_{i}_fallback.png")
                fig_w.savefig(wpath, dpi=200)
                plt.close(fig_w)
                waterfall_paths.append(wpath)
        plots["waterfalls"] = waterfall_paths
    except Exception:
        return {}
    return plots


# -------------------------
# AQI label
# -------------------------
def get_aqi_label(v):
    v = float(v)
    if v <= 50:
        return "Good"
    if v <= 100:
        return "Moderate"
    if v <= 150:
        return "Unhealthy (for Sensitive people)"
    if v <= 200:
        return "Unhealthy"
    if v <= 300:
        return "Very Unhealthy"
    return "Hazardous"


# -------------------------
# Main predict workflow (invoked by UI)
# -------------------------
def run_forecast_and_explain():
    out = {
        "status": "",
        "model_hint": "",
        "forecast_df": None,
        "history_plot": None,  # plotly figure
        "shap_bar": None,
        "shap_waterfalls": [],
        "error": None
    }

    try:
        model, scaler, model_source = load_latest_model()
        out["model_hint"] = model_source
    except Exception as e:
        out["error"] = f"Model load failed: {e}"
        return out

    try:
        latest_row, full_df = fetch_latest_processed_row()
    except Exception as e:
        out["error"] = f"Feature fetch failed: {e}"
        return out

    future_df = generate_future_features(latest_row, days=FORECAST_DAYS)

    X_future_df, feature_names = prepare_model_input(future_df, model)

    if scaler is not None:
        try:
            X_future_np = scaler.transform(X_future_df)
            X_future = pd.DataFrame(X_future_np, columns=X_future_df.columns)
        except Exception:
            X_future = X_future_df.copy()
    else:
        X_future = X_future_df.copy()

    X_future = X_future.select_dtypes(include=["number", "bool"]).fillna(0)

    try:
        if hasattr(model, "predict"):
            preds = model.predict(X_future)
        else:
            preds = model.predict(X_future.values).flatten()
    except Exception as e:
        out["error"] = f"Prediction failed: {e}"
        return out

    time_col = "time" if "time" in future_df.columns else "timestamp"
    future_df = future_df.reset_index(drop=True)
    future_df["predicted_AQI"] = np.round(preds, 2)
    future_df["AQI_Label"] = future_df["predicted_AQI"].apply(get_aqi_label)

    # history + forecast plot
    try:
        hist = full_df.copy()
        if "time" in hist.columns:
            hist["time"] = pd.to_datetime(hist["time"])
            hist_plot = hist.set_index("time")[["aqi"]].tail(7 * 24).reset_index().rename(columns={"time": "datetime"})
        elif "timestamp" in hist.columns:
            hist["timestamp"] = pd.to_datetime(hist["timestamp"])
            hist_plot = hist.set_index("timestamp")[["aqi"]].tail(7 * 24).reset_index().rename(columns={"timestamp": "datetime"})
        else:
            hist_plot = pd.DataFrame({"datetime": pd.date_range(end=pd.Timestamp.now(), periods=len(hist)), "aqi": hist["aqi"].values})

        fc = future_df.copy()
        fc[time_col] = pd.to_datetime(fc[time_col])
        fc_plot = fc[[time_col, "predicted_AQI"]].rename(columns={time_col: "datetime"})
        combined = pd.concat([hist_plot.rename(columns={"aqi": "value"}), fc_plot.rename(columns={"predicted_AQI": "value"})], ignore_index=True)
        combined = combined.sort_values("datetime")
        fig = px.line(combined, x="datetime", y="value", title="AQI (history + forecast)", markers=True)
        fig.update_layout(xaxis_title="Datetime", yaxis_title="AQI")
        out["history_plot"] = fig
    except Exception:
        out["history_plot"] = None

    # SHAP
    try:
        plots = compute_shap_and_return_paths(model, X_future)
        if plots.get("bar"):
            out["shap_bar"] = plots["bar"]
        if plots.get("waterfalls"):
            out["shap_waterfalls"] = plots["waterfalls"]
    except Exception as e:
        out["shap_error"] = str(e)

    out["status"] = "ok"
    out["forecast_df"] = future_df
    return out


# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="Pearls AQI Predictor — Live Dashboard") as demo:
    gr.Markdown("## 🌍 Pearls AQI Predictor — Live (Hopsworks)")
    gr.Markdown("Automatically uses the latest processed features + latest registered model to forecast AQI and explain predictions with SHAP.")

    with gr.Row():
        with gr.Column(scale=3):
            history_plot = gr.Plot(label="AQI trend (recent + forecast)")
            forecast_table = gr.Dataframe(headers=["datetime", "predicted_AQI", "AQI_Label"], interactive=False, label="3-day Forecast")
            alert_box = gr.Markdown("")
        with gr.Column(scale=2):
            model_info = gr.Textbox(label="Model source", interactive=False)
            shap_bar_img = gr.Image(type="filepath", label="SHAP — mean |value| importance")
            shap_waterfalls_gallery = gr.Gallery(label="Per-day SHAP contributions", columns=1)

    with gr.Row():
        btn_refresh = gr.Button("Reload model & data", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False)

    # handler returns outputs (order must match outputs list)
    def _load_and_render():
        res = run_forecast_and_explain()
        if res.get("error"):
            status = "ERROR: " + res["error"]
            # return placeholders
            return None, pd.DataFrame([]), "", None, [], status

        model_hint = res.get("model_hint", "unknown")
        status = "OK — loaded latest model & data"

        # plotly fig or None
        history_fig = res.get("history_plot")

        # forecast table -> formatted DataFrame
        df = res.get("forecast_df")
        if df is not None:
            time_col = "time" if "time" in df.columns else "timestamp"
            df_out = pd.DataFrame({
                "datetime": pd.to_datetime(df[time_col]).dt.strftime("%Y-%m-%d %H:%M"),
                "predicted_AQI": df["predicted_AQI"],
                "AQI_Label": df["AQI_Label"]
            })
        else:
            df_out = pd.DataFrame([])

        shap_bar = res.get("shap_bar")
        shap_waterfalls = res.get("shap_waterfalls", [])

        return history_fig, df_out, model_hint, shap_bar, shap_waterfalls, status

    # wire up events
    btn_refresh.click(fn=_load_and_render, inputs=None,
                      outputs=[history_plot, forecast_table, model_info, shap_bar_img, shap_waterfalls_gallery, status_box])

    # load once at startup
    demo.load(fn=_load_and_render, outputs=[history_plot, forecast_table, model_info, shap_bar_img, shap_waterfalls_gallery, status_box])

# Launch if run as script
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
