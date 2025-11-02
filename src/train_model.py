import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hopsworks
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import json

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:15s} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def build_ffnn(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation="relu"),
        Dropout(0.25),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=False),
        Dropout(0.25),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="mse", metrics=["mae"])
    return model


def make_sequences(X_df, y_series, seq_len=24):
    X_vals = X_df.values.astype("float32")
    y_vals = y_series.values.astype("float32")
    X_seq, y_seq = [], []
    for i in range(len(X_vals) - seq_len):
        X_seq.append(X_vals[i:i+seq_len])
        y_seq.append(y_vals[i+seq_len])
    if len(X_seq) == 0:
        return None, None
    return np.array(X_seq, dtype="float32"), np.array(y_seq, dtype="float32")


def main():
    start = time.time()
    print(" Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()
    processed_fg = fs.get_feature_group(name="air_quality_processed", version=2)
    df = processed_fg.read()
    print(" Fetched processed features shape:", df.shape)

    target_col = "aqi"
    assert target_col in df.columns, "Target column 'aqi' missing"

    drop_cols = ["time", "id",
                 "aqi_pm2_5", "aqi_pm10", "aqi_carbon_monoxide",
                 "aqi_nitrogen_dioxide", "aqi_sulphur_dioxide", "aqi_ozone"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target_col], errors="ignore")
    y = df[target_col].astype("float32")
    X = X.ffill().bfill()
    feature_names = X.columns.tolist()
    print(f" Number of features used: {len(feature_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    results = {}
    best_model = None
    best_score = -np.inf
    best_model_name = None

    ml_models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                random_state=RANDOM_STATE, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.08, num_leaves=31, random_state=RANDOM_STATE)
    }

    print("\n Training ML models...")
    for name, model in ml_models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(name, y_test, y_pred)
            results[name] = metrics
            if metrics["R2"] > best_score:
                best_score = metrics["R2"]
                best_model = model
                best_model_name = name
        except Exception as e:
            print(f"{name} failed: {e}")

    # ---------- Deep Learning: FFNN ----------
    print("\n Training Feedforward NN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ff_model = build_ffnn(X_train_scaled.shape[1])
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    ff_model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, validation_split=0.1, callbacks=[es], verbose=0)

    y_pred_ff = ff_model.predict(X_test_scaled).flatten()
    metrics = evaluate_model("DL_FFNN", y_test, y_pred_ff)
    results["DL_FFNN"] = metrics
    if metrics["R2"] > best_score:
        best_score = metrics["R2"]
        best_model = ff_model
        best_model_name = "DL_FFNN"

    # ---------- Deep Learning: LSTM ----------
    print("\n Preparing sequences for LSTM...")
    seq_len = 24
    X_all_scaled = scaler.transform(X)
    X_all_scaled = pd.DataFrame(X_all_scaled, columns=feature_names)
    X_seq, y_seq = make_sequences(X_all_scaled, y, seq_len=seq_len)

    if X_seq is not None:
        split_idx = int(0.8 * len(X_seq))
        X_train_seq, X_test_seq = X_seq[:split_idx], X_seq[split_idx:]
        y_train_seq, y_test_seq = y_seq[:split_idx], y_seq[split_idx:]

        print(" Training LSTM...")
        lstm_model = build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))
        es_l = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        lstm_model.fit(X_train_seq, y_train_seq, epochs=100, batch_size=64, validation_split=0.1, callbacks=[es_l], verbose=0)

        y_pred_lstm = lstm_model.predict(X_test_seq).flatten()
        metrics = evaluate_model("DL_LSTM", y_test_seq, y_pred_lstm)
        results["DL_LSTM"] = metrics
        if metrics["R2"] > best_score:
            best_score = metrics["R2"]
            best_model = lstm_model
            best_model_name = "DL_LSTM"
    else:
        print(" Not enough rows for LSTM sequences.")

    # ---------- Save best model ----------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    print(f"\n Best model: {best_model_name} (R²={best_score:.4f})")

    if best_model_name.startswith("DL_"):
        local_model_path = os.path.join(models_dir, f"{best_model_name}_{timestamp}.h5")
        best_model.save(local_model_path)
    else:
        local_model_path = os.path.join(models_dir, f"{best_model_name}_{timestamp}.pkl")
        joblib.dump(best_model, local_model_path)

    scaler_path = os.path.join(models_dir, f"scaler_{timestamp}.pkl")
    joblib.dump(scaler, scaler_path)

    print(f" Model saved at: {local_model_path}")
    print(f" Scaler saved at: {scaler_path}")

    # ---------- SHAP Explainability ----------
    print("\n Generating SHAP feature importance plots...")
    try:
        import shap
        import matplotlib.pyplot as plt

        shap_dir = "shap_outputs"
        os.makedirs(shap_dir, exist_ok=True)

        if not best_model_name.startswith("DL_"):
            explainer = shap.TreeExplainer(best_model, check_additivity=False)
            shap_values = explainer(X_test)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            shap_summary_path = os.path.join(shap_dir, f"shap_summary_{best_model_name}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(shap_summary_path, dpi=300)
            plt.close()

            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_values, show=False)
            shap_bar_path = os.path.join(shap_dir, f"shap_bar_{best_model_name}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(shap_bar_path, dpi=300)
            plt.close()

            # Waterfall Plot (first test instance)
            shap.waterfall_plot(shap_values[0], show=False)
            shap_waterfall_path = os.path.join(shap_dir, f"shap_waterfall_{best_model_name}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(shap_waterfall_path, dpi=300)
            plt.close()

            print(f" SHAP plots saved:\n - {shap_summary_path}\n - {shap_bar_path}\n - {shap_waterfall_path}")

        else:
            print("SHAP skipped for deep learning models.")

    except Exception as e:
        print(" SHAP explainability failed:", e)

    # ---------- Register in Hopsworks ----------
    print("\n Registering model in Hopsworks...")
    try:
        mr = project.get_model_registry()
        model_name = f"{best_model_name}_aqi_model"
        model_desc = "Best AQI prediction model trained on hybrid ML/DL pipeline."

        model_meta = mr.python.create_model(
            name=model_name,
            description=model_desc,
            metrics=results[best_model_name],
            model_schema=None
        )

        model_meta.save(local_model_path)
        print(f" Model uploaded to registry: {model_name}")
    except Exception as e:
        print(" Model upload failed:", e)

    # ---------- Save best model info for automation ----------
    print("\n Saving best model info for automation...")
    best_info = {
        "best_model_name": best_model_name,
        "r2_score": best_score,
        "metrics": results[best_model_name],
        "timestamp": timestamp
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/best_model_info.json", "w") as f:
        json.dump(best_info, f, indent=4)
    print(" Best model info saved at artifacts/best_model_info.json")

    # ---------- Summary ----------
    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"{k:12s}: MAE={v['MAE']:.3f}, RMSE={v['RMSE']:.3f}, R²={v['R2']:.3f}")

    print(f"\nTotal time: {time.time() - start:.1f}s Done!")


if __name__ == "__main__":
    main()
