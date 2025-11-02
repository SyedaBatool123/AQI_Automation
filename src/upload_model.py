import hopsworks, joblib, os, json, shutil

from datetime import datetime
from config import HOPSWORKS_PROJECT, DERIVED_FG_NAME

MODEL_LOCAL_PATH = "models/AQI_RF_regressor.joblib"
MODEL_PACKAGE_DIR = "model_package"

def main():

    project = hopsworks.login()
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    print("✅ Connected to Hopsworks")

    # 2. ensure model file exists
    if not os.path.exists(MODEL_LOCAL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_LOCAL_PATH}")

    # 3. Prepare package folder
    if os.path.exists(MODEL_PACKAGE_DIR):
        # remove old package if exists
        import shutil
        shutil.rmtree(MODEL_PACKAGE_DIR)
    os.makedirs(MODEL_PACKAGE_DIR, exist_ok=True)

    # copy model and minimal metadata
    shutil.copy(MODEL_LOCAL_PATH, os.path.join(MODEL_PACKAGE_DIR, "model.joblib"))

    # optional: save metadata (metrics, features list, commit hash)
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_file": "model.joblib",
    }
    with open(os.path.join(MODEL_PACKAGE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    # 4. Register model (python models)
    model = mr.python.create_model(name="aqi_rf_regressor", description="RF regressor for AQI")
    model.save(MODEL_PACKAGE_DIR)
    print("✅ Model uploaded to Hopsworks Model Registry:", model.name, "version:", model.version)

if __name__ == "__main__":
    main()
