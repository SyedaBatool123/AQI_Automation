import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

# --- Load new corrected raw CSV
data = pd.read_csv("data/raw/historical_air_weather_data_karachi.csv")

# --- Convert time and create ID key
data["time"] = pd.to_datetime(data["time"])
data["id"] = data["time"].astype(str)

# --- Create new feature group (version 2)
raw_fg = fs.create_feature_group(
    name="air_quality_raw",
    version=2,  # ✅ new version for corrected data
    primary_key=["id"],
    description="Corrected raw air quality data with pollutant and weather readings",
    online_enabled=True
)

# --- Insert corrected data
raw_fg.insert(data)
print("Raw data uploaded successfully as version 2!")
