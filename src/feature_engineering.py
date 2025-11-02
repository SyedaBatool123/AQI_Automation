import hopsworks
import pandas as pd
import numpy as np
import os

project = hopsworks.login()
fs = project.get_feature_store()

raw_csv_path = "data/raw/historical_air_weather_data_karachi.csv"
data = pd.read_csv(raw_csv_path)

data["time"] = pd.to_datetime(data["time"])
data["id"] = data["time"].astype(str)

# Handle missing values
data.fillna(method="ffill", inplace=True)
data.fillna(method="bfill", inplace=True)

# Pollutants list
pollutant_cols = ["pm2_5","pm10","carbon_monoxide","nitrogen_dioxide","sulphur_dioxide","ozone"]
for col in pollutant_cols:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())

# Time based features
data["hour"] = data["time"].dt.hour
data["day"] = data["time"].dt.day
data["month"] = data["time"].dt.month
data["weekday"] = data["time"].dt.weekday
data["is_weekend"] = data["weekday"].isin([5,6]).astype(int)


# Pollutant derived features
for col in pollutant_cols:
    if col in data.columns:
        data[f"{col}_roll3"] = data[col].rolling(window=3, min_periods=1).mean()
        data[f"{col}_diff"] = data[col].diff().fillna(0)

# EPA AQI calculation function
def calc_aqi(C, breakpoints):
    for bp in breakpoints:
        if bp["C_low"] <= C <= bp["C_high"]:
            I = ((bp["I_high"] - bp["I_low"]) / (bp["C_high"] - bp["C_low"])) * (C - bp["C_low"]) + bp["I_low"]
            return round(I)
    return None

# EPA breakpoints for each pollutant
# Units: PM2.5 & PM10 µg/m³, gases ppb
epa_breakpoints = {
    "pm2_5": [
        {"C_low":0.0,"C_high":12.0,"I_low":0,"I_high":50},
        {"C_low":12.1,"C_high":35.4,"I_low":51,"I_high":100},
        {"C_low":35.5,"C_high":55.4,"I_low":101,"I_high":150},
        {"C_low":55.5,"C_high":150.4,"I_low":151,"I_high":200},
        {"C_low":150.5,"C_high":250.4,"I_low":201,"I_high":300},
        {"C_low":250.5,"C_high":350.4,"I_low":301,"I_high":400},
        {"C_low":350.5,"C_high":500.4,"I_low":401,"I_high":500}
    ],
    "pm10": [
        {"C_low":0,"C_high":54,"I_low":0,"I_high":50},
        {"C_low":55,"C_high":154,"I_low":51,"I_high":100},
        {"C_low":155,"C_high":254,"I_low":101,"I_high":150},
        {"C_low":255,"C_high":354,"I_low":151,"I_high":200},
        {"C_low":355,"C_high":424,"I_low":201,"I_high":300},
        {"C_low":425,"C_high":504,"I_low":301,"I_high":400},
        {"C_low":505,"C_high":604,"I_low":401,"I_high":500}
    ],
    "carbon_monoxide": [
        {"C_low":0.0,"C_high":4.4,"I_low":0,"I_high":50},
        {"C_low":4.5,"C_high":9.4,"I_low":51,"I_high":100},
        {"C_low":9.5,"C_high":12.4,"I_low":101,"I_high":150},
        {"C_low":12.5,"C_high":15.4,"I_low":151,"I_high":200},
        {"C_low":15.5,"C_high":30.4,"I_low":201,"I_high":300},
        {"C_low":30.5,"C_high":40.4,"I_low":301,"I_high":400},
        {"C_low":40.5,"C_high":50.4,"I_low":401,"I_high":500}
    ],
    "nitrogen_dioxide": [
        {"C_low":0,"C_high":53,"I_low":0,"I_high":50},
        {"C_low":54,"C_high":100,"I_low":51,"I_high":100},
        {"C_low":101,"C_high":360,"I_low":101,"I_high":150},
        {"C_low":361,"C_high":649,"I_low":151,"I_high":200},
        {"C_low":650,"C_high":1249,"I_low":201,"I_high":300},
        {"C_low":1250,"C_high":1649,"I_low":301,"I_high":400},
        {"C_low":1650,"C_high":2049,"I_low":401,"I_high":500}
    ],
    "sulphur_dioxide": [
        {"C_low":0,"C_high":35,"I_low":0,"I_high":50},
        {"C_low":36,"C_high":75,"I_low":51,"I_high":100},
        {"C_low":76,"C_high":185,"I_low":101,"I_high":150},
        {"C_low":186,"C_high":304,"I_low":151,"I_high":200},
        {"C_low":305,"C_high":604,"I_low":201,"I_high":300},
        {"C_low":605,"C_high":804,"I_low":301,"I_high":400},
        {"C_low":805,"C_high":1004,"I_low":401,"I_high":500}
    ],
    "ozone": [
        {"C_low":0,"C_high":54,"I_low":0,"I_high":50},
        {"C_low":55,"C_high":70,"I_low":51,"I_high":100},
        {"C_low":71,"C_high":85,"I_low":101,"I_high":150},
        {"C_low":86,"C_high":105,"I_low":151,"I_high":200},
        {"C_low":106,"C_high":200,"I_low":201,"I_high":300},
        {"C_low":201,"C_high":300,"I_low":301,"I_high":400},
        {"C_low":301,"C_high":400,"I_low":401,"I_high":500}
    ]
}

# Compute AQI per pollutant
for pollutant in pollutant_cols:
    if pollutant in data.columns:
        data[f"aqi_{pollutant}"] = data[pollutant].apply(lambda x: calc_aqi(x, epa_breakpoints[pollutant]))

aqi_cols = [col for col in data.columns if col.startswith("aqi_")]
data["aqi"] = data[aqi_cols].max(axis=1)

os.makedirs("data/processed", exist_ok=True)
data.to_csv("data/processed/air_quality_processed.csv", index=False)
print(" Local CSV saved: data/processed/air_quality_processed.csv")

#  Insert into Hopsworks Feature Store

#processed_fg = fs.create_feature_group(
    #name="air_quality_processed",
    #version=2,
    #primary_key=["id"],
    #description="Processed features for AQI prediction (Full EPA standard)",
    #online_enabled=True
#)
#processed_fg.insert(data)
#print(" Processed features uploaded to Hopsworks successfully!")

# Instead of creating every time:

processed_fg = fs.get_feature_group("air_quality_processed", version=2)
processed_fg.insert(data, write_options={"upsert": True})
