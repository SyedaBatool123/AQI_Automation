import os
import pandas as pd
import numpy as np
import requests
import hopsworks
from datetime import datetime

print(" Starting Automated Feature Pipeline...")

project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
fs = project.get_feature_store()

# Fetch Real-Time Data
def fetch_openmeteo_data():
    print(" Fetching Air Quality and Weather Data...")

    aq_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality?"
        "latitude=24.8607&longitude=67.0011&"
        "hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
    )

    weather_url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=24.8607&longitude=67.0011&"
        "hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
        "apparent_temperature,precipitation,pressure_msl,cloud_cover,"
        "wind_speed_10m,wind_direction_10m"
    )

    aq_resp = requests.get(aq_url)
    weather_resp = requests.get(weather_url)

    aq_resp.raise_for_status()
    weather_resp.raise_for_status()

    aq_data = aq_resp.json()
    weather_data = weather_resp.json()

    df_aq = pd.DataFrame(aq_data["hourly"])
    df_weather = pd.DataFrame(weather_data["hourly"])

    df = pd.merge(df_aq, df_weather, on="time", how="inner")
    print(f" Combined Data Shape: {df.shape}")
    return df

# AQI Calculation
epa_breakpoints = {
    "pm2_5": [{"C_low": 0, "C_high": 12, "I_low": 0, "I_high": 50},
              {"C_low": 12.1, "C_high": 35.4, "I_low": 51, "I_high": 100}],
    "pm10": [{"C_low": 0, "C_high": 54, "I_low": 0, "I_high": 50},
             {"C_low": 55, "C_high": 154, "I_low": 51, "I_high": 100}],
    "carbon_monoxide": [{"C_low": 0, "C_high": 4.4, "I_low": 0, "I_high": 50}],
    "nitrogen_dioxide": [{"C_low": 0, "C_high": 53, "I_low": 0, "I_high": 50}],
    "sulphur_dioxide": [{"C_low": 0, "C_high": 35, "I_low": 0, "I_high": 50}],
    "ozone": [{"C_low": 0, "C_high": 54, "I_low": 0, "I_high": 50}],
}

def calc_aqi(concentration, pollutant_bp):
    if pd.isna(concentration):
        return np.nan
    for bp in pollutant_bp:
        if bp["C_low"] <= concentration <= bp["C_high"]:
            return ((bp["I_high"] - bp["I_low"]) /
                    (bp["C_high"] - bp["C_low"])) * (concentration - bp["C_low"]) + bp["I_low"]
    return np.nan


df = fetch_openmeteo_data()
df["time"] = pd.to_datetime(df["time"])

df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["weekday"] = df["time"].dt.weekday
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

pollutants = ["pm2_5", "pm10", "carbon_monoxide",
              "nitrogen_dioxide", "sulphur_dioxide", "ozone"]

for col in pollutants:
    df[f"{col}_roll3"] = df[col].rolling(3, min_periods=1).mean()
    df[f"{col}_diff"] = df[col].diff()

for pollutant in pollutants:
    df[f"aqi_{pollutant}"] = df[pollutant].apply(lambda x: calc_aqi(x, epa_breakpoints[pollutant]))

df["aqi"] = df[[f"aqi_{p}" for p in pollutants]].max(axis=1)

# Replace dropna with fillna
df = df.fillna(method='ffill').fillna(method='bfill')
df = df.reset_index(drop=True)

print(f" Processed Data Ready: {df.shape}")
print(df.head(2).T)

# Insert into Feature Group
df["is_weekend"] = df["is_weekend"].astype(int)
df["id"] = df.index.astype(str)

#  Ensure correct data types for AQI columns
if "aqi_pm2_5" in df.columns:
    df["aqi_pm2_5"] = df["aqi_pm2_5"].fillna(0).astype("int64")
if "aqi_ozone" in df.columns:
    df["aqi_ozone"] = df["aqi_ozone"].fillna(0).astype("int64")

try:
    fg = fs.get_feature_group("air_quality_processed", version=2)
    fg.insert(df, write_options={"wait_for_job": False})
    print(" Data successfully inserted into existing Feature Group (version 2).")
except Exception as e:
    print(f" Error inserting data: {e}")
    print(" Trying to recreate Feature Group...")

    #  Only recreate if it truly doesn’t exist
    try:
        fg = fs.create_feature_group(
            name="air_quality_processed",
            version=2,
            description="Automated hourly AQI + weather features (v2)",
            primary_key=["id", "time"],
            online_enabled=True
        )
        fg.insert(df, write_options={"wait_for_job": False})
        print(" Feature group recreated & data inserted successfully.")
    except Exception as e2:
        print(f" Still failed to insert or recreate: {e2}")

print(" Automated Feature Pipeline completed successfully!")
