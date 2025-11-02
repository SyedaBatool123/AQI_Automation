import requests
import pandas as pd

latitude = 24.8607  
longitude = 67.0011
start_date = "2024-10-15"
end_date = "2025-10-15"

# Air Quality data
air_url = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"
)

# Weather data
weather_url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
    "apparent_temperature,precipitation,pressure_msl,"
    "cloud_cover,wind_speed_10m,wind_direction_10m"
)

# === Fetch both ===
print("Fetching air quality data...")
air_data = requests.get(air_url).json()

print("Fetching weather data...")
weather_data = requests.get(weather_url).json()

# Convert to DataFrames 
df_air = pd.DataFrame(air_data["hourly"])
df_weather = pd.DataFrame(weather_data["hourly"])

# === Merge both on 'time' ===
df_air["time"] = pd.to_datetime(df_air["time"])
df_weather["time"] = pd.to_datetime(df_weather["time"])
df = pd.merge(df_air, df_weather, on="time", how="inner")

# === Save combined data in main folder ===
df.to_csv("historical_air_weather_data_karachi.csv", index=False)

print("Combined Air + Weather data saved as 'historical_air_weather_data_karachi.csv'")
print(df.head(3))
print("\nTotal records:", len(df))
