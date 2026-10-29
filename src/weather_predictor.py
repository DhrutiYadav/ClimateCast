import requests
import pandas as pd
import joblib

from pathlib import Path

# Load the trained Random Forest model

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "weather_random_forest.pkl"

final_model = joblib.load(MODEL_PATH)


# Convert weather codes into our training conditions
def weather_condition(code):
    if code == 0:
        return "Clear"
    elif code == 1:
        return "Mainly Clear"
    elif code == 2:
        return "Partly Cloudy"
    elif code == 3:
        return "Overcast"
    elif code in [51, 53, 55]:
        return "Drizzle"
    elif code in [61, 63, 65]:
        return "Rain"
    else:
        return "Clear"


def predict_weather(location):

    # 1. Convert location to coordinates
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"

    geocoding_params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json",
    }

    geocoding_response = requests.get(geocoding_url, params=geocoding_params)

    location_data = geocoding_response.json()

    if "results" not in location_data:
        return None

    latitude = location_data["results"][0]["latitude"]
    longitude = location_data["results"][0]["longitude"]
    location_name = location_data["results"][0]["name"]

    # 2. Get today's weather
    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
        ],
        "forecast_days": 1,
        "timezone": "auto",
    }

    weather_response = requests.get(weather_url, params=weather_params)

    # 3. Convert hourly weather data into daily features
    hourly_data = weather_response.json()["hourly"]

    today_df = pd.DataFrame(hourly_data)

    today_df["time"] = pd.to_datetime(today_df["time"])

    today_features = pd.DataFrame(
        {
            "temperature_2m (°C)": [today_df["temperature_2m"].mean()],
            "relative_humidity_2m (%)": [today_df["relative_humidity_2m"].mean()],
            "precipitation (mm)": [today_df["precipitation"].sum()],
            "surface_pressure (hPa)": [today_df["surface_pressure"].mean()],
            "cloud_cover (%)": [today_df["cloud_cover"].mean()],
            "wind_speed_10m (km/h)": [today_df["wind_speed_10m"].mean()],
        }
    )

    # Convert weather codes into conditions
    today_conditions = today_df["weather_code"].apply(weather_condition)

    today_features["weather_condition"] = today_conditions.mode()[0]

    # Add month
    today_features["month"] = today_df["time"].dt.month.iloc[0]

    # 4. Make prediction
    prediction = final_model.predict(today_features)

    probabilities = final_model.predict_proba(today_features)

    confidence = probabilities.max() * 100

    # 5. Return result
    return {
        "location": location_name,
        "prediction": prediction[0],
        "confidence": round(confidence, 2),
    }
