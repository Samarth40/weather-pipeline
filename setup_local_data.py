import os
import io
from datetime import datetime
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests

from local_storage_adapter import get_blob_service_client

def populate_local_environment():
    print("--- Setting up Local Weather Data from Open-Meteo API ---")
    blob_service = get_blob_service_client()
    
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    variables = [
        "temperature_2m_mean", "shortwave_radiation_sum", "cloud_cover_mean",
        "dew_point_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean",
        "surface_pressure_mean", "wind_gusts_10m_mean", "wet_bulb_temperature_2m_mean",
        "daylight_duration", "sunshine_duration", "snowfall_sum", "rain_sum"
    ]
    
    # 1. Fetch historical monthly dataset 1 (2023)
    print("Fetching historical weather dataset (2023)...")
    url_hist = "https://archive-api.open-meteo.com/v1/archive"
    params_hist1 = {
        "latitude": 40.4165,
        "longitude": -3.7026,
        "start_date": "2023-01-01",
        "end_date": "2023-06-30",
        "daily": variables
    }
    resp1 = openmeteo.weather_api(url_hist, params=params_hist1)[0]
    daily1 = resp1.Daily()
    
    data1 = {"date": pd.date_range(
        start=pd.to_datetime(daily1.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily1.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily1.Interval()),
        inclusive="left"
    )}
    for idx, var in enumerate(variables):
        data1[var] = daily1.Variables(idx).ValuesAsNumpy()
    df_hist1 = pd.DataFrame(data1)
    
    # Upload to local weather-data as weather_data_20230630.csv
    client_hist1 = blob_service.get_blob_client("weather-data", "weather_data_20230630.csv")
    csv_buf1 = io.StringIO()
    df_hist1.to_csv(csv_buf1, index=False)
    client_hist1.upload_blob(csv_buf1.getvalue())
    print(f"Saved weather_data_20230630.csv ({len(df_hist1)} rows)")

    # 2. Fetch historical monthly dataset 2 (2024 - recent)
    print("Fetching historical weather dataset (2024)...")
    params_hist2 = {
        "latitude": 40.4165,
        "longitude": -3.7026,
        "start_date": "2024-01-01",
        "end_date": "2024-06-30",
        "daily": variables
    }
    resp2 = openmeteo.weather_api(url_hist, params=params_hist2)[0]
    daily2 = resp2.Daily()
    data2 = {"date": pd.date_range(
        start=pd.to_datetime(daily2.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily2.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily2.Interval()),
        inclusive="left"
    )}
    for idx, var in enumerate(variables):
        data2[var] = daily2.Variables(idx).ValuesAsNumpy()
    df_hist2 = pd.DataFrame(data2)
    
    # Upload to local weather-data as weather_data_20240630.csv
    client_hist2 = blob_service.get_blob_client("weather-data", "weather_data_20240630.csv")
    csv_buf2 = io.StringIO()
    df_hist2.to_csv(csv_buf2, index=False)
    client_hist2.upload_blob(csv_buf2.getvalue())
    print(f"Saved weather_data_20240630.csv ({len(df_hist2)} rows)")

    # 3. Fetch current 14-day forecast for inference
    print("Fetching 14-day forecast for inference...")
    url_forecast = "https://api.open-meteo.com/v1/forecast"
    params_forecast = {
        "latitude": 40.4165,
        "longitude": -3.7026,
        "daily": variables,
        "forecast_days": 14
    }
    resp_fc = openmeteo.weather_api(url_forecast, params=params_forecast)[0]
    daily_fc = resp_fc.Daily()
    data_fc = {"date": pd.date_range(
        start=pd.to_datetime(daily_fc.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily_fc.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily_fc.Interval()),
        inclusive="left"
    )}
    for idx, var in enumerate(variables):
        data_fc[var] = daily_fc.Variables(idx).ValuesAsNumpy()
    df_forecast = pd.DataFrame(data_fc)
    
    today_str = datetime.now().strftime("%Y%m%d")
    forecast_blob_name = f"weather_data_{today_str}.csv"
    client_forecast = blob_service.get_blob_client("raw-daily-weather-data", forecast_blob_name)
    csv_buf_fc = io.StringIO()
    df_forecast.to_csv(csv_buf_fc, index=False)
    client_forecast.upload_blob(csv_buf_fc.getvalue())
    print(f"Saved raw-daily-weather-data/{forecast_blob_name} ({len(df_forecast)} rows)")

    print("\n[SUCCESS] Local datasets created in ./local_data!")
    return forecast_blob_name

if __name__ == "__main__":
    populate_local_environment()
