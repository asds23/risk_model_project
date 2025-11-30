"""
Файл: data_fetching_and_risk_mapping.py
Данные MET.NO → расширенная система рисков (0–10)
"""

import requests
from datetime import datetime


# ============================================================
# 1. Получение данных MET.NO
# ============================================================

def fetch_weather(lat: float, lon: float):
    url = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
    headers = {"User-Agent": "RiskModel/1.0 your_email@example.com"}
    params = {"lat": lat, "lon": lon}

    try:
        r = requests.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Ошибка MET.NO:", e)
        return None


# ============================================================
# 2. Маппинг параметров → частичные риски
# ============================================================

def normalize(value, min_v, max_v):
    """ Нормализация в диапазон 0–10 """
    if value <= min_v: return 0
    if value >= max_v: return 10
    return round((value - min_v) / (max_v - min_v) * 10, 2)


def map_weather_to_risk(weather_data):
    if weather_data is None:
        return {"total_risk": 7, "details": {"error": True}}

    ts = weather_data["properties"]["timeseries"][0]["data"]

    instant = ts["instant"]["details"]
    next1 = ts.get("next_1_hours", {})
    next6 = ts.get("next_6_hours", {})
    next12 = ts.get("next_12_hours", {})

    # Достаём значения
    wind_speed = instant.get("wind_speed", 0)
    wind_gust = instant.get("wind_speed_of_gust", 0)
    wind_dir = instant.get("wind_from_direction", 0)

    temp = instant.get("air_temperature", 0)
    dew_point = instant.get("dew_point_temperature", 0)
    humidity = instant.get("relative_humidity", 0)
    fog = instant.get("fog_area_fraction", 0)
    clouds = instant.get("cloud_area_fraction", 0)
    pressure = instant.get("air_pressure_at_sea_level", 1010)

    precipitation_1h = next1.get("details", {}).get("precipitation_amount", 0)
    precipitation_6h = next6.get("details", {}).get("precipitation_amount", 0)
    precipitation_12h = next12.get("details", {}).get("precipitation_amount", 0)

    prob_rain = next6.get("details", {}).get("probability_of_precipitation", 0)
    prob_thunder = next12.get("details", {}).get("probability_of_thunder", 0)
    uv_index = next12.get("details", {}).get("ultraviolet_index_clear_sky", 0)

    wave_height = next12.get("details", {}).get("significant_wave_height", 0)
    wave_period = next12.get("details", {}).get("wave_period", 0)
    wave_dir = next12.get("details", {}).get("wave_direction", 0)
    sea_temp = next12.get("details", {}).get("sea_water_temperature", 0)

    # ============================================================
    # 3. Расчёт частичных рисков
    # ============================================================

    risks = {
        "wind_speed_risk": normalize(wind_speed, 0, 25),
        "wind_gust_risk": normalize(wind_gust, 0, 35),
        "fog_risk": normalize(fog, 0, 100),
        "humidity_risk": normalize(humidity, 20, 100),
        "cloud_risk": normalize(clouds, 0, 100),
        "temperature_risk": normalize(abs(temp), 0, 35),
        "dew_point_risk": normalize(abs(temp - dew_point), 0, 10),
        "pressure_risk": normalize(1015 - (pressure - 960), 0, 55),

        "precipitation_1h_risk": normalize(precipitation_1h, 0, 10),
        "precipitation_6h_risk": normalize(precipitation_6h, 0, 30),
        "precipitation_12h_risk": normalize(precipitation_12h, 0, 50),

        "prob_rain_risk": normalize(prob_rain, 0, 100),
        "prob_thunder_risk": normalize(prob_thunder, 0, 100),

        "uv_risk": normalize(uv_index, 0, 11),

        "wave_height_risk": normalize(wave_height, 0, 12),
        "wave_period_risk": normalize(wave_period, 0, 15),
        "sea_temp_risk": normalize(abs(20 - sea_temp), 0, 25)
    }

    # ============================================================
    # 4. Интегральный риск по формуле со взвешиванием
    # ============================================================

    weights = {
        "wind_speed_risk": 0.15,
        "wind_gust_risk": 0.1,
        "fog_risk": 0.08,
        "humidity_risk": 0.05,
        "cloud_risk": 0.03,
        "temperature_risk": 0.07,
        "dew_point_risk": 0.05,
        "pressure_risk": 0.05,

        "precipitation_1h_risk": 0.1,
        "precipitation_6h_risk": 0.05,
        "precipitation_12h_risk": 0.05,

        "prob_rain_risk": 0.06,
        "prob_thunder_risk": 0.08,

        "uv_risk": 0.02,

        "wave_height_risk": 0.04,
        "wave_period_risk": 0.02,
        "sea_temp_risk": 0.0   # почти не влияет
    }

    total_risk = round(sum(risks[k] * weights[k] for k in risks), 2)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_risk": total_risk,
        "details": risks
    }


# ============================================================
# 5. Основная функция
# ============================================================

def get_point_risk(lat: float, lon: float):
    weather = fetch_weather(lat, lon)
    risk = map_weather_to_risk(weather)

    return {
        "coords": (lat, lon),
        "weather_raw": weather,
        "risk": risk
    }


# ============================================================
# Тест
# ============================================================
if __name__ == "__main__":
    print(get_point_risk(55.7558, 37.6176))
